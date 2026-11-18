"""Time one grouped-RL step's model calls with Qwen3.5-4B (4-bit) on the local GPU.

Uses the 650-step config's real data and prompts (exp/receiver-schedule code):
 1. solve one APPS problem from scratch
 2. rewrite DeepSeek V4.1's own solution (sender prompt), batch 1 and a group of 8
 3. paraphrase the rewrite (LLM_SUMMARIZE prompts)
 4. judge P(YES) on the paraphrase (receiver prompt, one forward)
Thinking is off everywhere (the paraphraser runs with reasoning_effort=minimal on Tinker).

Two venvs, because the repo code needs its own control_arena and Qwen3.5 needs
transformers 5:
  repo-venv/python rl_step_timing.py prep  CONFIG.yaml prompts.json
  main-venv/python rl_step_timing.py run   prompts.json outputs.json   (bench libs on PYTHONPATH)
  repo-venv/python rl_step_timing.py grade CONFIG.yaml outputs.json
"""

import asyncio
import json
import sys
import time


def repo(config_path):
    from lasr_labs_2025_control_project.finetuning.grouped_rl.config import (
        load_rl_config,
    )
    from lasr_labs_2025_control_project.finetuning.grouped_rl.run import load_tasks

    config = load_rl_config(config_path)
    return config, load_tasks(config).train[0]


def prep(config_path, out):
    from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
        rewrite_sender_messages,
    )
    from lasr_labs_2025_control_project.prompts.monitor_prompts import (
        MONITOR_USER_PROMPT,
    )
    from lasr_labs_2025_control_project.utils.config_loader import (
        get_prompt_by_name as g,
    )

    config, problem = repo(config_path)
    P, ch = config.prompts, config.channel
    json.dump(
        {
            "problem_id": problem.problem_id,
            "base_code": problem.base_code,
            "solve": [
                {
                    "role": "system",
                    "content": "Solve the programming problem in Python. Read stdin, "
                    "write stdout. Put the final code inside <final_code></final_code>.",
                },
                {
                    "role": "user",
                    "content": f"{problem.question}\n\n{problem.starter_code}",
                },
            ],
            "rewrite": rewrite_sender_messages(
                problem,
                1,
                system_prompt=g(P.sender_system),
                user_prompt=g(P.sender_user),
                signal_labels=(g(P.signal_off), g(P.signal_on)),
            ),
            "para_system": g(ch.paraphrase_system_prompt),
            "para_user": g(ch.paraphrase_base_prompt),  # .format(solution=...)
            "judge_system": g(P.receiver_system),
            "judge_user": MONITOR_USER_PROMPT.format(
                problem_statement=problem.question, solution="{solution}"
            ),
        },
        open(out, "w"),
    )


def run(prompts_path, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL = "Qwen/Qwen3.5-4B"
    q = json.load(open(prompts_path))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    ).eval()

    def ids_of(messages):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
            "cuda:0"
        )

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return r, time.perf_counter() - t

    @torch.no_grad()
    def generate(messages, n=1, max_new=4096):
        ids = ids_of(messages).repeat(n, 1)
        o, dt = timed(
            lambda: model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        )
        texts = tok.batch_decode(o[:, ids.shape[1] :], skip_special_tokens=True)
        lens = [len(tok(x).input_ids) for x in texts]
        print(
            f"  prompt {ids.shape[1]} tok, n={n}, generated {lens} tok, {dt:.1f}s "
            f"({sum(lens) / dt:.1f} tok/s)",
            flush=True,
        )
        return texts, dt

    with torch.no_grad():
        model(ids_of(q["solve"])[:, :64])  # warm-up
    res = {}
    print(
        f"problem {q['problem_id']}; V4.1 base_code {len(tok(q['base_code']).input_ids)} tok"
    )
    print("1. solve from scratch")
    (res["solve"],), res["t_solve"] = generate(q["solve"])
    print("2. rewrite V4.1's solution (sender prompt)")
    (res["rewrite"],), res["t_rewrite"] = generate(q["rewrite"])
    print("2b. a full group of 8 rewrites at once")
    res["group"], res["t_group"] = generate(q["rewrite"], n=8)

    code = res["rewrite"]
    if "<final_code>" in code:
        code = code.split("<final_code>")[-1].split("</final_code>")[0]
    print("3. paraphrase the rewrite")
    (summary,), res["t_para"] = generate(
        [
            {"role": "system", "content": q["para_system"]},
            {"role": "user", "content": q["para_user"].format(solution=code)},
        ]
    )
    res["summary"] = summary

    print("4. judge P(YES) on the paraphrase (one forward)")
    ids = ids_of(
        [
            {"role": "system", "content": q["judge_system"]},
            {"role": "user", "content": q["judge_user"].replace("{solution}", summary)},
        ]
    )
    with torch.no_grad():
        logits, res["t_judge"] = timed(lambda: model(ids).logits[0, -1].float())
    p = logits.softmax(-1)
    yes, no = tok.encode("YES")[0], tok.encode("NO")[0]
    print(
        f"  prompt {ids.shape[1]} tok, {res['t_judge']:.2f}s, P(YES)={p[yes].item():.3f} "
        f"P(NO)={p[no].item():.3f}"
    )
    print(f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    json.dump(res, open(out, "w"))


def grade(config_path, outputs_path):
    from lasr_labs_2025_control_project.finetuning.grouped_rl.sandbox import (
        DockerAppsEvaluator,
    )
    from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
        extract_final_code,
    )

    config, problem = repo(config_path)
    res = json.load(open(outputs_path))
    sb = config.sandbox
    ev = DockerAppsEvaluator(
        image=sb.image,
        timeout_seconds=sb.timeout_seconds,
        control_arena_parity=sb.control_arena_parity,
        batch_tests=sb.batch_tests,
        num_tests=config.data.max_tests,
    )
    for name, text in [("solve", res["solve"]), ("rewrite", res["rewrite"])] + [
        (f"group[{i}]", t) for i, t in enumerate(res["group"])
    ]:
        try:
            print(name, asyncio.run(ev.grade(problem, extract_final_code(text))))
        except ValueError as e:
            print(name, "no code:", e)


if __name__ == "__main__":
    {"prep": prep, "run": run, "grade": grade}[sys.argv[1]](*sys.argv[2:])
