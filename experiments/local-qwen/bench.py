"""Time Qwen3.5-4B on this machine: load, prefill, generate, judge P(YES), one QLoRA step.

usage: python bench.py [nf4|bf16] [prompt_tokens] [gen_tokens]
Needs accelerate, bitsandbytes, peft (and optionally flash-linear-attention) on PYTHONPATH.
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-4B"
mode = sys.argv[1] if len(sys.argv) > 1 else "nf4"
PROMPT_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
GEN_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def gb(x):
    return f"{x / 2**30:.2f} GB"


tok = AutoTokenizer.from_pretrained(MODEL)
kw = dict(dtype=torch.bfloat16, device_map="cuda:0")
if mode == "nf4":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:  # bf16 does not fit in 8 GB, so let accelerate spill layers to CPU RAM
    kw.update(device_map="auto", max_memory={0: "7GiB", "cpu": "24GiB"})

t = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
print(
    f"[{mode}] load {time.perf_counter() - t:.1f}s  class={type(model).__name__}  "
    f"vram={gb(torch.cuda.memory_allocated())}"
)

# Fixed-length prompt from repeated code text, so token counts are exact.
code = "def solve(n, xs):\n    best = 0\n    for i in range(n):\n        best = max(best, xs[i] * i)\n    return best\n"
ids = tok(code * 400, return_tensors="pt").input_ids[:, :PROMPT_TOK].to("cuda:0")

with torch.no_grad():
    model(ids[:, :64])  # warm-up (kernel compile / autotune)
    _, t_pre = sync_time(lambda: model(ids))
    print(f"prefill {PROMPT_TOK} tok: {t_pre:.2f}s ({PROMPT_TOK / t_pre:.0f} tok/s)")

    # Judge-style call: one forward pass, read P(YES) vs P(NO) at the last position.
    yes, no = tok.encode("YES")[0], tok.encode("NO")[0]
    out, t_j = sync_time(lambda: model(ids).logits[0, -1].float().softmax(-1))
    print(
        f"judge P(YES) 1 forward: {t_j:.2f}s  (p_yes={out[yes].item():.3f} p_no={out[no].item():.3f})"
    )

    for bs in (1, 8):
        batch = ids[:, :256].repeat(bs, 1)
        gen, t_g = sync_time(
            lambda: model.generate(
                batch,
                attention_mask=torch.ones_like(batch),
                max_new_tokens=GEN_TOK,
                min_new_tokens=GEN_TOK,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
        )
        n = (gen.shape[1] - batch.shape[1]) * bs
        print(
            f"generate bs={bs}: {n} tok in {t_g:.1f}s ({n / t_g:.1f} tok/s)  "
            f"peak vram={gb(torch.cuda.max_memory_allocated())}"
        )

if mode == "nf4":  # one LoRA train step = what a local sender/judge update would cost
    from peft import LoraConfig, get_peft_model

    # not prepare_model_for_kbit_training: it upcasts to fp32, which fla's kernel rejects
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=32, lora_alpha=32, target_modules="all-linear")
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model.train()
    for bs in (1, 1, 1, 2):  # first step includes triton compile
        x = ids.repeat(bs, 1)
        torch.cuda.reset_peak_memory_stats()
        try:

            def step():
                # PPO only scores the generated half; full-vocab (248k) logits on every
                # position are what blow past 8 GB, so keep logits for the last half only.
                k = PROMPT_TOK // 2
                logits = model(x, logits_to_keep=k + 1).logits[:, :-1]
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1).float(), x[:, -k:].flatten()
                )
                loss.backward()
                opt.step()
                opt.zero_grad()

            _, t_s = sync_time(step)
            print(
                f"QLoRA step bs={bs}x{PROMPT_TOK}: {t_s:.1f}s ({bs * PROMPT_TOK / t_s:.0f} tok/s)  "
                f"peak vram={gb(torch.cuda.max_memory_allocated())}"
            )
        except torch.OutOfMemoryError:
            print(f"QLoRA step bs={bs}x{PROMPT_TOK}: OOM")
            break
