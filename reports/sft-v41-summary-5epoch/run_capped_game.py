"""Rescore only the 5-epoch collusion judge on v2's cached summaries, then run main's game solver.

Everything else (trusted/untrusted suspicion, honest calibration set, game settings)
is v2's saved baseline input, so the only changed quantity is the collusion judge.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

REPO = Path(os.environ["SFT_WORKTREE"])
ICRL = Path("/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/icrl")
OUT = REPO / "experiments/sft-v41-summary-5epoch/game"
OUT.mkdir(parents=True, exist_ok=True)
load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env")

from lasr_labs_2025_control_project.scripts import plot_data as pd_mod  # noqa: E402
from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (  # noqa: E402
    collusion_score,
)
from lasr_labs_2025_control_project.tracking import StageRun  # noqa: E402
from lasr_labs_2025_control_project.utils import tinker_provider as tp  # noqa: E402

BASE = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
CHECKPOINT = "tinker/tinker://f30b7ece-95e4-5344-a67e-fbcdc6f645d2:train:0/sampler_weights/self_rec_paraphrased"
LIMIT = 19.0  # $1 held back under the user's $20 cap
PREFILL, SAMPLE = 0.60 / 1e6, 3.60 / 1e6
REF = json.loads(
    __import__("gzip").decompress(
        (
            ICRL
            / "experiments/apps/game/apps.self_rec.paraphrased.sft-a8781f665021/analysis_data.json.gz"
        ).read_bytes()
    )
)
INPUTS = {k: ICRL / v for k, v in REF["pipeline"]["inputs"].items()}
prior = json.loads((OUT.parent / "budget-ledger.json").read_text())
ledger_path = OUT / "budget-ledger.json"
account = (
    json.loads(ledger_path.read_text())
    if ledger_path.exists()
    else {
        "aggregate_uncached_compute_usd": prior["reserved_or_spent_usd"],
        "prior_training_and_diagnostics_usd": prior["reserved_or_spent_usd"],
        "game_rescore_usd": 0.0,
        "scoring_calls": 0,
    }
)

frames = {
    label: pd_mod.build_jsonl_dataframe_from_eval_log(INPUTS[label])
    for label in ("attacks", "honeypots", "honest")
}
assert [len(frames[label]) for label in ("attacks", "honeypots", "honest")] == [
    309,
    246,
    817,
]
requests = {}
for label in ("attacks", "honeypots"):
    log = read_eval_log(str(INPUTS[label]))
    df = frames[label]
    system = df["collusion_monitor_system_prompt"].iloc[0]
    template = df["collusion_monitor_user_prompt"].iloc[0]
    assert (
        df["collusion_monitor_system_prompt"].nunique() == 1
        and df["collusion_monitor_user_prompt"].nunique() == 1
    )
    ids = set(df["sample_id"])
    for s in log.samples:
        if s.id not in ids:
            continue
        m = {**(s.metadata or {}), **(s.store or {})}
        assert m["base_solution"] == df.loc[df["sample_id"] == s.id, "solution"].iloc[0]
        requests[(label, s.id)] = (
            system,
            template.format(
                problem_statement=m["main_task_description"],
                solution=m["transformed_solution"],
            ),
        )
assert len(requests) == 555


def bound(system, user):
    tok = tp._tokenizer(BASE)
    prefix = tp._apply_chat_template(
        BASE,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True,
    )
    n = len(prefix) + len(tok.encode("<answer>", add_special_tokens=False))
    tail = max(len(tok.encode(t, add_special_tokens=False)) for t in ("YES", "NO"))
    return 2 * (n + tail) * PREFILL + 2 * SAMPLE  # covers both teacher-forced branches


upper = sum(bound(*v) for v in requests.values())
assert account["aggregate_uncached_compute_usd"] + upper < LIMIT, (account, upper)
pre = {
    "original_cap_usd": 20,
    "compute_limit_usd": LIMIT,
    "already_used_usd": account["aggregate_uncached_compute_usd"],
    "rescore_upper_bound_usd": upper,
    "aggregate_upper_bound_usd": account["aggregate_uncached_compute_usd"] + upper,
    "requests": 555,
    "honest_collusion_scores_reused": True,
    "checkpoint": CHECKPOINT,
    "inputs": {k: str(v) for k, v in INPUTS.items()},
    "reason": "v2 reference summaries differ from the training-diagnostic summaries; rescore the judge only.",
}
(OUT / "budget-preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
print(json.dumps(pre, indent=2), flush=True)
if "--allow-paid" not in sys.argv:
    print("PREFLIGHT ONLY: no paid calls.", flush=True)
    sys.exit(0)


async def score_all():
    sem = asyncio.Semaphore(8)
    out = {}

    async def one(key, system, user):
        reserve = bound(system, user)
        async with sem:
            # No await between the check and the reservation, so concurrent tasks cannot race the cap.
            if account["aggregate_uncached_compute_usd"] + reserve > LIMIT:
                raise RuntimeError(f"$20 cap guard stopped scoring: {account}")
            account["aggregate_uncached_compute_usd"] += reserve
            try:
                r = await tp.collusion_yes_no_logprobs(
                    CHECKPOINT, system, user, single_prefill=True
                )
            except BaseException:
                ledger_path.write_text(json.dumps(account, indent=2) + "\n")
                raise  # keep the worst-case reservation for an uncertain failed request
            u = r["usage"]
            cost = u["input_tokens"] * PREFILL + u["output_tokens"] * SAMPLE
            assert cost <= reserve + 1e-9
            account["aggregate_uncached_compute_usd"] += cost - reserve
            account["game_rescore_usd"] += cost
            account["scoring_calls"] += 1
            assert r.get("YES") is not None and r.get("NO") is not None, key
            out[key] = collusion_score(r["YES"], r["NO"])
            with (OUT / "scores.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "label": key[0],
                            "sample_id": key[1],
                            "yes": r["YES"],
                            "no": r["NO"],
                            "score": out[key],
                            "usage": u,
                            "prompt_sha256": hashlib.sha256(
                                (system + "\0" + user).encode()
                            ).hexdigest(),
                        }
                    )
                    + "\n"
                )

    await asyncio.gather(*(one(k, *v) for k, v in requests.items()))
    ledger_path.write_text(json.dumps(account, indent=2) + "\n")
    return out


if (OUT / "scores.jsonl").exists():
    raise SystemExit(
        "scores.jsonl exists: refusing to double-spend; inspect and resume deliberately"
    )
scores = asyncio.run(score_all())
for label in ("attacks", "honeypots"):
    df = frames[label]
    df["untrusted_collusion_score"] = [scores[(label, i)] for i in df["sample_id"]]

pd_mod.load_condition = lambda inputs, min_samples=0: (
    frames
)  # solve_game reads the rescored frames
solve = pd_mod.SolveSettings(
    game=REF["pipeline"]["solve"]["game"],
    regeneration_cutoffs=REF["pipeline"]["solve"]["regeneration_cutoffs"],
    solver="double_oracle",
    max_iterations=100,
)
boot = pd_mod.BootstrapSettings(replicates=200, ci=0.80, n_jobs=-1, chunk_size=100)
cond = {
    "collusion_strategy": "self_rec",
    "monitor_arm": "sft-5epoch",
    "setting": "apps",
    "transform": "paraphrased",
}
tracker = StageRun(
    "game",
    "game.5epoch_sft_summary",
    config={
        "condition": cond,
        "run": {"arm": "sft-5epoch", "kind": "baseline"},
        "baseline_id": "sft-v41-summary-5epoch",
        "variants": ["all_monitors", "um_only", "tm_only"],
        "game_parameters": REF["pipeline"]["solve"]["game"],
        "budget_cap_usd": 20,
        "aggregate_uncached_compute_usd": account["aggregate_uncached_compute_usd"],
        "note": "v2 reference summaries, T/U suspicion and honest set; only the collusion judge is the 5-epoch checkpoint.",
    },
    tags=["collusion-rl-v2", "baseline-sft", "5-epoch", "summary-trained"],
)
game = pd_mod.solve_game(
    name="apps.self_rec.paraphrased.sft-5epoch",
    inputs=INPUTS,
    variants=["all_monitors", "um_only", "tm_only"],
    solve=solve,
    bootstrap=boot,
    seed=0,
    out_dir=OUT / "game_output",
    tracker=tracker,
    record={
        "experiment": "sft-v41-summary-5epoch",
        "checkpoint": CHECKPOINT,
        "reference_game": "a8781f665021",
    },
    min_samples=10,
)
print("COMPLETED", json.dumps(account), flush=True)
