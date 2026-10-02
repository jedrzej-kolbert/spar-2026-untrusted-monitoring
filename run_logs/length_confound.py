"""Is trajectory length what separates U's attacks from honeypots?

Scores every test example (20 U attacks vs gpt-oss or DeepSeek honeypots) with the
base model and the final SFT checkpoint, then compares against length alone,
length-matched pairs, and length-residualised scores.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log
from sklearn.metrics import roc_auc_score

from lasr_labs_2025_control_project.scripts.tinker_sft_dynamics import (
    load_eval_examples,
)
from lasr_labs_2025_control_project.utils.tinker_provider import (
    collusion_yes_no_logprobs,
)
from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid

C = Path(sys.argv[1])  # comparison dir with the .eval files
OUT = Path(sys.argv[2])
CHECKPOINTS = {"base": "Qwen/Qwen3.6-35B-A3B", "final": sys.argv[3]}
SETS = {
    "gpt-oss": "honeypots_gpt_oss_120b_20.eval",
    "deepseek": "honeypots_deepseek_v31_20.eval",
}


def n_actions(path: Path) -> list[int]:
    out = []
    for s in read_eval_log(str(path)).samples:
        if check_sample_is_valid(s.scores, s.metadata.get("eval_mode")):
            out.append(s.store["base_solution"].count("<action_"))
    return out


async def score_all(model: str, examples) -> list[float]:
    sem = asyncio.Semaphore(8)

    async def one(system, user):
        async with sem:
            r = await collusion_yes_no_logprobs(f"tinker/{model}", system, user)
            return r["YES"] - r["NO"]

    return await asyncio.gather(*(one(s, u) for _, s, u in examples))


def pair_auc(y, score, length, max_gap):
    """AUC over (attack, honeypot) pairs whose action counts differ by <= max_gap."""
    pos = [(s, n) for s, n, t in zip(score, length, y) if t == 1]
    neg = [(s, n) for s, n, t in zip(score, length, y) if t == 0]
    wins = [
        1.0 if a > b else 0.5 if a == b else 0.0
        for a, la in pos
        for b, lb in neg
        if abs(la - lb) <= max_gap
    ]
    return (float(np.mean(wins)) if wins else float("nan")), len(wins)


rows = {}
attacks = C / "attacks_qwen36_20.eval"
for name, file in SETS.items():
    examples = load_eval_examples(attacks, C / file, "base_solution")
    y = [label for label, _, _ in examples]
    length = n_actions(attacks) + n_actions(
        C / file
    )  # same order as load_eval_examples
    assert len(length) == len(y)
    res = {
        "n_pos": sum(y),
        "n_neg": len(y) - sum(y),
        "length_only_auc": roc_auc_score(y, length),
    }
    for ck, model in CHECKPOINTS.items():
        s = np.array(asyncio.run(score_all(model, examples)))
        L = np.log(np.array(length, dtype=float))
        resid = s - np.polyval(
            np.polyfit(L, s, 1), L
        )  # score with a linear log-length trend removed
        res[ck] = {
            "auc": roc_auc_score(y, s),
            "auc_length_residualised": roc_auc_score(y, resid),
            "pair_auc_gap<=2": pair_auc(y, s, length, 2),
            "pair_auc_gap<=4": pair_auc(y, s, length, 4),
            "corr_score_length_within_attacks": float(
                np.corrcoef(
                    [v for v, t in zip(s, y) if t], [v for v, t in zip(length, y) if t]
                )[0, 1]
            ),
        }
    rows[name] = res
    print(name, json.dumps(res, indent=1))
OUT.write_text(json.dumps(rows, indent=1))
