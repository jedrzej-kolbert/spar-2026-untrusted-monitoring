"""Offline checks on a self-recognition GBRL run's saved logs (no model calls).

Reads ``rollouts.jsonl`` and ``honeypot_scores.jsonl`` from each run directory
and reports, per window of training steps:

- spread of the judge's P(YES) within a problem's rewrites (what the sender
  learns from), between problems, and over the whole batch;
- a paraphrase-noise floor: rewrites in the same group with identical code
  (after whitespace normalisation) differ in P(YES) only through the
  paraphraser's sampling, since the judge is the same within a step;
- the share of informative groups: groups whose passing rewrites' P(YES)
  variance exceeds what paraphrase noise alone gives 5% of the time (a
  chi-square test against the noise floor; with no noise estimate, a P(YES)
  range above ``--min-range``), and the within-group signal-to-noise ratio
  (mean within-group variance / noise variance; 1 means pure noise);
- sender diversity: identical rewrites per group, mean within-group similarity,
  response length and pass rate;

and, per validation step, the all-rewrite AUROC against that step's honeypot
scores, the mean P(YES) gap (own - honeypot), the mean log-odds gap and how
close the judge stays to 0.5.

It ends with the plateau rule: the first step after which the informative
share stays under ``--informative-floor`` for ``--patience`` training steps.

    python -m lasr_labs_2025_control_project.scripts.gbrl_free_checks \\
        experiments/.../benign-para-rx-s1 experiments/.../benign-para-rx-frozen-s1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(code: str | None) -> str | None:
    if code is None:
        return None
    return "\n".join(
        line.rstrip() for line in code.strip().splitlines() if line.strip()
    )


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUROC; ties count half."""
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid = (i + j + 1) / 2  # average 1-based rank of the tie block
        rank_sum += mid * sum(label for _, label in ranked[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def chi2_upper(df: int, z: float = 1.645) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (z=1.645: 95%)."""
    a = 2 / (9 * df)
    return df * (1 - a + z * math.sqrt(a)) ** 3


def informative_group(
    ok: list[float], noise_var: float | None, min_range: float
) -> bool:
    if len(ok) < 2:
        return False
    if noise_var is None:
        return max(ok) - min(ok) > min_range
    df = len(ok) - 1
    return statistics.variance(ok) > noise_var * chi2_upper(df) / df


def similarity(a: str, b: str, limit: int = 3000) -> float:
    return SequenceMatcher(None, a[:limit], b[:limit], autojunk=False).ratio()


def train_steps(rollouts: list[dict]) -> dict[int, dict]:
    groups: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rollouts:
        if row.get("phase") == "train" and row.get("p_yes") is not None:
            groups[row["step"]][row["problem_id"]].append(row)

    steps = {}
    for step, by_problem in sorted(groups.items()):
        within, means, all_p, noise_d2, view_d2 = [], [], [], [], []
        passing, variances, identical_groups, sims, tokens, passed = (
            [],
            [],
            0,
            [],
            [],
            [],
        )
        for rows in by_problem.values():
            ps = [r["p_yes"] for r in rows]
            all_p += ps
            means.append(statistics.fmean(ps))
            if len(ps) > 1:
                within.append(statistics.pstdev(ps))
                variances.append(statistics.variance(ps))
            passing.append([r["p_yes"] for r in rows if r.get("passed")])
            codes = [normalise(r.get("code")) for r in rows]
            has_identical = False
            for (a, ca), (b, cb) in combinations(zip(rows, codes), 2):
                if ca is None or cb is None:
                    continue
                if ca == cb:
                    has_identical = True
                    noise_d2.append((a["p_yes"] - b["p_yes"]) ** 2)
                    if a.get("receiver_code") == b.get("receiver_code"):
                        view_d2.append((a["p_yes"] - b["p_yes"]) ** 2)
                else:
                    sims.append(similarity(ca, cb))
            identical_groups += has_identical
            tokens += [r["response_tokens"] for r in rows if r.get("response_tokens")]
            passed += [bool(r.get("passed")) for r in rows]
        steps[step] = {
            "within_sd": statistics.fmean(within) if within else None,
            "between_sd": statistics.pstdev(means) if len(means) > 1 else None,
            "total_sd": statistics.pstdev(all_p) if len(all_p) > 1 else None,
            "mean_p": statistics.fmean(all_p),
            "noise_d2": noise_d2,
            "view_d2": view_d2,
            "passing": passing,
            "within_var": statistics.fmean(variances) if variances else None,
            "identical_share": identical_groups / len(by_problem),
            "similarity": statistics.fmean(sims) if sims else None,
            "tokens": statistics.fmean(tokens) if tokens else None,
            "pass_rate": statistics.fmean(passed) if passed else None,
        }
    return steps


def validation_steps(rollouts: list[dict], honeypots: list[dict]) -> dict[int, dict]:
    own: dict[int, list[float]] = defaultdict(list)
    for row in rollouts:
        if row.get("phase") == "validation" and row.get("p_yes") is not None:
            own[row["step"]].append(row["p_yes"])
    neg: dict[int, list[float]] = defaultdict(list)
    for row in honeypots:
        if row.get("p_yes") is not None:
            neg[row["step"]].append(row["p_yes"])
    out = {}
    for step in sorted(set(own) & set(neg)):
        pos, hp = own[step], neg[step]
        out[step] = {
            "auroc": auroc(pos, hp),
            "gap": statistics.fmean(pos) - statistics.fmean(hp),
            "logit_gap": statistics.fmean(map(logit, pos))
            - statistics.fmean(map(logit, hp)),
            "distance_from_half": statistics.fmean(abs(p - 0.5) for p in pos + hp),
            "n": (len(pos), len(hp)),
        }
    return out


def fmt(value: float | None, spec: str = ".3f") -> str:
    return "  -  " if value is None else format(value, spec)


def report(
    run_dir: Path, window: int, min_range: float, floor: float, patience: int
) -> None:
    rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    honeypots = read_jsonl(run_dir / "honeypot_scores.jsonl")
    print(
        f"\n=== {run_dir}  ({len(rollouts)} rollouts, {len(honeypots)} honeypot scores)"
    )
    steps = train_steps(rollouts)
    if not steps:
        print("no training rollouts with p_yes")
        return

    noise_d2 = [d for s in steps.values() for d in s["noise_d2"]]
    view_d2 = [d for s in steps.values() for d in s["view_d2"]]
    noise_sd = math.sqrt(statistics.fmean(noise_d2) / 2) if noise_d2 else None
    print(
        f"noise floor: {len(noise_d2)} identical-code pairs within a step -> "
        f"paraphrase noise sd {fmt(noise_sd)}"
        + (
            f"; {len(view_d2)} pairs with identical judge input, rms diff "
            f"{fmt(math.sqrt(statistics.fmean(view_d2)))} (should be ~0)"
            if view_d2
            else ""
        )
    )
    noise_var = noise_sd**2 if noise_sd else None
    print(
        "informative group: passing rewrites' P(YES) variance above the 95% "
        "noise-only level"
        if noise_var
        else f"informative group: passing rewrites' P(YES) range > {min_range} "
        "(--min-range; no identical-code pairs, so no noise estimate)"
    )

    print(
        f"\n{'steps':>9} {'within':>7} {'between':>7} {'total':>7} {'meanP':>6} "
        f"{'SNR':>5} {'inform':>6} {'ident':>6} {'simil':>6} {'tokens':>6} {'pass':>5}"
    )
    keys = sorted(steps)
    informative = {
        k: statistics.fmean(
            informative_group(ok, noise_var, min_range) for ok in steps[k]["passing"]
        )
        for k in keys
    }
    for start in range(keys[0], keys[-1] + 1, window):
        block = [steps[k] for k in keys if start <= k < start + window]
        if not block:
            continue

        def mean(field: str) -> float | None:
            vals = [b[field] for b in block if b[field] is not None]
            return statistics.fmean(vals) if vals else None

        within_var = mean("within_var")
        snr = within_var / noise_var if noise_var and within_var is not None else None
        inf = statistics.fmean(
            informative[k] for k in keys if start <= k < start + window
        )
        print(
            f"{start:>4}-{start + window - 1:<4} {fmt(mean('within_sd'))} {fmt(mean('between_sd'))} "
            f"{fmt(mean('total_sd'))} {fmt(mean('mean_p'), '.2f'):>6} {fmt(snr, '5.1f')} {inf:6.2f} "
            f"{fmt(mean('identical_share'), '.2f'):>6} {fmt(mean('similarity'), '.2f'):>6} "
            f"{fmt(mean('tokens'), '.0f'):>6} {fmt(mean('pass_rate'), '.2f'):>5}"
        )

    # Rolling mean: with 8 groups a step, one noise-only false positive is 12.5%.
    stop = None
    for i in range(patience - 1, len(keys)):
        if (
            statistics.fmean(informative[k] for k in keys[i - patience + 1 : i + 1])
            < floor
        ):
            stop = keys[i - patience + 1]
            break
    print(
        f"\nplateau rule (mean informative share < {floor:.0%} over {patience} steps; "
        "noise alone gives ~5%): "
        + (f"triggered from step {stop}" if stop is not None else "not triggered")
    )

    val = validation_steps(rollouts, honeypots)
    if val:
        print(
            f"\n{'val step':>8} {'AUROC':>6} {'gap':>7} {'logit gap':>9} {'|P-0.5|':>7} {'n own/hp':>9}"
        )
        for step, v in val.items():
            print(
                f"{step:>8} {fmt(v['auroc'])} {v['gap']:+.3f} {v['logit_gap']:+9.3f} "
                f"{v['distance_from_half']:7.3f} {v['n'][0]:>4}/{v['n'][1]:<4}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=10, help="training steps per row")
    parser.add_argument(
        "--min-range",
        type=float,
        default=0.05,
        help="informative-group threshold when no identical-code pairs exist",
    )
    parser.add_argument("--informative-floor", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        report(
            run_dir, args.window, args.min_range, args.informative_floor, args.patience
        )


if __name__ == "__main__":
    main()
