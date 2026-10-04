"""Paired comparison of self-recognition ROC AUC between Tinker models.

Scores every model on the *same* attacks (U's own APPS solutions, label 1) vs
honeypots (another model's, label 0) with the collusion monitor's exact
``log P(YES) - log P(NO)`` margin, then reports each model's AUC and, for each
model after the first, a paired bootstrap 95% CI of its AUC minus the first
model's. Put the base model first, e.g.

  python -m lasr_labs_2025_control_project.scripts.compare_self_rec_auc \\
      --attacks-eval attacks.eval --honeypots-eval honeypots.eval \\
      --model deepseek-ai/DeepSeek-V3.1 --model tinker://<run>/sampler_weights/<name> \\
      --out results/introspection_deepseek/

The bootstrap resamples attacks and honeypots separately (class sizes fixed)
with the same indices for every model, so the CI accounts for both models being
scored on one finite set of solutions. Writes margins.csv (one row per example)
and summary.json to --out.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from pathlib import Path

import click
import numpy as np
from sklearn.metrics import roc_auc_score

from lasr_labs_2025_control_project.scripts.tinker_sft_dynamics import (
    load_eval_examples,
)
from lasr_labs_2025_control_project.utils.tinker_provider import (
    collusion_yes_no_logprobs,
    strip_prefix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _margins(
    model: str, examples: list[tuple[int, str, str]], concurrency: int
) -> list[float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def margin(system: str, user: str) -> float:
        async with semaphore:
            result = await collusion_yes_no_logprobs(
                model_name=f"tinker/{model}", system_prompt=system, user_prompt=user
            )
        if result.get("YES") is None or result.get("NO") is None:
            raise ValueError(f"{model}: Tinker returned no YES/NO logprobs")
        return float(result["YES"]) - float(result["NO"])

    return list(
        await asyncio.gather(*(margin(system, user) for _, system, user in examples))
    )


def paired_bootstrap_delta(
    labels: np.ndarray,
    base: np.ndarray,
    other: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (delta, ci_low, ci_high) for AUC(other) - AUC(base).

    Resamples within each class so every replicate has both classes.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        raise ValueError("need both attacks and honeypots")
    delta = roc_auc_score(labels, other) - roc_auc_score(labels, base)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        y = labels[idx]
        deltas[b] = roc_auc_score(y, other[idx]) - roc_auc_score(y, base[idx])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(delta), float(low), float(high)


@click.command()
@click.option(
    "--attacks-eval", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--honeypots-eval", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--model",
    "models",
    multiple=True,
    required=True,
    help="Base model id or tinker:// checkpoint; repeat. The first is the baseline.",
)
@click.option(
    "--transformed",
    is_flag=True,
    help="Score transformed_solution instead of base_solution.",
)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--concurrency", type=click.IntRange(1, 64), default=16, show_default=True
)
@click.option("--n-boot", type=int, default=2000, show_default=True)
def main(
    attacks_eval: Path,
    honeypots_eval: Path,
    models: tuple[str, ...],
    transformed: bool,
    out: Path,
    concurrency: int,
    n_boot: int,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    models = tuple(strip_prefix(m) for m in models)
    solution_field = "transformed_solution" if transformed else "base_solution"
    examples = load_eval_examples(attacks_eval, honeypots_eval, solution_field)
    labels = np.array([label for label, _, _ in examples])
    logger.info(
        "%d attacks / %d honeypots", int(labels.sum()), int((labels == 0).sum())
    )

    margins = {}
    for model in models:
        margins[model] = np.array(asyncio.run(_margins(model, examples, concurrency)))
        logger.info("%s: AUC %.4f", model, roc_auc_score(labels, margins[model]))

    base = models[0]
    summary = {
        "attacks_eval": str(attacks_eval),
        "honeypots_eval": str(honeypots_eval),
        "solution_field": solution_field,
        "n_attacks": int(labels.sum()),
        "n_honeypots": int((labels == 0).sum()),
        "baseline": base,
        "models": [],
    }
    for model in models:
        entry = {"model": model, "auc": float(roc_auc_score(labels, margins[model]))}
        if model != base:
            delta, low, high = paired_bootstrap_delta(
                labels, margins[base], margins[model], n_boot=n_boot
            )
            entry.update(delta_vs_baseline=delta, ci95=[low, high])
            logger.info(
                "%s - %s: dAUC %+.4f [95%% CI %+.4f, %+.4f]",
                model,
                base,
                delta,
                low,
                high,
            )
        summary["models"].append(entry)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "margins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *models])
        for i, label in enumerate(labels):
            writer.writerow([int(label), *(margins[m][i] for m in models)])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
