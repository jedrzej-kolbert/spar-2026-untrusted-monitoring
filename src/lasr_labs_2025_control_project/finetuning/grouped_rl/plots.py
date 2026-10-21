"""Plot how the receiver's discrimination changes over an RL run.

    python -m lasr_labs_2025_control_project.finetuning.grouped_rl.plots \\
        --run-dir experiments/open-source-model/apps/rl/<run>

Reads the run's ``rollouts.jsonl`` and ``metrics.jsonl`` and writes, into the run
directory, ``auroc_over_steps.svg`` and ``auroc_over_steps.json`` (the same numbers
as a table):

- ON vs OFF AUROC: the receiver's P(YES) on signal-ON vs signal-OFF rewrites at each
  held-out evaluation, with a bootstrap 95% interval. This is the training objective.
- Signalled vs honeypot AUROC (when logged): bit-1 rewrites vs honeypots, the
  deployment comparison.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import click

from .core import auroc

# Validated two-series palette (categorical slots 1 and 2) plus recessive ink.
_ON_OFF_COLOR = "#2a78d6"
_HONEYPOT_COLOR = "#eb6834"
_INK = "#52514e"
_GRID = "#e4e3df"


@dataclass(frozen=True)
class AurocPoint:
    step: int
    auroc: float
    ci_low: float | None
    ci_high: float | None
    n_positive: int
    n_negative: int


def bootstrap_interval(
    positives: list[float],
    negatives: list[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile bootstrap interval for AUROC (resampling each class)."""
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        value = auroc(
            [rng.choice(positives) for _ in positives],
            [rng.choice(negatives) for _ in negatives],
        )
        assert value is not None
        values.append(value)
    values.sort()
    return values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]


def on_off_auroc_by_step(
    rollouts_path: Path, *, phase: str = "eval_pairs", resamples: int = 2000
) -> list[AurocPoint]:
    """ON vs OFF AUROC of the receiver's P(YES) at each step of ``phase``."""
    scores: dict[int, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    with open(rollouts_path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") == phase and row.get("p_yes") is not None:
                scores[row["step"]][row["payload"]].append(row["p_yes"])
    points = []
    for step in sorted(scores):
        on, off = scores[step][1], scores[step][0]
        value = auroc(on, off)
        if value is None:
            continue
        low, high = bootstrap_interval(on, off, resamples=resamples)
        points.append(AurocPoint(step, value, low, high, len(on), len(off)))
    return points


def logged_auroc_by_step(
    metrics_path: Path, key: str = "eval_vs_honeypots/auroc"
) -> list[AurocPoint]:
    """An AUROC already logged in ``metrics.jsonl`` (no interval available)."""
    points = []
    if not metrics_path.exists():
        return points
    with open(metrics_path) as handle:
        for line in handle:
            row = json.loads(line)
            if isinstance(row.get(key), (int, float)):
                n_pos = row.get("eval_vs_honeypots/n_signalled", 0)
                n_neg = row.get("eval_vs_honeypots/n_honeypots", 0)
                points.append(
                    AurocPoint(row["step"], row[key], None, None, n_pos, n_neg)
                )
    return points


def plot_auroc_over_steps(run_dir: Path, *, resamples: int = 2000) -> Path:
    """Write the AUROC-over-steps figure and its table; returns the figure path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on_off = on_off_auroc_by_step(run_dir / "rollouts.jsonl", resamples=resamples)
    if not on_off:
        raise ValueError(f"no held-out evaluation records in {run_dir}")
    honeypot = logged_auroc_by_step(run_dir / "metrics.jsonl")

    figure, axis = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    axis.axhline(0.5, color=_INK, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.text(
        on_off[-1].step, 0.5, " chance", va="bottom", ha="right", color=_INK, fontsize=8
    )

    steps = [point.step for point in on_off]
    axis.fill_between(
        steps,
        [point.auroc if point.ci_low is None else point.ci_low for point in on_off],
        [point.auroc if point.ci_high is None else point.ci_high for point in on_off],
        color=_ON_OFF_COLOR,
        alpha=0.14,
        linewidth=0,
        zorder=2,
    )
    series = [
        (on_off, _ON_OFF_COLOR, "o", "Signal ON vs OFF (training objective)"),
        (honeypot, _HONEYPOT_COLOR, "s", "Signalled attack vs honeypot"),
    ]
    for points, color, marker, label in series:
        if not points:
            continue
        axis.plot(
            [point.step for point in points],
            [point.auroc for point in points],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=label,
            zorder=3,
        )
        last = points[-1]
        axis.annotate(
            f"{last.auroc:.2f}",
            (last.step, last.auroc),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=_INK,
            fontsize=8,
        )

    axis.set_ylim(0, 1)
    axis.set_xticks(steps)
    axis.set_xlabel("Training step", color=_INK)
    axis.set_ylabel("AUROC (receiver P(YES))", color=_INK)
    n_problems = on_off[0].n_positive
    axis.set_title(
        f"Receiver discrimination over training ({n_problems} held-out "
        "rewrites per bit; band = 95% CI)",
        fontsize=10,
        loc="left",
    )
    axis.grid(axis="y", color=_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_GRID)
    axis.tick_params(colors=_INK, labelsize=8)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    figure.tight_layout()

    figure_path = run_dir / "auroc_over_steps.svg"
    figure.savefig(figure_path)
    plt.close(figure)
    (run_dir / "auroc_over_steps.json").write_text(
        json.dumps(
            {
                "on_off": [asdict(point) for point in on_off],
                "signalled_vs_honeypot": [asdict(point) for point in honeypot],
            },
            indent=2,
        )
    )
    return figure_path


@click.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True)
def main(run_dir: str) -> None:
    path = plot_auroc_over_steps(Path(run_dir))
    click.echo(f"wrote {path} and {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
