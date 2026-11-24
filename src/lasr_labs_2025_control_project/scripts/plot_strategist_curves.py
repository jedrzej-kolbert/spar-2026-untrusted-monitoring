"""Side-by-side safety/usefulness and collusion ROC for a strategist run's signals.

The default comparison figure for GBRL strategist test scoring: pass the
experiment and the signals (measurement arms) to plot, and get one figure with

    left:  usefulness vs safety (the solved game's point estimate per
           usefulness level, with a percentile bootstrap band on safety)
    right: the collusion monitor's ROC, attacks (true positives) against
           honeypots (false positives), with a pointwise bootstrap band

for one channel (paraphrased by default). Every arm of the experiment is
plotted when no --arm is given.

    python -m lasr_labs_2025_control_project.scripts.plot_strategist_curves \
        --experiment rl_gbrl_strategist_test \
        --arm "step40=Lookup table (step 40)" --arm "step1=Recursion (step 1)" \
        --out experiments/open-source-model/apps/rl/gbrl_strategist/curves_paraphrased.png

Curves from elsewhere (another project's run, or points read off a figure)
go in with --reference FILE.json:

    {"label": "...", "approximate": true,
     "safety_usefulness": [[safety, usefulness], ...],
     "roc": [[false_positive_rate, true_positive_rate], ...]}

drawn dashed with hollow markers and no band. The game stage keeps only a few
bootstrap replicates (optimization.bootstrap.replicates), so the safety band's
replicates are re-run here (local CPU, no API calls; about 4 minutes per arm
for 200 on 8 cores) and cached next to the figure, seeded as the game stage
seeds them. --wandb-run logs the figure as an image on that run.
"""

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Optional

# One maths thread per bootstrap worker: joblib already runs a worker per core,
# and per-worker BLAS/OpenMP pools oversubscribe the CPU (~6x slower).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from lasr_labs_2025_control_project.scripts.plot_data import (  # noqa: E402
    VARIANT_FLAGS,
    BootstrapSettings,
    bootstrap_variant,
    condition_of,
    load_saved_condition,
    load_scores,
    saved_conditions,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (  # noqa: E402
    OPTIMIZERS,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (  # noqa: E402
    OI_BLUE,
    OI_ORANGE,
    OI_PURPLE,
    OI_SKY_BLUE,
    OI_TEAL,
    OI_VERMILION,
)

#: Series colours in the order the arms are passed (an arm keeps its colour
#: across figures when it is passed first in each); references follow the arms.
PALETTE = [OI_BLUE, OI_VERMILION, OI_TEAL, OI_ORANGE, OI_PURPLE, OI_SKY_BLUE]
CONTROL_COLOR = "#8a8a8a"
CONTROL_ARMS = {"no_strategy", "baseline", "base"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
ROC_GRID = np.linspace(0.0, 1.0, 401)


def find_game(root: Path, experiment: str, arm: str, transform: str) -> Path:
    """The newest saved game folder of an experiment's arm on one channel."""
    matches = []
    for folder in saved_conditions(root):
        record = load_saved_condition(folder)["pipeline"]
        condition = condition_of(record)
        if (
            record.get("experiment") == experiment
            and condition["monitor_arm"] == arm
            and condition["transform"] == transform
        ):
            matches.append(folder)
    if not matches:
        raise SystemExit(f"no saved game for {experiment} / {arm} / {transform}")
    return max(matches, key=lambda f: (f / "analysis_data.json.gz").stat().st_mtime)


def experiment_arms(root: Path, experiment: str, transform: str) -> list[str]:
    arms = []
    for folder in saved_conditions(root):
        record = load_saved_condition(folder)["pipeline"]
        condition = condition_of(record)
        if (
            record.get("experiment") == experiment
            and condition["transform"] == transform
        ):
            arms.append(condition["monitor_arm"])
    # controls first, then as named
    return sorted(set(arms), key=lambda a: (a not in CONTROL_ARMS, a))


def safety_curve(
    folder: Path, replicates: int, ci: float, cache: Path
) -> dict[str, np.ndarray]:
    """Point-estimate safety per usefulness level, plus the bootstrap band."""
    payload = load_saved_condition(folder)
    record = payload["pipeline"]
    usefulness = np.array([e["usefulness"] for e in payload["sweep"]], dtype=float)
    safety = np.array([e["safety"] for e in payload["sweep"]], dtype=float)
    frames = load_scores(folder / "scores.json.gz")
    disable_tm, disable_um = VARIANT_FLAGS[record["variant"]]
    solve = record["solve"]
    cache.mkdir(parents=True, exist_ok=True)
    reps = bootstrap_variant(
        cache / f"{folder.name}_bootstrap_{record['variant']}.jsonl.gz",
        BootstrapSettings(replicates=replicates, ci=ci, n_jobs=-1, chunk_size=25),
        condition_seed=record["bootstrap_seed"],
        attacks_df=frames["attacks"],
        honeypots_df=frames["honeypots"],
        benign_df=frames["honest"],
        disable_trusted_monitor=disable_tm,
        disable_untrusted_monitor=disable_um,
        game=dict(solve["game"]),
        regeneration_cutoffs=solve["regeneration_cutoffs"],
        optimizer=OPTIMIZERS[solve["solver"]],
        max_iterations=solve["max_iterations"],
    )
    sweeps = [r["sweep"] for r in reps if "sweep" in r]
    boot = np.array(
        [[np.nan if e["safety"] is None else e["safety"] for e in s] for s in sweeps],
        dtype=float,
    )
    tail = 100.0 * (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boot, [tail, 100.0 - tail], axis=0)
    order = np.argsort(usefulness)
    return {
        "usefulness": usefulness[order],
        "safety": safety[order],
        "lo": lo[order],
        "hi": hi[order],
        "n_replicates": np.array(len(sweeps)),
    }


def roc_at(attacks: np.ndarray, honeypots: np.ndarray) -> np.ndarray:
    """Best true positive rate at each false positive rate in ROC_GRID
    (flag a sample when its collusion score is above the threshold)."""
    h = np.sort(honeypots)[::-1]
    k = np.floor(ROC_GRID * len(h)).astype(int)
    thresholds = np.where(k < len(h), h[np.minimum(k, len(h) - 1)], -np.inf)
    return (attacks[None, :] > thresholds[:, None]).mean(axis=1)


def auroc(attacks: np.ndarray, honeypots: np.ndarray) -> float:
    diff = attacks[:, None] - honeypots[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def roc_curve(folder: Path, ci: float, resamples: int, seed: int = 0) -> dict[str, Any]:
    frames = load_scores(folder / "scores.json.gz")
    a = frames["attacks"]["untrusted_collusion_score"].to_numpy(dtype=float)
    h = frames["honeypots"]["untrusted_collusion_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [roc_at(rng.choice(a, len(a)), rng.choice(h, len(h))) for _ in range(resamples)]
    )
    tail = 100.0 * (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [tail, 100.0 - tail], axis=0)
    return {"tpr": roc_at(a, h), "lo": lo, "hi": hi, "auroc": auroc(a, h)}


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for name, spine in ax.spines.items():
        spine.set_visible(name in {"left", "bottom"})
        spine.set_color("#9a9a9a")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])


def draw(
    curves: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    title: str,
    ci: float,
    note: Optional[str],
    out: Path,
) -> None:
    fig, (su, roc) = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.patch.set_facecolor("white")
    for ax in (su, roc):
        style(ax)
    su.axhline(0.95, color="#9a9a9a", linestyle=(0, (1.5, 2)), linewidth=1, zorder=0.5)
    su.text(
        0.99, 0.955, "usefulness 0.95", ha="right", va="bottom", color=MUTED, fontsize=9
    )
    roc.plot(
        [0, 1], [0, 1], color="#b0b0b0", linestyle=(0, (2, 3)), linewidth=1, zorder=0.5
    )

    handles: list[Line2D] = []
    for c in curves:
        s = c["safety"]
        dash = (0, (4, 2)) if c["control"] else "-"
        su.fill_betweenx(
            s["usefulness"],
            s["lo"],
            s["hi"],
            color=c["color"],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        su.plot(
            s["safety"],
            s["usefulness"],
            color=c["color"],
            linewidth=2,
            linestyle=dash,
            zorder=2,
        )
        su.scatter(
            s["safety"],
            s["usefulness"],
            s=42,
            color=c["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        r = c["roc"]
        roc.fill_between(
            ROC_GRID,
            r["lo"],
            r["hi"],
            color=c["color"],
            alpha=0.13,
            linewidth=0,
            step="post",
            zorder=1,
        )
        roc.step(
            ROC_GRID,
            r["tpr"],
            where="post",
            color=c["color"],
            linewidth=2,
            linestyle=dash,
            zorder=2,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=c["color"],
                linestyle=dash,
                linewidth=2,
                marker="o",
                markeredgecolor="white",
                label=f"{c['label']}  (AUROC {r['auroc']:.2f})",
            )
        )
    for ref in references:
        style_kw = dict(
            color=ref["color"], linewidth=2, linestyle=(0, (5, 2.5)), zorder=2
        )
        if ref.get("safety_usefulness"):
            pts = np.array(sorted(ref["safety_usefulness"], key=lambda p: p[1]))
            su.plot(pts[:, 0], pts[:, 1], **style_kw)
            su.scatter(
                pts[:, 0],
                pts[:, 1],
                s=42,
                facecolor="white",
                edgecolor=ref["color"],
                linewidth=1.6,
                zorder=3,
            )
        if ref.get("roc"):
            pts = np.array(sorted(ref["roc"]))
            roc.plot(pts[:, 0], pts[:, 1], **style_kw)
        label = ref["label"] + (
            "  (approximate, no band)" if ref.get("approximate") else ""
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=ref["color"],
                label=label,
                **{k: v for k, v in style_kw.items() if k != "zorder"},
            )
        )

    su.set_xlabel("Safety (blue-team win rate)", color=INK)
    su.set_ylabel("Usefulness", color=INK)
    roc.set_xlabel("False positive rate (honeypots)", color=INK)
    roc.set_ylabel("True positive rate (attacks)", color=INK)
    su.set_title(
        f"Usefulness vs safety, {ci:.0%} bootstrap band on safety",
        fontsize=10,
        color=MUTED,
        loc="left",
    )
    roc.set_title(
        f"Collusion monitor ROC, {ci:.0%} pointwise bootstrap band",
        fontsize=10,
        color=MUTED,
        loc="left",
    )
    fig.suptitle(title, x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK)

    legend_rows = int(np.ceil(len(handles) / 2))
    note_lines = textwrap.wrap(note, 140) if note else []
    note_h = 0.032 * len(note_lines) + (0.02 if note_lines else 0.0)
    if note_lines:
        fig.text(
            0.06,
            0.01,
            "\n".join(note_lines),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=MUTED,
            linespacing=1.4,
        )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.01 + note_h),
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=INK,
    )
    bottom = 0.03 + note_h + 0.045 * legend_rows
    fig.tight_layout(rect=(0.02, bottom, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def parse_arm(spec: str) -> tuple[str, Optional[str]]:
    arm, _, label = spec.partition("=")
    return arm, label or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="the test config's `name` (e.g. rl_gbrl_strategist_test)",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="ARM or ARM=Legend label; repeat; order sets colour (default: every arm)",
    )
    parser.add_argument(
        "--transform", default="paraphrased", help="channel: none, paraphrased, ..."
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        type=Path,
        help="external curve JSON (see module doc); repeat",
    )
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--title", default=None, help="default: the channel, e.g. Paraphrased"
    )
    parser.add_argument("--note", default=None, help="text printed under the legend")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--replicates",
        type=int,
        default=200,
        help="game bootstrap replicates for the safety band",
    )
    parser.add_argument("--roc-resamples", type=int, default=1000)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="bootstrap cache dir (default: <out dir>/bootstrap_cache)",
    )
    parser.add_argument(
        "--wandb-run", default=None, help="W&B run id to log the figure on"
    )
    parser.add_argument(
        "--wandb-key",
        default=None,
        help="key for the logged image (default: test/<out stem>)",
    )
    parser.add_argument("--wandb-entity", default="spar-untrusted-monitoring-collusion")
    parser.add_argument("--wandb-project", default="collusion-rl-v2")
    args = parser.parse_args()

    arms = [parse_arm(s) for s in args.arm] or [
        (a, None) for a in experiment_arms(args.root, args.experiment, args.transform)
    ]
    cache = args.cache or args.out.parent / "bootstrap_cache"
    curves, colors = [], iter(PALETTE)
    for arm, label in arms:
        folder = find_game(args.root, args.experiment, arm, args.transform)
        print(f"{arm}: {folder}")
        control = arm in CONTROL_ARMS
        curves.append(
            {
                "label": label or arm,
                "control": control,
                "color": CONTROL_COLOR if control else next(colors),
                "safety": safety_curve(folder, args.replicates, args.ci, cache),
                "roc": roc_curve(folder, args.ci, args.roc_resamples),
            }
        )
    references = []
    for path in args.reference:
        ref = json.loads(path.read_text())
        ref.setdefault("color", next(colors))
        references.append(ref)

    draw(
        curves,
        references,
        title=args.title or args.transform.capitalize(),
        ci=args.ci,
        note=args.note,
        out=args.out,
    )
    if args.wandb_run:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run,
            resume="must",
        )
        run.log({args.wandb_key or f"test/{args.out.stem}": wandb.Image(str(args.out))})
        run.finish()


if __name__ == "__main__":
    main()
