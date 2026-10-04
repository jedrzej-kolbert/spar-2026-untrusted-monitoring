"""Plot the introspection experiment's results from runs/ (see README.md).

  uv run python experiments/introspection_deepseek/plot_results.py

Reads accuracy_base.json, {self,cross}/accuracy.json, {self,cross}/dynamics/
dynamics.csv and compare/summary.json; arms that have not run are left out.
Writes runs/figures/results.{png,svg} and prints the same numbers as a table.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).parent / "runs"

# Colour follows the model, in every panel. Base is the reference, so it is gray.
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BASE = "#898781"
ARMS = {
    "self": ("Trained on its own answers", "#2a78d6"),
    "cross": ("Control: trained on gpt-oss-120b's answers", "#eb6834"),
}


def load() -> dict:
    data: dict = {"base_acc": json.loads((RUNS / "accuracy_base.json").read_text())}
    summary_path = RUNS / "compare" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary:
        data["n"] = (summary["n_attacks"], summary["n_honeypots"])
        data["base_auc"] = summary["models"][0]["auc"]
    data["arms"] = {}
    for arm in ARMS:
        ckpt_path = RUNS / arm / "checkpoint_path.txt"
        if not ckpt_path.exists():
            continue
        entry: dict = {}
        acc = RUNS / arm / "accuracy.json"
        if acc.exists():
            entry["acc"] = json.loads(acc.read_text())
        with (RUNS / arm / "dynamics" / "dynamics.csv").open() as f:
            entry["dynamics"] = [
                (int(r["updates"]), float(r["roc_auc"]))
                for r in csv.DictReader(f)
                if r["roc_auc"]
            ]
        ckpt = ckpt_path.read_text().strip()
        for model in summary["models"][1:] if summary else []:
            if ckpt.endswith(model["model"]) or model["model"].endswith(ckpt):
                entry["auc"] = model
        data["arms"][arm] = entry
    return data


def style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=24)
    ax.annotate(
        subtitle,
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9,
        color=INK2,
    )


def plot_accuracy(ax, data: dict) -> None:
    base = data["base_acc"]
    rows = [("Base DeepSeek-V3.1", base["accuracy"], BASE)]
    rows += [
        (ARMS[arm][0], entry["acc"]["accuracy"], ARMS[arm][1])
        for arm, entry in data["arms"].items()
        if "acc" in entry
    ]
    style(
        ax,
        "Did training instil self-prediction?",
        f"Accuracy on held-out tasks, each model vs its own answers (n={base['n']:,})",
    )
    for y, (label, value, colour) in enumerate(rows):
        ax.barh(y, value, height=0.42, color=colour)
        ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK2, fontsize=9.5)
    ax.invert_yaxis()
    mode = base["mode_baseline"]
    ax.axvline(mode, color=INK2, lw=1)
    ax.text(
        mode + 0.012,
        -0.8,
        f"mode baseline (most common answer) {mode:.3f}",
        fontsize=8.5,
        color=INK2,
        va="center",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.5, -1.1)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)


def plot_per_task(ax, data: dict) -> None:
    base = data["base_acc"]["per_task"]
    trained = data["arms"].get("self", {}).get("acc", {}).get("per_task")
    style(
        ax,
        "Self-prediction accuracy by held-out task",
        "Base to trained on its own answers; tick = mode baseline",
    )
    tasks = sorted(base, key=lambda t: (trained or base)[t]["accuracy"])
    for y, task in enumerate(tasks):
        b = base[task]["accuracy"]
        ax.plot(
            [base[task]["mode_baseline"]] * 2,
            [y - 0.28, y + 0.28],
            color=AXIS,
            lw=1.5,
            zorder=1,
        )
        if trained and task in trained:
            t = trained[task]["accuracy"]
            ax.plot([b, t], [y, y], color=AXIS, lw=2, zorder=2)
            ax.scatter(
                t, y, s=60, color=ARMS["self"][1], edgecolor=SURFACE, lw=1.5, zorder=4
            )
        ax.scatter(b, y, s=60, color=BASE, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.set_yticks(range(len(tasks)), tasks, color=INK2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("accuracy", color=MUTED, fontsize=9)


def plot_dynamics(ax, data: dict) -> None:
    n_att, n_hp = data.get("n", ("?", "?"))
    style(
        ax,
        "Self-recognition on APPS during training",
        f"ROC AUC, {n_att} own attacks vs {n_hp} gpt-oss-120b honeypots",
    )
    values = [0.5]
    last = 0
    for arm, entry in data["arms"].items():
        xs, ys = zip(*entry["dynamics"])
        ax.plot(xs, ys, color=ARMS[arm][1], lw=2, solid_capstyle="round")
        ax.scatter(
            xs[-1],
            ys[-1],
            s=60,
            color=ARMS[arm][1],
            edgecolor=SURFACE,
            lw=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{ys[-1]:.3f}",
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            color=INK,
        )
        values += ys
        last = max(last, xs[-1])
    for level, label in ((data.get("base_auc"), "base"), (0.5, "chance")):
        if level is None:
            continue
        ax.axhline(
            level,
            color=BASE if label == "base" else AXIS,
            lw=2 if label == "base" else 1,
        )
        ax.text(
            0,
            level,
            f"{label} {level:.3f} ",
            ha="left",
            va="bottom",
            fontsize=9,
            color=INK2,
        )
        values.append(level)
    ax.set_xlim(0, last * 1.12)
    ax.set_ylim(min(values) - 0.03, max(values) + 0.04)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("gradient updates", color=MUTED, fontsize=9)


def plot_delta(ax, data: dict) -> None:
    style(
        ax,
        "Change in self-recognition AUC vs base",
        "Paired bootstrap 95% CI; an interval crossing 0 is no detectable change",
    )
    rows = [(arm, e["auc"]) for arm, e in data["arms"].items() if "auc" in e]
    span = [0.0]
    for y, (arm, m) in enumerate(rows):
        low, high = m["ci95"]
        ax.plot([low, high], [y, y], color=ARMS[arm][1], lw=2, solid_capstyle="round")
        ax.scatter(
            m["delta_vs_baseline"],
            y,
            s=70,
            color=ARMS[arm][1],
            edgecolor=SURFACE,
            lw=1.5,
            zorder=3,
        )
        ax.text(
            high,
            y,
            f"   {m['delta_vs_baseline']:+.3f}  [{low:+.3f}, {high:+.3f}]",
            va="center",
            fontsize=9.5,
            color=INK,
        )
        span += [low, high]
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(
        range(len(rows)), [ARMS[a][0] for a, _ in rows], color=INK2, fontsize=9.5
    )
    ax.set_ylim(len(rows) - 0.4, -0.6)
    width = max(span) - min(span) or 0.1
    ax.set_xlim(min(span) - 0.15 * width, max(span) + 0.75 * width)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("ΔAUC (trained − base)", color=MUTED, fontsize=9)


def table(data: dict) -> str:
    base = data["base_acc"]
    lines = [
        "model | held-out self-prediction acc | mode baseline | APPS self-rec AUC | dAUC vs base [95% CI]",
        f"base | {base['accuracy']:.3f} | {base['mode_baseline']:.3f} | "
        f"{data.get('base_auc', float('nan')):.3f} | -",
    ]
    for arm, e in data["arms"].items():
        acc, auc = e.get("acc"), e.get("auc")
        lines.append(
            f"{arm}-trained | "
            + (
                f"{acc['accuracy']:.3f} | {acc['mode_baseline']:.3f}"
                if acc
                else "- | -"
            )
            + " | "
            + (
                f"{auc['auc']:.3f} | {auc['delta_vs_baseline']:+.3f} "
                f"[{auc['ci95'][0]:+.3f}, {auc['ci95'][1]:+.3f}]"
                if auc
                else "- | -"
            )
        )
    return "\n".join(lines)


def main() -> None:
    data = load()
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig = plt.figure(figsize=(14, 10.5), facecolor=SURFACE)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1, 1.05],
        hspace=0.75,
        wspace=0.62,
        left=0.235,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )
    plot_accuracy(fig.add_subplot(grid[0, 0]), data)
    plot_dynamics(fig.add_subplot(grid[1, 0]), data)
    plot_delta(fig.add_subplot(grid[2, 0]), data)
    plot_per_task(fig.add_subplot(grid[:, 1]), data)
    keys = [("Base DeepSeek-V3.1", BASE)] + [ARMS[arm] for arm in data["arms"]]
    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", ms=8, color=colour, label=label)
            for label, colour in keys
        ],
        loc="upper left",
        bbox_to_anchor=(0.225, 0.995),
        ncols=len(keys),
        frameon=False,
        fontsize=10,
        labelcolor=INK2,
    )
    out = RUNS / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"results.{ext}", dpi=150, facecolor=SURFACE)
    print(table(data))
    print(f"wrote {out}/results.png and .svg")


if __name__ == "__main__":
    main()
