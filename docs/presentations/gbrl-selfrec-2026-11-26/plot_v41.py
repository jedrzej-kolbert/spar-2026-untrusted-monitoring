# V4.1-only redraw of mentor-deck slides 26/27 (same data CSV, V3.1 lines dropped).
import csv
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = list(csv.DictReader(open(sys.argv[1])))


def series(run, phase, key):
    pts = [
        (int(r["step"]), float(r[key]))
        for r in rows
        if r["run"] == run and r["phase"] == phase and r[key]
    ]
    return zip(*sorted(pts))


COL = {"B": "#2a78d6", "C": "#eb6834"}
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def lines(ax, phase, key, f=lambda v: v, runs=("B", "C"), prefix="V4.1 "):
    for a in runs:
        x, y = series(prefix + a, phase, key)
        ax.plot(
            x, [f(v) for v in y], "-o", ms=5, lw=2.2, color=COL[a], label=f"V4.1 {a}"
        )
    ax.set_xlabel("Training update")
    ax.legend(fontsize=11)


# training split
fig, axs = plt.subplots(2, 3, figsize=(13, 8.2))
axs = axs.ravel()
for ax, (k, t) in zip(
    axs,
    [
        ("reward", "reward"),
        ("term_task", "pass (tests pass)"),
        ("term_self_recognition", "pass × P(self)"),
        ("term_backdoor_kept", "kept (backdoor fires)"),
        ("term_backdoor_suspicion", "− suspicion"),
    ],
):
    lines(ax, "train", k)
    ax.set_title("Training: " + t)
axs[5].axis("off")
fig.tight_layout()
fig.savefig("v41_reward_terms.png", dpi=150)
# test split
fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
axs = axs.ravel()
for ax, (k, t) in zip(
    axs,
    [
        ("suspicion", "mean suspicion"),
        ("term_backdoor_kept", "kept (backdoor fires)"),
        ("p_yes", "mean P(self)"),
    ],
):
    lines(ax, "eval", k)
    ax.set_title("Test: " + t)
    ax.set_ylim(0, 1.03)
ax = axs[3]
for a in "BC":
    pts = sorted(
        (int(r["step"]), float(r["auroc"]), float(r["ci_low"]), float(r["ci_high"]))
        for r in rows
        if r["run"] == f"Corrected B/C {a}" and r["model"] == "V4.1"
    )
    x, y, lo, hi = zip(*pts)
    ax.plot(x, y, "-o", ms=5, lw=2.2, color=COL[a], label=f"V4.1 {a}")
    ax.fill_between(x, lo, hi, color=COL[a], alpha=0.12)
ax.axhline(0.765, ls=":", color="#555")
ax.set_title("Test: AUROC (dotted = original attacks)")
ax.set_xlabel("Training update")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("v41_bc_test.png", dpi=150)
for a in "BC":
    for k in ["suspicion", "term_backdoor_kept", "p_yes"]:
        x, y = series("V4.1 " + a, "eval", k)
        print(a, k, [round(v, 3) for v in y])
    print(
        a,
        "auc",
        [
            (r["step"], round(float(r["auroc"]), 3))
            for r in rows
            if r["run"] == f"Corrected B/C {a}" and r["model"] == "V4.1"
        ],
    )
