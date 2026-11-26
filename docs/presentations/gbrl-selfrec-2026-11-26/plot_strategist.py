# Strategist GBRL run #2 (W&B collusion-rl-v2 7wg314xz) training curves, per step, no smoothing.
# train = pooled over the 8 strategies sampled that step; validation = that step's best strategy
# on validation problems. No validation loss is logged.
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = json.load(open(sys.argv[1]))
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
for ax, key, title in ((axes[0], "reward", "Reward"), (axes[1], "auroc", "AUROC")):
    ax.plot(*zip(*H["train_" + key]), color="#2a78d6", lw=2, label="train")
    ax.plot(
        *zip(*H["val_" + key]), "-o", color="#eb6834", lw=3, ms=9, label="validation"
    )
    ((_, base),) = H["base_" + key]
    ax.axhline(base, ls=":", color="#555", lw=2.5, label="validation, no strategy")
    ax.set_title(title)
axes[0].set_ylim(0, 0.65)
axes[1].set_ylim(0.4, 1.0)
axes[2].plot(*zip(*H["loss"]), color="#2a78d6", lw=1.8)
axes[2].axhline(0, color="#555", lw=1)
axes[2].set_title("PPO loss (train only)")
for ax in axes:
    ax.set_xlabel("RL step")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=21)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("strategist_s1_curves.png", dpi=100)
