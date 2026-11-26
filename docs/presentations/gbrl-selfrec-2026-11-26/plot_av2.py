# collusion-rl-v2 attack GBRL runs (slide a-v2): train reward with a 0-25 step zoom,
# and validation AUROC / working rate. Input: W&B history dump {run_id: [rows]}.
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = json.load(open(sys.argv[1]))
RUNS = {  # run id -> (label, colour)
    "lmi2pue1": ("valid_sigmoid s0", "#2a78d6"),
    "xvv9fm3a": ("valid_sigmoid s1", "#2a78d6"),
    "y30wbe0h": ("valid_sigmoid s2", "#2a78d6"),
    "11r71on7": ("additive", "#eb6834"),
    "dyldu3i5": ("recognition_only", "#1baf7a"),
}
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def series(rid, key):
    pts = sorted((r["_step"], r[key]) for r in H[rid] if key in r)
    return [p[0] for p in pts], [p[1] for p in pts]


def rolling(y, n=10):
    return [
        sum(y[max(0, i - n + 1) : i + 1]) / len(y[max(0, i - n + 1) : i + 1])
        for i in range(len(y))
    ]


# train reward: full run (10-step mean) and zoom on steps 0-25 (raw per step)
fig, (a, b) = plt.subplots(
    1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.4, 1]}
)
seen = set()
for rid, (lab, col) in RUNS.items():
    x, y = series(rid, "train/reward")
    name = lab.split(" s")[0]
    a.plot(x, rolling(y), color=col, lw=2, label=None if name in seen else name)
    b.plot(x, y, "-o", color=col, lw=1.6, ms=4, alpha=0.9)
    seen.add(name)
a.set_title("Train reward (10-step mean)")
b.set_title("Zoom: steps 0–25 (per step)")
b.set_xlim(-0.5, 25)
for ax in (a, b):
    ax.set_xlabel("GBRL step")
a.set_ylabel("train reward")
a.legend(fontsize=11)
fig.tight_layout()
fig.savefig("av2_train_reward.png", dpi=150)

# validation: AUROC (signalled attacks vs honeypots) and working rate
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
seen = set()
for rid, (lab, col) in RUNS.items():
    name = lab.split(" s")[0]
    for ax, key in ((a, "validation/auroc"), (b, "validation/working_rate")):
        x, y = series(rid, key)
        ax.plot(
            x,
            y,
            "-o",
            color=col,
            lw=2,
            ms=6,
            label=None if name in seen or ax is b else name,
        )
    seen.add(name)
a.axhline(0.5, ls=":", color="#555")
a.set_title("Validation AUROC")
b.set_title("Validation working rate")
a.set_ylim(0.4, 0.85)
b.set_ylim(-0.02, 1.02)
for ax in (a, b):
    ax.set_xlabel("GBRL step")
a.legend(fontsize=11)
fig.tight_layout()
fig.savefig("av2_validation.png", dpi=150)
for rid, (lab, _) in RUNS.items():
    print(
        lab,
        "val auroc",
        [round(v, 3) for v in series(rid, "validation/auroc")[1]],
        "working",
        [round(v, 2) for v in series(rid, "validation/working_rate")[1]],
    )
