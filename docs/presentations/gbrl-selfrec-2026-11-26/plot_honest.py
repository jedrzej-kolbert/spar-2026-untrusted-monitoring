# Honest-policy plots (slides h-frozen, h-trained, h-long, h-joint, h-temp) with readable text.
# Input: metrics.json {"rows": [{run, step, phase, reward, p_self_rec, auroc?}]}, built from each
# run's local metrics.jsonl, validation rollouts and honeypot scores (all items, ties count half).
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = json.load(open(sys.argv[1]))["rows"]
RUNS = {  # run key -> (letter, colour, marker)
    "selfrec": ("A", "#2a78d6", "o"),
    "sft": ("B", "#eb6834", "s"),
    "raw": ("C", "#14946a", "^"),
    "rawjoint": ("D", "#b77400", "D"),
    "parajoint": ("E", "#c0137a", "o"),
    "control": ("F", "#4a4a4a", "v"),
    "lowtemp": ("G", "#7a4fd0", "D"),
}


def series(run, phase, key):
    pts = sorted(
        (r["step"], r[key]) for r in ROWS if r["run"] == run and r["phase"] == phase
    )
    assert pts, (run, phase, key)
    return [p[0] for p in pts], [p[1] for p in pts]


def rolling(y, n=10):
    return [
        sum(y[max(0, i - n + 1) : i + 1]) / len(y[max(0, i - n + 1) : i + 1])
        for i in range(len(y))
    ]


def style(size):
    plt.rcParams.update(
        {
            "font.size": size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def train_pself(name, labels):
    style(17)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for run, label in labels.items():
        letter, col, mark = RUNS[run]
        x, y = series(run, "train", "p_self_rec")
        ax.plot(
            x,
            rolling(y),
            color=col,
            lw=2.5,
            marker=mark,
            markevery=max(1, len(x) // 10),
            ms=8,
            label=f"{letter} · {label}",
        )
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("GBRL step")
    ax.set_ylabel("train P(self | rewrite), 10-step mean")
    ax.legend(
        fontsize=16, loc="lower right" if "rawjoint" in labels else "center right"
    )
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    plt.close(fig)


def reward_auroc(name, labels):
    style(20)
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.4))
    for run, label in labels.items():
        letter, col, mark = RUNS[run]
        x, y = series(run, "train", "reward")
        a.plot(x, y, color=col, lw=1.8, label=f"{letter} · {label}")
        x, y = series(run, "validation", "auroc")
        b.plot(x, y, color=col, lw=3, marker=mark, ms=9, label=f"{letter} · {label}")
    b.axhline(0.5, ls=":", color="#555", lw=2)
    a.set_title("Training reward (per step)")
    b.set_title("Validation AUROC (dotted: chance)")
    for ax in (a, b):
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training step")
    handles, names = a.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, fontsize=21)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(name, dpi=100)
    plt.close(fig)


train_pself(
    "honest_frozen_um.png",
    {
        "selfrec": "base, paraphrased",
        "sft": "SFT U_M, paraphrased",
        "raw": "base, raw code",
    },
)
train_pself(
    "honest_trained_um.png",
    {
        "rawjoint": "raw code",
        "parajoint": "paraphrased",
        "control": "paraphrased, U_G frozen",
        "lowtemp": "paraphrased, T = 0.1",
    },
)
reward_auroc(
    "honest_long.png", {"selfrec": "paraphrased, base U_M", "raw": "raw code, base U_M"}
)
reward_auroc(
    "honest_joint.png", {"raw": "raw code, base U_M", "rawjoint": "raw code, joint U_M"}
)
reward_auroc(
    "honest_temp.png",
    {"parajoint": "paraphrased T = 1, joint", "lowtemp": "paraphrased T = 0.1, joint"},
)
for run, (letter, _, _) in RUNS.items():
    x, y = series(run, "validation", "reward")
    _, a = series(run, "validation", "auroc")
    print(
        letter,
        "best val reward",
        round(max(y), 2),
        "| AUROC first",
        round(a[0], 2),
        "last",
        round(a[-1], 2),
        "max",
        round(max(a), 2),
    )
