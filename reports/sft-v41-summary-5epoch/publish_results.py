"""Copy the completed five-epoch SFT result and add two named Collusion RL panels.

Run with ephemeral wandb-workspaces and scikit-learn dependencies. This script
makes no model calls and refuses to publish an incomplete or over-budget result.
"""

from collections import defaultdict
import json
from pathlib import Path

import wandb
from sklearn.metrics import roc_auc_score
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "spar-untrusted-monitoring-collusion"
SOURCE = f"{ENTITY}/untrusted-runs/l0dbt9nf"
DESTINATION = "collusion-rl-v2"
MIRROR_ID = "sft5p1105"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/sft-v41-summary-5epoch"
REPORT = Path(__file__).resolve().parent
LABEL = "5-epoch SFT | V4.1 | summarised attacks"

api = wandb.Api(timeout=60)
source = api.run(SOURCE)
assert source.state == "finished", f"Source is not finished: {source.state}"
summary = dict(source.summary)
assert summary.get("complete") is True and source.config["epochs"] == 5
preflight = json.loads((OUT / "budget-preflight.json").read_text())
ledger = json.loads((OUT / "budget-ledger.json").read_text())
assert summary["usage/train_tokens"] == preflight["train_tokens"]
assert ledger["reserved_or_spent_usd"] <= preflight["compute_limit_usd"] <= 20
assert summary["train/step"] == 50 and ledger["scoring_calls"] == 3360

scores = defaultdict(list)
for line in (OUT / "scores.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["label"] is not None:
        scores[row["checkpoint"]].append(row)
checkpoints = []
for checkpoint, rows in scores.items():
    assert len(rows) == len({r["prompt_sha256"] for r in rows}) == 555
    auroc = roc_auc_score(
        [r["label"] for r in rows], [r["yes"] - r["no"] for r in rows]
    )
    step = int(checkpoint.rsplit("-step-", 1)[1]) if "-step-" in checkpoint else 50
    checkpoints.append({"step": step, "auroc": float(auroc), "checkpoint": checkpoint})
checkpoints.sort(key=lambda r: r["step"])
assert [r["step"] for r in checkpoints] == [9, 18, 27, 36, 45, 50]
assert abs(checkpoints[-1]["auroc"] - summary["checkpoint/roc_auc"]) < 1e-12

history = [row for row in source.scan_history() if "train/step" in row]
assert any(row.get("train/step") == 50 and "train/loss" in row for row in history)
assert sum("checkpoint/roc_auc" in row for row in history) == 6
(OUT / "source-history.json").write_text(json.dumps(history, indent=2) + "\n")
result = {
    "label": LABEL,
    "source_run": source.url,
    "main_commit": preflight["main_commit"],
    "epochs": 5,
    "train_examples": 580,
    "heldout_examples": 30,
    "test_examples": 555,
    "checkpoint_metrics": checkpoints,
    "final_auroc": checkpoints[-1]["auroc"],
    "uncached_compute_estimate_usd": ledger["reserved_or_spent_usd"],
    "cap_usd": 20,
    "checkpoint_retention_excluded": True,
    "selection": preflight["selection"],
    "scope": "Training and self-recognition AUROC on cached summarised attacks; no new safety-game evaluation.",
}
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
summary_updates = {
    "budget/cap_usd": 20,
    "budget/uncached_compute_estimate_usd": ledger["reserved_or_spent_usd"],
    "baseline/epochs": 5,
    "baseline/channel": "paraphrased",
    "baseline/final_checkpoint_selected_by": "predetermined epoch 5",
}
source.summary.update(summary_updates)
source.name = LABEL
source.update()

with wandb.init(
    entity=ENTITY,
    project=DESTINATION,
    id=MIRROR_ID,
    resume="allow",
    name=LABEL,
    job_type="finetune",
    tags=["baseline-sft", "5-epoch", "summary-trained", "signal-jamming"],
    config={
        **source.config,
        "baseline_id": "sft-v41-summary-5epoch",
        "source_run": source.url,
        "source_project": "untrusted-runs",
        "run": {"arm": "sft-5epoch", "kind": "baseline"},
        "condition": {"transform": "paraphrased"},
        "budget_cap_usd": 20,
    },
    notes="Imported result, not a second training run. Fixed five-epoch endpoint; test AUROC is diagnostic. No new safety-game scores.",
) as mirror:
    mirror.define_metric("train/step")
    mirror.define_metric("train/loss", step_metric="train/step")
    mirror.define_metric("checkpoint/roc_auc", step_metric="train/step")
    for i, row in enumerate(history):
        mirror.log({k: v for k, v in row.items() if not k.startswith("_")}, step=i)
    mirror.summary.update({k: v for k, v in summary.items() if not k.startswith("_")})
    mirror.summary.update(summary_updates)
    artifact = wandb.Artifact(
        "sft-v41-summary-5epoch-results",
        type="sft-results",
        metadata={"source_run": source.url, "epochs": 5, "cap_usd": 20},
    )
    for name in [
        "results.json",
        "budget-preflight.json",
        "budget-ledger.json",
        "scores.jsonl",
        "source-history.json",
    ]:
        artifact.add_file(str(OUT / name), name=name)
    artifact.add_file(
        str(ROOT / "conf/experiment/self_rec_v41_summary_5epoch.yaml"),
        name="experiment.yaml",
    )
    artifact.add_file(
        "/home/jkk/.claude/jobs/2ee5e883/tmp/run_capped_sft.py",
        name="run_capped_sft.py",
    )
    mirror.log_artifact(artifact)
    result["mirrored_run"] = mirror.url

panels = [
    wr.LinePlot(
        title="5-epoch SFT — training loss (summarised attacks)",
        x="train/step",
        y=["train/loss"],
        title_x="Optimizer update (50 = 5 epochs)",
        title_y="Cross-entropy loss",
        range_x=(0, 50),
    ),
    wr.LinePlot(
        title="5-epoch SFT — collusion AUROC (summarised test attacks)",
        x="train/step",
        y=["checkpoint/roc_auc"],
        title_x="Optimizer update (50 = 5 epochs)",
        title_y="Self-recognition AUROC",
        range_x=(0, 50),
        range_y=(0, 1),
        line_marks={"checkpoint/roc_auc": "points"},
    ),
]
view = ws.Workspace(
    entity=ENTITY,
    project=DESTINATION,
    name="5-epoch SFT — summarised attacks",
    sections=[
        ws.Section(name="5-epoch SFT · summary-trained", panels=panels, is_open=True)
    ],
    runset_settings=ws.RunsetSettings(
        filters=[ws.Config("baseline_id") == "sft-v41-summary-5epoch"]
    ),
).save_as_new_view()
assert len(view.sections) == 1 and len(view.sections[0].panels) == 2
# Preserve the /wandb reverse-proxy prefix that the Workspace SDK drops.
base_ui = source.url.split(f"/{ENTITY}/", 1)[0]
view_query = view.url.split("?", 1)[1]
result["plots_view"] = f"{base_ui}/{ENTITY}/{DESTINATION}?{view_query}"
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
(REPORT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
