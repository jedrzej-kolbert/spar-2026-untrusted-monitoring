"""V4.1 honest train generations -> JSONL of passing solutions (same extraction the SFT step uses)."""

import json
from inspect_ai.log import read_eval_log
from lasr_labs_2025_control_project.finetuning.utils.eval_to_dataset import (
    eval_log_to_dataset,
)

log = read_eval_log(
    "experiments/apps/generate/apps-262144_ft_a1611f-train-honest-honest-c7c7f56a99bd/honest.eval"
)
ds = eval_log_to_dataset(log, "deepseek-ai/DeepSeek-V4.1-Flash")
print(type(ds), [a for a in dir(ds) if not a.startswith("_")][:15])
rows = [
    r.model_dump() if hasattr(r, "model_dump") else dict(r)
    for r in getattr(ds, "solutions", ds)
]
with open("datasets/v41_honest_train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, default=str) + "\n")
print(len(log.samples), "samples ->", len(rows), "passing rows")
