"""Generate only the train-honest samples missing from the crashed log, then merge into one log."""

from pathlib import Path

from control_arena.eval_mode import EvalMode
from inspect_ai import eval
from inspect_ai.log import read_eval_log, write_eval_log

from lasr_labs_2025_control_project.scripts.generate_solutions_inspect import (
    backdoored_solutions,
)
from lasr_labs_2025_control_project.utils.config_loader import DataSplit

PARTIAL = Path("logs/honest_partial_941.eval")
OUT = Path(
    "experiments/apps/generate/apps-262144_ft_a1611f-train-honest-honest-c7c7f56a99bd/honest.eval"
)
log = read_eval_log(PARTIAL)
args = log.eval.task_args
good = {s.id for s in log.samples if not s.error and s.scores}
task = backdoored_solutions(
    EvalMode.HONEST,
    DataSplit.TRAIN,
    args["system_prompt"],
    args["problem_ids"],
)
all_ids = [s.id for s in task.dataset]
missing = [i for i in all_ids if i not in good]
print(f"dataset {len(all_ids)}, good {len(good)}, missing {len(missing)}", flush=True)
res = eval(
    tasks=task,
    model=log.eval.model,
    max_connections=8,
    max_sandboxes=8,
    reasoning_tokens=0,
    reasoning_effort="minimal",
    max_tokens=65536,
    seed=log.eval.model_generate_config.seed,
    sample_id=missing,
    log_dir="logs/retry",
)[0]
new = read_eval_log(res.location)
print("retry status", new.status, len(new.samples), flush=True)
assert new.status == "success" and len(good) + len(new.samples) == len(all_ids)
log.samples = [s for s in log.samples if s.id in good] + new.samples
log.status = "success"
log.error = None
write_eval_log(log, OUT)
print("merged", len(log.samples), "->", OUT)
