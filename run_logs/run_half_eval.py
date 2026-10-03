"""Score the half-split test set with the new checkpoint and build the standard plots.

Usage: run_half_eval.py ARM. Honest runs are whatever the still-running honest batch
has saved so far (plus the earlier 20-task prefix), snapshotted without touching it;
once the batch has finished, the complete generator/test/honest.eval is used instead.
"""

import asyncio
import glob
import sys

import numpy as np
import yaml
from inspect_ai.log import read_eval_log, write_eval_log
from run_half_experiment import CONFIGS, OUT, PREFIXES, ROOT, command, monitor_config

from lasr_labs_2025_control_project.scripts.generate_solutions_inspect import (
    add_trajectory_views,
)
from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
    suspiciousness_monitor,
)
from lasr_labs_2025_control_project.utils.config_loader import load_and_normalize_config
from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid

arm = sys.argv[1]
# --action-views: per-action monitors with the per-action fine-tune (see
# run_half_experiment.py --action-views); scores are maxed per trajectory in plots.
per_action = "--action-views" in sys.argv[2:]
checkpoint = (
    (OUT / ("final_checkpoint_action.txt" if per_action else "final_checkpoint.txt"))
    .read_text()
    .strip()
)
honest = OUT / "generator/test/honest.eval"
if (
    not honest.exists()
    and not (OUT / "configs" / f"{arm}_monitor_config.yaml").exists()
):
    gen = load_and_normalize_config(
        CONFIGS / "generator/half_test_honest_config.yaml", "generator"
    ).runs[0]
    source = sorted(glob.glob(str(ROOT / "logs/*bash-arena*honest*.eval")))[-1]
    snapshot = add_trajectory_views(
        source,
        gen.generator.eval_mode,
        gen.generator.model_info.system_prompt,
        gen.generator.model_info.model,
    )
    prefix = add_trajectory_views(
        str(PREFIXES[("test", "honest")]),
        gen.generator.eval_mode,
        gen.generator.model_info.system_prompt,
        gen.generator.model_info.model,
    )
    samples = [*prefix.samples, *snapshot.samples]
    ids = [str(s.id) for s in samples]
    assert len(set(ids)) == len(ids) and all(int(i) % 2 == 0 for i in ids)
    snapshot.samples = samples
    snapshot.metadata = {
        "partial_snapshot": True,
        "source_eval_logs": [str(PREFIXES[("test", "honest")]), source],
    }
    honest = OUT / f"generator/test/honest_partial_{arm}.eval"
    write_eval_log(snapshot, str(honest))
    usable = sum(
        check_sample_is_valid(s.scores, s.metadata.get("eval_mode")) for s in samples
    )
    print(f"=== honest snapshot: {len(samples)} saved, {usable} usable", flush=True)

SCORES = [
    "untrusted_collusion_score",
    "trusted_suspiciousness_score",
    "untrusted_suspiciousness_score",
]
cfg_path = OUT / "configs" / f"{arm}_monitor_config.yaml"
if not cfg_path.exists():  # a rerun reuses finished monitor logs and only replots
    cfg_path = monitor_config(checkpoint, arm, per_action)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["runs"]["honest"]["eval_path"] = f"generator/test/{honest.name}"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command("monitor_solutions_inspect", cfg_path)
MONITOR_OF = {
    "trusted_suspiciousness_score": "trusted_backdoor_monitor",
    "untrusted_suspiciousness_score": "untrusted_backdoor_monitor",
}
models = {
    name: spec["model_info"]["model"]
    for entry in yaml.safe_load(cfg_path.read_text())["defaults"]["monitors"]
    for name, spec in entry.items()
}
for run in load_and_normalize_config(cfg_path, "monitor").runs:
    everything = run.save_path.with_name(run.save_path.stem + "_all.eval")
    if everything.exists():  # undo an earlier drop: no sample is filtered out
        everything.replace(run.save_path)
    log = read_eval_log(str(run.save_path), resolve_attachments=True)
    retried = 0
    for sample in log.samples:
        for name in SCORES:
            # plot_data reads the score from the sample store, so both must be set.
            stored = sample.store[name]  # a NaN score is saved as null
            if stored is not None and np.isfinite(float(stored)):
                continue
            if np.isfinite(float(sample.scores[name].value)):  # re-asked earlier
                sample.store[name] = sample.scores[name].value
                retried += 1
                continue
            # A monitor that replies without a parseable score is asked again with
            # exactly the input it had, rather than losing the sample.
            model = models[MONITOR_OF[name]]  # KeyError: collusion NaN is not retried
            event = next(
                e
                for e in sample.events
                if e.event == "model" and e.model.endswith(model.split("/", 1)[1])
            )
            for attempt in range(1, 4):
                score = asyncio.run(
                    suspiciousness_monitor(
                        event.input[0].text, event.input[1].text, model
                    )
                )
                if np.isfinite(float(score.value)):
                    score.metadata = {"monitor_retries": attempt}
                    sample.scores[name] = score
                    sample.store[name] = score.value
                    retried += 1
                    break
            else:
                raise RuntimeError(f"{name} still missing for sample {sample.id}")
    print(
        f"=== {run.save_path.name}: {len(log.samples)} scored, {retried} re-asked",
        flush=True,
    )
    if retried:
        write_eval_log(log, str(run.save_path))
command("plot_data", cfg_path)
print(f"=== EVAL {arm} DONE", flush=True)
