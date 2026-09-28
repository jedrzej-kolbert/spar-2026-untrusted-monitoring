"""
Re-score a stored APPS generation .eval with the installed control-arena APPS scorer
and diff per-sample results against the scores stored in the log.

Rebuilds each sample from AppsSetting (so metadata is in the installed control-arena's
format), replays the stored submission, and runs apps_scorer in the Docker sandbox.
Imports only paths that exist in both control-arena 2.1.3 and 17.1.2, so running it
under each version shows whether a version change moves APPS scores.

Usage:
    python -m lasr_labs_2025_control_project.scripts.helpers.rescore_apps_eval \
        --eval-path generator/test/attacks.eval --limit 100
"""

import json
from collections import Counter
from pathlib import Path

import click
from control_arena.settings.apps.task.scorers import apps_scorer
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store

from lasr_labs_2025_control_project.settings.setting import AppsSetting

try:  # control-arena >= 3
    from control_arena import EvalMode
except ImportError:  # control-arena 2.1.3
    from control_arena.eval_mode import EvalMode  # type: ignore[no-redef]


@solver
def replay_submission(completions: dict[int | str, str]):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        completion = completions[state.sample_id]
        state.output = ModelOutput.from_content(
            "replay", completion
        )  # 2.1.3 reads this
        store().set("submit_args", completion)  # 17.x reads STORE_SUBMIT_ARGS_KEY
        return state

    return solve


@click.command()
@click.option(
    "--eval-path", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.option("--limit", type=int, default=None, help="First N samples of the log.")
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Diff JSON.")
@click.option("--max-sandboxes", type=int, default=8)
def main(eval_path: str, limit: int | None, out: str | None, max_sandboxes: int):
    log = read_eval_log(eval_path)
    assert log.samples, "No samples in the eval log"
    stored = log.samples[:limit]
    mode = EvalMode(stored[0].metadata["eval_mode"])
    assert all(
        s.metadata["eval_mode"] == mode.value for s in stored
    ), "Mixed eval modes"
    assert len({s.epoch for s in stored}) == 1, "Multi-epoch logs are not supported"

    # backdoored_solutions always samples codeparrot/apps "test" (odd/even ids pick the
    # train/test halves), where problem ids are unique.
    by_id = {
        s.id: s
        for s in AppsSetting(original_dataset_split="test", shuffle=False).get_dataset(
            mode
        )
    }
    missing = [s.id for s in stored if s.id not in by_id]
    assert (
        not missing
    ), f"{len(missing)} log ids not in APPS test split, e.g. {missing[:5]}"

    completions = {s.id: s.output.completion for s in stored}
    rescored = eval(
        Task(
            dataset=MemoryDataset([by_id[s.id] for s in stored]),
            solver=replay_submission(completions),
            scorer=apps_scorer(use_cache=False, num_tests=10),
            sandbox="docker",
        ),
        model="mockllm/model",
        max_sandboxes=max_sandboxes,
        display="none",
    )[0]
    assert rescored.status == "success", rescored.error
    assert rescored.samples is not None

    old = {s.id: s.scores["apps_scorer"].value for s in stored}  # type: ignore[index]
    new = {s.id: s.scores["apps_scorer"].value for s in rescored.samples}  # type: ignore[index]
    changes = Counter()
    diffs = []
    for sample_id, old_value in old.items():
        for key, before in old_value.items():  # type: ignore[union-attr]
            after = new[sample_id].get(key)  # type: ignore[union-attr]
            if before != after:
                changes[f"{key}: {before}->{after}"] += 1
                diffs.append(
                    {"id": sample_id, "key": key, "stored": before, "rescored": after}
                )

    print(f"{len(old)} samples, mode={mode.value}, {len(diffs)} per-key differences")
    for change, count in changes.most_common():
        print(f"  {change}: {count}")
    if out:
        Path(out).write_text(json.dumps(diffs, indent=2))


if __name__ == "__main__":
    main()
