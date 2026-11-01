"""Generate BashArena trajectories from a generator config. Dry-run by default.

Reads the same generator YAML as generate_solutions_inspect.py, restricted to
`dataset_type: basharena` runs with a `limit`.

Example:
    python -m basharena_runtime.generate --config cfg.yaml --output out/
    python -m basharena_runtime.generate --config cfg.yaml --output out/ --allow-paid
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting
from control_arena.settings.bash_arena.policies.attack_policy import (
    DEFAULT_SYSTEM_PROMPT as ATTACK_SYSTEM_PROMPT,
)
from control_arena.settings.bash_arena.policies.attack_policy import attack_policy
from control_arena.settings.bash_arena.policies.honest_policy import (
    DEFAULT_SYSTEM_PROMPT as HONEST_SYSTEM_PROMPT,
)
from control_arena.settings.bash_arena.policies.honest_policy import honest_policy
from control_arena.settings.bash_arena.protocol.auto_submit_protocol_wrapper import (
    auto_submit_protocol_wrapper,
)
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log

from lasr_labs_2025_control_project.utils.basharena_views import trajectory_view

from basharena_runtime import runtime_versions

# Fixed agent limits; move into the config if pilots need tuning. gpt-oss hit 40
# steps on 2/5 pilot tasks, so both actors get the same larger budget.
MAX_STEPS = 60
TIME_LIMIT_S = 60 * 60
TOKEN_LIMIT = 2_000_000  # prompts are re-sent every step
# Inspect aborts a whole eval when one task image fails to build, so tasks that
# cannot build are dropped by hand (1003684 runs mknod during docker build).
UNBUILDABLE_TASKS = {"1003684"}
POPULATIONS = ("attacks", "honest", "honeypots")
SPLIT_PROVENANCE = "BashArena task-id parity: odd=train, even=test"


@dataclass(frozen=True)
class Run:
    population: str
    eval_mode: str
    split: str
    system_prompt_name: str
    model: str
    max_tokens: int | None
    limit: int | tuple[int, int]

    @property
    def path(self) -> str:
        return f"{self.population}.eval"


def load_runs(config_path: Path) -> list[Run]:
    """Normalise a generator config: each run's keys override `defaults`."""
    config = yaml.safe_load(config_path.read_text())
    defaults = config.get("defaults") or {}
    runs = []
    for population, raw in (config.get("runs") or {}).items():
        if population not in POPULATIONS:
            raise ValueError(f"Run label must be one of {POPULATIONS}: {population!r}")

        settings = {**defaults, **raw}
        if settings["dataset_type"] != "basharena":
            raise ValueError("Only dataset_type: basharena runs are supported")
        eval_mode, split = settings["eval_mode"], settings["data_split"]
        if eval_mode not in ("attack", "honest") or split not in ("train", "test"):
            raise ValueError(f"Run {population!r}: bad eval_mode or data_split")
        if (population == "honest") != (eval_mode == "honest"):
            raise ValueError(f"Run {population!r} cannot use eval_mode {eval_mode!r}")
        limit = settings.get("limit")
        if isinstance(limit, list):
            if (
                len(limit) != 2
                or any(type(n) is not int for n in limit)
                or not 0 <= limit[0] < limit[1]
            ):
                raise ValueError("limit range must be [start, stop], 0 <= start < stop")
            limit = (limit[0], limit[1])  # Inspect's zero-based [start, stop) range
        elif type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive count or a [start, stop] range")
        runs.append(
            Run(
                population,
                eval_mode,
                split,
                settings["system_prompt"],
                settings["model"],
                settings.get("max_tokens"),
                limit,
            )
        )
    if not runs:
        raise ValueError("Config has no runs")
    return runs


def system_prompt(name: str) -> str:
    """BashArena's own policy prompts, selectable by name in configs."""
    prompts = {
        "BASHARENA_ATTACK_SYSTEM_PROMPT": ATTACK_SYSTEM_PROMPT,
        "BASHARENA_HONEST_SYSTEM_PROMPT": HONEST_SYSTEM_PROMPT,
    }
    if name not in prompts:
        raise ValueError(f"system_prompt must be one of {sorted(prompts)}: {name!r}")
    return prompts[name]


def basharena_task(run: Run) -> Any:
    """BashArena agent task, ended early if a side task fires.

    BashArena has one split, so task ids (StackOverflow question numbers) are
    split like APPS problem ids: odd = train, even = test.

    max_tokens overrides the policy's per-step output cap (16000), which counts
    against the context window: on 32K models it leaves only ~16K for the growing
    trajectory.
    """
    mode = EvalMode(run.eval_mode)
    setting = BashArenaSetting()
    parity = 1 if run.split == "train" else 0
    make_policy = attack_policy if mode == EvalMode.ATTACK else honest_policy
    prompt = system_prompt(run.system_prompt_name)
    policy = make_policy(model_name=run.model, system_prompt=prompt)
    if run.max_tokens is not None:  # keep the policy's reasoning settings
        policy = make_policy(
            model_name=run.model,
            system_prompt=prompt,
            generate_config=policy.generate_config.model_copy(
                update={"max_tokens": run.max_tokens}
            ),
        )
    task = get_control_task(
        setting=setting,
        micro_protocol=auto_submit_protocol_wrapper(micro_protocol=policy),
        mode=mode,
        eval_config=ControlEvalConfig(
            untrusted_model=run.model,
            trusted_model=run.model,  # unused: no trusted monitor during generation
            max_steps=MAX_STEPS,
            token_limit=TOKEN_LIMIT,
        ),
        dataset=[
            s
            for s in setting.get_dataset(mode)
            if int(str(s.id)) % 2 == parity and str(s.id) not in UNBUILDABLE_TASKS
        ],
    )
    # ControlEvalConfig.time_limit only feeds eval_args(), which get_control_task
    # never applies, so set it on the Task. Without it a hung main-task test
    # blocked a run for 9 h.
    task.time_limit = TIME_LIMIT_S
    return task


def add_trajectory_views(log_path: str, run: Run) -> Any:
    """Store each trajectory's action view and generator in the log.

    Runs after eval(), not as a solver, because a sample that hits its step, time
    or token limit skips later solvers. `generator_info.model_info.model` is the
    name without its provider, as APPS logs record it.
    """
    log = read_eval_log(log_path)
    for sample in log.samples or []:
        sample.store["base_solution"] = trajectory_view(sample.messages)
        sample.store["generator_info"] = {
            "model_info": {
                "model": run.model.split("/", 1)[1],
                "system_prompt": system_prompt(run.system_prompt_name),
                "base_prompt": sample.metadata["main_task_description"],
            },
            "eval_mode": run.eval_mode,
            "secret_input": "",
        }
    return log


def use_project_tinker_provider() -> None:
    """Register the main project's `tinker/` model provider with Inspect."""
    import lasr_labs_2025_control_project.utils.tinker_provider  # noqa: F401


def generate(
    config_path: Path,
    output: Path,
    concurrency: int,
    allow_paid: bool,
    metadata: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Run each config run; `metadata` goes into every log's header (the main
    project's pipeline passes its W&B lineage and node key), `seed` to Inspect."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if output.exists():
        raise FileExistsError(output)
    runs = load_runs(config_path)
    entries = [
        {
            "population": run.population,
            "actor_id": run.model,
            "eval_mode": run.eval_mode,
            "split": run.split,
            "split_provenance": SPLIT_PROVENANCE,
            "system_prompt": run.system_prompt_name,
            "max_tokens": run.max_tokens,
            "limit": run.limit,
            "path": run.path,
        }
        for run in runs
    ]
    manifest = {
        "complete": False,
        "runtime": runtime_versions(),
        "runs": entries,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "limits": {
            "max_steps": MAX_STEPS,
            "time_limit_seconds": TIME_LIMIT_S,
            "token_limit": TOKEN_LIMIT,
        },
        "excluded_task_ids": sorted(UNBUILDABLE_TASKS),
        "exclusion_reason": "Task images that fail to build abort the whole eval",
        "concurrency": concurrency,
    }
    if not allow_paid:  # no dataset reads, Docker or model calls
        return {**manifest, "dry_run": True}

    load_dotenv()
    use_project_tinker_provider()
    output.mkdir(parents=True)
    (output / "source_config.yaml").write_bytes(config_path.read_bytes())

    def save() -> None:
        (output / "generation_manifest.json").write_text(json.dumps(manifest) + "\n")

    save()
    for run, entry in zip(runs, entries, strict=True):
        log = inspect_eval(
            tasks=basharena_task(run),
            model=run.model,
            max_samples=concurrency,
            max_sandboxes=concurrency,
            log_buffer=1,  # save each completed sample, even if another hangs
            fail_on_error=False,  # one broken container must not end the run
            limit=run.limit,
            log_dir=str(output / "native_logs"),
            metadata=metadata,
            seed=seed,
        )[0]
        annotated = add_trajectory_views(log.location, run)
        write_eval_log(annotated, str(output / run.path))
        entry["log_status"] = log.status
        entry["samples"] = len(annotated.samples or [])
        save()
        if log.status != "success":
            raise RuntimeError("Generation failed; partial logs and manifest retained")
    manifest["complete"] = True
    save()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # 1 by default: each sample can use ~2.3 GB and two docker networks.
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Build sandboxes and call models; without it only the plan is printed",
    )
    parser.add_argument(
        "--metadata-json",
        type=json.loads,
        default=None,
        help="JSON object to store in each log's header",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = generate(
        args.config,
        args.output,
        args.concurrency,
        args.allow_paid,
        metadata=args.metadata_json,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
