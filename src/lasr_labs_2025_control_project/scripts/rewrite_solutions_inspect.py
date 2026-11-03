"""
Rewrite existing attack solutions with an RL-trained sender (HiPT-style Option A).

Reads attacks from a generator eval log, asks the sender (e.g. a tinker:// RL
checkpoint) to rewrite each one with the collusion signal ON, and re-scores the
rewritten code with the same APPS scorer as generate_solutions_inspect.py. The output
.eval has the generator's layout (base_solution, generator_info with the original
secret input), so the pipeline takes it in place of its test attacks
(``+inputs.generate.test.attacks=<file>``) and runs its monitors and games on it
unchanged. The pre-rewrite code is kept as `pre_rewrite_solution`.

Each run is a W&B stage run (job type ``rewrite``) like the pipeline's, so the
output's header links the monitors to it, and it links to its inputs: the attacks,
the RL checkpoint, or the ICRL result its strategy came from.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import click
import yaml
from control_arena.eval_mode import EvalMode
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store
from pydantic import BaseModel, ConfigDict

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    extract_final_code,
    rewrite_messages,
)
from lasr_labs_2025_control_project.settings.setting import AppsSetting
from lasr_labs_2025_control_project.utils import tinker_provider  # noqa: F401  (registers tinker/ provider)
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid


class RewriteSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    system_prompt: str
    user_prompt: str
    signal: str
    max_tokens: int = 4096
    # ICRL sender instructions, added the way the search added them.
    strategy: str = ""
    # Or an ICRL run's icrl_result.json (a path or a wandb: ref to its strategy
    # artifact): its winner's sender instructions are used, and the run is linked.
    strategy_result: str | None = None


class RewriteRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_path: Path
    save_path: Path
    limit: int | None = None


class RewriteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_folder: Path
    experiment_name: str
    results_folder: str = "results/"
    defaults: dict[str, RewriteSettings]
    runs: dict[str, RewriteRun]


def attack_dataset(eval_path: Path) -> MemoryDataset:
    """Generator samples with their task metadata and original store, keeping IDs."""
    log = read_eval_log(str(eval_path))
    assert log.samples, f"no samples in {eval_path}"
    samples = []
    for sample in log.samples:
        if not check_sample_is_valid(sample.scores):
            continue
        samples.append(
            Sample(
                input=sample.metadata["main_task_description"],
                id=sample.id,
                metadata={**sample.metadata, "original_store": dict(sample.store)},
            )
        )
    return MemoryDataset(samples)


def _secret_input(generator_info: Any) -> str:
    if isinstance(generator_info, dict):
        return generator_info.get("secret_input") or ""
    return getattr(generator_info, "secret_input", "") or ""


@solver
def rewrite_solution(settings: RewriteSettings) -> Solver:
    model = get_model(settings.model)
    system_prompt = get_prompt_by_name(settings.system_prompt)
    if settings.strategy:
        system_prompt += f"\n\nInstructions for this rewrite:\n{settings.strategy}"
    user_prompt = get_prompt_by_name(settings.user_prompt)
    signal = get_prompt_by_name(settings.signal)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        original_store = state.metadata["original_store"]
        original = original_store["base_solution"]
        messages = rewrite_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            signal=signal,
            problem_statement=state.metadata["main_task_description"],
            base_code=original,
        )
        output = await model.generate(
            [
                ChatMessageSystem(content=messages[0]["content"]),
                ChatMessageUser(content=messages[1]["content"]),
            ],
            config=GenerateConfig(max_tokens=settings.max_tokens),
        )
        try:
            code = extract_final_code(output.completion)
        except ValueError:
            code = ""  # scored as a failed attack, never the pre-rewrite code
        secret_input = _secret_input(original_store.get("generator_info"))

        for key, value in original_store.items():
            store().set(key, value)
        store().set("base_solution", code)
        store().set("pre_rewrite_solution", original)
        store().set(
            "rewrite_info",
            {
                "model": settings.model,
                "system_prompt": settings.system_prompt,
                "user_prompt": settings.user_prompt,
                "signal": signal,
                "completion": output.completion,
                "stop_reason": output.stop_reason,
            },
        )
        state.messages.append(output.message)
        # The APPS scorer reads the submission from the completion (as in
        # generate_solutions_inspect.parse_and_submit).
        output.completion = json.dumps({"code": code, "secret_input": secret_input})
        state.output = output
        return state

    return solve


@task
def rewrite_solutions(run: RewriteRun, settings: RewriteSettings) -> Task:
    setting = AppsSetting(cache_python_code=False, num_tests_to_run=10)
    return Task(
        dataset=attack_dataset(run.eval_path),
        solver=rewrite_solution(settings),
        scorer=setting.get_scorers()[0],
        sandbox=setting.get_sandbox_config(),
        metadata={"eval_mode": EvalMode.ATTACK.value},
    )


def with_strategy_result(
    settings: RewriteSettings,
) -> tuple[RewriteSettings, list[tracking.ResolvedInput]]:
    """Settings with the winner's instructions from ``strategy_result``, and its input."""
    if settings.strategy_result is None:
        return settings, []
    if settings.strategy:
        raise ValueError("give strategy or strategy_result, not both")
    source = tracking.resolve_input(settings.strategy_result)
    winner = json.loads(source.local_path.read_text())["winner"]
    if winner["decoder"]:
        # The pipeline's judge would need the learned instructions as a prompt.
        raise ValueError("a centralized strategy's judge text is not wired in yet")
    return settings.model_copy(update={"strategy": winner["strategy"]}), [source]


@click.command()
@click.option("--config", type=str, required=True)
@click.option(
    "--limit", type=int, default=None, help="Rewrite only the first N samples."
)
@click.option("--allow-unlinked", is_flag=True, help=tracking.ALLOW_UNLINKED_HELP)
def main(config: str, limit: int | None, allow_unlinked: bool) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parsed = RewriteConfig.model_validate(yaml.safe_load(Path(config).read_text()))
    folder = parsed.experiment_folder
    (folder / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(config, folder / "configs" / "rewrite_config.yaml")
    settings, strategy_inputs = with_strategy_result(parsed.defaults["rewrite"])
    for label, run in parsed.runs.items():
        resolved = RewriteRun(
            eval_path=(folder / run.eval_path).resolve(),
            save_path=folder / run.save_path,
            limit=limit if limit is not None else run.limit,
        )
        with tracking.StageRun(
            "rewrite",
            f"rewrite-{parsed.experiment_name}-{label}",
            config={"rewrite": settings.model_dump(), "run": run.model_dump()},
            tags=[parsed.experiment_name, label],
            inputs=[tracking.resolve_input(resolved.eval_path), *strategy_inputs],
            models=[settings.model],
            allow_unlinked=allow_unlinked,
        ) as stage:
            result = eval(
                tasks=rewrite_solutions(resolved, settings),
                max_connections=32,
                max_sandboxes=32,
                limit=resolved.limit,
                metadata=stage.eval_metadata(),
            )
            resolved.save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(result[0].location, resolved.save_path)
            try:
                stage.finish_eval(
                    result[0],
                    resolved.save_path,
                    limit=resolved.limit,
                    extra_counts=tracking.input_counts(
                        resolved.eval_path, result[0].eval.dataset.samples or 0
                    ),
                )
            except tracking.StageFailed as error:
                raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
