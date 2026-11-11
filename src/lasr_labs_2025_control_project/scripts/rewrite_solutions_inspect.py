"""
Rewrite existing attack solutions with an RL-trained sender (HiPT-style Option A).

Reads attacks from a generator eval log, asks the sender (e.g. a tinker:// RL
checkpoint) to rewrite each one with the collusion signal ON, and re-scores the
rewritten code with the same APPS scorer as generate_solutions_inspect.py. The output
.eval has the generator's layout (base_solution, generator_info with the original
secret input), so transform_solutions_inspect.py and monitor_solutions_inspect.py
run on it unchanged. The pre-rewrite code is kept as `pre_rewrite_solution`.

Run as the pipeline's ``rewrite`` stage: a monitor arm with ``rewrite:`` set
(``conf/experiment/rl/``) rewrites the test attacks before they are transformed
and monitored.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Optional

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

from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    extract_final_code,
    rewrite_messages,
)
from lasr_labs_2025_control_project.settings.setting import AppsSetting
from lasr_labs_2025_control_project.tracking import StageRun
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
def rewrite_solution(
    settings: RewriteSettings,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Solver:
    model = get_model(settings.model)
    system_prompt = get_prompt_by_name(settings.system_prompt)
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
            config=GenerateConfig(
                max_tokens=settings.max_tokens, seed=seed, temperature=temperature
            ),
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
def rewrite_solutions(
    eval_path: Path,
    settings: RewriteSettings,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Task:
    setting = AppsSetting(cache_python_code=False, num_tests_to_run=10)
    return Task(
        dataset=attack_dataset(eval_path),
        solver=rewrite_solution(settings, seed, temperature),
        scorer=setting.get_scorers()[0],
        sandbox=setting.get_sandbox_config(),
        metadata={"eval_mode": EvalMode.ATTACK.value},
    )


def run_rewrite(
    eval_path: Path,
    save_path: Path,
    settings: RewriteSettings,
    tracker: StageRun,
    *,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    max_connections: int = 32,
) -> Any:
    """Rewrite the attacks in ``eval_path`` under ``tracker``; save to ``save_path``."""
    with tracker:
        result = eval(
            tasks=rewrite_solutions(eval_path, settings, seed, temperature),
            max_connections=max_connections,
            max_sandboxes=max_connections,
            limit=limit,
            metadata=tracker.eval_metadata(),
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result[0].location, save_path)
        tracker.finish_eval(
            result[0], save_path, limit=limit, metadata={"label": "attacks"}
        )
    return result[0]
