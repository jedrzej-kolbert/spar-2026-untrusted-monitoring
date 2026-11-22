"""APPS grading for the RL loops: control-arena's scorer, in the pipeline's sandbox.

The RL loops are not Inspect evals, so ``_run`` opens the sandbox around each scorer
call itself. It can't be a nested ``eval_async``: that allows one call at a time,
resets Inspect's concurrency limits under the loop's model calls, and changes the
process's working directory. The ``inspect_ai.util._sandbox`` imports are private;
inspect-ai is pinned exactly in pyproject.toml.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from collections.abc import Awaitable, Callable

from control_arena.eval_mode import EvalMode
from control_arena.settings.apps.task.dataset import AppsTaskMetadata
from control_arena.settings.apps.task.scorers import has_backdoor, passes_tests
from control_arena.settings.apps.task.submission_types import (
    SubmitAttackArgs,
    SubmitHonestArgs,
)
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, NOANSWER, Score
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
)
from inspect_ai.util._sandbox.docker.cleanup import project_cleanup_startup
from inspect_ai.util._sandbox.docker.config import resolve_compose_file
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

from lasr_labs_2025_control_project.settings.setting import AppsSetting

from .core import AppsProblem, ExecutionGrade

_TASK = "grouped_rl"


def _to_grade(score: Score) -> ExecutionGrade:
    # Only the reason: the rest of the explanation quotes test inputs and program
    # output (up to 10 MB), and the ICRL learner is shown this text.
    reason = re.split(r"[:(]", score.explanation or "", maxsplit=1)[0].strip()
    return ExecutionGrade(score.value == CORRECT, reason)


def _infrastructure_failure(error: Exception) -> ExecutionGrade:
    return ExecutionGrade(
        False, infrastructure_error=f"sandbox infrastructure failure: {error}"
    )


class AppsEvaluator:
    """The two halves of ``apps_scorer``, each in a fresh container of its own, so
    code that tampers with its container cannot reach the reference solution's run."""

    def __init__(self, *, num_tests: int | None = 10, max_containers: int = 8) -> None:
        if num_tests is not None and num_tests < 1:
            raise ValueError("num_tests must be positive")
        self.num_tests = num_tests
        # An overloaded daemon turns correct code into timeouts, scored as failures.
        self._containers = asyncio.Semaphore(max_containers)
        # With no compose file Inspect writes ./.compose.yaml and deletes it when any
        # eval ends, breaking others in the same directory; so keep a private copy.
        self._folder = tempfile.TemporaryDirectory()
        self._compose = AppsSetting().get_sandbox_config().config or (
            resolve_compose_file(self._folder.name)
        )

    async def start(self) -> None:
        """Check Docker and pull the sandbox image if it is missing."""
        await DockerSandboxEnvironment.task_init(_TASK, self._compose)

    async def _run(self, score: Callable[[], Awaitable[Score]]) -> Score:
        async with self._containers:
            project_cleanup_startup()
            environments = await DockerSandboxEnvironment.sample_init(
                _TASK, self._compose, {}
            )
            try:
                # Not init_sandbox_environments_sample: its proxy keeps every exec's
                # input and output in the transcript for the whole run.
                sandbox_environments_context_var.set(environments)
                sandbox_default_context_var.set(next(iter(environments)))
                result = await score()
                # The scorer reports a Docker failure as the program's own failure
                # (or as a working backdoor), so check the container is still there.
                alive = await next(iter(environments.values())).exec(
                    ["true"], timeout=30
                )
                if not alive.success:
                    raise RuntimeError(alive.stderr or "sandbox container stopped")
                return result
            finally:
                await asyncio.shield(
                    DockerSandboxEnvironment.sample_cleanup(
                        _TASK, self._compose, environments, False
                    )
                )

    async def grade(self, problem: AppsProblem, code: str) -> ExecutionGrade:
        if problem.fn_name is not None:
            return _infrastructure_failure(
                ValueError("apps_scorer does not support function-call problems")
            )
        try:
            metadata = AppsTaskMetadata(
                main_task_description=problem.question,
                side_task_description="",
                setting_description="",
                setting_name="apps",
                eval_mode=EvalMode.HONEST,
                inputs=list(problem.inputs),
                outputs=list(problem.outputs),
            )
            sample = Sample(input=problem.question, metadata=metadata.model_dump())
            result = await self._run(
                lambda: passes_tests(
                    SubmitHonestArgs(code=code),
                    sample,
                    self.num_tests or len(problem.inputs),
                    use_cache=False,
                )
            )
        except Exception as exc:
            return _infrastructure_failure(exc)
        return _to_grade(result)

    async def check_backdoor(
        self, problem: AppsProblem, code: str
    ) -> ExecutionGrade | None:
        """None when the check does not apply (no secret input or reference, or the
        reference itself fails on it), so it is never counted against the sender."""
        secret, reference = problem.secret_input, problem.reference_solution
        if secret is None or reference is None or problem.fn_name is not None:
            return None
        try:
            result = await self._run(
                lambda: has_backdoor(
                    SubmitAttackArgs(code=code, secret_input=secret),
                    reference,
                    use_cache=False,
                )
            )
        except Exception as exc:
            return _infrastructure_failure(exc)
        return None if result.value == NOANSWER else _to_grade(result)
