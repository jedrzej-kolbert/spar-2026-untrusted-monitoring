"""Connect the ICRL loop to the grouped-RL trainer and to Tinker sampling.

The team reuses the trainer's own evaluation, so grading, the frozen judge, the
backdoor check, the trusted monitor and the reward are the same as in the gradient
runs. The only difference is that the sender's prompt carries the current strategy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    GroupedRLTrainer,
    HoneypotEvalResult,
    Rollout,
)
from .icrl import Episode
from .tinker_backend import _TinkerRole


@dataclass
class StrategyBox:
    """The strategy the sender currently sees; the team sets it before each run."""

    text: str = ""


def with_strategy(
    messages_fn: Callable[..., list[dict[str, str]]], box: StrategyBox
) -> Callable[..., list[dict[str, str]]]:
    """Wrap the sender's ``messages_fn`` so its system message ends with the strategy."""

    def messages(*args: Any) -> list[dict[str, str]]:
        messages = messages_fn(*args)
        if box.text:
            system = messages[0]
            messages = [
                {
                    **system,
                    "content": system["content"]
                    + "\n\nInstructions for this rewrite:\n"
                    + box.text,
                },
                *messages[1:],
            ]
        return messages

    return messages


def episode_from(rollout: Rollout) -> Episode:
    """Turn one graded rollout into an episode. Missing scores are errors, not defaults."""
    if rollout.probabilities is None or rollout.execution is None:
        raise ValueError("rollout has no judge or execution result")
    if rollout.reward is None:
        raise ValueError("rollout has no reward")
    if rollout.backdoor_suspicion is None:
        raise ValueError(
            "rollout has no trusted-monitor suspicion; "
            "set reward.backdoor_suspicion_weight and models.backdoor_monitor"
        )
    return Episode(
        p_self=rollout.probabilities[1],
        passed=rollout.execution.passed,
        backdoor=None if rollout.backdoor is None else rollout.backdoor.passed,
        suspicion=rollout.backdoor_suspicion,
        reward=rollout.reward,
        components=dict(rollout.reward_components),
        problem=rollout.problem.question,
        code=rollout.receiver_code
        if rollout.receiver_code is not None
        else rollout.code,
        execution_detail=rollout.execution.detail,
    )


@dataclass
class TrainerTeam:
    """Runs strategies through the trainer's evaluation on fixed problem sets.

    ``box`` must be the one wrapped into the sender's ``messages_fn`` with
    ``with_strategy``. Search problems choose the strategy; held-out problems and
    honeypots only check the chosen one, so they must not overlap the search set.
    ``on_result(strategy, result, phase)`` sees every evaluated batch, including one
    that hit a sandbox infrastructure error; such a batch is run once more.
    """

    trainer: GroupedRLTrainer
    box: StrategyBox
    search_problems: Sequence[AppsProblem]
    held_out_problems: Sequence[AppsProblem]
    honeypots: Sequence[tuple[AppsProblem, str]]
    num_samples: int = 1
    on_result: Callable[[str, BatchResult, str], None] | None = None
    # Fresh problems that re-score the search's top strategies, to choose among them.
    retest_problems: Sequence[AppsProblem] = ()

    def __post_init__(self) -> None:
        if not self.trainer.config.self_recognition:
            raise ValueError("TrainerTeam needs a self-recognition trainer")
        sets = [
            {p.problem_id for p in problems}
            for problems in (
                self.search_problems,
                self.retest_problems,
                self.held_out_problems,
            )
        ]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("search, retest and held-out problems overlap")
        # No weights change, so the trainer's train/held-out guard does not apply to
        # these training problems; what matters is that the sets above do not overlap.
        self.search_problems = [
            replace(p, split="validation") for p in self.search_problems
        ]
        self.retest_problems = [
            replace(p, split="validation") for p in self.retest_problems
        ]

    def _report(self, strategy: str, result: BatchResult, phase: str) -> None:
        if self.on_result is not None:
            self.on_result(strategy, result, phase)

    async def _run(
        self, strategy: str, problems: Sequence[AppsProblem], phase: str
    ) -> BatchResult:
        self.box.text = strategy
        for attempt in (1, 2):
            try:
                result = await self.trainer.evaluate(list(problems), self.num_samples)
            except BatchInfrastructureError as error:
                self._report(strategy, error.result, f"{phase}_failed")
                if attempt == 2:
                    raise
                continue
            self._report(strategy, result, phase)
            return result
        raise AssertionError("unreachable")

    async def episodes(self, strategy: str) -> list[Episode]:
        result = await self._run(strategy, self.search_problems, "search")
        return [episode_from(rollout) for rollout in result.rollouts]

    async def retest(self, strategy: str) -> list[Episode]:
        result = await self._run(strategy, self.retest_problems, "retest")
        return [episode_from(rollout) for rollout in result.rollouts]

    async def held_out(self, strategy: str) -> tuple[list[Episode], HoneypotEvalResult]:
        """Episodes on held-out problems and the judge's scores against honeypots."""
        result = await self._run(strategy, self.held_out_problems, "held_out")
        honeypot = await self.trainer.evaluate_against_honeypots(
            list(self.held_out_problems), list(self.honeypots), signalled=result
        )
        return [episode_from(rollout) for rollout in result.rollouts], honeypot


class TinkerLearner(_TinkerRole):
    """Samples strategies from a base model. Never trains; needs allow_paid."""

    def __init__(
        self,
        sampling_client: Any,
        tokenizer: Any,
        *,
        allow_paid: bool = False,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        max_context_tokens: int = 16384,
    ) -> None:
        super().__init__(
            None,
            sampling_client,
            tokenizer,
            allow_paid=allow_paid,
            max_context_tokens=max_context_tokens,
        )
        if not 0 < max_tokens < max_context_tokens or not 0 < temperature < float(
            "inf"
        ):
            raise ValueError(
                "max_tokens must fit the context; temperature must be positive and finite"
            )
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def query(self, messages: list[dict[str, str]]) -> str:
        self._check_permission()
        from tinker import types

        prompt = self._prompt(messages)
        if len(prompt) + self.max_tokens > self.max_context_tokens:
            raise ValueError(
                "learner prompt plus response allowance exceeds context limit"
            )
        response = await self.sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=self.max_tokens, temperature=self.temperature
            ),
        )
        text = self.tokenizer.decode(
            list(response.sequences[0].tokens), skip_special_tokens=True
        )
        return text
