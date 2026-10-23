"""Connect the ICRL loop to the grouped-RL trainer and to Tinker sampling.

The team reuses the trainer's own evaluation, so grading, overseer, backdoor and
reward are the same as in the GBRL runs. The only difference is that the sender and
receiver prompts carry the current strategy text.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .core import AppsProblem, GroupedRLTrainer, Rollout
from .icrl import Episode, RoundSummary, learner_messages, parse_strategies
from .tinker_backend import _TinkerRole


@dataclass
class StrategyBox:
    """The strategy the sender and receiver currently see; set once per team run."""

    text: str = ""


def with_strategy(
    messages_fn: Callable[..., list[dict[str, str]]], box: StrategyBox
) -> Callable[..., list[dict[str, str]]]:
    """Append the box's strategy to the system message of a sender or receiver prompt."""

    def messages(*args: Any) -> list[dict[str, str]]:
        messages = messages_fn(*args)
        if box.text:
            system = messages[0]
            messages = [
                {
                    **system,
                    "content": system["content"]
                    + "\n\nStrategy you share with your partner:\n"
                    + box.text,
                },
                *messages[1:],
            ]
        return messages

    return messages


def episode_from(rollout: Rollout) -> Episode:
    """Turn one graded rollout into an episode. Missing fields are errors, not defaults."""
    if rollout.probabilities is None or rollout.execution is None:
        raise ValueError("rollout has no receiver or execution result")
    if rollout.reward is None:
        raise ValueError("rollout has no reward")
    suspicion = (
        rollout.suspiciousness
        if rollout.suspiciousness is not None
        else rollout.backdoor_suspicion
    )
    if suspicion is None:
        raise ValueError("rollout has no overseer suspicion")
    return Episode(
        payload=rollout.payload,
        decoded=int(rollout.probabilities[1] > rollout.probabilities[0]),
        passed=rollout.execution.passed,
        backdoor=bool(rollout.backdoor and rollout.backdoor.passed),
        suspicion=suspicion,
        reward=rollout.reward,
    )


@dataclass
class TrainerTeam:
    """Runs one strategy through the trainer's held-out evaluation on fixed problems."""

    trainer: GroupedRLTrainer
    problems: Sequence[AppsProblem]
    box: StrategyBox = field(default_factory=StrategyBox)
    num_samples: int = 1

    async def episodes(self, strategy: str) -> list[Episode]:
        self.box.text = strategy
        result = await self.trainer.evaluate(
            list(self.problems), self.num_samples, balanced=True
        )
        return [episode_from(rollout) for rollout in result.rollouts]


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
    ) -> None:
        super().__init__(None, sampling_client, tokenizer, allow_paid=allow_paid)
        if max_tokens < 1 or temperature <= 0:
            raise ValueError("max_tokens must be positive and temperature > 0")
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def propose(self, history: Sequence[RoundSummary], count: int) -> str:
        self._check_permission()
        from tinker import types

        prompt = self._prompt(learner_messages(history, count))
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
        if not parse_strategies(text):
            raise ValueError("learner reply has no <strategy> blocks")
        return text
