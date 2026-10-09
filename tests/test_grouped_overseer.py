"""Offline overseer reward, channel, failure, and frozen-judge checks."""

import math
from typing import Any
from unittest.mock import AsyncMock

import pytest

from experiments.grouped_apps_smoke.minification_train import (
    ChannelEvaluator,
    ChannelReceiver,
    MinificationChannel,
)
from experiments.grouped_apps_smoke.run import Recorder
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    BatchInfrastructureError,
    GroupedRLConfig,
    GroupedRLTrainer,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    TinkerOverseer,
    overseer_messages,
)
from tests.test_minification_train import Evaluator, Receiver, Sender


class Judge:
    def __init__(self, scores: list[float | Exception]) -> None:
        self.scores = iter(scores)
        self.messages: list[str] = []

    async def suspiciousness(self, problem: AppsProblem, message: str) -> float:
        self.messages.append(message)
        value = next(self.scores)
        if isinstance(value, Exception):
            raise value
        return value


def problem() -> AppsProblem:
    return AppsProblem("p", "Print hello", ("",), ("hello",))


async def test_penalty_changes_advantages_and_judge_sees_original(
    tmp_path: Any,
) -> None:
    sender, receiver, evaluator = Sender(), Receiver(), Evaluator()
    channel = MinificationChannel()
    judge = Judge([0.0, 1.0])
    trainer = GroupedRLTrainer(
        sender,
        ChannelReceiver(receiver, channel),
        ChannelEvaluator(evaluator, channel),
        GroupedRLConfig(group_size=2, overseer_weight=2),
        overseer=judge,
    )
    result = await trainer.train_step([problem()])
    assert judge.messages == [
        r.generation.text for r in result.rollouts if r.generation
    ]
    assert all("# Secret bit:" in s for s in judge.messages)
    assert all("#" not in s for s in receiver.codes)
    assert [r.reward for r in result.rollouts] == pytest.approx([1.3, -0.7])
    assert [r.advantage for r in result.rollouts] == pytest.approx([1, -1])
    assert [x.advantage for x in sender.updated] == pytest.approx([1, -1])
    recorder = Recorder(tmp_path / "records")
    metrics = recorder.batch("train", result)
    assert metrics["train/reward_overseer_penalty_mean"] == 1
    assert metrics["train/suspiciousness_mean"] == 0.5
    assert metrics["train/reward_communication_mean"] == pytest.approx(0.3)
    assert metrics["train/overseer_scored_records"] == 2


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -0.1, 1.1, RuntimeError("judge unavailable")]
)
async def test_missing_or_invalid_judge_stops_all_updates(
    bad: float | Exception,
) -> None:
    sender, receiver = Sender(), Receiver()
    trainer = GroupedRLTrainer(
        sender,
        receiver,
        Evaluator(),
        GroupedRLConfig(group_size=2, overseer_weight=1),
        overseer=Judge([bad, 0.5]),
    )
    with pytest.raises(BatchInfrastructureError) as caught:
        await trainer.train_step([problem()])
    assert len(caught.value.result.rollouts) == 2
    assert caught.value.result.rollouts[0].reward is None
    assert caught.value.result.rollouts[0].overseer_penalty is None
    assert not sender.updated and not receiver.examples


async def test_constant_penalty_does_not_create_group_signal() -> None:
    trainer = GroupedRLTrainer(
        Sender(),
        Receiver(),
        Evaluator(),
        GroupedRLConfig(group_size=2, overseer_weight=1),
        overseer=Judge([1, 1]),
    )
    result = await trainer.train_step([problem()])
    assert [r.advantage for r in result.rollouts] == [0, 0]
    assert not result.sender_updated


async def test_failed_program_still_gets_suspiciousness_penalty() -> None:
    result = await GroupedRLTrainer(
        Sender(),
        Receiver(),
        Evaluator(alternate=True),
        GroupedRLConfig(group_size=2, overseer_weight=1),
        overseer=Judge([0, 0.8]),
    ).train_step([problem()])
    assert result.rollouts[1].reward == pytest.approx(-0.8)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weight_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="overseer weight"):
        GroupedRLConfig(overseer_weight=weight)


def test_missing_required_overseer_rejected() -> None:
    with pytest.raises(ValueError, match="requires an overseer"):
        GroupedRLTrainer(
            Sender(), Receiver(), Evaluator(), GroupedRLConfig(overseer_weight=1)
        )


class Tokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> list[int]:
        self.messages = messages
        return [7]

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [10 if text == "0" else 11]


async def test_frozen_judge_binary_scoring_without_label_leakage() -> None:
    sampler, tokenizer = AsyncMock(), Tokenizer()
    sampler.compute_logprobs_async.side_effect = [[None, -2.0], [None, -0.1]]
    judge = TinkerOverseer(None, sampler, tokenizer, allow_paid=True)
    score = await judge.suspiciousness(problem(), "# Secret bit: 1\nprint('hello')")
    assert score == pytest.approx(1 / (1 + math.exp(-1.9)))
    assert tokenizer.messages == overseer_messages(
        problem(), "# Secret bit: 1\nprint('hello')"
    )
    assert "Secret bit:" not in tokenizer.messages[0]["content"]
    with pytest.raises(PermissionError, match="frozen"):
        await judge.update([])


async def test_judge_calls_require_paid_permission() -> None:
    sampler = AsyncMock()
    judge = TinkerOverseer(None, sampler, Tokenizer())
    with pytest.raises(PermissionError):
        await judge.suspiciousness(problem(), "print('hello')")
    sampler.compute_logprobs_async.assert_not_called()
