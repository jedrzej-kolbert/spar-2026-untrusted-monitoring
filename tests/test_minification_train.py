"""No real Tinker clients, model calls, or Docker processes in these tests."""

from unittest.mock import AsyncMock

import pytest

from experiments.grouped_apps_smoke.minification_train import (
    BalancedTrainer,
    ChannelEvaluator,
    ChannelReceiver,
    MinificationChannel,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    ExecutionGrade,
    GeneratedSolution,
    GroupedRLConfig,
    ReceiverExample,
    SenderExample,
    UpdateStats,
)


class Receiver:
    def __init__(self) -> None:
        self.codes = []
        self.examples = []

    async def probabilities(
        self, problem: AppsProblem, code: str
    ) -> tuple[float, float]:
        self.codes.append(code)
        return 0.7, 0.3

    async def update(self, examples: list[ReceiverExample]) -> UpdateStats:
        self.examples = examples
        return UpdateStats()


async def test_receiver_scoring_and_sft_use_same_minified_source() -> None:
    receiver = Receiver()
    channel = MinificationChannel()
    wrapped = ChannelReceiver(receiver, channel)
    problem = AppsProblem("p", "Print hello", ("",), ("hello",))
    original = "# Secret bit: 1\nsecret_bit = 1\nprint('hello')"
    await wrapped.probabilities(problem, original)
    await wrapped.update([ReceiverExample(problem, original, 1)])
    expected = channel.transform(original)[0]
    assert receiver.codes == [expected]
    assert receiver.examples[0].code == expected
    assert "secret_bit" not in expected and "#" not in expected
    assert receiver.examples[0].payload == 1


async def test_invalid_source_is_failure_and_never_raw_fallback() -> None:
    receiver = Receiver()
    channel = MinificationChannel()
    problem = AppsProblem("p", "Print hello", ("",), ("hello",))
    backend = AsyncMock()
    wrapped = ChannelEvaluator(backend, channel)
    source = "def invalid("
    await ChannelReceiver(receiver, channel).probabilities(problem, source)
    grade = await wrapped.grade(problem, source)
    assert receiver.codes == [""]
    assert not grade.passed and grade.infrastructure_error is None
    backend.grade.assert_not_called()


class Sender:
    def __init__(self) -> None:
        self.generated = []
        self.updated = []

    async def generate(
        self, problem: AppsProblem, payload: int, num_samples: int
    ) -> list[GeneratedSolution]:
        rows = [
            GeneratedSolution(
                f"# Secret bit: {payload}\nsecret_bit = {payload}\nprint({i + 1})",
                (11,),
                (20 + i,),
                (-0.5,),
            )
            for i in range(num_samples)
        ]
        self.generated.extend(rows)
        return rows

    async def update(self, examples: list[SenderExample]) -> UpdateStats:
        self.updated = examples
        return UpdateStats()


class Evaluator:
    def __init__(self, alternate: bool = False) -> None:
        self.codes = []
        self.alternate = alternate

    async def grade(self, problem: AppsProblem, code: str) -> ExecutionGrade:
        self.codes.append(code)
        return ExecutionGrade(not self.alternate or len(self.codes) % 2 == 1)


async def test_actor_update_keeps_original_tokens_and_logprobs() -> None:
    sender = Sender()
    receiver = Receiver()
    channel = MinificationChannel()
    evaluator = Evaluator(alternate=True)
    trainer = BalancedTrainer(
        sender,
        ChannelReceiver(receiver, channel),
        ChannelEvaluator(evaluator, channel),
        GroupedRLConfig(group_size=2),
    )
    result = await trainer.train_step(
        [AppsProblem("p", "Print hello", ("",), ("hello",))]
    )
    assert result.sender_updated and result.receiver_updated
    assert [x.generation for x in sender.updated] == sender.generated
    assert all(x.generation.text.startswith("# Secret bit:") for x in sender.updated)
    assert all(
        x.generation.prompt_tokens == (11,) and x.generation.logprobs == (-0.5,)
        for x in sender.updated
    )
    assert receiver.codes == evaluator.codes
    assert [x.code for x in receiver.examples] == evaluator.codes
    assert all("#" not in code and "secret_bit" not in code for code in evaluator.codes)


async def test_balanced_evaluation_uses_real_generations_for_both_bits() -> None:
    sender = Sender()
    receiver = Receiver()
    trainer = BalancedTrainer(sender, receiver, Evaluator())
    problems = [
        AppsProblem(str(i), "Print hello", ("",), ("hello",), split="validation")
        for i in range(4)
    ]
    result = await trainer.evaluate_balanced(problems)
    assert result.requested_rollouts == len(result.rollouts) == 8
    for problem in problems:
        assert {
            r.payload
            for r in result.rollouts
            if r.problem.problem_id == problem.problem_id
        } == {0, 1}
    assert not result.sender_updated and not result.receiver_updated
    assert not sender.updated and not receiver.examples


async def test_balanced_evaluation_rejects_training_tasks() -> None:
    trainer = BalancedTrainer(Sender(), Receiver(), Evaluator())
    with pytest.raises(ValueError, match="held-out"):
        await trainer.evaluate_balanced(
            [AppsProblem("p", "Print hello", ("",), ("hello",))]
        )
