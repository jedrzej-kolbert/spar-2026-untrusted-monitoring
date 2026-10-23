"""Offline-only tests: no Tinker, Docker, model downloads, or generated-code execution."""

import asyncio
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    BatchInfrastructureError,
    ExecutionGrade,
    GeneratedSolution,
    GroupedRLConfig,
    GroupedRLTrainer,
    ReceiverExample,
    SenderExample,
    UpdateStats,
    extract_code,
    group_advantages,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.sandbox import (
    DockerAppsEvaluator,
    _OutputLimitError,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    TinkerReceiver,
    TinkerSender,
    create_tinker_team,
    policy_inputs,
    receiver_messages,
    sender_messages,
    supervised_inputs,
)


@pytest.fixture(autouse=True)
def forbid_live_tinker(monkeypatch: pytest.MonkeyPatch) -> None:
    import tinker

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline tests must not create a live Tinker service")

    monkeypatch.setattr(tinker, "ServiceClient", forbidden)


@pytest.fixture
def problem() -> AppsProblem:
    return AppsProblem("7", "Read a number and print it.", ("12\n",), ("12\n",))


class FakeSender:
    def __init__(self, constant: bool = False) -> None:
        self.constant = constant
        self.groups: list[tuple[str, int]] = []
        self.updates: list[list[SenderExample]] = []

    async def generate(
        self, problem: AppsProblem, payload: int, num_samples: int
    ) -> list[GeneratedSolution]:
        self.groups.append((problem.problem_id, payload))
        return [
            GeneratedSolution(
                f"# bit={payload if self.constant or index % 2 == 0 else 1 - payload}\n"
                f"print({0 if self.constant else index})",
                (1, 2),
                (20 + index,),
                (-0.5,),
            )
            for index in range(num_samples)
        ]

    async def update(self, examples: list[SenderExample]) -> UpdateStats:
        self.updates.append(examples)
        return UpdateStats({"fake_update_count": float(len(self.updates))})


class FakeReceiver:
    def __init__(self) -> None:
        self.updates: list[list[ReceiverExample]] = []

    async def probabilities(
        self, problem: AppsProblem, code: str
    ) -> tuple[float, float]:
        return (0.1, 0.9) if "# bit=1" in code else (0.9, 0.1)

    async def update(self, examples: list[ReceiverExample]) -> UpdateStats:
        self.updates.append(examples)
        return UpdateStats({"fake_update_count": float(len(self.updates))})


class FakeEvaluator:
    async def grade(self, problem: AppsProblem, code: str) -> ExecutionGrade:
        return ExecutionGrade("print(2)" not in code)


def trainer(constant: bool = False) -> Any:
    return GroupedRLTrainer(FakeSender(constant), FakeReceiver(), FakeEvaluator())


@pytest.mark.parametrize(
    "rewards, expected", [([0, 1, 2], [-1, 0, 1]), ([0.8] * 4, [0] * 4)]
)
def test_centering(rewards: list[float], expected: list[float]) -> None:
    assert group_advantages(rewards) == pytest.approx(expected)
    assert sum(group_advantages(rewards)) == pytest.approx(0)


@pytest.mark.parametrize("rewards", [[], [1], [1, math.nan], [math.inf, 1]])
def test_centering_rejects_missing_nonfinite_or_singleton(rewards: list[float]) -> None:
    with pytest.raises(ValueError):
        group_advantages(rewards)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"task_weight": -1},
        {"task_weight": math.inf},
        {"task_weight": 0, "communication_weight": 0},
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GroupedRLConfig(**kwargs)


def test_record_parsing_drops_no_test_cases_or_requires_reference_solutions() -> None:
    record = {
        "problem_id": 0,
        "question": "Public problem",
        "starter_code": "",
        "input_output": '{"inputs": ["PRIVATE_IN"], "outputs": ["PRIVATE_OUT"]}',
        "solutions": '["PRIVATE_REFERENCE"]',
    }
    problem = AppsProblem.from_record(record, split="train")
    assert problem.problem_id == "0"
    assert problem.inputs == ("PRIVATE_IN",)
    assert not hasattr(problem, "solutions")
    for messages in (
        sender_messages(problem, 1),
        receiver_messages(problem, "print(1)"),
    ):
        rendered = str(messages)
        assert all(
            secret not in rendered
            for secret in ("PRIVATE_IN", "PRIVATE_OUT", "PRIVATE_REFERENCE")
        )
    assert "Secret bit" not in str(receiver_messages(problem, "print(1)"))


def test_function_call_record_preserves_arguments_and_outputs() -> None:
    problem = AppsProblem.from_record(
        {
            "problem_id": 1,
            "question": "Add two numbers.",
            "input_output": {"fn_name": "add", "inputs": [[1, 2]], "outputs": [[3]]},
        },
        split="validation",
    )
    assert problem.fn_name == "add"
    assert problem.inputs == ([1, 2],)
    assert problem.outputs == ([3],)


@pytest.mark.parametrize(
    "inputs, outputs, fn_name",
    [((), (), None), (("a",), (), None), (([1],), (1,), None), ((1,), (2,), "add")],
)
def test_bad_test_metadata_fails_loudly(
    inputs: tuple[Any, ...], outputs: tuple[Any, ...], fn_name: str | None
) -> None:
    with pytest.raises(ValueError):
        AppsProblem("7", "Question", inputs, outputs, fn_name=fn_name)


async def test_alternating_updates_keep_broken_code_and_all_metrics(
    problem: AppsProblem,
) -> None:
    loop = trainer()
    result = await loop.train_step([problem])
    assert len(result.rollouts) == 4
    assert result.metrics()["correctness"] == 0.75
    assert result.metrics()["decoding_accuracy"] == 0.5
    assert result.metrics()["joint_success"] == 0.25
    assert result.metrics()["reward_min"] == 0
    assert result.metrics()["reward_max"] == pytest.approx(1.9)
    assert result.rollouts[2].reward == 0
    assert sum(item.advantage for item in result.rollouts) == pytest.approx(0)
    assert result.sender_updated is True and result.receiver_updated is True
    assert len(loop.sender.updates[0]) == len(loop.receiver.updates[0]) == 4
    assert "print(2)" in loop.receiver.updates[0][2].code
    assert {item.payload for item in result.rollouts} == {loop.sender.groups[0][1]}


async def test_constant_rewards_train_receiver_but_skip_zero_signal_actor(
    problem: AppsProblem,
) -> None:
    loop = trainer(constant=True)
    result = await loop.train_step([problem])
    assert result.receiver_updated is True and result.sender_updated is False
    assert result.metrics()["nonzero_advantages"] == 0
    assert not loop.sender.updates


async def test_rewards_are_measured_before_receiver_sft(problem: AppsProblem) -> None:
    loop = trainer()
    original = loop.receiver.probabilities

    async def scoring(problem: AppsProblem, code: str) -> tuple[float, float]:
        assert not loop.receiver.updates
        return await original(problem, code)

    loop.receiver.probabilities = scoring
    await loop.train_step([problem])


async def test_groups_do_not_share_baselines(problem: AppsProblem) -> None:
    loop = trainer()
    other = AppsProblem("8", "Another question", ("a",), ("b",))
    result = await loop.train_step([problem, other])
    for problem_id in ("7", "8"):
        group = [
            item for item in result.rollouts if item.problem.problem_id == problem_id
        ]
        assert sum(item.advantage for item in group) == pytest.approx(0)
        assert len({item.payload for item in group}) == 1


async def test_payload_rng_is_independent_of_problem_and_eval(
    problem: AppsProblem,
) -> None:
    a, b = trainer(), trainer()
    held_out = AppsProblem("9", "Held-out", ("a",), ("b",), split="validation")
    await a.evaluate([held_out])
    await a.train_step([problem])
    await b.train_step([problem])
    assert a.sender.groups[-1][1] == b.sender.groups[-1][1]
    first = await a.evaluate([held_out])
    second = await a.evaluate([held_out])
    assert first.rollouts[0].payload == second.rollouts[0].payload
    assert not first.sender_updated and not first.receiver_updated


async def test_split_overlap_and_duplicates_are_rejected_before_generation(
    problem: AppsProblem,
) -> None:
    loop = trainer()
    held_out = AppsProblem("9", "Held-out", ("a",), ("b",), split="validation")
    with pytest.raises(ValueError):
        await loop.train_step([held_out])
    with pytest.raises(ValueError):
        await loop.train_step([problem, problem])
    assert not loop.sender.groups
    await loop.evaluate([held_out])
    alias = AppsProblem("9", "Alias", ("a",), ("b",))
    with pytest.raises(ValueError):
        await loop.train_step([alias])
    await loop.train_step([problem])
    alias = AppsProblem("7", "Alias", ("a",), ("b",), split="test")
    with pytest.raises(ValueError):
        await loop.evaluate([alias])


async def test_infrastructure_failure_retains_records_and_aborts_updates(
    problem: AppsProblem,
) -> None:
    loop = trainer()

    async def grade(problem: AppsProblem, code: str) -> ExecutionGrade:
        return ExecutionGrade(False, infrastructure_error="Docker unavailable")

    loop.evaluator.grade = grade
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    result = error.value.result
    assert len(result.rollouts) == 4
    assert result.metrics()["infrastructure_failures"] == 4
    assert result.metrics()["mean_reward"] is None
    assert all(item.reward is None for item in result.rollouts)
    assert not loop.sender.updates and not loop.receiver.updates


async def test_missing_scores_are_not_imputed_or_dropped(problem: AppsProblem) -> None:
    loop = trainer()
    loop.receiver.probabilities = AsyncMock(return_value=(math.nan, 0.5))
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    result = error.value.result
    assert result.metrics()["recorded_rollouts"] == 4
    assert result.metrics()["decoding_accuracy"] == 0
    assert result.metrics()["mean_reward"] is None
    assert all(item.reward is None for item in result.rollouts)
    assert not loop.receiver.updates


async def test_generation_failure_records_missing_requested_samples(
    problem: AppsProblem,
) -> None:
    loop = trainer()
    loop.sender.generate = AsyncMock(side_effect=RuntimeError("sampler failure"))
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    result = error.value.result
    assert result.metrics()["recorded_rollouts"] == 4
    assert all(item.generation is None for item in result.rollouts)
    assert not loop.receiver.updates


async def test_excess_samples_are_preserved_but_no_updates_occur(
    problem: AppsProblem,
) -> None:
    loop = trainer()
    samples = await loop.sender.generate(problem, 0, 5)
    loop.sender.generate = AsyncMock(return_value=samples)
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    assert len(error.value.result.rollouts) == 5
    assert error.value.result.requested_rollouts == 4
    assert error.value.result.metrics()["batch_infrastructure_errors"] == 1
    assert not loop.receiver.updates


async def test_failed_update_reports_uncertain_partial_state(
    problem: AppsProblem,
) -> None:
    loop = trainer()
    loop.sender.update = AsyncMock(side_effect=RuntimeError("lost optimizer response"))
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    assert error.value.result.receiver_updated is True
    assert error.value.result.sender_updated is None
    assert len(loop.receiver.updates) == 1


def test_code_extraction_never_passes_surrounding_prose() -> None:
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    with pytest.raises(ValueError):
        extract_code("```python\nprint(1)\n```\nSecret bit 1")


def test_policy_and_sft_alignment() -> None:
    sample = GeneratedSolution("code", (1, 2, 3), (4, 5), (-0.2, -0.3))
    inputs = policy_inputs(sample, 0.8)
    assert inputs == {
        "target_tokens": [2, 3, 4, 5],
        "logprobs": [0, 0, -0.2, -0.3],
        "advantages": [0, 0, 0.8, 0.8],
    }
    assert supervised_inputs([1, 2, 3], [4, 5]) == {
        "target_tokens": [2, 3, 4, 5],
        "weights": [0, 0, 1, 1],
    }


class FakeFuture:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def result_async(self) -> Any:
        return self.value


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, list[list[int]]]:
        self.messages.append(messages)
        assert kwargs["enable_thinking"] is False
        return {"input_ids": [[1, 2, 3]]}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return {"0": [10], "1": [20, 21]}[text]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "print(1)"


class FakeSampler:
    async def sample_async(
        self, prompt: Any, num_samples: int, sampling_params: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(tokens=[30], logprobs=[-0.5])
                for _ in range(num_samples)
            ]
        )

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        return [None] + [
            {10: -0.2, 20: -0.4, 21: -0.6}.get(token, -0.01)
            for token in prompt.to_ints()[1:]
        ]


class FakeTrainingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.tokenizer = FakeTokenizer()

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    async def forward_backward_async(self, data: list[Any], loss_fn: str) -> FakeFuture:
        self.calls.append(("backward", loss_fn, data))
        return FakeFuture(SimpleNamespace(metrics={"fake_loss": 1.0}))

    async def optim_step_async(self, params: Any) -> FakeFuture:
        self.calls.append(("optimizer", params))
        return FakeFuture(SimpleNamespace())

    async def save_weights_and_get_sampling_client_async(self) -> FakeSampler:
        self.calls.append(("refresh",))
        return FakeSampler()

    async def save_weights_for_sampler_async(self, name: str) -> FakeFuture:
        return FakeFuture(SimpleNamespace(path=f"tinker://offline-fixture/{name}"))


async def test_paid_gate_precedes_service_creation(problem: AppsProblem) -> None:
    with pytest.raises(PermissionError):
        await create_tinker_team()
    sender = TinkerSender(None, None, None)
    with pytest.raises(PermissionError):
        await sender.generate(problem, 0, 4)
    with pytest.raises(PermissionError):
        await sender.update([])
    with pytest.raises(PermissionError):
        await sender.save_checkpoint("not-approved")
    receiver = TinkerReceiver(None, None, None)
    with pytest.raises(PermissionError):
        await receiver.probabilities(problem, "code")
    with pytest.raises(PermissionError):
        await receiver.update([])


async def test_tinker_sender_uses_is_masks_and_refreshes_sampler(
    problem: AppsProblem,
) -> None:
    client = FakeTrainingClient()
    sender = TinkerSender(client, FakeSampler(), client.tokenizer, allow_paid=True)
    samples = await sender.generate(problem, 1, 4)
    assert len(samples) == 4
    old_sampler = sender.sampling_client
    await sender.update([SenderExample(samples[0], 0.7)])
    assert [call[0] for call in client.calls] == ["backward", "optimizer", "refresh"]
    _, loss_fn, data = client.calls[0]
    assert loss_fn == "importance_sampling"
    assert list(data[0].loss_fn_inputs["advantages"].data) == [0, 0, pytest.approx(0.7)]
    assert sender.sampling_client is not old_sampler
    assert await sender.save_checkpoint("sender") == "tinker://offline-fixture/sender"


async def test_receiver_multitoken_label_scoring_and_answer_only_sft(
    problem: AppsProblem,
) -> None:
    client = FakeTrainingClient()
    receiver = TinkerReceiver(client, FakeSampler(), client.tokenizer, allow_paid=True)
    p0, p1 = await receiver.probabilities(problem, "print(1)")
    assert p0 / p1 == pytest.approx(math.exp(0.8))
    assert p0 + p1 == pytest.approx(1)
    await receiver.update([ReceiverExample(problem, "print(1)", 1)])
    _, loss_fn, data = client.calls[0]
    assert loss_fn == "cross_entropy"
    assert list(data[0].loss_fn_inputs["weights"].data) == [0, 0, 1, 1, 1]
    assert list(data[0].loss_fn_inputs["target_tokens"].data) == [2, 3, 20, 21, 9]
    assert "Secret bit" not in str(client.tokenizer.messages)


async def test_context_overflow_is_not_silently_truncated(problem: AppsProblem) -> None:
    client = FakeTrainingClient()
    sender = TinkerSender(
        client,
        FakeSampler(),
        client.tokenizer,
        allow_paid=True,
        max_tokens=2,
        max_context_tokens=4,
    )
    with pytest.raises(ValueError, match="context"):
        await sender.generate(problem, 0, 4)
    assert not client.calls


async def test_missing_rollout_logprobs_are_not_fabricated(
    problem: AppsProblem,
) -> None:
    client = FakeTrainingClient()
    sampler = FakeSampler()
    sampler.sample_async = AsyncMock(
        return_value=SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[30], logprobs=None)]
        )
    )
    sender = TinkerSender(client, sampler, client.tokenizer, allow_paid=True)
    samples = await sender.generate(problem, 0, 1)
    assert samples[0].logprobs == pytest.approx((-0.01,))
    assert not client.calls


async def test_missing_label_scores_abort_before_training(problem: AppsProblem) -> None:
    client = FakeTrainingClient()
    sampler = FakeSampler()
    sampler.compute_logprobs_async = AsyncMock(return_value=[None] * 4)
    receiver = TinkerReceiver(client, sampler, client.tokenizer, allow_paid=True)
    with pytest.raises(ValueError, match="log probabilities"):
        await receiver.probabilities(problem, "print(1)")
    assert not client.calls


async def test_backward_failure_does_not_submit_optimizer(problem: AppsProblem) -> None:
    client = FakeTrainingClient()
    client.forward_backward_async = AsyncMock(
        side_effect=RuntimeError("backward rejected")
    )
    receiver = TinkerReceiver(client, FakeSampler(), client.tokenizer, allow_paid=True)
    with pytest.raises(RuntimeError):
        await receiver.update([ReceiverExample(problem, "print(1)", 0)])
    assert not client.calls


async def test_factory_creates_independent_adapters_with_fake_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tinker

    clients: list[FakeTrainingClient] = []

    class FakeService:
        async def create_lora_training_client_async(
            self, **kwargs: Any
        ) -> FakeTrainingClient:
            client = FakeTrainingClient()
            clients.append(client)
            return client

    monkeypatch.setattr(tinker, "ServiceClient", FakeService)
    sender, receiver = await create_tinker_team(allow_paid=True)
    assert sender.training_client is clients[0]
    assert receiver.training_client is clients[1]
    assert clients[0] is not clients[1]


async def test_sandbox_stdin_all_tests_and_security_flags(
    problem: AppsProblem, tmp_path: Path
) -> None:
    evaluator = DockerAppsEvaluator(temp_root=tmp_path)
    evaluator._run = AsyncMock(return_value=(0, b"12\n", b""))
    assert (await evaluator.grade(problem, "print(input())")).passed
    assert evaluator._run.call_args is not None
    command, name, input_text = evaluator._run.call_args.args
    for option in (
        "--pull=never",
        "--read-only",
        "--cap-drop",
        "--pids-limit",
        "--log-driver",
    ):
        assert option in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert input_text == "12\n"
    assert name.startswith("apps-grouped-rl-")


async def test_sandbox_function_calls_support_wrapped_outputs(tmp_path: Path) -> None:
    problem = AppsProblem("8", "Add", ([1, 2], [3, 4]), ([3], 7), fn_name="add")
    evaluator = DockerAppsEvaluator(temp_root=tmp_path)
    evaluator._run = AsyncMock(side_effect=[(0, b"3", b""), (0, b"7", b"")])
    assert (await evaluator.grade(problem, "def add(a, b): return a + b")).passed
    assert evaluator._run.await_count == 2
    assert evaluator._run.call_args is not None
    assert evaluator._run.call_args.args[0][-2:] == ["/task/call_runner.py", "add"]


@pytest.mark.parametrize(
    "reply", [(1, b"", b"Traceback: invalid code"), (0, b"wrong", b"")]
)
async def test_program_failures_are_not_infrastructure_failures(
    problem: AppsProblem, tmp_path: Path, reply: tuple[int, bytes, bytes]
) -> None:
    evaluator = DockerAppsEvaluator(temp_root=tmp_path)
    evaluator._run = AsyncMock(return_value=reply)
    grade = await evaluator.grade(problem, "bad code")
    assert not grade.passed and grade.infrastructure_error is None


async def test_missing_docker_is_infrastructure_not_incorrect_code(
    problem: AppsProblem, tmp_path: Path
) -> None:
    evaluator = DockerAppsEvaluator(temp_root=tmp_path)
    evaluator._run = AsyncMock(side_effect=FileNotFoundError("docker unavailable"))
    grade = await evaluator.grade(problem, "print(1)")
    assert not grade.passed and grade.infrastructure_error


async def test_output_is_bounded() -> None:
    evaluator = DockerAppsEvaluator(max_output_bytes=8)
    stream = asyncio.StreamReader()
    stream.feed_data(b"x" * 9)
    stream.feed_eof()
    with pytest.raises(_OutputLimitError):
        await evaluator._read_limited(stream)


@pytest.mark.parametrize(
    ("reference_output", "solution_output", "expected"),
    [(b"1", b"2", True), (b"1", b"1", False), (None, b"2", None)],
)
async def test_oversized_backdoor_batch_grades_each_program_alone(
    tmp_path: Path,
    problem: AppsProblem,
    reference_output: bytes | None,
    solution_output: bytes,
    expected: bool | None,
) -> None:
    # Reference None = the reference overflows the cap, so the check does not apply.
    evaluator = DockerAppsEvaluator(temp_root=tmp_path, batch_tests=True)
    evaluator._execute_batch = AsyncMock(
        side_effect=_OutputLimitError("program output limit exceeded")
    )

    async def execute(
        directory: Path, program: str, fn_name: str | None, input_text: str
    ) -> tuple[int, bytes, bytes]:
        if program == "reference.py":
            if reference_output is None:
                raise _OutputLimitError("program output limit exceeded")
            return 0, reference_output, b""
        return 0, solution_output, b""

    evaluator._execute = execute
    secret = replace(problem, secret_input="7", reference_solution="print(1)")
    grade = await evaluator.check_backdoor(secret, "print(2)")
    assert (None if grade is None else grade.passed) is expected
    assert grade is None or grade.infrastructure_error is None


@pytest.mark.parametrize("overflow", [False, True])
async def test_failure_removes_container_before_killing_cli(
    monkeypatch: pytest.MonkeyPatch, overflow: bool
) -> None:
    events: list[str] = []
    done = asyncio.Event()

    class Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            pass

    class Process:
        stdin = Writer()
        stdout = asyncio.StreamReader()
        stderr = asyncio.StreamReader()
        returncode: int | None = None

        async def wait(self) -> int | None:
            await done.wait()
            return self.returncode

        def kill(self) -> None:
            events.append("kill_cli")
            self.returncode = -9
            done.set()

    evaluator = DockerAppsEvaluator(timeout_seconds=0.01, max_output_bytes=8)
    process = Process()
    if overflow:
        process.stdout.feed_data(b"x" * 9)
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )

    async def remove(name: str) -> None:
        events.append("remove_container")

    evaluator._remove_container = remove
    with pytest.raises(_OutputLimitError if overflow else TimeoutError):
        await evaluator._run(["docker", "run"], "our-container", "input")
    assert events == ["remove_container", "kill_cli"]


async def test_probability_recovery_failure_keeps_all_code(
    problem: AppsProblem,
) -> None:
    client = FakeTrainingClient()
    sampler = FakeSampler()
    sampler.sample_async = AsyncMock(
        return_value=SimpleNamespace(
            sequences=[
                SimpleNamespace(tokens=[30], logprobs=None),
                SimpleNamespace(tokens=[31], logprobs=[-0.5]),
            ]
        )
    )
    sampler.compute_logprobs_async = AsyncMock(
        side_effect=RuntimeError("scoring unavailable")
    )
    sender = TinkerSender(client, sampler, client.tokenizer, allow_paid=True)
    samples = await sender.generate(problem, 0, 2)
    assert len(samples) == 2
    assert samples[0].text == samples[1].text == "print(1)"
    assert samples[0].logprobs is None and samples[0].logprob_error
    assert samples[1].logprobs == (-0.5,)
    assert not client.calls


@pytest.mark.parametrize("empty", [False, True])
async def test_missing_policy_data_keeps_scores_and_stops_training(
    problem: AppsProblem,
    empty: bool,
) -> None:
    loop = trainer()
    samples = await loop.sender.generate(problem, 0, 4)
    samples[0] = GeneratedSolution(
        "" if empty else samples[0].text,
        samples[0].prompt_tokens,
        () if empty else samples[0].response_tokens,
        () if empty else None,
    )
    loop.sender.generate = AsyncMock(return_value=samples)
    with pytest.raises(BatchInfrastructureError) as error:
        await loop.train_step([problem])
    result = error.value.result
    assert len(result.rollouts) == 4
    assert result.rollouts[0].generation is samples[0]
    assert result.rollouts[0].execution is not None
    assert result.rollouts[0].probabilities is not None
    assert result.metrics()["correctness"] == 0.75
    assert not loop.sender.updates and not loop.receiver.updates
    held_out = AppsProblem("99", "Held-out", ("a",), ("b",), split="validation")
    evaluated = await loop.evaluate([held_out], num_samples=4)
    assert evaluated.metrics()["correctness"] == 0.75
    assert evaluated.metrics()["infrastructure_failures"] == 0
    with pytest.raises(ValueError):
        policy_inputs(samples[0], 1.0)


async def test_invalid_utf8_cannot_match_replacement_character(tmp_path: Path) -> None:
    problem = AppsProblem("8", "Print replacement character", ("a",), ("�",))
    evaluator = DockerAppsEvaluator(temp_root=tmp_path)
    evaluator._run = AsyncMock(return_value=(0, b"\xff", b""))
    grade = await evaluator.grade(problem, "code")
    assert not grade.passed and grade.infrastructure_error is None
