"""Offline unit tests for utils/tinker_provider.py (no network / no API key).

Live end-to-end tests against the real Tinker API live in test_tinker_live.py.
"""

from __future__ import annotations

import types as pytypes

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolInfo

from lasr_labs_2025_control_project.utils import tinker_provider as tp


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_is_tinker_model():
    assert tp.is_tinker_model("tinker/Qwen/Qwen3-32B")
    assert not tp.is_tinker_model("openai/gpt-4.1")
    assert not tp.is_tinker_model("openai-api/together/Qwen/Qwen2.5-7B-Instruct-Turbo")


def test_strip_prefix():
    assert tp.strip_prefix("tinker/Qwen/Qwen3-32B") == "Qwen/Qwen3-32B"
    # idempotent / no-op for unprefixed names
    assert tp.strip_prefix("Qwen/Qwen3-32B") == "Qwen/Qwen3-32B"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([1, 2, 3], [1, 2, 3]),  # already flat
        ({"input_ids": [4, 5]}, [4, 5]),  # BatchEncoding-as-dict
        ({"input_ids": [[6, 7, 8]]}, [6, 7, 8]),  # nested batch dim
    ],
)
def test_normalize_ids(raw, expected):
    assert tp._normalize_ids(raw) == expected


def test_normalize_ids_batchencoding_attr():
    class FakeBatchEncoding:
        input_ids = [9, 10, 11]

    assert tp._normalize_ids(FakeBatchEncoding()) == [9, 10, 11]


# --------------------------------------------------------------------------- #
# Provider registration (needs inspect only; no key)
# --------------------------------------------------------------------------- #
def test_provider_registered_and_resolves(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    model = get_model("tinker/Qwen/Qwen3-32B")
    assert type(model.api).__name__ == "TinkerAPI"
    # prefix is stripped by get_model before reaching the provider
    assert model.api.base_model == "Qwen/Qwen3-32B"


def test_service_client_requires_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    tp._service_client = None  # reset process-wide cache
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        tp._service()


def test_sampling_client_routes_checkpoint_vs_base(monkeypatch):
    """Base names go via base_model; tinker:// checkpoints via model_path."""
    calls = []

    class FakeService:
        def create_sampling_client(self, model_path=None, base_model=None):
            calls.append(("path", model_path) if model_path else ("base", base_model))
            return object()

    monkeypatch.setattr(tp, "_service", lambda: FakeService())
    tp._sampling_client.cache_clear()
    try:
        tp._sampling_client("Qwen/Qwen3-8B")
        tp._sampling_client("tinker://run-id/sampler_weights/self_rec_u")
    finally:
        tp._sampling_client.cache_clear()
    assert calls == [
        ("base", "Qwen/Qwen3-8B"),
        ("path", "tinker://run-id/sampler_weights/self_rec_u"),
    ]


# --------------------------------------------------------------------------- #
# Forced-decoding collusion logprobs (stubbed clients)
# --------------------------------------------------------------------------- #
async def test_collusion_logprobs_sums_multitoken(stub_tinker):
    module, _sc, _tok = stub_tinker
    result = await module.collusion_yes_no_logprobs(
        "tinker/Qwen/Qwen3-32B", "system", "user"
    )
    # YES is one token (-0.5); NO is two tokens (-1.0 + -2.0) summed.
    assert result["YES"] == pytest.approx(-0.5)
    assert result["NO"] == pytest.approx(-3.0)
    # YES has higher logprob -> completion reports YES
    assert result["completion"] == "<answer>YES</answer>"


async def test_collusion_logprobs_shared_prefix(stub_tinker):
    """Both candidates must be scored against the same prompt+<answer> prefix."""
    _module, sc, _tok = stub_tinker
    await _module.collusion_yes_no_logprobs("tinker/m", "system", "user")
    # two compute_logprobs calls (YES, NO); both share the 4-token prefix
    # ([1,2,3] chat template + [900] for "<answer>")
    assert len(sc.logprob_calls) == 2
    assert sc.logprob_calls[0][:4] == [1, 2, 3, 900]
    assert sc.logprob_calls[1][:4] == [1, 2, 3, 900]
    assert sc.logprob_calls[0][4:] == [10]  # YES
    assert sc.logprob_calls[1][4:] == [20, 21]  # NO


async def test_generate_via_provider(stub_tinker):
    module, _sc, _tok = stub_tinker
    api = get_model("tinker/Qwen/Qwen3-32B").api
    out = await api.generate(
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(max_tokens=8, temperature=0.0),
    )
    # stub emits token id 10 -> tokenizer decodes to "YES"
    assert out.completion == "YES"
    assert out.stop_reason == "stop"
    # one sampled token, plus whatever the chat template produced for the prompt
    assert out.usage is not None
    assert out.usage.output_tokens == 1
    assert out.usage.total_tokens == out.usage.input_tokens + 1


# --------------------------------------------------------------------------- #
# Agent runs: fitting the history to the context window
# --------------------------------------------------------------------------- #
def _agent_messages(steps: int, output: str = "x" * 200) -> list:
    messages: list = [
        ChatMessageSystem(content="system"),
        ChatMessageUser(content="task"),
    ]
    for i in range(steps):
        call = ToolCall(id=f"call{i}", function="bash", arguments={"cmd": "ls"})
        messages.append(ChatMessageAssistant(content=f"step {i}", tool_calls=[call]))
        messages.append(
            ChatMessageTool(content=output, tool_call_id=call.id, function="bash")
        )
    return messages


def _chars(messages: list) -> pytypes.SimpleNamespace:
    """A prompt whose token length is the number of characters in the messages."""
    return pytypes.SimpleNamespace(length=sum(len(m.text) for m in messages))


def test_shortened_omits_oldest_outputs_first():
    messages = _agent_messages(4)
    short = tp._shortened(messages, 2)
    assert [m.text for m in short if m.role == "tool"] == [
        tp._OMITTED_OUTPUT,
        tp._OMITTED_OUTPUT,
        "x" * 200,
        "x" * 200,
    ]
    assert [m.role for m in short] == [m.role for m in messages]
    assert messages[3].text == "x" * 200  # the caller's messages are not changed


def test_shortened_then_drops_oldest_steps():
    messages = _agent_messages(4)  # three earlier outputs, three earlier steps
    short = tp._shortened(messages, 3 + 2)
    assert short[:2] == messages[:2]
    assert short[2].role == "user" and short[2].text.startswith("[2 earlier steps")
    assert [m.text for m in short[3:]] == [
        "step 2",
        tp._OMITTED_OUTPUT,
        "step 3",
        "x" * 200,  # the latest step keeps its output
    ]
    # past the last level there is nothing more to drop
    assert [m.text for m in tp._shortened(messages, 99)[3:]] == ["step 3", "x" * 200]


def test_fit_history_omits_no_more_than_needed():
    messages = _agent_messages(6)
    full = _chars(messages).length

    assert tp._fit_history(_chars, messages, full) == (_chars(messages), None)

    budget = full - 300  # two 200-character outputs must go
    prompt, omitted = tp._fit_history(_chars, messages, budget)
    assert omitted == {"outputs": 2, "steps": 0}
    assert prompt.length <= budget < _chars(tp._shortened(messages, 1)).length

    # room for the task and the latest step only
    latest = _chars(messages[:2] + messages[-2:]).length
    prompt, omitted = tp._fit_history(_chars, messages, latest + 60)
    assert omitted == {"outputs": 5, "steps": 5}
    assert prompt.length <= latest + 60

    assert tp._fit_history(_chars, messages, latest - 1) is None


def test_context_window_reads_server_capabilities(monkeypatch):
    models = [
        pytypes.SimpleNamespace(model_name="a/small", max_context_length=32768),
        pytypes.SimpleNamespace(model_name="a/large", max_context_length=65536),
    ]
    service = pytypes.SimpleNamespace(
        get_server_capabilities=lambda: pytypes.SimpleNamespace(supported_models=models)
    )
    monkeypatch.setattr(tp, "_service", lambda: service)
    tp._context_window.cache_clear()
    try:
        assert tp._context_window("a/small") == 32768
        assert tp._context_window("a/large") == 65536
        assert tp._context_window("a/unlisted") is None
    finally:
        tp._context_window.cache_clear()


class _StubRenderer:
    """Renders a conversation to one token per character of message content."""

    def create_conversation_prefix_with_tools(self, specs, system_prompt):
        return []

    def build_generation_prompt(self, messages):
        return pytypes.SimpleNamespace(length=sum(len(m["content"]) for m in messages))

    def get_stop_sequences(self):
        return []

    def parse_response(self, tokens):
        return {"role": "assistant", "content": "ok"}, True


@pytest.fixture
def agent_api(monkeypatch):
    """The provider with a stub renderer; returns (api, recorded sample calls).

    Uses the real `tinker` types: the tool-call path imports tinker-cookbook, which
    the fake module of `stub_tinker` would break.
    """
    calls = []

    class Client:
        async def sample_async(self, prompt, num_samples, sampling_params):
            calls.append((prompt, sampling_params))
            return pytypes.SimpleNamespace(
                sequences=[pytypes.SimpleNamespace(tokens=[1], stop_reason="stop")]
            )

    monkeypatch.setattr(tp, "_renderer", lambda model_ref: _StubRenderer())
    monkeypatch.setattr(tp, "_sampling_client", lambda model_ref: Client())
    monkeypatch.setattr(tp, "_AGENT_MIN_OUTPUT_TOKENS", 100)
    return get_model("tinker/Qwen/Qwen3-32B").api, calls


_BASH = [ToolInfo(name="bash", description="Run a command.")]
_STEP_CONFIG = GenerateConfig(max_tokens=400)


async def test_generate_fits_agent_history_to_window(agent_api, monkeypatch):
    api, calls = agent_api
    monkeypatch.setattr(tp, "_context_window", lambda model_ref: 1000)
    messages = _agent_messages(6)  # 1240 characters; omitting an output saves 150
    before = [m.text for m in messages]

    out = await api.generate(messages, _BASH, "auto", _STEP_CONFIG)

    prompt, params = calls[0]
    assert prompt.length <= 1000 - 100  # room for a step is kept
    assert params.max_tokens == min(400, 1000 - prompt.length)
    assert out.message.metadata == {"history_omitted": {"outputs": 3, "steps": 0}}
    assert out.usage is not None and out.usage.input_tokens == prompt.length
    assert [m.text for m in messages] == before  # the trajectory itself is untouched


async def test_generate_sends_a_short_history_unchanged(agent_api, monkeypatch):
    api, calls = agent_api
    monkeypatch.setattr(tp, "_context_window", lambda model_ref: 100_000)
    messages = _agent_messages(6)

    out = await api.generate(messages, _BASH, "auto", _STEP_CONFIG)

    prompt, params = calls[0]
    # the system message is the renderer's prefix, which the stub leaves empty
    assert prompt.length == sum(len(m.text) for m in messages[1:])
    assert params.max_tokens == 400
    assert out.message.metadata is None


async def test_generate_ends_the_run_when_one_step_is_too_long(agent_api, monkeypatch):
    api, calls = agent_api
    monkeypatch.setattr(tp, "_context_window", lambda model_ref: 1000)
    messages = _agent_messages(1, output="x" * 2000)

    out = await api.generate(messages, _BASH, "auto", _STEP_CONFIG)

    assert out.stop_reason == "model_length"  # Inspect's agent loop stops on this
    assert calls == []


async def test_generate_without_a_reported_window_sends_everything(
    agent_api, monkeypatch
):
    api, calls = agent_api
    monkeypatch.setattr(tp, "_context_window", lambda model_ref: None)
    messages = _agent_messages(6)

    await api.generate(messages, _BASH, "auto", _STEP_CONFIG)

    prompt, params = calls[0]
    assert prompt.length == sum(len(m.text) for m in messages[1:])
    assert params.max_tokens == 400
