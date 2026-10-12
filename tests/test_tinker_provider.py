"""Offline unit tests for utils/tinker_provider.py (no network / no API key).

Live end-to-end tests against the real Tinker API live in test_tinker_live.py.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import GenerateConfig, get_model

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
# Tool calling (stubbed tinker-cookbook renderer)
# --------------------------------------------------------------------------- #
class _FakeRendererToolCall:
    class FunctionBody:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    def __init__(self, function, id=None):
        self.function, self.id = function, id


class _FakeRenderer:
    """Records the rendered conversation; replies with a fixed parsed message."""

    def __init__(self, reply):
        self.reply = reply
        self.rendered = None

    def create_conversation_prefix_with_tools(self, specs, system_prompt):
        return [{"role": "system", "content": system_prompt, "tools": specs}]

    def build_generation_prompt(self, messages):
        from tinker import types

        self.rendered = messages
        return types.ModelInput.from_ints([1, 2, 3, 4])

    def get_stop_sequences(self):
        return ["<stop>"]

    def parse_response(self, tokens):
        return self.reply, True


@pytest.fixture
def stub_renderer(stub_tinker, monkeypatch):
    import sys
    import types

    module, _sc, _tok = stub_tinker
    monkeypatch.setitem(sys.modules, "tinker_cookbook", types.ModuleType("cookbook"))
    renderers = types.ModuleType("tinker_cookbook.renderers")
    renderers.ToolCall = _FakeRendererToolCall
    renderers.get_text_content = lambda message: message["content"]
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", renderers)

    def install(reply):
        renderer = _FakeRenderer(reply)
        monkeypatch.setattr(module, "_renderer", lambda model_ref: renderer)
        return renderer

    return install


def _bash_tool():
    from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

    return ToolInfo(
        name="bash",
        description="Run a command.",
        parameters=ToolParams(
            properties={"cmd": ToolParam(type="string", description="The command.")},
            required=["cmd"],
        ),
    )


async def test_generate_with_tools_returns_tool_call(stub_renderer):
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall

    call = _FakeRendererToolCall(
        _FakeRendererToolCall.FunctionBody("bash", '{"cmd": "ls /tmp"}'), id="c1"
    )
    renderer = stub_renderer({"content": "Listing.", "tool_calls": [call]})
    history = [
        ChatMessageSystem(content="Use bash."),
        ChatMessageUser(content="List /tmp."),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c0", function="bash", arguments={"cmd": "pwd"})],
        ),
        ChatMessageTool(content="/root", tool_call_id="c0", function="bash"),
    ]
    out = await get_model("tinker/Qwen/Qwen3-32B").api.generate(
        input=history,
        tools=[_bash_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=8),
    )
    [parsed] = out.message.tool_calls
    assert (parsed.id, parsed.function, parsed.arguments) == (
        "c1",
        "bash",
        {"cmd": "ls /tmp"},
    )
    assert out.message.text == "Listing."
    assert out.stop_reason == "tool_calls"
    assert out.usage.input_tokens == 4 and out.usage.output_tokens == 1
    # The earlier call and its result are rendered back in the native format.
    assert [m["role"] for m in renderer.rendered] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert renderer.rendered[0]["tools"][0]["name"] == "bash"
    assert renderer.rendered[2]["tool_calls"][0].function.arguments == '{"cmd": "pwd"}'
    assert renderer.rendered[3]["tool_call_id"] == "c0"


async def test_malformed_tool_arguments_become_parse_errors(stub_renderer):
    bad = _FakeRendererToolCall(_FakeRendererToolCall.FunctionBody("bash", "[1]"))
    stub_renderer({"content": "", "tool_calls": [bad]})
    out = await get_model("tinker/Qwen/Qwen3-32B").api.generate(
        input=[],
        tools=[_bash_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=8),
    )
    [parsed] = out.message.tool_calls
    assert parsed.arguments == {} and "JSON object" in parsed.parse_error
    assert parsed.id.startswith("bash_")
