"""Offline unit tests for utils/tinker_provider.py (no network / no API key).

Live end-to-end tests against the real Tinker API live in test_tinker_live.py.
"""

from __future__ import annotations

import sys

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


def test_nothink_template_kwargs():
    assert tp._NOTHINK_TEMPLATE_KWARGS.get("enable_thinking") is False
    assert tp._NOTHINK_TEMPLATE_KWARGS.get("thinking") is False


def test_tiktoken_blobfile_patch_fallback(monkeypatch, tmp_path):
    """Verify that tiktoken.load.read_file falls back to open() when blobfile is missing."""
    import tiktoken.load

    # Create a test file
    test_file = tmp_path / "test.bpe"
    test_file.write_bytes(b"hello bpe")

    # Force unpatched state and patch
    monkeypatch.setattr(tiktoken.load, "_spar_patched", False)
    tp._ensure_tiktoken_blobfile_patch()

    # Simulate blobfile missing in sys.modules
    monkeypatch.setitem(sys.modules, "blobfile", None)

    # Calling read_file on local path should succeed via open()
    data = tiktoken.load.read_file(str(test_file))
    assert data == b"hello bpe"


def test_tokenizer_fallback_on_tokenizers_backend_error(monkeypatch):
    """When sampling_client.get_tokenizer() fails with TokenizersBackend error, fallback is called."""
    called_with = []

    class FakeSamplingClient:
        def get_tokenizer(self):
            raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

        def get_base_model(self):
            return "zai-org/GLM-5.3:peft:262144"

    monkeypatch.setattr(tp, "_sampling_client", lambda model_ref: FakeSamplingClient())
    monkeypatch.setattr(
        tp,
        "_load_fast_tokenizer_fallback",
        lambda base_model: called_with.append(base_model) or "mock_fast_tokenizer",
    )
    tp._tokenizer.cache_clear()
    try:
        tok = tp._tokenizer("zai-org/GLM-5.3:peft:262144")
        assert tok == "mock_fast_tokenizer"
        assert called_with == ["zai-org/GLM-5.3:peft:262144"]
    finally:
        tp._tokenizer.cache_clear()


async def test_collusion_logprobs_closes_unclosed_think(stub_tinker, monkeypatch):
    """If chat template ends with <think>, collusion_yes_no_logprobs appends </think>."""
    module, sc, tok = stub_tinker

    # Enhance stub tokenizer to support convert_tokens_to_ids
    tok.convert_tokens_to_ids = lambda t: 501 if t == "<think>" else 502 if t == "</think>" else None
    # Chat template returns prefix ending in 501 (<think>)
    monkeypatch.setattr(
        tok,
        "apply_chat_template",
        lambda messages, add_generation_prompt, tokenize, return_dict=False, **kwargs: [1, 2, 501],
    )

    await module.collusion_yes_no_logprobs("tinker/m", "system", "user")
    # prefix should have [1, 2, 501] + [502] (</think>) + [900] (<answer>) = 5 tokens
    assert sc.logprob_calls[0][:5] == [1, 2, 501, 502, 900]
