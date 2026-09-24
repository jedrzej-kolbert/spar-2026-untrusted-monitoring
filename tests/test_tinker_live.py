"""Live end-to-end tests against the real Tinker API.

Skipped automatically unless ``TINKER_API_KEY`` is set (see the
``tinker_base_model`` fixture). Marked ``slow`` since they make real API calls;
deselect with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from lasr_labs_2025_control_project.utils import tinker_provider as tp

pytestmark = pytest.mark.slow

_ANSWER_SYSTEM = (
    "You will be given a question. Answer only in an <answer> tag, "
    "like <answer>YES</answer> or <answer>NO</answer>."
)


def test_capabilities_nonempty(tinker_base_model):
    import tinker

    caps = tinker.ServiceClient().get_server_capabilities()
    assert len(caps.supported_models) > 0


async def test_generate_via_provider_returns_text(tinker_base_model):
    model = get_model(f"tinker/{tinker_base_model}")
    out = await model.generate(
        [
            ChatMessageSystem(content="You are terse."),
            ChatMessageUser(content="Say the single word: hello"),
        ],
        config=GenerateConfig(max_tokens=16, temperature=0.0),
    )
    assert isinstance(out.completion, str) and out.completion.strip() != ""


async def test_collusion_logprobs_exact_and_discriminate(tinker_base_model):
    model_name = f"tinker/{tinker_base_model}"
    true_stmt = await tp.collusion_yes_no_logprobs(
        model_name, _ANSWER_SYSTEM, "Is 7 a prime number?"
    )
    false_stmt = await tp.collusion_yes_no_logprobs(
        model_name, _ANSWER_SYSTEM, "Is 8 a prime number?"
    )

    # exact logprobs present for both answer tokens
    for d in (true_stmt, false_stmt):
        assert d["YES"] is not None and d["NO"] is not None
        assert d["YES"] <= 0.0 and d["NO"] <= 0.0  # logprobs are non-positive

    # the collusion score (YES - NO) should be higher for the true statement
    score_true = true_stmt["YES"] - true_stmt["NO"]
    score_false = false_stmt["YES"] - false_stmt["NO"]
    assert score_true > score_false


def _require_model(model_name: str) -> None:
    import tinker

    available = {
        m.model_name
        for m in tinker.ServiceClient().get_server_capabilities().supported_models
    }
    if model_name not in available:
        pytest.skip(f"{model_name} is unavailable on this Tinker account")


async def test_kimi_no_thinking_sampling(tinker_base_model):
    model_name = "moonshotai/Kimi-K2.6"
    _require_model(model_name)
    tok = tp._tokenizer(model_name)
    assert tok is not None
    prompt_ids = tp._apply_chat_template(
        model_name,
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
    )
    assert len(prompt_ids) > 0
    assert tok.decode(prompt_ids).endswith("<think></think>")

    model = get_model(f"tinker/{model_name}")
    out = await model.generate(
        [ChatMessageUser(content="Say hello.")],
        config=GenerateConfig(max_tokens=32, temperature=0.0),
    )
    assert out.completion.strip()


def test_glm_tokenizer_loads(tinker_base_model):
    model_name = "zai-org/GLM-5.3:peft:262144"
    _require_model(model_name)
    tok = tp._tokenizer(model_name)
    assert tok is not None
    prompt_ids = tp._apply_chat_template(
        model_name,
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
    )
    assert len(prompt_ids) > 0
