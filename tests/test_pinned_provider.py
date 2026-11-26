"""Offline tests for the pinned/ OpenRouter provider and the Inspect ICRL roles."""

from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage, get_model
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

from lasr_labs_2025_control_project.finetuning.grouped_rl import usage
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_adapters import (
    InspectLearner,
    InspectSender,
)
from lasr_labs_2025_control_project.utils import tinker_provider as tp
from tests.test_hipt_rl import attack_problem
from tests.test_icrl_driver import icrl_config

PINNED = "pinned/deepseek/deepseek-v4.1-flash"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return get_model(PINNED).api


def test_requests_carry_the_pin_and_tinkers_sampling_defaults(api: Any) -> None:
    params = api.completion_params(GenerateConfig(reasoning_effort="minimal"), False)
    assert params["model"] == "deepseek/deepseek-v4.1-flash"
    assert (
        params["extra_body"]["provider"]
        == tp.OPENROUTER_PINS["deepseek/deepseek-v4.1-flash"][1]
    )
    assert params["extra_body"]["reasoning"] == {"enabled": False}
    assert params["extra_body"]["top_k"] == 0
    assert (params["temperature"], params["top_p"], params["max_tokens"]) == (
        1.0,
        1.0,
        16384,
    )
    caller = api.completion_params(GenerateConfig(temperature=0.1, max_tokens=7), False)
    assert (caller["temperature"], caller["max_tokens"]) == (0.1, 7)


def test_unpinned_models_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="no OpenRouter pin"):
        get_model("pinned/qwen/qwen3.5-9b")
    with pytest.raises(ValueError, match="no OpenRouter pin"):
        tp.tinker_base("pinned/qwen/qwen3.5-9b")


def test_tinker_base_maps_every_spelling_to_the_tinker_id() -> None:
    base = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
    assert tp.tinker_base(PINNED) == base
    assert tp.tinker_base(f"tinker/{base}") == base
    assert tp.tinker_base(base) == base


async def test_generate_counts_reported_tokens_and_respects_the_cap(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    meter = usage.Usage()
    monkeypatch.setattr(usage, "USAGE", meter)

    async def fake_generate(*args: Any) -> ModelOutput:
        output = ModelOutput.from_content(model="m", content="ok")
        output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
        return output

    monkeypatch.setattr(OpenAICompatibleAPI, "generate", fake_generate)
    await api.generate([], [], "none", GenerateConfig(max_tokens=50))
    assert meter.tokens == {PINNED: [100, 20, 0]}
    meter.max_cost_usd = meter.cost_usd  # nothing left for the next reservation
    with pytest.raises(usage.BudgetExceeded):
        await api.generate([], [], "none", GenerateConfig(max_tokens=50))


async def test_inspect_sender_returns_one_untrainable_sample_each() -> None:
    sender = InspectSender(
        "mockllm/model",
        allow_paid=True,
        messages_fn=lambda problem, payload: [
            {"role": "system", "content": "rewrite"},
            {"role": "user", "content": problem.question},
        ],
    )
    samples = await sender.generate(attack_problem(), 1, 3)
    assert len(samples) == 3
    assert all(s.text and s.logprobs is None and s.logprob_error for s in samples)
    with pytest.raises(PermissionError):
        await sender.update([])


async def test_inspect_roles_refuse_to_sample_without_approval() -> None:
    with pytest.raises(PermissionError):
        await InspectLearner("mockllm/model").query([{"role": "user", "content": "x"}])
    sender = InspectSender("mockllm/model", messages_fn=lambda *_: [])
    with pytest.raises(PermissionError):
        await sender.generate(attack_problem(), 1, 1)


def test_pinned_sender_is_in_context_only(tmp_path: Path) -> None:
    config = icrl_config(tmp_path, models={"sender": PINNED})
    assert config.models.sender == PINNED  # same base model as the Tinker judge
    with pytest.raises(ValueError, match="can't be trained"):
        icrl_config(tmp_path, models={"sender": PINNED}, method="gradient")
