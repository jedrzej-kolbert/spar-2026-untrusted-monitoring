"""Offline tests for opt-in single-prefill YES/NO scoring; never create a real client."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tinker.types import TensorData

# Stub chat template followed by "<answer>". Long enough that the SDK encodes
# the targets as sparse CSR, as it does for real prompts.
PREFIX = [1, 2, 3, 4, 5, 6, 7, 8, 900]


@pytest.fixture
def single_token_tinker(stub_tinker: Any, monkeypatch: pytest.MonkeyPatch):
    module, sc, tok = stub_tinker
    vocab = {"<answer>": [900], "YES": [10], "NO": [20], " no": [22, 23]}
    monkeypatch.setattr(tok, "encode", lambda text, **kw: list(vocab[text]))
    monkeypatch.setattr(tok, "apply_chat_template", lambda *a, **kw: PREFIX[:-1])
    monkeypatch.setattr(
        sys.modules["tinker"].types, "TensorData", TensorData, raising=False
    )
    calls: list[Any] = []

    async def sample_async(
        prompt: Any, num_samples: int, sampling_params: Any, **kwargs: Any
    ):
        calls.append((prompt.to_ints(), num_samples, sampling_params, kwargs))
        logprobs = torch.zeros(len(PREFIX), 2)
        logprobs[-1] = torch.tensor([-0.5, -3.0])
        return SimpleNamespace(
            target_prompt_logprobs=TensorData.from_torch_sparse(logprobs)
        )

    monkeypatch.setattr(sc, "sample_async", sample_async)
    return module, sc, calls


async def test_single_prefill_scores_both_answers_in_one_request(
    single_token_tinker: Any,
):
    module, sc, calls = single_token_tinker
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "system", "user", single_prefill=True
    )
    usage = {"input_tokens": len(PREFIX) + 1, "output_tokens": 1}
    assert result == {
        "YES": -0.5,
        "NO": -3.0,
        "completion": "<answer>YES</answer>",
        "usage": usage,
    }
    assert not sc.logprob_calls and len(calls) == 1
    prompt, num_samples, params, kwargs = calls[0]
    assert prompt == PREFIX + [10]  # YES only fills the answer slot
    assert num_samples == 1 and params.max_tokens == 1
    assert kwargs["include_prompt_logprobs"] is True
    targets = kwargs["target_prompt_logprobs"]
    assert targets.sparse_crow_indices is not None
    expected = torch.full((len(PREFIX), 2), -1)
    expected[-1] = torch.tensor([10, 20])
    assert torch.equal(targets.to_torch(pad_value=-1), expected)


async def test_teacher_forcing_stays_the_default(single_token_tinker: Any):
    module, sc, calls = single_token_tinker
    result = await module.collusion_yes_no_logprobs("tinker/m", "system", "user")
    usage = {"input_tokens": 2 * len(PREFIX) + 2, "output_tokens": 0}
    assert result == {
        "YES": -0.5,
        "NO": -1.0,
        "completion": "<answer>YES</answer>",
        "usage": usage,
    }
    assert sc.logprob_calls == [PREFIX + [10], PREFIX + [20]]
    assert not calls


async def test_multi_token_answer_uses_teacher_forcing(single_token_tinker: Any):
    module, sc, calls = single_token_tinker
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "system", "user", no_text=" no", single_prefill=True
    )
    usage = {"input_tokens": 2 * len(PREFIX) + 3, "output_tokens": 0}
    assert result == {
        "YES": -0.5,
        "NO": -0.02,
        "completion": "<answer> no</answer>",
        "usage": usage,
    }
    assert sc.logprob_calls == [PREFIX + [10], PREFIX + [22, 23]]
    assert not calls
