"""Offline exact YES/NO single-prefill regressions; never create a real client."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from lasr_labs_2025_control_project.utils import tinker_provider as tp


class TensorData:
    """SDK-shaped dense/CSR payload, also usable when Tinker 0.29 is installed."""

    def __init__(self, data: Any, shape: Any, crow: Any = None, col: Any = None):
        self.data = data
        self.shape = shape
        self.sparse_crow_indices = crow
        self.sparse_col_indices = col

    @classmethod
    def from_torch_sparse(cls, tensor: torch.Tensor, pad_value: int = 0):
        assert tensor.dtype == torch.int64
        assert pad_value == -1
        shifted = (tensor - pad_value).to_sparse_csr()
        return cls(
            (shifted.values() + pad_value).tolist(),
            list(tensor.shape),
            shifted.crow_indices().tolist(),
            shifted.col_indices().tolist(),
        )


@pytest.fixture
def single_token_tinker(stub_tinker: Any, monkeypatch: pytest.MonkeyPatch):
    module, sc, tok = stub_tinker
    vocab = {"<answer>": [900], "YES": [10], "NO": [20], "": []}
    monkeypatch.setattr(tok, "encode", lambda text, **kw: list(vocab[text]))
    fake_types = sys.modules["tinker"].types
    monkeypatch.setattr(fake_types, "TensorData", TensorData, raising=False)
    calls: list[Any] = []

    async def sample_async(
        prompt: Any,
        num_samples: int,
        sampling_params: Any,
        include_prompt_logprobs: bool = False,
        target_prompt_logprobs: Any = None,
    ):
        calls.append(
            (
                prompt.to_ints(),
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                target_prompt_logprobs,
            )
        )
        return SimpleNamespace(
            target_prompt_logprobs=TensorData(
                [-0.5, -3.0], [4, 2], [0, 0, 0, 0, 2], [0, 1]
            ),
            # Deliberately irrelevant generated answer.
            sequences=[SimpleNamespace(tokens=[20])],
        )

    monkeypatch.setattr(sc, "sample_async", sample_async)
    return module, sc, tok, calls


async def test_single_prefill_exact_prefix_and_targets(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch
):
    module, sc, tok, calls = single_token_tinker
    templates: list[Any] = []

    def template(messages: Any, **kwargs: Any):
        templates.append((messages, kwargs))
        return [1, 2, 3]

    monkeypatch.setattr(tok, "apply_chat_template", template)
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "system", "user", single_prefill=True
    )
    assert result == {"YES": -0.5, "NO": -3.0, "completion": "<answer>YES</answer>"}
    assert result["YES"] - result["NO"] == 2.5  # unchanged monitor score
    assert not sc.logprob_calls
    assert len(calls) == 1
    prompt, n, params, include, targets = calls[0]
    assert prompt == [1, 2, 3, 900, 10]
    assert n == 1 and params.max_tokens == 1 and include is True
    assert targets.shape == [4, 2]
    assert targets.data == [10, 20]
    assert targets.sparse_crow_indices == [0, 0, 0, 0, 2]
    assert targets.sparse_col_indices == [0, 1]
    assert templates == [
        (
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": False,
            },
        )
    ]


@pytest.mark.parametrize("old_api", ["signature", "tensor", "sparse_method"])
async def test_old_sdk_teacher_forcing(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch, old_api: str
):
    module, sc, tok, calls = single_token_tinker
    types = sys.modules["tinker"].types
    if old_api == "signature":

        async def old_sample(prompt: Any, num_samples: int, sampling_params: Any):
            pytest.fail("old SDK must not sample")

        monkeypatch.setattr(sc, "sample_async", old_sample)
    elif old_api == "tensor":
        monkeypatch.delattr(types, "TensorData")
    else:
        monkeypatch.setattr(types, "TensorData", SimpleNamespace())
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "system", "user", single_prefill=True
    )
    assert result == {"YES": -0.5, "NO": -1.0, "completion": "<answer>YES</answer>"}
    assert sc.logprob_calls == [[1, 2, 3, 900, 10], [1, 2, 3, 900, 20]]
    assert calls == []


async def test_multitoken_and_custom_scaffold(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch
):
    module, sc, tok, calls = single_token_tinker
    vocab = {"open": [901, 902], " yes": [11, 12], " no": [22]}
    monkeypatch.setattr(tok, "encode", lambda text, **kw: vocab[text])
    result = await module.collusion_yes_no_logprobs(
        "tinker/m",
        "system",
        "user",
        answer_open="open",
        yes_text=" yes",
        no_text=" no",
        single_prefill=True,
    )
    assert result == {"YES": -0.02, "NO": -0.01, "completion": "open no</answer>"}
    assert sc.logprob_calls == [[1, 2, 3, 901, 902, 11, 12], [1, 2, 3, 901, 902, 22]]
    assert calls == []


async def test_empty_candidate_preserves_missing(single_token_tinker: Any):
    module, sc, _, calls = single_token_tinker
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "s", "u", yes_text="", single_prefill=True
    )
    assert result == {"YES": None, "NO": -1.0, "completion": "<answer>NO</answer>"}
    assert len(sc.logprob_calls) == 1 and not calls


@pytest.mark.parametrize(
    "values,answer",
    [
        ([-2.0, -0.5], "NO"),
        ([-1.0, -1.0], "YES"),
        ([None, -1.0], "NO"),
        ([-1.0, float("nan")], "YES"),
        ([float("nan"), None], ""),
        ([float("inf"), -1.0], "YES"),
        ([float("-inf"), -1.0], "NO"),
        ([0.0, -1.0], "YES"),
    ],
)
async def test_target_values_preserve_completion(
    single_token_tinker: Any,
    monkeypatch: pytest.MonkeyPatch,
    values: list[float | None],
    answer: str,
):
    module, sc, _, calls = single_token_tinker

    async def sample_async(
        *args: Any, target_prompt_logprobs: Any = None, **kwargs: Any
    ):
        calls.append(1)
        return SimpleNamespace(
            target_prompt_logprobs=TensorData([0.0] * 6 + values, [4, 2])
        )

    monkeypatch.setattr(sc, "sample_async", sample_async)
    result = await module.collusion_yes_no_logprobs(
        "tinker/m", "s", "u", single_prefill=True
    )
    for key, value in zip(("YES", "NO"), values):
        assert result[key] == (None if value is None or math.isnan(value) else value)
    assert result["completion"] == (f"<answer>{answer}</answer>" if answer else "")
    assert len(calls) == 1 and not sc.logprob_calls


def test_sparse_missing_is_not_zero():
    assert tp._answer_target_logprobs(
        TensorData([-1.0], [4, 2], [0, 0, 0, 0, 1], [1]), 4
    ) == [None, -1.0]
    assert tp._answer_target_logprobs(None, 4) == [None, None]


@pytest.mark.parametrize(
    "tensor",
    [
        TensorData([0.0, 0.0], [1, 2]),
        TensorData([0.0], [4, 2]),
        TensorData([0.0], [4, 2], [0, 0, 0, 0, 2], [1]),
        TensorData([0.0, 0.0], [4, 2], [0, 0, 0, 0, 2], [1, 1]),
        TensorData([0.0], [4, 2], [0, 1, 0, 0, 1], [1]),
        TensorData([0.0], [4, 2], [0, 0, 0, 0, 1], [2]),
    ],
)
def test_malformed_target_response_rejected(tensor: TensorData):
    with pytest.raises(ValueError):
        tp._answer_target_logprobs(tensor, 4)


@pytest.mark.parametrize(
    "message",
    ["auth", "quota", "context", "model", "target_prompt_logprobs unsupported"],
)
async def test_backend_errors_never_paid_fallback(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch, message: str
):
    module, sc, _, calls = single_token_tinker

    async def sample_async(
        *args: Any, target_prompt_logprobs: Any = None, **kwargs: Any
    ):
        calls.append(1)
        raise RuntimeError(message)

    monkeypatch.setattr(sc, "sample_async", sample_async)
    with pytest.raises(RuntimeError, match=message):
        await module.collusion_yes_no_logprobs(
            "tinker/m", "s", "u", single_prefill=True
        )
    assert len(calls) == 1 and not sc.logprob_calls


async def test_legacy_is_default_even_with_target_capability(single_token_tinker: Any):
    module, sc, _, calls = single_token_tinker
    result = await module.collusion_yes_no_logprobs("tinker/m", "system", "user")
    assert result == {"YES": -0.5, "NO": -1.0, "completion": "<answer>YES</answer>"}
    assert sc.logprob_calls == [[1, 2, 3, 900, 10], [1, 2, 3, 900, 20]]
    assert not calls


async def test_custom_single_tokens_are_encoded_not_assumed(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch
):
    module, sc, tok, calls = single_token_tinker
    vocab = {"open": [901, 902], " yes": [123], " no": [456]}
    monkeypatch.setattr(tok, "encode", lambda text, **kw: vocab[text])

    async def sample_async(
        prompt: Any, target_prompt_logprobs: Any = None, **kwargs: Any
    ):
        assert prompt.to_ints() == [1, 2, 3, 901, 902, 123]
        assert target_prompt_logprobs.shape == [5, 2]
        assert target_prompt_logprobs.data == [123, 456]
        assert target_prompt_logprobs.sparse_crow_indices == [0, 0, 0, 0, 0, 2]
        assert kwargs["include_prompt_logprobs"] is True
        calls.append(1)
        return SimpleNamespace(
            target_prompt_logprobs=TensorData([0.0] * 8 + [-1.0, -0.5], [5, 2])
        )

    monkeypatch.setattr(sc, "sample_async", sample_async)
    result = await module.collusion_yes_no_logprobs(
        "tinker/m",
        "s",
        "u",
        answer_open="open",
        yes_text=" yes",
        no_text=" no",
        single_prefill=True,
    )
    assert result == {"YES": -1.0, "NO": -0.5, "completion": "open no</answer>"}
    assert len(calls) == 1 and not sc.logprob_calls


@pytest.mark.parametrize("malformed", [False, True])
async def test_missing_or_malformed_response_no_retry(
    single_token_tinker: Any, monkeypatch: pytest.MonkeyPatch, malformed: bool
):
    module, sc, _, calls = single_token_tinker

    async def sample_async(
        *args: Any, target_prompt_logprobs: Any = None, **kwargs: Any
    ):
        calls.append(1)
        tensor = TensorData([], [4, 2]) if malformed else None
        return SimpleNamespace(target_prompt_logprobs=tensor)

    monkeypatch.setattr(sc, "sample_async", sample_async)
    if malformed:
        with pytest.raises(ValueError, match="Malformed"):
            await module.collusion_yes_no_logprobs(
                "tinker/m", "s", "u", single_prefill=True
            )
    else:
        result = await module.collusion_yes_no_logprobs(
            "tinker/m", "s", "u", single_prefill=True
        )
        assert result == {"YES": None, "NO": None, "completion": ""}
    assert len(calls) == 1 and not sc.logprob_calls


def test_installed_sdk_tensor_contract():
    # Imports types only, no clients/network. Skip just this compatibility check
    # with 0.29; all fast-path and fallback regressions above still run offline.
    from tinker import types

    if not hasattr(types.TensorData, "from_torch_sparse"):
        pytest.skip("installed SDK predates targeted prompt logprobs")
    target = torch.full((16, 2), -1, dtype=torch.int64)
    target[-1] = torch.tensor([51346, 13909])
    packed = types.TensorData.from_torch_sparse(target, pad_value=-1)
    assert torch.equal(packed.to_torch(pad_value=-1), target)
    response = types.TensorData.from_torch_sparse(
        torch.cat((torch.zeros(15, 2), torch.tensor([[-0.5, -3.0]])))
    )
    assert tp._answer_target_logprobs(response, 16) == [-0.5, -3.0]
