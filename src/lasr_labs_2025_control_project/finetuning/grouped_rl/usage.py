"""Running token and cost totals for every Tinker request in this process.

``install()`` wraps the SDK's sample, compute_logprobs and forward_backward calls,
so requests made through the inspect provider (the trusted monitor) are counted
too. Costs are estimates from published prices, not billing data.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

# USD per 1M tokens: (prefill, sample, train). Tinker pricing page, 2026-10-19.
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-ai/DeepSeek-V3.1": (1.695, 4.215, 3.718),
    "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),
}
# A model missing from PRICES is charged the highest known rates, so the budget
# stop never undercounts.
FALLBACK_PRICE = tuple(max(price[i] for price in PRICES.values()) for i in range(3))
KINDS = ("prefill", "sample", "train")


class Usage:
    def __init__(self) -> None:
        self.tokens: dict[str, list[int]] = {}  # model -> [prefill, sample, train]

    def add(self, model: str, *, prefill: int = 0, sample: int = 0, train: int = 0):
        counts = self.tokens.setdefault(model, [0, 0, 0])
        for index, value in enumerate((prefill, sample, train)):
            counts[index] += value

    @property
    def cost_usd(self) -> float:
        return math.fsum(
            count * PRICES.get(model, FALLBACK_PRICE)[index] / 1e6
            for model, counts in self.tokens.items()
            for index, count in enumerate(counts)
        )

    def snapshot(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {"cost_usd": self.cost_usd}
        for index, kind in enumerate(KINDS):
            out[f"tokens_{kind}"] = sum(c[index] for c in self.tokens.values())
        out["unpriced_models"] = sum(model not in PRICES for model in self.tokens)
        return out


USAGE = Usage()


def install(usage: Usage = USAGE) -> Usage:
    """Count every Tinker request from now on. Safe to call more than once."""
    import tinker

    sampler, trainer = tinker.SamplingClient, tinker.TrainingClient
    if getattr(sampler.sample_async, "_counts_usage", False):
        return usage
    sample, logprobs = sampler.sample_async, sampler.compute_logprobs_async
    backward = trainer.forward_backward_async
    sample_signature = inspect.signature(sample)

    async def sampler_model(client: Any) -> str:
        if getattr(client, "_usage_model", None) is None:
            client._usage_model = await client.get_base_model_async()
        return client._usage_model

    async def trainer_model(client: Any) -> str:
        if getattr(client, "_usage_model", None) is None:
            info = await client.get_info_async()
            client._usage_model = info.model_data.model_name or info.model_name or ""
        return client._usage_model

    async def sample_async(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = sample_signature.bind(self, *args, **kwargs).arguments
        result = await sample(self, *args, **kwargs)
        usage.add(
            await sampler_model(self),
            # shortcut: counts the prompt once per sample, an upper bound if Tinker
            # prefills it once per request; switch to billing data if it matters.
            prefill=len(bound["prompt"].to_ints()) * bound["num_samples"],
            sample=sum(len(sequence.tokens) for sequence in result.sequences),
        )
        return result

    async def compute_logprobs_async(self: Any, prompt: Any) -> Any:
        result = await logprobs(self, prompt)
        usage.add(await sampler_model(self), prefill=len(prompt.to_ints()))
        return result

    async def forward_backward_async(self: Any, data: Any, *args: Any, **kw: Any):
        future = await backward(self, data, *args, **kw)
        usage.add(
            await trainer_model(self),
            # +1: the target of the last input position.
            train=sum(len(datum.model_input.to_ints()) + 1 for datum in data),
        )
        return future

    for function in (sample_async, compute_logprobs_async, forward_backward_async):
        function._counts_usage = True  # type: ignore[attr-defined]
    sampler.sample_async = sample_async  # type: ignore[method-assign]
    sampler.compute_logprobs_async = compute_logprobs_async  # type: ignore[method-assign]
    trainer.forward_backward_async = forward_backward_async  # type: ignore[method-assign]
    return usage
