"""Running token and cost totals for every Tinker request in this process.

``install()`` wraps the SDK's sample, compute_logprobs and forward_backward calls,
so requests made through the inspect provider (the trusted monitor) are counted
too. Costs are estimates from published prices, not billing data.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# USD per 1M tokens: (prefill, sample, train). Tinker pricing page, 2026-10-19.
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-ai/DeepSeek-V3.1": (1.695, 4.215, 3.718),
    "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),
}
# Unknown models get an indicative estimate only; capped runs reject them.
FALLBACK_PRICE = tuple(max(price[i] for price in PRICES.values()) for i in range(3))
KINDS = ("prefill", "sample", "train")


class BudgetExceeded(RuntimeError):
    """A request was refused before dispatch to stay within the token-cost budget."""


class Usage:
    def __init__(self) -> None:
        self.tokens: dict[str, list[int]] = {}  # model -> [prefill, sample, train]
        self.max_cost_usd: float | None = None
        self.reserved_usd = 0.0
        self.exhausted = False

    @contextmanager
    def request(
        self, model: str, *, prefill: int = 0, sample: int = 0, train: int = 0
    ) -> Iterator[dict[str, int]]:
        """Reserve the maximum cost, then account actual tokens (maximum on error)."""
        if self.max_cost_usd is not None and model not in PRICES:
            raise ValueError(f"no budget price for model {model!r}; refusing request")
        counts = dict(prefill=prefill, sample=sample, train=train)
        prices = PRICES.get(model, FALLBACK_PRICE)
        reserved = math.fsum(
            counts[kind] * prices[i] / 1e6 for i, kind in enumerate(KINDS)
        )
        if self.max_cost_usd is not None and (
            self.exhausted
            or self.cost_usd + self.reserved_usd + reserved > self.max_cost_usd
        ):
            self.exhausted = True
            raise BudgetExceeded(
                f"budget: ${self.cost_usd:.4f} spent, ${self.reserved_usd:.4f} in flight, "
                f"request up to ${reserved:.4f}, cap ${self.max_cost_usd:.2f}"
            )
        # No await between checking and reserving: concurrent asyncio calls share the cap.
        self.reserved_usd += reserved
        try:
            yield counts
        finally:
            self.reserved_usd -= reserved
            self.add(model, **counts)

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
        params = bound.get("sampling_params", kwargs.get("sampling_params"))
        max_tokens = params.max_tokens if params is not None else None
        if max_tokens is None and usage.max_cost_usd is not None:
            raise ValueError("capped sampling requires an explicit max_tokens")
        with usage.request(
            await sampler_model(self),
            # shortcut: counts the prompt once per sample, an upper bound if Tinker
            # prefills it once per request; switch to billing data if it matters.
            prefill=len(bound["prompt"].to_ints()) * bound["num_samples"],
            sample=(max_tokens or 0) * bound["num_samples"],
        ) as counts:
            result = await sample(self, *args, **kwargs)
            counts["sample"] = sum(
                len(sequence.tokens) for sequence in result.sequences
            )
            return result

    async def compute_logprobs_async(self: Any, prompt: Any) -> Any:
        with usage.request(await sampler_model(self), prefill=len(prompt.to_ints())):
            return await logprobs(self, prompt)

    async def forward_backward_async(self: Any, data: Any, *args: Any, **kw: Any):
        with usage.request(
            await trainer_model(self),
            # +1: the target of the last input position.
            train=sum(len(datum.model_input.to_ints()) + 1 for datum in data),
        ):
            return await backward(self, data, *args, **kw)

    for function in (sample_async, compute_logprobs_async, forward_backward_async):
        function._counts_usage = True  # type: ignore[attr-defined]
    sampler.sample_async = sample_async  # type: ignore[method-assign]
    sampler.compute_logprobs_async = compute_logprobs_async  # type: ignore[method-assign]
    trainer.forward_backward_async = forward_backward_async  # type: ignore[method-assign]
    return usage
