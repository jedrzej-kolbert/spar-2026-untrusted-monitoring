"""Client-side token counts and optional training cost estimates for Tinker."""

from __future__ import annotations

import math
from typing import Any


class CostTracker:
    """Count submitted training tokens; estimate USD only with an explicit rate."""

    def __init__(self, train_rate_usd_per_million: float | None = None) -> None:
        if train_rate_usd_per_million is not None and (
            not math.isfinite(train_rate_usd_per_million)
            or train_rate_usd_per_million < 0
        ):
            raise ValueError("training rate must be finite and non-negative")
        self.train_rate_usd_per_million = train_rate_usd_per_million
        self.total_tokens = 0

    @property
    def estimated_training_cost_usd(self) -> float | None:
        if self.train_rate_usd_per_million is None:
            return None
        return self.total_tokens * self.train_rate_usd_per_million / 1_000_000

    def record_batch(self, batch: list[Any]) -> dict[str, float]:
        """Record the model-input tokens in a completed forward/backward batch."""
        tokens = sum(len(datum.model_input.to_ints()) for datum in batch)
        self.total_tokens += tokens
        metrics = {
            "cost/step_tokens": float(tokens),
            "cost/cumulative_tokens": float(self.total_tokens),
        }
        if self.train_rate_usd_per_million is not None:
            metrics["cost/estimated_step_usd"] = (
                tokens * self.train_rate_usd_per_million / 1_000_000
            )
            metrics["cost/estimated_cumulative_usd"] = (
                self.total_tokens * self.train_rate_usd_per_million / 1_000_000
            )
        return metrics
