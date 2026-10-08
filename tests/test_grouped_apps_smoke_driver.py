"""Offline coverage for the smoke driver; never creates a real service or run."""

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from experiments.grouped_apps_smoke.run import (
    Budget,
    BudgetStop,
    Recorder,
    SamplingProxy,
    TrainingProxy,
    numeric_metrics,
)


@pytest.mark.parametrize("limit", [0, 2, 11, float("nan"), float("inf")])
def test_budget_rejects_unapproved_limits(limit: float) -> None:
    with pytest.raises(ValueError):
        Budget(limit, lambda *args: None)


def test_budget_reserves_before_work_and_never_refunds() -> None:
    rows = []
    budget = Budget(10, lambda kind, data: rows.append(data))
    budget.sampling(1000, 4, 1536)
    assert budget.reserved > Decimal(2)
    assert rows[0]["operation"] == "sample"
    before = budget.reserved
    with pytest.raises(BudgetStop):
        budget.reserve("oversize", Decimal(10))
    assert budget.reserved == before
    with pytest.raises(BudgetStop):
        budget.reserve("after_stop", Decimal(0))


def test_ledger_preserves_reservations_across_attempts(tmp_path: Path) -> None:
    ledger = tmp_path / "budget.json"
    first = Budget(10, lambda *args: None, ledger)
    first.reserve("first", Decimal(3))
    second = Budget(10, lambda *args: None, ledger)
    assert second.reserved == Decimal(5)
    second.reserve("second", Decimal(2))
    first.reserve("later_first", Decimal(1))
    assert first.reserved == Decimal(8)
    third = Budget(10, lambda *args: None, ledger)
    assert third.reserved == Decimal(8)
    with pytest.raises(BudgetStop):
        Budget(10, lambda *args: None, ledger)


def test_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "budget.json"
    ledger.write_text("not json")
    with pytest.raises(ValueError):
        Budget(10, lambda *args: None, ledger)
    assert ledger.read_text() == "not json"


def test_receiver_components_use_actual_masked_outputs(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "run")
    datum = SimpleNamespace(
        loss_fn_inputs={
            "weights": SimpleNamespace(data=[0, 1, 1]),
            "target_tokens": SimpleNamespace(data=[2, 10, 9]),
        }
    )
    output = SimpleNamespace(
        metrics={"loss:sum": 0.7},
        loss_fn_outputs=[
            {
                "logprobs": SimpleNamespace(data=[-100, -0.2, -0.5]),
            }
        ],
    )
    recorder.backward("receiver", [datum], output, eos=9)
    metrics = recorder.losses["receiver"]
    assert metrics["receiver/label_nll_mean"] == pytest.approx(0.2)
    assert metrics["receiver/eos_nll_mean"] == pytest.approx(0.5)
    assert metrics["receiver/loss_per_active_token"] == pytest.approx(0.35)


def test_missing_components_are_not_fabricated(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "run")
    datum = SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=[1])})
    output = SimpleNamespace(metrics={"loss:sum": 0.7}, loss_fn_outputs=[{}])
    recorder.backward("receiver", [datum], output, eos=9)
    assert "receiver/label_nll_mean" not in recorder.losses["receiver"]
    with pytest.raises(ValueError):
        numeric_metrics({"loss:sum": float("nan")})


async def test_sampling_budget_stops_before_fake_call() -> None:
    client = SimpleNamespace(sample_async=AsyncMock())
    budget = Budget(3, lambda *args: None)
    proxy = SamplingProxy(client, budget)
    prompt = SimpleNamespace(to_ints=lambda: [1, 2])
    with pytest.raises(BudgetStop):
        await proxy.sample_async(
            prompt=prompt,
            num_samples=4,
            sampling_params=SimpleNamespace(max_tokens=1_000_000),
        )
    client.sample_async.assert_not_called()


async def test_training_reservation_survives_fake_api_failure(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "run")
    budget = Budget(10, recorder.event)
    client = SimpleNamespace(
        forward_backward_async=AsyncMock(side_effect=RuntimeError("failed"))
    )
    proxy = TrainingProxy(client, None, budget, recorder, "sender", None)
    datum = SimpleNamespace(model_input=SimpleNamespace(to_ints=lambda: [1, 2, 3]))
    with pytest.raises(RuntimeError):
        await proxy.forward_backward_async([datum], "importance_sampling")
    assert budget.reserved > Decimal(2)
    assert client.forward_backward_async.await_count == 1
