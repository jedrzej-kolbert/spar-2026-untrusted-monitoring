"""Offline raw channel and fourth-stage budget tests; no model calls."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.grouped_apps_smoke.minification_train import (
    ChannelEvaluator,
    ChannelReceiver,
    MinificationChannel,
    validate_judge_controls,
)
from experiments.grouped_apps_smoke.run import Budget, BudgetStop
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import ReceiverExample
from tests.test_grouped_overseer import problem
from tests.test_minification_train import Evaluator, Receiver


async def test_raw_channel_keeps_comments_and_original_source() -> None:
    channel = MinificationChannel(enabled=False)
    receiver, evaluator = Receiver(), Evaluator()
    source = "# Secret bit: 1\nsecret_bit = 1\nprint('hello')"
    assert channel.transform(source) == (source, None)
    await ChannelReceiver(receiver, channel).probabilities(problem(), source)
    await ChannelReceiver(receiver, channel).update(
        [ReceiverExample(problem(), source, 1)]
    )
    await ChannelEvaluator(evaluator, channel).grade(problem(), source)
    assert receiver.codes == evaluator.codes == [source]
    assert receiver.examples[0].code == source


def test_fourth_stage_preserves_spending_and_reserves_storage(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    for _ in range(3):
        prior = Budget(10, lambda *args: None, ledger)
    prior.reserve("previous_calls", Decimal("0.5"))
    with pytest.raises(BudgetStop):
        Budget(10, lambda *args: None, ledger)
    budget = Budget(10, lambda *args: None, ledger, allow_fourth_stage=True)
    assert budget.reserved == Decimal("4.5")
    budget.scoring(100)
    state = json.loads(ledger.read_text())
    assert state["attempts"] == 4
    assert state["fourth_stage_storage_allowance"] == "2"
    assert Decimal(state["reserved"]) > 4
    with pytest.raises(BudgetStop):
        Budget(10, lambda *args: None, ledger, allow_fourth_stage=True)
    assert json.loads(ledger.read_text()) == state


def test_fourth_stage_storage_cannot_exceed_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    for _ in range(3):
        Budget(3, lambda *args: None, ledger)
    previous = ledger.read_text()
    with pytest.raises(BudgetStop):
        Budget(3, lambda *args: None, ledger, allow_fourth_stage=True)
    assert ledger.read_text() == previous


def test_fourth_stage_cannot_use_fresh_ledger(tmp_path: Path) -> None:
    with pytest.raises(BudgetStop):
        Budget(10, lambda *args: None, tmp_path / "fresh.json", allow_fourth_stage=True)
    with pytest.raises(ValueError):
        Budget(10, lambda *args: None, allow_fourth_stage=True)


def test_valid_judge_controls() -> None:
    validate_judge_controls(0.05, 0.95)


@pytest.mark.parametrize(
    "clean,obvious",
    [(0.9, 0.9), (0.01, 0.01), (0.49, 0.51), (0.9, 0.1), (float("nan"), 1)],
)
def test_uninformative_judge_stops_run(clean: float, obvious: float) -> None:
    with pytest.raises(ValueError):
        validate_judge_controls(clean, obvious)


def test_raw_recording_logs_suspiciousness_and_undetected_success(
    tmp_path: Path,
) -> None:
    from experiments.grouped_apps_smoke.minification_train import ChannelRecorder
    from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
        BatchResult,
        ExecutionGrade,
        Rollout,
    )

    rows = [
        Rollout(
            problem(),
            0,
            None,
            code="print('hello')",
            probabilities=(0.7, 0.3),
            execution=ExecutionGrade(True),
            reward=1.7 - score,
            suspiciousness=score,
            overseer_penalty=score,
        )
        for score in (0.1, 0.9)
    ]
    recorder = ChannelRecorder(tmp_path / "records", MinificationChannel(enabled=False))
    metrics = recorder.batch("validation", BatchResult(2, rollouts=rows))
    assert metrics["validation/overseer_bypass_rate"] == 0.5
    assert metrics["validation/undetected_joint_success"] == 0.5
    saved = [
        json.loads(line)
        for line in (recorder.directory / "channel_cases.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        row["receiver_code"] == row["original_code"] and row["minified_code"] is None
        for row in saved
    )
