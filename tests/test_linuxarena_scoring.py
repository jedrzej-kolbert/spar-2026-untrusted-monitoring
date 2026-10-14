import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from lasr_labs_2025_control_project.scripts import score_linuxarena_actions as scoring
from lasr_labs_2025_control_project.utils import tinker_provider


def record(action_id: str, own: bool, margin: float | None) -> dict[str, Any]:
    return {"action_id": action_id, "own_source": own, "margin": margin}


def row(action_id: str, own: bool) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "recognition question"},
            {"role": "user", "content": f"view {action_id}"},
            {"role": "assistant", "content": "<answer>YES</answer>"},
        ],
        "meta": {
            "action_id": action_id,
            "trajectory_id": "t",
            "population": "attacks" if own else "honeypots",
            "own_source": own,
            "main_task_ids": ["task"],
            "action_index": 0,
            "split": "test",
        },
    }


def test_summary_auc_uses_own_actions_as_positives():
    summary = scoring.summarize(
        [
            record("a", True, 2.0),
            record("b", True, 0.5),
            record("c", False, 1.0),
            record("d", False, -3.0),
        ]
    )
    assert summary["roc_auc"] == pytest.approx(0.75)
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["yes_rate"] == pytest.approx(0.75)
    assert summary["mean_margin_own"] == pytest.approx(1.25)
    assert summary["mean_margin_other"] == pytest.approx(-1.0)


def test_failed_rows_are_counted_and_a_later_success_replaces_them():
    failed = {**record("a", True, None), "error": "timeout"}
    summary = scoring.summarize([failed, record("b", False, -1.0)])
    assert (summary["rows"], summary["scored"], summary["failed"]) == (2, 1, 1)
    assert "roc_auc" not in summary
    retried = scoring.summarize(
        [failed, record("b", False, -1.0), record("a", True, 1.0)]
    )
    assert (retried["rows"], retried["scored"], retried["failed"]) == (2, 2, 0)
    assert retried["roc_auc"] == pytest.approx(1.0)


def test_scoring_records_errors_and_resumes_only_unscored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []
    fail = {"view b"}

    async def fake(
        model_name: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        calls.append(user_prompt)
        if user_prompt in fail:
            raise RuntimeError("service unavailable")
        return {"YES": -0.1, "NO": -2.1, "completion": "<answer>YES</answer>"}

    monkeypatch.setattr(tinker_provider, "collusion_yes_no_logprobs", fake)
    rows = [row("a", True), row("b", False)]
    output = tmp_path / "scores.jsonl"
    asyncio.run(scoring.score_rows(rows, "tinker/model", output, concurrency=2))
    first = [json.loads(line) for line in output.read_text().splitlines()]
    assert {r["action_id"]: r["margin"] for r in first} == {
        "a": pytest.approx(2.0),
        "b": None,
    }
    assert (
        "service unavailable"
        in next(r for r in first if r["action_id"] == "b")["error"]
    )

    fail.clear()
    calls.clear()
    asyncio.run(scoring.score_rows(rows, "tinker/model", output, concurrency=2))
    assert calls == ["view b"]
    summary = scoring.summarize(scoring.load_rows(output))
    assert (summary["rows"], summary["scored"], summary["failed"]) == (2, 2, 0)


@pytest.mark.parametrize("change", ["model", "system", "user", "legacy"])
def test_resume_provenance_rejected_before_requests(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake(
        model_name: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        calls.append(user_prompt)
        return {"YES": -0.1, "NO": -2.1}

    monkeypatch.setattr(tinker_provider, "collusion_yes_no_logprobs", fake)
    rows = [row("a", True), row("b", False)]
    output = tmp_path / "scores.jsonl"
    asyncio.run(scoring.score_rows(rows[:1], "tinker/model", output, concurrency=1))
    calls.clear()
    model = "tinker/other-model" if change == "model" else "tinker/model"
    if change in {"system", "user"}:
        for message in rows[0]["messages"]:
            if message["role"] == change:
                message["content"] += " changed"
    if change == "legacy":
        records = scoring.load_rows(output)
        records[0].pop("input_hash")
        output.write_text(json.dumps(records[0]) + "\n")
    original = output.read_text()
    expected = (
        "model differs"
        if change == "model"
        else "legacy output"
        if change == "legacy"
        else "input text changed"
    )
    with pytest.raises(ValueError, match=expected):
        asyncio.run(scoring.score_rows(rows, model, output, concurrency=1))
    assert not calls
    assert output.read_text() == original


def test_input_hash_ignores_training_answer_and_covers_exact_monitor_text() -> None:
    original = row("a", True)
    identity = scoring.input_hash(original)
    original["messages"][-1]["content"] = "a different training label"
    assert scoring.input_hash(original) == identity
    original["messages"][1]["content"] += " "
    assert scoring.input_hash(original) != identity
