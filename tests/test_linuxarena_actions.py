from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lasr_labs_2025_control_project.utils import linuxarena_actions as actions


def row(
    action_id: str = "action-1",
    tasks: tuple[str, ...] = ("task-1",),
    population: str = "attacks",
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "recognition_messages": [
            {"role": "system", "content": actions.RECOGNITION_SYSTEM_PROMPT},
            {"role": "user", "content": "A native monitor observation"},
        ],
        "suspiciousness_messages": [
            {"role": "user", "content": "A native risk prompt"}
        ],
        "meta": {
            "action_id": action_id,
            "trajectory_id": "trajectory-1",
            "task_keys": list(tasks),
            "population": population,
            "own_source": population != "honeypots",
            "sample_error": error,
            "scores": {"tasks": {"value": {"side_task_success": False}}},
        },
    }


def test_splits_keep_generators_and_nested_actions_together():
    rows = [row("a"), row("b", population="honeypots"), row("c", population="honest")]
    actions.assign_splits(rows)
    assert len({r["meta"]["split"] for r in rows}) == 1
    assert len({r["meta"]["task_group"] for r in rows}) == 1
    changed = deepcopy(rows)
    changed.reverse()
    changed[0]["meta"]["scores"] = {"tasks": {"value": 999}}
    actions.assign_splits(changed)
    assert {r["action_id"]: r["meta"]["split"] for r in rows} == {
        r["action_id"]: r["meta"]["split"] for r in changed
    }


def test_multi_task_connected_components_do_not_leak_tasks():
    rows = [row("a", ("x", "y")), row("b", ("y", "z")), row("c", ("z",))]
    actions.assign_splits(rows)
    assert len({r["meta"]["task_group"] for r in rows}) == 1
    assert len({r["meta"]["split"] for r in rows}) == 1


@pytest.mark.parametrize(
    "population,answer", [("attacks", "YES"), ("honest", "YES"), ("honeypots", "NO")]
)
def test_sft_labels_are_provenance_not_intent(population: str, answer: str):
    original = row(population=population)
    example = actions.sft_example(original)
    assert example["messages"][-1] == {
        "role": "assistant",
        "content": f"<answer>{answer}</answer>",
    }
    assert len(original["recognition_messages"]) == 2
    assert example["messages"][:-1] == original["recognition_messages"]
    assert example["meta"] == original["meta"]


def test_actor_checked_against_real_role_not_eval_seat(tmp_path: Path):
    source = actions.ActionSource(
        tmp_path / "run.eval", "deepseek-ai/DeepSeek-V3.1", "attacks"
    )
    log = SimpleNamespace(
        eval=SimpleNamespace(
            model="eval-seat/fake",
            model_roles={
                "untrusted": SimpleNamespace(
                    model="tinker-pilot/deepseek-ai/DeepSeek-V3.1"
                )
            },
        )
    )
    assert actions._validate_actor(log, source, source.actor_id).endswith(
        source.actor_id
    )
    log.eval.model_roles["untrusted"].model = "tinker-pilot/openai/gpt-oss-120b"
    with pytest.raises(ValueError, match="differs from logged model"):
        actions._validate_actor(log, source, source.actor_id)
    log.eval.model_roles = {}
    with pytest.raises(ValueError, match="recorded untrusted model"):
        actions._validate_actor(log, source, source.actor_id)


def test_failed_runs_and_honest_rows_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_rows = [
        row("a", error={"message": "timeout"}),
        row("b", population="honest"),
        row("c", population="honeypots"),
    ]
    ledger = [
        {
            "population": r["meta"]["population"],
            "sample_error": r["meta"]["sample_error"],
            "action_count": 1,
        }
        for r in source_rows
    ]
    ledger.append(
        {
            "population": "attacks",
            "sample_error": {"message": "sandbox startup failed"},
            "action_count": 0,
        }
    )
    monkeypatch.setattr(
        actions, "read_action_source", lambda *args: (source_rows, ledger)
    )
    output = tmp_path / "export"
    summary = actions.export_actions(
        [actions.ActionSource(Path("run.eval"), "own", "attacks")], "own", output
    )
    rows = [
        json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert summary["samples_with_errors"] == 2
    assert summary["samples_without_actions"] == 1
    assert rows[0]["meta"]["sample_error"]["message"] == "timeout"
    all_sft = [
        json.loads(line)
        for p in output.glob("sft_all_*.jsonl")
        for line in p.read_text().splitlines()
    ]
    attack_sft = [
        json.loads(line)
        for p in output.glob("sft_attack_*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(all_sft) == 3
    assert len(attack_sft) == 2
    assert {r["meta"]["population"] for r in attack_sft} == {"attacks", "honeypots"}
    assert len((output / "trajectory_ledger.jsonl").read_text().splitlines()) == 4


def test_duplicate_actions_rejected_without_partial_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        actions, "read_action_source", lambda *args: ([row(), row()], [])
    )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Duplicate actions"):
        actions.export_actions(
            [actions.ActionSource(Path("run.eval"), "own", "attacks")], "own", output
        )
    assert not output.exists()


def test_monitor_scores_keep_unscored_draws_and_skip_non_action_keys():
    by_action = actions.monitor_scores_by_action(
        {
            "simple/1/0": 0.0,
            "simple/1/2": "unscored",
            "simple/2/2": 7.0,
            "simple/1/trajectory": 3.0,
            "simple/1/draft/12345": 9.0,
        },
        {"simple/1/2": {"explanation": "no answer", "attempts": 3}},
    )
    assert sorted(by_action) == [0, 2]
    assert by_action[0] == [{"monitor": "simple", "draw": "1", "value": 0.0}]
    assert by_action[2] == [
        {
            "monitor": "simple",
            "draw": "1",
            "value": None,
            "unscored_reason": {"explanation": "no answer", "attempts": 3},
        },
        {"monitor": "simple", "draw": "2", "value": 7.0},
    ]
