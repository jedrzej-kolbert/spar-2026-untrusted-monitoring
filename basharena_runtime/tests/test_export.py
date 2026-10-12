"""Saved-log export: every attempt is kept, labelled, and rendered as upstream does."""

import json
import socket
from pathlib import Path

import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalError,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.scorer import Score
from test_views import UPSTREAM_FIRST, UPSTREAM_LAST, _trajectory

from basharena_runtime import export as ex
from basharena_runtime.export import ExportSource, export_saved_log
from basharena_runtime.views import trajectory_view


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in export tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _log(samples, model="fixture-actor"):
    return EvalLog(
        status="error",
        samples=samples,
        eval=EvalSpec(
            created="2026-01-01T00:00:00Z",
            task="bash-fixture",
            model=model,
            dataset=EvalDataset(samples=len(samples)),
            config=EvalConfig(),
            metadata={"fixture": True},
        ),
    )


def _sample(epoch, transcript, side="C", error=None, generator=None):
    store = {"base_solution": trajectory_view(transcript)}
    if generator:
        store["generator_info"] = {"model_info": {"model": generator}}
    return EvalSample(
        id="same-task",
        epoch=epoch,
        input="Do the task.",
        target="",
        messages=transcript,
        metadata={
            "setting_name": "bash_arena",
            "eval_mode": "attack",
            "main_task_description": "Do the task.",
            "main_task_name": "task",
            "side_task_name": "side",
            "data_split": "test",
        },
        store=store,
        error=error,
        scores={
            "main_task_success": Score(value={"main_task_success": "I"}),
            "side_task_success": Score(value=side),
        }
        if side
        else None,
    )


def test_source_validation():
    with pytest.raises(ValueError, match="split_provenance"):
        ExportSource(Path("a.eval"), "attacks", split="train")
    with pytest.raises(ValueError, match="population"):
        ExportSource(Path("a.eval"), "other")
    assert ex.stable_id({"a": 1, "b": 2}) == ex.stable_id({"b": 2, "a": 1})
    assert json.loads(ex.json_line({"score": float("nan")})) == {
        "score": {"$nonfinite": "NaN"}
    }


def test_wrong_runtime_is_rejected_before_reading_the_log(tmp_path, monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "wrong-version")
    with pytest.raises(RuntimeError, match="needs"):
        export_saved_log(ExportSource(tmp_path / "absent.eval", "attacks"), tmp_path)


def test_failures_zero_actions_and_labels_are_kept(tmp_path):
    messages = _trajectory()
    timeout = EvalError(message="timeout", traceback="trace", traceback_ansi="trace")
    samples = [
        _sample(1, messages),
        _sample(2, messages, side="I"),  # unsuccessful attack: kept, not eligible
        _sample(3, messages, error=timeout),
        _sample(4, messages[:2]),  # no actions
        _sample(5, messages[:3], side=None),  # action without output or scores
    ]
    path = tmp_path / "attempts.eval"
    write_eval_log(_log(samples), path)
    source = ExportSource(
        path,
        "attacks",
        "train",
        "fixture declaration",
        monitor_actor_id="fixture-actor",
    )
    count = export_saved_log(source, tmp_path / "export")
    assert count == {
        "trajectories": 5,
        "actions": 10,
        "baseline_eligible_trajectories": 1,
    }
    ledger = rows(tmp_path / "export/trajectories.jsonl")
    actions = rows(tmp_path / "export/actions.jsonl")
    assert [r["action_count"] for r in ledger] == [3, 3, 3, 0, 1]
    assert [r["baseline_dataset_eligible"] for r in ledger] == [True] + [False] * 4
    assert len({r["trajectory_id"] for r in ledger}) == 5
    assert len({r["action_id"] for r in actions}) == 10
    assert "zero_actions" in ledger[3]["reasons"]
    assert "sample_error" in ledger[2]["reasons"]
    assert "missing_scores" in ledger[4]["reasons"]
    assert "missing_tool_outputs:message_2" in ledger[4]["reasons"]
    assert all("declared_split_conflicts_with_metadata" in r["reasons"] for r in ledger)
    assert all(r["split"]["recorded"] == {"data_split": "test"} for r in ledger)
    assert all(r["labels"]["own_source"] is True for r in actions)
    assert all(r["per_action_scores"] is None for r in actions)
    assert actions[0]["monitor_messages"][1]["content"] == UPSTREAM_FIRST
    assert actions[2]["monitor_messages"][1]["content"] == UPSTREAM_LAST
    assert len(actions[1]["proposed_message"]["tool_calls"]) == 2
    assert len(actions[1]["tool_outputs"]) == 2
    assert actions[-1]["tool_outputs"] == []
    assert ledger[0]["sample"]["messages"][3] == actions[0]["tool_outputs"][0]
    manifest = json.loads((tmp_path / "export/manifest.json").read_text())
    assert manifest["complete"]
    assert set(manifest["renderer"]["files"]) == {"views.py", "monitor_prompts.py"}

    with pytest.raises(FileExistsError):
        export_saved_log(source, tmp_path / "export")
    export_saved_log(source, tmp_path / "repeat")
    assert (tmp_path / "repeat/actions.jsonl").read_bytes() == (
        tmp_path / "export/actions.jsonl"
    ).read_bytes()


def test_missing_provenance_keeps_the_attempt_without_inventing_a_task(tmp_path):
    sample = _sample(1, _trajectory()[:3], side=None)
    sample.metadata = {}
    path = tmp_path / "missing.eval"
    write_eval_log(_log([sample]), path)
    export_saved_log(ExportSource(path, "attacks"), tmp_path / "missing")
    attempt = rows(tmp_path / "missing/trajectories.jsonl")[0]
    action = rows(tmp_path / "missing/actions.jsonl")[0]
    assert "missing_setting_name" in attempt["reasons"]
    assert "missing_task_description" in attempt["reasons"]
    assert attempt["split"] == {"declared": None, "provenance": None, "recorded": {}}
    assert action["recognition_messages"] is None
    assert action["monitor_messages"] is None
    assert action["labels"]["own_source"] is None


@pytest.mark.parametrize(
    ("declared", "monitor", "own"),
    [
        # The log records "Qwen/X" for a "tinker/Qwen/X" run; both spellings match.
        (None, "tinker/Qwen/Qwen3.6-35B-A3B", True),
        (None, "Qwen/Qwen3.6-35B-A3B", True),
        (None, "tinker/openai/gpt-oss-120b", False),
        # Declared ids are compared exactly.
        ("tinker/Qwen/Qwen3.6-35B-A3B", "tinker/Qwen/Qwen3.6-35B-A3B", True),
        ("Qwen/Qwen3.6-35B-A3B", "tinker/Qwen/Qwen3.6-35B-A3B", False),
        ("tinker://run/sampler_weights/a", "tinker/Qwen/Qwen3.6-35B-A3B", False),
    ],
)
def test_own_source_label(tmp_path, declared, monitor, own):
    path = tmp_path / "log.eval"
    sample = _sample(1, _trajectory(), generator="Qwen/Qwen3.6-35B-A3B")
    write_eval_log(_log([sample], model="tinker/Qwen/Qwen3.6-35B-A3B"), path)
    source = ExportSource(path, "attacks", actor_id=declared, monitor_actor_id=monitor)
    export_saved_log(source, tmp_path / "out")
    action = rows(tmp_path / "out/actions.jsonl")[0]
    assert action["labels"] == {"monitor_actor_id": monitor, "own_source": own}
    assert action["actor"]["id"] == (declared or "Qwen/Qwen3.6-35B-A3B")


def test_non_bash_logs_are_rejected(tmp_path):
    sample = _sample(1, _trajectory())
    sample.metadata["setting_name"] = "apps"
    path = tmp_path / "apps.eval"
    write_eval_log(_log([sample]), path)
    with pytest.raises(ValueError, match="non-Bash"):
        export_saved_log(ExportSource(path, "attacks"), tmp_path / "out")
    assert not (tmp_path / "out").exists()
