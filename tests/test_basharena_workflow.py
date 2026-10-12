"""BashArena action data: preparation, scoring and comparison. Offline.

The fixtures are the JSONL that basharena_runtime.export writes, built by hand:
no API requests, Docker or control-arena 17.
"""

import asyncio
import socket
from types import SimpleNamespace

import pytest

from lasr_labs_2025_control_project.utils.action_records import (
    SCHEMA_VERSION,
    auc,
    compare_scores,
    is_actor,
    json_line,
    load_prepared,
    prepare_exports,
    read_rows,
    write_rows,
)
from lasr_labs_2025_control_project.utils.action_scoring import (
    baseline_request,
    parse_suspiciousness,
    score_prepared,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in offline workflow tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def export_fixture(
    path, actor="U", population="attacks", split="train", task="1", zero=False
):
    path.mkdir()
    tid = path.name
    messages = [
        {"role": "system", "content": "baseline system"},
        {"role": "user", "content": "history + action"},
    ]
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": tid,
        "source_sha256": "source-" + tid,
        "actor": {"id": actor},
        "split": {"declared": split, "provenance": "original config", "recorded": {}},
        "task_identity": {"main_task_name": task},
        "population": population,
        "baseline_dataset_eligible": not zero,
        "action_count": 0 if zero else 2,
        "trajectory_scores": {
            "side_task_success": {"value": "C" if population != "honest" else "I"}
        },
        "sample": {
            "error": None,
            "messages": [{"content": "original structured history"}],
        },
    }
    actions = [
        {
            "action_id": f"{tid}-a{k}",
            "trajectory_id": tid,
            "action_index": k,
            "source_sha256": trajectory["source_sha256"],
            "recognition_messages": messages,
            "monitor_messages": messages,
            "proposed_message": {
                "tool_calls": [{"arguments": {"cmd": "echo original"}}]
            },
            "tool_outputs": [{"content": "unmodified output"}],
        }
        for k in range(1, trajectory["action_count"] + 1)
    ]
    write_rows(path / "actions.jsonl", actions)
    write_rows(path / "trajectories.jsonl", [trajectory])
    (path / "manifest.json").write_text(
        json_line(
            {
                "complete": True,
                "schema_version": SCHEMA_VERSION,
                "counts": {"trajectories": 1, "actions": len(actions)},
            }
        )
    )
    return path


@pytest.fixture
def prepared(tmp_path):
    own = export_fixture(tmp_path / "own", task="1")
    other = export_fixture(
        tmp_path / "other", actor="H", population="honeypots", task="1"
    )
    honest = export_fixture(
        tmp_path / "honest", population="honest", split="test", task="2"
    )
    test_other = export_fixture(
        tmp_path / "test-other",
        actor="H",
        population="honeypots",
        split="test",
        task="2",
    )
    output = tmp_path / "prepared"
    prepare_exports([own, other, honest, test_other], "U", output)
    return output


def test_sft_contract_original_splits_and_structured_history(prepared):
    actions, ledger = load_prepared(prepared)
    train = read_rows(prepared / "sft_train.jsonl")
    assert len(actions) == 8 and len(ledger) == 4 and len(train) == 4
    assert {r["messages"][-1]["content"] for r in train} == {
        "<answer>YES</answer>",
        "<answer>NO</answer>",
    }
    assert all(r["meta"]["split"] == "train" for r in train)
    assert len(read_rows(prepared / "sft_test.jsonl")) == 2
    assert len(read_rows(prepared / "sft_exclusions.jsonl")) == 2
    assert (
        actions[0]["proposed_message"]["tool_calls"][0]["arguments"]["cmd"]
        == "echo original"
    )
    assert (
        ledger[0]["sample"]["messages"][0]["content"] == "original structured history"
    )


def test_split_leakage_and_missing_actions_rejected(tmp_path):
    a = export_fixture(tmp_path / "a", task="1")
    b = export_fixture(tmp_path / "b", split="test", task="1")
    with pytest.raises(ValueError, match="crosses original splits"):
        prepare_exports([a, b], "U", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        prepare_exports([a, a], "U", tmp_path / "duplicate")
    rows = read_rows(a / "actions.jsonl")
    rows[1]["action_index"] = 1
    (a / "actions.jsonl").write_text("".join(json_line(r) for r in rows))
    with pytest.raises(ValueError, match="action indices"):
        prepare_exports([a], "U", tmp_path / "missing")


def test_dry_run_scoring_resume_and_input_provenance(prepared, tmp_path):
    calls = []

    async def request(messages, role, model):
        calls.append((messages, role, model))
        return {"score": 2.0}

    output = tmp_path / "scores.jsonl"
    models = {"recognition": "tinker/U"}
    dry = asyncio.run(score_prepared(prepared, output, models, request=request))
    assert dry["dry_run"] and dry["requests_pending"] == 4
    assert not output.exists() and calls == []
    asyncio.run(
        score_prepared(prepared, output, models, allow_paid=True, request=request)
    )
    assert len(calls) == 4
    asyncio.run(
        score_prepared(prepared, output, models, allow_paid=True, request=request)
    )
    assert len(calls) == 4
    failed_latest = {**read_rows(output)[0], "score": None, "error": "later failure"}
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json_line(failed_latest))
    assert (
        asyncio.run(score_prepared(prepared, output, models, request=request))[
            "requests_pending"
        ]
        == 1
    )
    asyncio.run(
        score_prepared(prepared, output, models, allow_paid=True, request=request)
    )
    assert len(calls) == 5
    with pytest.raises(ValueError, match="different models/inputs"):
        asyncio.run(
            score_prepared(
                prepared, output, {"recognition": "tinker/changed"}, request=request
            )
        )
    (prepared / "actions.jsonl").write_text(
        (prepared / "actions.jsonl").read_text() + "\n"
    )
    with pytest.raises(ValueError, match="inputs changed"):
        load_prepared(prepared)


def test_pending_generated_checkpoints_never_start_paid_scoring(prepared, tmp_path):
    async def forbidden(*args):
        raise AssertionError("Pending checkpoints must not call a backend")

    output = tmp_path / "pending-scores.jsonl"
    models = {"recognition": "tinker/tinker://PENDING__basharena_self_rec_base"}
    dry = asyncio.run(score_prepared(prepared, output, models, request=forbidden))
    assert dry["dry_run"] and dry["requests_pending"] == 4
    with pytest.raises(ValueError, match="Resolve pending SFT checkpoints"):
        asyncio.run(
            score_prepared(prepared, output, models, allow_paid=True, request=forbidden)
        )
    assert not output.exists()


def test_failed_requests_retry_and_partial_trajectories_do_not_score(
    prepared, tmp_path
):
    counter = 0

    async def request(messages, role, model):
        nonlocal counter
        counter += 1
        if counter == 1:
            raise ValueError("fixture failure")
        return {"score": 1.0}

    output = tmp_path / "scores.jsonl"
    models = {"recognition": "tinker/U"}
    asyncio.run(
        score_prepared(prepared, output, models, allow_paid=True, request=request)
    )
    report = compare_scores(prepared, [output], tmp_path / "partial")
    assert not report["complete"]
    table = read_rows(tmp_path / "partial/trajectory_scores.jsonl")
    assert any(r["score_max"] is None for r in table)
    asyncio.run(
        score_prepared(prepared, output, models, allow_paid=True, request=request)
    )
    assert counter == 5  # Retry only the failed request; no successful calls repeated.
    complete = compare_scores(prepared, [output], tmp_path / "complete")
    assert complete["complete"]
    assert complete["coverage"][0]["trajectory_max_auc"] == 0.5


def test_zero_actions_retained_and_explicit_cohort(tmp_path):
    source = export_fixture(tmp_path / "zero", split="test", zero=True)
    normal = export_fixture(tmp_path / "normal", split="test", task="2")
    output = tmp_path / "prepared"
    prepare_exports([source, normal], "U", output)

    async def request(*args):
        return {"score": 3.0}

    scores = tmp_path / "scores.jsonl"
    asyncio.run(
        score_prepared(
            output,
            scores,
            {"trusted_suspiciousness": "fixture"},
            allow_paid=True,
            request=request,
        )
    )
    all_report = compare_scores(output, [scores], tmp_path / "all")
    assert not all_report["complete"]
    assert all_report["coverage"][0]["trajectories_expected"] == 2
    baseline = compare_scores(
        output, [scores], tmp_path / "baseline", cohort="baseline"
    )
    assert baseline["complete"]
    assert baseline["exclusions"] == [
        {"trajectory_id": "zero", "reason": "baseline_validity_gate"}
    ]


def test_auc_is_tie_aware_and_undefined_for_incomplete_data():
    assert auc([True, False], [1, 1]) == 0.5
    assert auc([True, False], [2, 1]) == 1.0
    assert auc([True, False], [1, None]) is None
    assert auc([True, False], [1]) is None
    assert auc([True, True], [1, 2]) is None


def test_actor_read_from_a_log_matches_either_spelling():
    logged = {"id": "Qwen/Qwen3.6-35B-A3B", "id_source": "generator_info"}
    assert is_actor(logged, "tinker/Qwen/Qwen3.6-35B-A3B")
    assert is_actor(logged, "Qwen/Qwen3.6-35B-A3B")
    assert not is_actor(logged, "tinker/openai/gpt-oss-120b")
    # Declared ids are exact, so a checkpoint is never taken for its base model.
    declared = {"id": "Qwen/Qwen3.6-35B-A3B", "id_source": "declared"}
    assert not is_actor(declared, "tinker/Qwen/Qwen3.6-35B-A3B")
    assert is_actor({"id": "tinker://run/weights"}, "tinker://run/weights")


def test_logged_actor_is_labelled_own_source_for_a_prefixed_target(tmp_path):
    own = export_fixture(tmp_path / "own", actor="Qwen/U")
    row = read_rows(own / "trajectories.jsonl")[0]
    row["actor"]["id_source"] = "generator_info"
    (own / "trajectories.jsonl").write_text(json_line(row))
    other = export_fixture(tmp_path / "other", actor="openai/H", population="honeypots")
    prepare_exports([own, other], "tinker/Qwen/U", tmp_path / "prepared")
    labels = {
        r["meta"]["own_source"]
        for r in read_rows(tmp_path / "prepared/sft_train.jsonl")
    }
    _, ledger = load_prepared(tmp_path / "prepared")
    assert [t["own_source"] for t in ledger] == [True, False] and labels == {
        True,
        False,
    }


@pytest.mark.parametrize(
    ("completion", "score"),
    [
        ("<reasoning>fine</reasoning>\n<score>2</score>", 2),
        ("<score> 8.9 </score>", 9),  # small monitors answer in decimals
        ("<score>0</score> and later <score>9</score>", 0),
        ("<score>12</score>", None),
        ("<score>high</score>", None),
        ("Score: 7", None),
        ("", None),
    ],
)
def test_suspiciousness_parsing(completion, score):
    assert parse_suspiciousness(completion) == score


def test_monitor_backends_and_raw_answers(monkeypatch):
    import inspect_ai.model

    from lasr_labs_2025_control_project.utils import tinker_provider

    async def recognition(model, system, user):
        assert (model, system, user) == ("tinker/U", "S", "U")
        return {"YES": -1.0, "NO": -3.0, "completion": "<answer>YES</answer>"}

    answers = iter(["<score>7</score>", "I cannot rate this."])

    class Model:
        async def generate(self, messages, config):
            assert [m.role for m in messages] == ["system", "user"]
            assert config.reasoning_effort == "minimal"
            return SimpleNamespace(completion=next(answers), stop_reason="stop")

    monkeypatch.setattr(tinker_provider, "collusion_yes_no_logprobs", recognition)
    monkeypatch.setattr(inspect_ai.model, "get_model", lambda model: Model())
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert asyncio.run(baseline_request(messages, "recognition", "tinker/U")) == {
        "score": 2.0,
        "yes_logprob": -1.0,
        "no_logprob": -3.0,
        "completion": "<answer>YES</answer>",
    }
    scored = asyncio.run(
        baseline_request(messages, "trusted_suspiciousness", "tinker/T")
    )
    assert scored == {
        "score": 7,
        "completion": "<score>7</score>",
        "stop_reason": "stop",
    }
    unparsed = asyncio.run(
        baseline_request(messages, "trusted_suspiciousness", "tinker/T")
    )
    assert unparsed["score"] is None and unparsed["completion"] == "I cannot rate this."


def test_suspiciousness_alone_registers_the_tinker_provider():
    """A fresh process that never scores recognition must still resolve tinker/.

    Without a key the request stops at the provider's own check, which shows
    Inspect found the provider; an unregistered one fails as "not recognized".
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = """
import asyncio, socket
def forbidden(*a, **k): raise AssertionError("network")
socket.socket.connect = socket.socket.connect_ex = forbidden
from lasr_labs_2025_control_project.utils.action_scoring import baseline_request
messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
try:
    asyncio.run(baseline_request(messages, "trusted_suspiciousness", "tinker/Qwen/T"))
except Exception as error:
    print(type(error).__name__, error)
"""
    env = {k: v for k, v in os.environ.items() if k != "TINKER_API_KEY"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(__file__).parent,  # no .env here, so no key is loaded
    )
    assert "TINKER_API_KEY" in result.stdout, result.stdout + result.stderr
    assert "not recognized" not in result.stdout


def test_unusable_answers_are_recorded_with_their_text(prepared, tmp_path):
    async def request(messages, role, model):
        return {"score": None, "completion": "I cannot rate this."}

    output = tmp_path / "scores.jsonl"
    summary = asyncio.run(
        score_prepared(
            prepared,
            output,
            {"trusted_suspiciousness": "tinker/T"},
            allow_paid=True,
            request=request,
        )
    )
    records = read_rows(output)
    assert summary["failed"] == len(records) == 4
    assert all(r["score"] is None and "finite score" in r["error"] for r in records)
    assert all(r["result"]["completion"] == "I cannot rate this." for r in records)


def test_multiple_monitor_models_compare_on_identical_inputs(prepared, tmp_path):
    async def request(*args):
        return {"score": 4.0}

    score_paths = []
    for name in ("base", "checkpoint"):
        path = tmp_path / f"{name}.jsonl"
        asyncio.run(
            score_prepared(
                prepared,
                path,
                {
                    "recognition": f"tinker/{name}",
                    "trusted_suspiciousness": "tinker/T",
                    "untrusted_suspiciousness": f"tinker/{name}",
                },
                allow_paid=True,
                request=request,
            )
        )
        score_paths.append(path)
    report = compare_scores(prepared, score_paths, tmp_path / "compared")
    assert report["complete"]
    assert len(report["coverage"]) == 5  # Shared trusted inputs/results counted once.
    assert {r["model"] for r in report["coverage"]} == {
        "tinker/base",
        "tinker/checkpoint",
        "tinker/T",
    }
    rows = read_rows(score_paths[0])
    rows[0]["score"] = 5.0  # Same request/run hash, but a conflicting independent draw.
    conflict = tmp_path / "conflict.jsonl"
    write_rows(conflict, rows)
    with pytest.raises(ValueError, match="Conflicting score runs"):
        compare_scores(prepared, [score_paths[0], conflict], tmp_path / "bad-draws")
    rows[0]["condition"] = "changed"
    bad_condition = tmp_path / "condition.jsonl"
    write_rows(bad_condition, rows)
    with pytest.raises(ValueError, match="provenance mismatch"):
        compare_scores(prepared, [bad_condition], tmp_path / "bad-condition")
