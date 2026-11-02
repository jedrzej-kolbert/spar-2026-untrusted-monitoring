"""Replay recorded APPS fields through the documented SFT preparation scripts.

Native .eval files are rebuilt with Inspect's writer; recorded code is never
executed. This covers preparation, not generation, scoring or training.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.scorer import Score

from lasr_labs_2025_control_project.prompts.monitor_prompts import (
    MONITOR_USER_PROMPT,
    SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src/lasr_labs_2025_control_project/finetuning/binary_tuning"


def test_sft_steps_prepare_recorded_apps(tmp_path, monkeypatch):
    fixture = json.loads((ROOT / "tests/data/sft_steps/apps_train.json").read_text())
    logs_dir = tmp_path / "evals"
    train_dir = logs_dir / "train"
    train_dir.mkdir(parents=True)

    for recorded_log in fixture["logs"]:
        samples = [
            EvalSample(
                id=record["id"],
                epoch=1,
                input=fixture["problem_statements"][str(record["id"])],
                target="",
                metadata={
                    "main_task_description": fixture["problem_statements"][
                        str(record["id"])
                    ],
                },
                store={
                    "base_solution": record["solution"],
                    "generator_info": {
                        "model_info": {"model": recorded_log["model"]},
                    },
                },
                scores={"apps_scorer": Score(value=record["score"])},
            )
            for record in recorded_log["samples"]
        ]
        log = EvalLog(
            status="success",
            eval=EvalSpec(
                created="2026-10-15T00:00:00Z",
                task="sft_steps_replay",
                dataset=EvalDataset(samples=len(samples)),
                # Model attribution must come from generator_info, not eval.model.
                model="mockllm/model",
                config=EvalConfig(),
            ),
            samples=samples,
        )
        write_eval_log(log, train_dir / f"{recorded_log['name']}.eval")

    # The shell scripts must reuse pytest's environment without downloads/sync.
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", sys.prefix)
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    target_model = fixture["logs"][0]["model"]
    prepare_command = [
        "bash",
        str(SCRIPTS / "transform_raw_to_prepared.sh"),
        str(raw_dir),
        target_model,
    ]
    commands = [
        [
            "bash",
            str(SCRIPTS / "transform_logs_to_raw.sh"),
            str(logs_dir),
            str(raw_dir),
        ],
        [*prepare_command, str(prepared_dir)],
    ]
    for command in commands:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr

    expected_rows = []
    for recorded_log in fixture["logs"]:
        raw = json.loads(
            (raw_dir / "train" / f"{recorded_log['name']}.json").read_text()
        )
        assert [p["apps_problem_id"] for p in raw["problems"]] == recorded_log[
            "included_ids"
        ]
        for problem in raw["problems"]:
            problem_id = problem["apps_problem_id"]
            assert (
                problem["problem_statement"]
                == fixture["problem_statements"][str(problem_id)]
            )
            assert len(problem["solutions"]) == 1
            solution = problem["solutions"][0]
            assert solution["model"] == recorded_log["model"]
            code = (
                solution["solution"].removeprefix("```python\n").removesuffix("\n```")
            )
            original = next(
                record["solution"]
                for record in recorded_log["samples"]
                if record["id"] == problem_id
            )
            assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(original))
            assert "# Subtle special case" not in code
            answer = "YES" if solution["model"] == target_model else "NO"
            expected_rows.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": MONITOR_USER_PROMPT.format(
                                problem_statement=problem["problem_statement"],
                                solution=solution["solution"],
                            ),
                        },
                        {"role": "assistant", "content": f"<answer>{answer}</answer>"},
                    ]
                }
            )

    rows = [
        json.loads(line)
        for line in (prepared_dir / "train/train_messages.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 2
    assert sorted(rows, key=lambda row: row["messages"][-1]["content"]) == sorted(
        expected_rows, key=lambda row: row["messages"][-1]["content"]
    )

    prepare_command[-1] = "tinker/wrong-target-model"
    result = subprocess.run(
        [*prepare_command, str(tmp_path / "wrong_target")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "matches no solution" in result.stderr
