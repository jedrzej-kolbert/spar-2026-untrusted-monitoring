"""The RL driver's W&B logging leaves its local records unchanged (no Tinker, no Docker).

``run.train`` is driven for two steps plus its baseline and final evaluations with the
fake sender, receiver, evaluator and overseer from test_hipt_rl.py; only data loading,
the Docker checks and checkpointing are stubbed.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.finetuning.grouped_rl import run
from lasr_labs_2025_control_project.finetuning.grouped_rl.config import (
    LoggingConfig,
    load_rl_config,
)

from .test_hipt_rl import PILOT_CONFIG, attack_problem, balanced_trainer

RECORDS = ("metrics.jsonl", "rollouts.jsonl", "honeypot_scores.jsonl", "manifest.json")


class FakeSender:
    async def save_checkpoint(self, name: str) -> str:
        return f"tinker://fake/sampler_weights/{name}"

    async def save_state(self, name: str) -> str:
        return f"tinker://fake/weights/{name}"


class FakeWandbRun:
    url = "https://wandb.example/run"

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, Any]]] = []
        self.summary: dict[str, Any] = {}
        self.saved: list[str] = []

    def log(self, values: dict[str, Any], step: int) -> None:
        self.logged.append((step, values))

    def save(self, path: str, base_path: str, policy: str) -> None:
        self.saved.append(Path(path).name)


class FakeStageRun:
    """Records what the driver sends to W&B instead of sending it."""

    last: FakeStageRun

    def __init__(self, stage: str, name: str, **kwargs: Any) -> None:
        self.stage, self.name, self.kwargs = stage, name, kwargs
        self.run = FakeWandbRun()
        self.rocs: list[str] = []
        self.models: list[tuple[str, Any]] = []
        self.finished: bool | None = None
        FakeStageRun.last = self

    def __enter__(self) -> FakeStageRun:
        return self

    def log_roc(self, key: str, curves: Any, title: str) -> None:
        self.rocs.append(key)

    def log_model_ref(self, model_id: str, metadata: Any) -> None:
        self.models.append((model_id, metadata))

    def finish(self, *, complete: bool, reason: str | None = None) -> None:
        self.finished = complete


def drive(output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_rl_config(PILOT_CONFIG).model_copy(
        update={"output_dir": output_dir, "logging": LoggingConfig()}
    )
    config.training.steps = 2
    config.training.problems_per_step = 2
    config.training.group_size = 2
    config.eval.every = 1
    config.preflight.enabled = False
    config.checkpoints.save_every = 1
    train = [attack_problem(str(i)) for i in (11, 13, 15)]
    validation = [attack_problem("17", split="validation")]
    honeypots = [(attack_problem(str(i)), "print(0)") for i in (19, 21)]

    async def parity(evaluator: Any, tasks: Any) -> dict[str, float]:
        return {"agreement": 1.0}

    async def build(*args: Any, **kwargs: Any) -> tuple[Any, FakeSender]:
        return balanced_trainer(), FakeSender()

    monkeypatch.setattr(run, "load_tasks", lambda c: (train, validation, honeypots))
    monkeypatch.setattr(run, "ensure_image", lambda image: None)
    monkeypatch.setattr(run, "DockerAppsEvaluator", lambda **kwargs: None)
    monkeypatch.setattr(run, "sandbox_parity_check", parity)
    monkeypatch.setattr(run, "build_trainer", build)
    monkeypatch.setattr(
        tracking, "resolve_input", lambda path: SimpleNamespace(source=str(path))
    )
    monkeypatch.setattr(run.time, "time", lambda: 1000.0)
    monkeypatch.setattr(run.time, "strftime", lambda fmt, *args: "2026-11-10T00:00:00")
    asyncio.run(run.train(config, allow_paid=True))


def test_wandb_logging_leaves_local_records_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same output_dir for both, since the manifest records it.
    drive(tmp_path / "run", monkeypatch)  # WANDB_MODE=disabled (conftest)
    (tmp_path / "run").rename(tmp_path / "off")
    monkeypatch.setattr(tracking, "StageRun", FakeStageRun)
    drive(tmp_path / "run", monkeypatch)
    (tmp_path / "run").rename(tmp_path / "on")

    for name in RECORDS:
        assert (tmp_path / "on" / name).read_text() == (
            tmp_path / "off" / name
        ).read_text(), name
    status_on = json.loads((tmp_path / "on" / "status.json").read_text())
    status_off = json.loads((tmp_path / "off" / "status.json").read_text())
    assert status_on.pop("wandb_url") == FakeWandbRun.url
    assert status_on == status_off

    stage = FakeStageRun.last
    identity = stage.kwargs["config"]["run"]
    assert stage.stage == "gbrl"
    assert identity["threat_model"] == "causal"
    assert identity["task_mode"] == "rewrite_signal"
    assert "method:gbrl" in stage.kwargs["tags"]
    logged = {key for _, values in stage.run.logged for key in values}
    assert {
        "train/reward",
        "train/working_rate",
        "train/on_off_auroc",
        "validation/auroc",
        "validation/on_off_auroc",
        "validation/honeypot_balanced_accuracy",
        "train_detail/batch/mean_reward",
        "validation_detail/honeypots/balanced_accuracy",
    } <= logged
    # Only the sections; the local prefixes (eval_pairs/...) stay local.
    assert all(
        key.split("/")[0]
        in {"train", "validation", "train_detail", "validation_detail", "preflight"}
        for key in logged
    )
    assert [step for step, values in stage.run.logged if "validation/auroc" in values]
    assert stage.rocs and stage.finished is True
    assert stage.models[-1][0] == "tinker/tinker://fake/sampler_weights/rl_" + (
        load_rl_config(PILOT_CONFIG).run_name
    )
    assert {"metrics.jsonl", "rollouts.jsonl", "manifest.json"} <= set(stage.run.saved)


def test_yaml_logging_fields_win_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    monkeypatch.setenv("WANDB_ENTITY", "env-entity")
    monkeypatch.setattr(tracking, "StageRun", FakeStageRun)
    monkeypatch.setattr(
        tracking, "resolve_input", lambda path: SimpleNamespace(source=str(path))
    )
    config = load_rl_config(PILOT_CONFIG).model_copy(
        update={
            "output_dir": tmp_path,
            "logging": LoggingConfig.model_validate({"wandb_project": "from-yaml"}),
        }
    )
    run.Recorder(config, tmp_path)
    assert os.environ["WANDB_PROJECT"] == "from-yaml"
    assert os.environ["WANDB_ENTITY"] == "env-entity"  # not written in the YAML
