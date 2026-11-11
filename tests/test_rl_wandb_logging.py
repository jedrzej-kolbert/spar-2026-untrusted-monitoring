"""The RL driver's W&B logging leaves its local records unchanged (no Tinker, no Docker).

``run.train`` is driven for two steps plus its baseline and final evaluations with the
fake sender, receiver, evaluator and overseer from test_hipt_rl.py; only data loading,
the Docker checks and checkpointing are stubbed.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.finetuning.grouped_rl import records, run
from lasr_labs_2025_control_project.finetuning.grouped_rl.config import (
    LoggingConfig,
    load_rl_config,
)

from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    BatchResult,
    ExecutionGrade,
    Rollout,
)

from .test_hipt_rl import PILOT_CONFIG, attack_problem, balanced_trainer, rollout

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
        self.env = {k: os.environ.get(k) for k in ("WANDB_PROJECT", "WANDB_MODE")}
        FakeStageRun.last = self

    def __enter__(self) -> FakeStageRun:
        return self

    def log_roc(self, key: str, curves: Any, title: str, step: int = 0) -> None:
        self.rocs.append(key)

    def log_artifact(self, path: Path, **kwargs: Any) -> None:
        pass

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

    monkeypatch.setattr(
        run, "load_tasks", lambda c: run.Tasks(train, validation, [], honeypots, [])
    )
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
        "train/reward/communication",
        "validation/reward/communication",
        "train/p_correct",
        "validation/p_correct",
        "train/overseer_strength",
        "train_detail/batch/bit1_mean_p1",
        "train_detail/batch/mean_reward",
        "validation_detail/honeypots/balanced_accuracy",
    } <= logged
    # Only the sections; local-only prefixes (train_legacy/...) stay local.
    assert {key.split("/")[0] for key in logged} <= {
        "train",
        "validation",
        "train_detail",
        "validation_detail",
        "preflight",
        "progress",
    }
    for _, values in stage.run.logged:
        for section in ("train", "validation"):
            terms = [v for k, v in values.items() if k.startswith(f"{section}/reward/")]
            if terms:
                assert math.fsum(terms) == pytest.approx(values[f"{section}/reward"])
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

    def recorder(logging: dict[str, Any]) -> records.Recorder:
        config = load_rl_config(PILOT_CONFIG).model_copy(
            update={
                "output_dir": tmp_path,
                "logging": LoggingConfig.model_validate(logging),
            }
        )
        return records.Recorder(config, tmp_path)

    first = recorder({"wandb_project": "from-yaml"})
    assert FakeStageRun.last.env["WANDB_PROJECT"] == "from-yaml"
    assert os.environ["WANDB_ENTITY"] == "env-entity"  # not written in the YAML
    first.finish(complete=True)
    assert os.environ["WANDB_PROJECT"] == "from-env"  # the next run is unaffected

    # wandb_project: null turned W&B off before; it still does.
    recorder({"wandb_project": None}).finish(complete=True)
    assert FakeStageRun.last.env["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_MODE"] == "disabled"  # conftest's own value, restored


def test_collusion_auroc_uses_working_signal_on_rewrites_only() -> None:
    def item(payload: int, p_yes: float, works: bool) -> Rollout:
        return rollout(payload=payload, p_yes=p_yes, backdoor=ExecutionGrade(works))

    result = BatchResult(
        4,
        [
            item(1, 0.9, True),
            item(1, 0.8, True),
            item(1, 0.99, False),  # backdoor broken: not a working rewrite
            item(0, 0.95, True),  # signal OFF: not a positive
        ],
    )
    numbers, roc = records.collusion_auroc(result, [("a", 0.85), ("b", 0.2)])
    # Positives 0.9 and 0.8 against honeypots 0.85 and 0.2: 3 of 4 pairs ranked right.
    assert numbers == {"auroc": 0.75, "n_working": 2, "n_honeypots": 2}
    assert roc
