import json
from pathlib import Path
from typing import Any

import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl import usage
from lasr_labs_2025_control_project.finetuning.grouped_rl.config import RLRunConfig
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_adapters import (
    StrategyBox,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.run import (
    Recorder,
    run_in_context,
)
from tests.test_hipt_rl import attack_problem
from tests.test_icrl_adapters import OneShotLearner, self_rec_trainer


def icrl_config(output_dir: Path, **overrides: Any) -> RLRunConfig:
    values: dict[str, Any] = {
        "run_name": "icrl",
        "output_dir": str(output_dir),
        "task_mode": "self_recognition",
        "method": "in_context",
        "data": {"attacks_eval": "a.eval", "honeypots_eval": "h.eval"},
        "reward": {
            "overseer_weight": 0,
            "overseer_scale": 0,
            "kl_coef": 0,
            "backdoor_kept_weight": 1,
            "backdoor_suspicion_weight": 1,
        },
        "icrl": {"rounds": 1, "candidates": 1, "search_problems": 2},
        "logging": {"wandb_mode": "disabled"},
    }
    return RLRunConfig.model_validate({**values, **overrides})


def test_in_context_needs_a_frozen_judge_and_the_trusted_monitor(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen judge"):
        icrl_config(tmp_path, training={"train_receiver": True})
    with pytest.raises(ValueError, match="backdoor_suspicion_weight"):
        icrl_config(
            tmp_path,
            reward={"overseer_weight": 0, "overseer_scale": 0, "kl_coef": 0},
        )
    assert icrl_config(tmp_path).method == "in_context"


async def test_driver_logs_every_strategy_reply_and_rewrite(tmp_path: Path) -> None:
    config = icrl_config(tmp_path)
    held_out = attack_problem("13", "validation")
    await run_in_context(
        config,
        self_rec_trainer(),
        StrategyBox(),
        OneShotLearner(),
        Recorder(config, tmp_path),
        usage.Usage(),
        [attack_problem("11"), attack_problem("15"), attack_problem("17")],
        [held_out],
        [(held_out, "# another model\nprint(7)")],
    )

    def rows(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]

    strategies = rows("strategies.jsonl")
    assert [s["strategy"] for s in strategies] == ["", "keep the original helper names"]
    assert all(s["episodes"] == 2 for s in strategies)
    (reply,) = rows("learner.jsonl")
    assert reply["strategies"] == ["keep the original helper names"]
    assert "(no instructions: the baseline)" in reply["prompt"][1]["content"]
    rollouts = rows("rollouts.jsonl")
    assert {r["phase"] for r in rollouts} == {"search", "held_out"}
    assert {r["strategy"] for r in rollouts if r["phase"] == "search"} == {
        "",
        "keep the original helper names",
    }
    result = json.loads((tmp_path / "icrl_result.json").read_text())
    assert set(result["held_out"]) == {"best", "baseline"}
    assert result["held_out"]["best"]["auroc"] == pytest.approx(1.0)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "finished"
    assert len(status["search_problem_ids"]) == 2


class FakeRun:
    url = None

    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict[str, Any]]] = []
        self.saved: list[tuple[str, str | None]] = []
        self.summary: dict[str, Any] = {}

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((step, values))

    def save(self, path: str, base_path: str | None = None, policy: str | None = None):
        self.saved.append((Path(path).name, policy))

    def finish(self) -> None:
        pass


async def test_driver_sends_proposed_and_scored_strategies_to_wandb(
    tmp_path: Path,
) -> None:
    config = icrl_config(tmp_path)
    recorder = Recorder(config, tmp_path)
    run = FakeRun()
    recorder.wandb = run
    held_out = attack_problem("13", "validation")
    await run_in_context(
        config,
        self_rec_trainer(),
        StrategyBox(),
        OneShotLearner(),
        recorder,
        usage.Usage(),
        [attack_problem("11"), attack_problem("15")],
        [held_out],
        [(held_out, "# another model\nprint(7)")],
    )

    def last_table(key: str) -> list[list[Any]]:
        return [values[key] for _, values in run.logged if key in values][-1].data

    # Proposed strategies reach W&B before they are scored.
    assert last_table("icrl/proposed_strategies") == [
        [1, 1, "keep the original helper names"]
    ]
    scored = last_table("icrl/strategies")
    assert [row[2] for row in scored] == ["", "keep the original helper names"]
    assert ("learner.jsonl", "live") in run.saved
    assert ("strategies.jsonl", "live") in run.saved
    assert ("rollouts.jsonl", "live") in run.saved
    assert ("icrl_result.json", "live") in run.saved
    steps = [step for step, _ in run.logged if step is not None]
    assert steps == sorted(steps)  # W&B drops rows whose step goes backwards

    def table(key: str) -> Any:
        return [values[key] for _, values in run.logged if key in values][-1]

    # Each strategy row: its text, total reward, and every reward term.
    strategies = table("icrl/strategies")
    assert strategies.columns[:4] == ["step", "round", "strategy", "reward_mean"]
    terms = [c for c in strategies.columns if c.startswith("reward/")]
    assert set(terms) == {
        "reward/task",
        "reward/self_recognition",
        "reward/backdoor_kept",
        "reward/backdoor_suspicion",
    }
    for row in strategies.data:
        values = dict(zip(strategies.columns, row))
        assert sum(values[c] for c in terms) == pytest.approx(values["reward_mean"])
    held_out = table("icrl/held_out")
    assert [row[0] for row in held_out.data] == ["best", "baseline"]
    assert {"strategy", "reward_mean", "reward/task", "auroc"} <= set(held_out.columns)


class FailingSender:
    async def generate(self, problem: Any, payload: int, num_samples: int) -> list:
        raise ValueError("tokenizer.chat_template is not set")


async def test_a_failed_batch_records_its_cause(tmp_path: Path) -> None:
    from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
        BatchInfrastructureError,
    )

    config = icrl_config(tmp_path)
    trainer = self_rec_trainer()
    trainer.sender = FailingSender()
    held_out = attack_problem("13", "validation")
    with pytest.raises(BatchInfrastructureError):
        await run_in_context(
            config,
            trainer,
            StrategyBox(),
            OneShotLearner(),
            Recorder(config, tmp_path),
            usage.Usage(),
            [attack_problem("11"), attack_problem("15")],
            [held_out],
            [(held_out, "# another model\nprint(7)")],
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "batch_errors.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows] == ["search_failed", "search_failed"]
    assert "chat_template is not set" in rows[0]["errors"][0]
