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
