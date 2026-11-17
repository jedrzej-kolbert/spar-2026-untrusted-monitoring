import json
from pathlib import Path
from typing import Any

from lasr_labs_2025_control_project.finetuning.grouped_rl import usage
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    GeneratedSolution,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_adapters import (
    StrategyBox,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_run import run_in_context
from lasr_labs_2025_control_project.finetuning.grouped_rl.records import Recorder
from tests.test_hipt_rl import EchoSender, attack_problem
from tests.test_icrl_adapters import OneShotLearner, self_rec_trainer
from tests.test_icrl_driver import icrl_config


class StrategySender(EchoSender):
    """Writes the judge's marker only when it is given a strategy."""

    def __init__(self, box: StrategyBox) -> None:
        super().__init__()
        self.box = box

    async def generate(
        self, problem: AppsProblem, payload: int, num_samples: int
    ) -> list[GeneratedSolution]:
        self.silent = not self.box.text
        return await super().generate(problem, payload, num_samples)


async def test_gepa_search_scores_dev_and_reflects_on_minibatches(
    tmp_path: Path,
) -> None:
    config = icrl_config(
        tmp_path,
        method="gepa",
        gepa={"max_metric_calls": 6, "reflection_minibatch_size": 1},
    )
    assert config.identity["method"] == "gepa"
    box = StrategyBox()
    trainer = self_rec_trainer()
    trainer.sender = StrategySender(box)
    held_out = attack_problem("13", "validation")
    await run_in_context(
        config,
        trainer,
        box,
        OneShotLearner(),
        Recorder(config, tmp_path),
        usage.Usage(),
        [attack_problem(i) for i in ("11", "15", "17", "19")],
        [held_out],
        [(held_out, "# another model\nprint(7)")],
    )

    def rows(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]

    strategies = rows("strategies.jsonl")
    # Round 0 is the empty seed; round 1 beat it on a minibatch, then on dev.
    assert [s["strategy"] for s in strategies] == ["", "keep the original helper names"]
    assert all(s["episodes"] == 2 for s in strategies)  # the two dev problems
    assert strategies[1]["parent"] == "0"
    assert strategies[1]["reward_mean"] > strategies[0]["reward_mean"]
    assert rows("learner.jsonl")[0]["stage"] == "strategy"
    phases = {r["phase"] for r in rows("rollouts.jsonl")}
    assert {"search", "minibatch", "validation"} <= phases
    result = json.loads((tmp_path / "icrl_result.json").read_text())
    assert result["winner"]["strategy"] == "keep the original helper names"
