"""The RL arms in experiments/open_models/make_configs.py (no files written)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def make_configs() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_configs", REPO / "experiments/open_models/make_configs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def test_rl_arm_checkpoints_are_accepted_and_resolved(
    make_configs: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "ckpts.json"
    path.write_text(
        json.dumps({"rl_rewrite_signal_pilot": "tinker://run/sampler_weights/x"})
    )
    checkpoints = make_configs.load_checkpoints(path)
    models = make_configs.ModelResolver("open", checkpoints)
    assert (
        models.resolve("rl_rewrite_signal_pilot")
        == "tinker/tinker://run/sampler_weights/x"
    )
    with pytest.raises(SystemExit):
        make_configs.ModelResolver("closed", checkpoints).resolve(
            "rl_rewrite_signal_pilot"
        )


def test_rl_experiments_are_open_only_and_share_negatives(
    make_configs: ModuleType,
) -> None:
    models = make_configs.ModelResolver("open", {})
    untrained, pilot = make_configs.OPEN_ONLY["apps"][:2]
    document = make_configs.build_rewrite(untrained.rewrites[0], models, "apps")
    assert (
        document["defaults"]["rewrite"]["model"]
        == make_configs.ROLE_MODELS["open"]["U"]
    )
    assert (
        document["runs"]["attacks"]["eval_path"] == "../self_rec/generator/attacks.eval"
    )
    pending = make_configs.build_rewrite(pilot.rewrites[0], models, "apps")
    assert "PENDING__rl_rewrite_signal_pilot" in pending["defaults"]["rewrite"]["model"]
    monitor = make_configs.build_monitor(pilot.monitors[0], models, "apps")
    collusion = monitor["defaults"]["monitors"][0]["untrusted_collusion_monitor"]
    assert (
        collusion["model_info"]["system_prompt"]
        == "RL_SELF_SIGNAL_RECEIVER_SYSTEM_PROMPT"
    )
    assert collusion["model_info"]["model"] == make_configs.ROLE_MODELS["open"]["U"]
    assert monitor["runs"]["honeypots"]["eval_path"].startswith("../self_rec/")
    assert monitor["runs"]["attacks"]["eval_path"] == "generator/attacks.eval"


def test_rl_arms_reuse_reference_negatives(make_configs: ModuleType) -> None:
    models = make_configs.ModelResolver("open", {})
    arms = {
        spec.rewrites[0].experiment: spec for spec in make_configs.OPEN_ONLY["apps"]
    }
    reference = arms[make_configs.RL_REFERENCE_ARM]
    pilot = arms["rl_rewrite_signal_pilot"]
    for monitor in reference.monitors:
        runs = make_configs.build_monitor(monitor, models, "apps")["runs"]
        assert all("reuse_scores_from" not in run for run in runs.values())
    for monitor in pilot.monitors:
        runs = make_configs.build_monitor(monitor, models, "apps")["runs"]
        assert "reuse_scores_from" not in runs["attacks"]  # attacks always re-scored
        for label in ("honest", "honeypots"):
            source = runs[label]["reuse_scores_from"]
            assert source.startswith(f"../{make_configs.RL_REFERENCE_ARM}/outputs/")
            assert source.endswith(runs[label]["save_path"].split("/", 1)[1])
