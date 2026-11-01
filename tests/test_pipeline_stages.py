"""The real stage functions under the pipeline, offline.

Generations and monitor outputs are supplied as ``inputs`` (Inspect logs
built here from recorded APPS solutions and synthetic scores), the minifier
runs for real, the monitors run against Inspect's mock model (their scores
come back NaN, which is fine for the wiring), and the analysis solves a real,
small game.
"""

import json
import random
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalResults,
    EvalSample,
    EvalSpec,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.scorer import Score
from omegaconf import OmegaConf

from lasr_labs_2025_control_project.pipeline import run as pipeline_run
from lasr_labs_2025_control_project.pipeline.schema import Config, register
from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
    load_game,
    load_saved_condition,
    resolve_saved_condition,
)
from lasr_labs_2025_control_project.tracking import PipelineContext

ROOT = Path(__file__).resolve().parents[1]
register()


def _log(samples: list[EvalSample], path: Path) -> str:
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-10-16T00:00:00Z",
            task="fixture",
            dataset=EvalDataset(samples=len(samples)),
            model="mockllm/model",
            config=EvalConfig(),
        ),
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples)),
        samples=samples,
    )
    write_eval_log(log, str(path))
    return str(path)


def _generation(tmp_path: Path, label: str, n: int = 12) -> str:
    """APPS-shaped generation log from the recorded fixture's solutions."""
    fixture = json.loads((ROOT / "tests/data/sft_steps/apps_train.json").read_text())
    records = [s for log in fixture["logs"] for s in log["samples"]]
    samples = []
    for i in range(n):
        record = records[i % len(records)]
        problem = fixture["problem_statements"][str(record["id"])]
        samples.append(
            EvalSample(
                id=i,
                epoch=1,
                input=problem,
                target="",
                metadata={"main_task_description": problem},
                store={
                    "base_solution": record["solution"],
                    "generator_info": {"model_info": {"model": "U"}},
                },
                scores={
                    "apps_scorer": Score(
                        value={"main_task_success": "C", "side_task_success": "C"}
                    )
                },
            )
        )
    return _log(samples, tmp_path / f"gen_{label}.eval")


def _monitored(tmp_path: Path, label: str, n: int = 40) -> str:
    """A monitor output with separable synthetic scores."""
    rng = random.Random(label)
    collusion = {"attacks": 2.0, "honeypots": -2.0, "honest": 0.0}[label]
    suspicion = {"attacks": 7, "honeypots": 6, "honest": 2}[label]
    samples = [
        EvalSample(
            id=i,
            epoch=1,
            input="x",
            target="",
            metadata={"main_task_description": "p", "base_solution": "print(1)"},
            store={
                "untrusted_collusion_score": rng.gauss(collusion, 1.0),
                "trusted_suspiciousness_score": min(
                    9, max(0, suspicion + rng.randint(-2, 2))
                ),
                "untrusted_suspiciousness_score": min(
                    9, max(0, suspicion + rng.randint(-2, 2))
                ),
                "untrusted_collusion_score_metadata": {
                    "model_info": {"system_prompt": "s", "base_prompt": "b"}
                },
            },
        )
        for i in range(n)
    ]
    return _log(samples, tmp_path / f"mon_{label}.eval")


def _run(
    tmp_path: Path, *overrides: str, experiment: str = "acausal"
) -> list[pipeline_run.Step]:
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        raw = compose(
            config_name="config",
            overrides=[
                f"experiment={experiment}",
                f"experiments_dir={tmp_path / 'experiments'}",
                "models.U=mockllm/model",
                "models.T=mockllm/model",
                "allow_paid=true",
                *overrides,
            ],
        )
    cfg: Config = OmegaConf.to_object(raw)  # type: ignore[assignment]
    return pipeline_run.run_pipeline(cfg, PipelineContext(name="test", config={}))


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_minify_and_monitor_stages(tmp_path: Path) -> None:
    gens = {
        label: _generation(tmp_path, label)
        for label in ("attacks", "honest", "honeypots")
    }
    steps = _run(
        tmp_path,
        "measurement.transforms=[minified]",
        "stages=[generate,transform,monitor]",
        *(f"+inputs.generate.test.{label}={path}" for label, path in gens.items()),
    )
    status = {s.node.id: s.status for s in steps}
    assert status["generate.test.attacks"] == pipeline_run.INPUT
    assert status["transform.test.minified.attacks"] == pipeline_run.RUN
    assert status["game.minified.base"] == pipeline_run.MISSING  # stage not selected

    store = pipeline_run.Store(tmp_path / "experiments", "apps")
    by_id = {s.node.id: s.node for s in steps}
    minified = read_eval_log(
        str(store.output_path(by_id["transform.test.minified.attacks"]))
    )
    assert (minified.eval.metadata or {})["pipeline"]["hash"] == (
        by_id["transform.test.minified.attacks"].hash
    )
    assert minified.samples and minified.samples[0].store["transformed_solution"]
    monitored = read_eval_log(
        str(store.output_path(by_id["monitor.minified.base.attacks"]))
    )
    assert monitored.samples
    store_keys = monitored.samples[0].store
    assert {"untrusted_collusion_score", "trusted_suspiciousness_score"} <= set(
        store_keys
    )

    # all done now: the same command runs nothing
    again = _run(
        tmp_path,
        "measurement.transforms=[minified]",
        "stages=[generate,transform,monitor]",
        *(f"+inputs.generate.test.{label}={path}" for label, path in gens.items()),
    )
    assert pipeline_run.RUN not in {s.status for s in again}


def test_shared_suspiciousness_scores_reach_each_arm(tmp_path: Path) -> None:
    gens = {
        label: _generation(tmp_path, label)
        for label in ("attacks", "honest", "honeypots")
    }
    steps = _run(
        tmp_path,
        "measurement.transforms=[none]",
        "measurement.share_suspiciousness=true",
        "measurement.arms.sft.model=mockllm/other",
        "stages=[generate,monitor]",
        *(f"+inputs.generate.test.{label}={path}" for label, path in gens.items()),
        experiment="self_rec",
    )
    ran = [s.node for s in steps if s.status == pipeline_run.RUN]
    assert len(ran) == 3 + 2 * 3  # one shared step per label, one per arm and label

    store = pipeline_run.Store(tmp_path / "experiments", "apps")
    frame = build_jsonl_dataframe_from_eval_log(
        store.output_path(next(n for n in ran if n.id == "monitor.none.sft.attacks"))
    )
    assert len(frame) == 12
    assert {
        "untrusted_collusion_score",
        "trusted_suspiciousness_score",
        "untrusted_suspiciousness_score",
    } <= set(frame.columns)


def test_game_and_analysis_stages(tmp_path: Path) -> None:
    monitored = {
        label: _monitored(tmp_path, label)
        for label in ("attacks", "honest", "honeypots")
    }
    overrides = [
        "measurement.transforms=[none]",
        "optimization.regeneration_cutoffs=[0.0,0.5]",
        "optimization.max_iterations=3",
        "optimization.bootstrap.replicates=2",
        "optimization.bootstrap.chunk_size=1",
        "optimization.bootstrap.n_jobs=1",
        *(
            f"+inputs.monitor.none.base.{label}={path}"
            for label, path in monitored.items()
        ),
    ]
    steps = _run(tmp_path, *overrides)
    store = pipeline_run.Store(tmp_path / "experiments", "apps")
    (node,) = [s.node for s in steps if s.node.stage == "game"]
    folder = store.folder(node)
    names = {p.name for p in folder.iterdir()}
    assert names == {
        "analysis_data.json.gz",
        "analysis_data_um_only.json.gz",
        "bootstrap_all_monitors.jsonl.gz",
        "bootstrap_um_only.jsonl.gz",
        "scores.json.gz",
    }

    # the analysis draws the figures once, from the saved games
    (analysis,) = [s.node for s in steps if s.node.stage == "analysis"]
    record = json.loads(store.output_path(analysis).read_text())
    assert (
        "strategies.png" in record["figures"]
        and "safety_usefulness_curves.png" in record["figures"]
    )
    assert all(
        (store.folder(analysis) / "figures" / f).exists() for f in record["figures"]
    )
    # a saved game is rebuilt for plotting without re-solving
    game = load_game(folder)
    assert [r["safety"] for r in game.result.results] == [
        e["safety"] for e in load_saved_condition(folder)["sweep"]
    ]
    assert len(game.replicates) == 2 and game.result.config.parameter_spaces

    saved = load_saved_condition(folder)
    assert saved["pipeline"]["hash"] == node.hash
    assert [e["usefulness"] for e in saved["sweep"]] == [1.0, 0.5]
    # re-solving with the recorded settings and seed gives the same safety
    again = resolve_saved_condition(folder)
    assert [r["safety"] for r in again.results] == [e["safety"] for e in saved["sweep"]]
    # and the done check accepts it
    assert {s.status for s in _run(tmp_path, *overrides)} <= {
        pipeline_run.DONE,
        pipeline_run.INPUT,
    }
