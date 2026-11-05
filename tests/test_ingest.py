"""Importing existing .eval files as pipeline steps (pipeline/ingest.py), offline."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score
from omegaconf import OmegaConf

from lasr_labs_2025_control_project.pipeline import ingest
from lasr_labs_2025_control_project.pipeline import run as pipeline_run
from lasr_labs_2025_control_project.pipeline.graph import build_graph
from lasr_labs_2025_control_project.pipeline.schema import Config, register
from lasr_labs_2025_control_project.tracking import PipelineContext
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name

ROOT = Path(__file__).resolve().parents[1]
register()
# acausal on APPS, every model the offline mock, the minifier as the only transform
OVERRIDES = [
    "experiment=acausal",
    "models.U=mockllm/model",
    "models.T=mockllm/model",
    "models.H=mockllm/model",
    "measurement.transforms=[none,minified]",
    "measurement.limits.generate=4",
]
GENERATIONS = {
    # label: (eval mode, prompt)
    "attacks": ("attack", "acausal_attack_base_prompt"),
    "honest": ("honest", "honest_policy_prompt"),
    "honeypots": ("attack", "self_rec_subtle_attack_policy_prompt"),
}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")


def config(tmp: Path, *extra: str) -> Config:
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        raw = compose(
            config_name="config",
            overrides=[*OVERRIDES, f"experiments_dir={tmp}", *extra],
        )
    return OmegaConf.to_object(raw)  # type: ignore[return-value]


def generation(
    path: Path, label: str, *, prompt: str = "", status: str = "success"
) -> Path:
    """A generation log as generate_solutions_inspect writes one (4 even ids)."""
    mode, name = GENERATIONS[label]
    samples = [
        EvalSample(
            id=2 * i,
            epoch=1,
            input=f"problem {i}",
            target="",
            metadata={"main_task_description": f"problem {i}"},
            store={
                "base_solution": f"print({label!r}, {i})",
                "generator_info": {
                    "model_info": {
                        "model": "model",
                        "system_prompt": prompt or get_prompt_by_name(name),
                        "base_prompt": f"problem {i}",
                    },
                    "eval_mode": mode,
                },
            },
            scores={
                "apps_scorer": Score(
                    value={"main_task_success": "C", "side_task_success": "C"}
                )
            },
        )
        for i in range(4)
    ]
    log = EvalLog(
        status=status,  # type: ignore[arg-type]
        eval=EvalSpec(
            created="2026-09-01T12:00:00+00:00",
            task="backdoored_solutions",
            dataset=EvalDataset(samples=1500),
            model="mockllm/model",
            config=EvalConfig(limit=4),
        ),
        plan=EvalPlan(config=GenerateConfig(max_tokens=16384)),
        results=EvalResults(total_samples=4, completed_samples=4),
        samples=samples,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(path))
    return path


def strip(source: Path, dest: Path) -> None:
    """Copy a log without the pipeline and W&B headers (as if from before both)."""
    log = read_eval_log(str(source))
    log.eval.metadata = {
        k: v
        for k, v in (log.eval.metadata or {}).items()
        if k not in ("pipeline", "tracking")
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(dest))


def ingest_dir(directory: Path, tmp: Path, *flags: str) -> None:
    ingest.main([str(directory), *OVERRIDES, f"experiments_dir={tmp}", *flags])


def statuses(tmp: Path, *extra: str) -> dict[str, str]:
    cfg = config(tmp, *extra)
    steps = pipeline_run.run_pipeline(cfg, PipelineContext(name=cfg.name, config={}))
    return {s.node.id: s.status for s in steps}


def test_files_match_steps_and_the_rest_are_reported(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    for label in GENERATIONS:
        generation(drive / f"{label}.eval", label)
    generation(drive / "cancelled.eval", "attacks", status="cancelled")
    generation(drive / "other_prompt.eval", "attacks", prompt="a prompt no step uses")
    cfg = config(tmp_path / "exp")
    ctx = ingest.RunContext(
        cfg,
        ingest.Store(cfg.experiments_dir, "apps"),
        PipelineContext(name="t", config={}),
    )
    plan, _ = ingest.plan_directory(build_graph(cfg), drive, ctx)

    assert {node.id for node, _, _ in plan.steps} == {
        "generate.test.attacks",
        "generate.test.honest",
        "generate.test.honeypots",
    }
    reasons = {Path(p).name: why for p, why in plan.skipped}
    assert "cancelled" in reasons["cancelled.eval"]
    assert "system_prompt" in reasons["other_prompt.eval"]
    record = next(r for n, _, r in plan.steps if n.id == "generate.test.attacks")
    assert record["created"].startswith("2026-09-01") and record["md5"]


def test_limits_must_match_the_declared_config(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    generation(drive / "attacks.eval", "attacks")
    cfg = config(tmp_path / "exp", "measurement.limits.generate=200")
    ctx = ingest.RunContext(
        cfg,
        ingest.Store(cfg.experiments_dir, "apps"),
        PipelineContext(name="t", config={}),
    )
    plan, _ = ingest.plan_directory(build_graph(cfg), drive, ctx)
    assert not plan.steps
    assert "limit" in plan.skipped[0][1]


def test_imported_steps_are_pipeline_steps(tmp_path: Path, capsys: Any) -> None:
    """Generations, then the pipeline's own transforms and (mock) monitors,
    stripped of every header and imported again: every step matches its key."""
    first = tmp_path / "first"
    drive = tmp_path / "drive"
    for label in GENERATIONS:
        generation(drive / f"{label}.eval", label)
    ingest_dir(drive, first)
    assert "Nothing was written" in capsys.readouterr().out
    assert not (first / "apps").exists()

    ingest_dir(drive, first, "--apply")
    pending = statuses(first, "stages=[transform,monitor]", "allow_paid=true")
    assert {pending[f"generate.test.{label}"] for label in GENERATIONS} == {
        pipeline_run.DONE
    }

    # everything the first store holds, as loose files without headers
    second_drive = tmp_path / "second_drive"
    for index, path in enumerate(sorted((first / "apps").rglob("*.eval"))):
        strip(path, second_drive / f"{index}_{path.name}")
    second = tmp_path / "second"
    ingest_dir(second_drive, second, "--apply")
    out = capsys.readouterr().out
    assert "Skipped" not in out

    no_analysis = "stages=[generate,transform,monitor]"
    now = statuses(second, no_analysis)
    before = statuses(first, no_analysis)
    eval_steps = {
        i for i in before if i.split(".")[0] in ("generate", "transform", "monitor")
    }
    assert eval_steps and all(now[i] == pipeline_run.DONE for i in eval_steps)
    # the imported copies carry the step's key, so the store accepts them as done
    some = next((second / "apps" / "monitor").rglob("*.eval"))
    header = read_eval_log(str(some), header_only=True).eval.metadata or {}
    assert header["pipeline"]["hash"]


def test_pipeline_run_folders_relog_their_steps(tmp_path: Path, capsys: Any) -> None:
    exp = tmp_path / "exp"
    drive = tmp_path / "drive"
    for label in GENERATIONS:
        generation(drive / f"{label}.eval", label)
    ingest_dir(drive, exp, "--apply")
    (run_dir,) = (exp / "runs" / "acausal").iterdir()
    capsys.readouterr()
    ingest.main([str(run_dir)])
    out = capsys.readouterr().out
    assert out.count("store") == 3  # the three generations, from the store
    shutil.rmtree(exp / "apps" / "generate")
    ingest.main([str(run_dir)])
    assert "no finished output" in capsys.readouterr().out


def test_setting_skips_samples_without_scores() -> None:
    scored = {
        "main_task_success": Score(value=0.5),
        "side_task_success": Score(value="C"),
    }
    log = SimpleNamespace(
        samples=[SimpleNamespace(scores={}), SimpleNamespace(scores=scored)],
        eval=SimpleNamespace(task="bash_arena"),
    )  # e.g. a BashArena log whose first sample timed out
    assert ingest._setting(log) == "basharena"
