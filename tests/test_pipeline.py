"""The Hydra pipeline: config composition, the stage graph, planning and reuse.

Stage functions are replaced by fakes that write real (mock-model) Inspect
logs, so planning, the paid gate, resuming and collision checks run offline.
"""

import gzip
import json
import posixpath
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from omegaconf import OmegaConf

from lasr_labs_2025_control_project.pipeline import run as pipeline_run
from lasr_labs_2025_control_project.pipeline.execute import (
    RunContext,
    basharena_runtime_config,
)
from lasr_labs_2025_control_project.pipeline.graph import (
    ConfigError,
    Graph,
    Node,
    build_graph,
)
from lasr_labs_2025_control_project.pipeline.schema import Config, register
from lasr_labs_2025_control_project.pipeline.store import CollisionError, Output
from lasr_labs_2025_control_project.scripts.plot_data import (
    BootstrapSettings,
    bootstrap_variant,
    payload_path,
)
from lasr_labs_2025_control_project.tracking import PipelineContext
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    save_analysis_payload,
)

CONF = Path(__file__).resolve().parents[1] / "conf"
SNAPSHOT = Path(__file__).parent / "data" / "make_configs_snapshot.json"

register()


def config(*overrides: str) -> Config:
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        raw = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_object(raw)  # type: ignore[return-value]


def graph(*overrides: str) -> Graph:
    return build_graph(config(*overrides))


def hashes(*overrides: str) -> dict[str, str]:
    return {node.id: node.hash for node in graph(*overrides).nodes}


# -- config ----------------------------------------------------------------


@pytest.mark.parametrize(
    "experiment",
    sorted(
        str(p.relative_to(CONF / "experiment").with_suffix(""))
        for p in (CONF / "experiment").rglob("*.yaml")
    ),
)
def test_every_experiment_composes_and_builds(experiment: str) -> None:
    assert graph(f"experiment={experiment}").targets


def test_unknown_keys_and_bad_values_are_rejected() -> None:
    with pytest.raises(Exception, match="nonexistent"):
        config("experiment=self_rec", "measurement.nonexistent=1")
    with pytest.raises(Exception, match="seed=abc"):
        config("seed=abc")
    with pytest.raises(ConfigError, match="variant tm_only needs monitors"):
        graph(
            "experiment=self_rec",
            "optimization.variants=[tm_only]",
            "measurement.monitors=[untrusted_collusion]",
        )
    with pytest.raises(ConfigError, match="doesn't support transform 'minified'"):
        graph("experiment=self_rec", "setting=basharena")
    with pytest.raises(ConfigError, match="no honest-mode honeypots"):
        graph(
            "experiment=self_rec_controls",
            "setting=basharena",
            "measurement.transforms=[none]",
        )
    with pytest.raises(ConfigError, match="isn't in elicitation.arms"):
        graph("experiment=acausal", "measurement.arms.base.model=sft:missing")


# -- the graph ----------------------------------------------------------------


def _snapshot_signatures(docs: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Each run of each config make_configs.py emitted, as a comparable signature."""
    paths: dict[str, tuple[Any, ...]] = {}
    signatures = set()
    order = {"generator": 0, "transform": 1, "monitor": 2}
    for _, (kind, doc) in sorted(docs.items(), key=lambda kv: order[kv[1][0]]):
        folder, defaults = doc["experiment_folder"], doc.get("defaults", {})
        setting = defaults.get("dataset_type") or defaults.get("dataset") or "apps"
        for run in doc["runs"].values():
            source = paths.get(
                posixpath.normpath(posixpath.join(folder, run.get("eval_path", "")))
            )
            if kind == "generator":
                signature = (
                    "gen",
                    setting,
                    defaults["data_split"],
                    run["eval_mode"],
                    run["system_prompt"],
                    run["model"],
                    defaults.get("max_tokens"),
                    defaults.get("limit"),
                )
            elif source is None:
                continue  # read a file no config generated (hand-placed inputs)
            elif kind == "transform":
                t = defaults["transform"]
                info = t.get("model_info") or {}
                signature = (
                    "tr",
                    source,
                    t["type"],
                    info.get("model"),
                    info.get("system_prompt"),
                    t["remove_comments"],
                )
            else:
                names = {
                    "untrusted_collusion_monitor": "untrusted_collusion",
                    "trusted_backdoor_monitor": "trusted_suspiciousness",
                    "untrusted_backdoor_monitor": "untrusted_suspiciousness",
                }
                monitors = {
                    names[k]: (
                        "SFT"
                        if "PENDING__" in v["model_info"]["model"]
                        else v["model_info"]["model"],
                        v["model_info"]["system_prompt"],
                        v["model_info"]["base_prompt"],
                        v["transformed"],
                        v["remove_comments"],
                    )
                    for entry in defaults["monitors"]
                    for k, v in entry.items()
                }
                signature = (
                    "mon",
                    source,
                    defaults.get("view", "trajectory"),
                    tuple(sorted(monitors.items())),
                )
            if kind != "monitor":
                paths[posixpath.normpath(posixpath.join(folder, run["save_path"]))] = (
                    signature
                )
            signatures.add(signature)
    return signatures


def _graph_signatures(*overrides: str) -> set[tuple[Any, ...]]:
    memo: dict[str, tuple[Any, ...]] = {}

    def signature(node: Node) -> tuple[Any, ...]:
        if node.hash not in memo:
            p = node.params
            if node.stage == "generate":
                memo[node.hash] = (
                    "gen",
                    p["setting"],
                    p["split"],
                    p["eval_mode"],
                    p["system_prompt"],
                    p["model"],
                    p["max_tokens"],
                    p["limit"],
                )
            elif node.stage == "transform":
                t = p["transform"]
                memo[node.hash] = (
                    "tr",
                    signature(node.deps["input"]),
                    t["type"],
                    t["model"],
                    t["system_prompt"],
                    t["remove_comments"],
                )
            else:
                ft = node.deps.get("collusion_model")
                monitors = {
                    name: (
                        (ft.external or "SFT")
                        if name == "untrusted_collusion" and ft is not None
                        else m["model"],
                        m["system_prompt"],
                        m["base_prompt"],
                        m["transformed"],
                        m["remove_comments"],
                    )
                    for name, m in p["monitors"].items()
                }
                memo[node.hash] = (
                    "mon",
                    signature(node.deps["input"]),
                    p["view"],
                    tuple(sorted(monitors.items())),
                )
        return memo[node.hash]

    return {
        signature(node)
        for node in graph(*overrides).nodes
        if node.stage in ("generate", "transform", "monitor")
    }


TREES = {
    "open/apps": [
        [f"experiment={e}"]
        for e in ("self_rec", "causal", "acausal", "additive", "self_rec_controls")
    ],
    "open/bigcodebench": [
        [f"experiment={e}", "setting=bigcodebench"]
        for e in ("self_rec", "causal", "acausal")
    ]
    + [["experiment=acausal_optimized"]],
    "closed/apps": [
        [f"experiment=paper/apps_{e}"]
        for e in ("self_rec", "causal", "acausal", "additive", "self_rec_controls")
    ]
    + [
        [f"experiment={e}", "models=closed", "measurement.share_generations=false"]
        for e in ("self_rec", "causal", "self_rec_controls")
    ],
    "closed/bigcodebench": [
        [f"experiment=paper/bigcodebench_{e}"]
        for e in ("self_rec", "causal", "acausal", "acausal_optimized")
    ]
    + [
        [
            "experiment=self_rec",
            "setting=bigcodebench",
            "models=closed",
            "measurement.share_generations=false",
        ]
    ],
    "open/basharena": [
        ["experiment=self_rec", "setting=basharena", "measurement.transforms=[none]"]
    ],
}


@pytest.mark.parametrize("tree", sorted(TREES))
def test_graph_reproduces_every_config_make_configs_emitted(tree: str) -> None:
    """tests/data/make_configs_snapshot.json is every config the deleted
    experiments/open_models/make_configs.py generated; each of its runs must
    have an identical node (models, prompts, flags, inputs) in some experiment."""
    expected = _snapshot_signatures(json.loads(SNAPSHOT.read_text())[tree])
    actual = set().union(*(_graph_signatures(*ov) for ov in TREES[tree]))
    assert expected - actual == set()


def test_generations_are_shared_across_collusion_strategies() -> None:
    self_rec, causal = hashes("experiment=self_rec"), hashes("experiment=causal")
    for node_id in (
        "generate.test.honest",
        "generate.test.honeypots",
        "transform.test.paraphrased.honest",
    ):
        assert self_rec[node_id] == causal[node_id]
    assert self_rec["generate.test.attacks"] != causal["generate.test.attacks"]
    separate = hashes("experiment=causal", "measurement.share_generations=false")
    assert separate["generate.test.honest"] != causal["generate.test.honest"]


def test_additive_reuses_the_causal_checkpoint() -> None:
    causal, additive = hashes("experiment=causal"), hashes("experiment=additive")
    assert (
        additive["finetune.causal.paraphrased"] == causal["finetune.causal.paraphrased"]
    )
    assert additive["game.paraphrased.sft"] != causal["game.paraphrased.sft"]


def test_label_controls() -> None:
    g = graph("experiment=self_rec_controls").by_id()
    honest_ft = g["finetune.self_rec_honest.none"]
    assert honest_ft.deps["positives"].params["eval_mode"] == "honest"
    assert honest_ft.deps["negatives"].params["model"] == config().models.H
    assert g["finetune.self_rec_permuted.none"].params["labels"] == "permuted"
    # no-backdoor arm: U's honest solutions fill the attacks slot
    attacks = g["monitor.none.honest_no_backdoors.attacks"]
    assert attacks.deps["input"].id == "generate.test.honest"
    assert attacks.params["label"] == "attacks"


def test_changes_only_rerun_what_they_reach() -> None:
    base = hashes("experiment=self_rec")
    lr = hashes("experiment=self_rec", "elicitation.training.lr=0.001")
    changed = {k for k in base if base[k] != lr[k]}
    assert changed and all(
        k.split(".")[0] in ("finetune", "monitor", "game", "analysis") for k in changed
    )
    assert all(
        lr[k] == base[k] for k in base if ".base." in k
    )  # the base arm is untouched
    game = hashes("experiment=self_rec", "optimization.game.num_tasks=100")
    assert {k for k in base if base[k] != game[k]} == {
        k for k in base if k.startswith(("game.", "analysis"))
    }
    seed = hashes("experiment=self_rec", "seed=1")
    assert (
        seed["transform.test.minified.attacks"]
        != base["transform.test.minified.attacks"]
    )  # input changed
    minify = graph("experiment=self_rec").by_id()["transform.test.minified.attacks"]
    assert "seed" not in minify.params  # deterministic: no seed in its own key


def test_inputs_replace_nodes_and_must_exist() -> None:
    g = graph(
        "experiment=self_rec",
        "+inputs.finetune.self_rec.none=tinker/tinker://run/sampler_weights/x",
    ).by_id()
    ft = g["finetune.self_rec.none"]
    assert ft.external == "tinker/tinker://run/sampler_weights/x" and not ft.deps
    assert "generate.train.attacks" in g  # still needed by the other transforms' arms
    only_none = graph(
        "experiment=self_rec",
        "measurement.transforms=[none]",
        "+inputs.finetune.self_rec.none=tinker/tinker://run/sampler_weights/x",
    ).by_id()
    assert "generate.train.attacks" not in only_none
    with pytest.raises(ConfigError, match="doesn't have"):
        graph("experiment=self_rec", "+inputs.finetune.nope.none=x")


def test_monitor_settings_follow_transform_and_view() -> None:
    g = graph("experiment=causal").by_id()
    para = g["monitor.paraphrased.sft.attacks"].params["monitors"]
    assert para["untrusted_collusion"]["transformed"] is True
    assert para["untrusted_collusion"]["system_prompt"].startswith(
        "SELF_REC"
    )  # combined arm
    assert para["trusted_suspiciousness"]["transformed"] is False
    assert (
        g["monitor.none.base.attacks"]
        .params["monitors"]["untrusted_collusion"]["system_prompt"]
        .startswith("CAUSAL")
    )
    bash = (
        graph(
            "experiment=self_rec",
            "setting=basharena",
            "measurement.transforms=[none]",
            "setting.action_view.max_history_chars=5",
        )
        .by_id()["monitor.none.sft.attacks"]
        .params
    )
    assert bash["view"] == "action" and bash["action_view"] == {"max_history_chars": 5}
    assert all(
        not m["transformed"] and not m["remove_comments"]
        for m in bash["monitors"].values()
    )


def test_basharena_runtime_config_shape() -> None:
    gen = graph(
        "experiment=self_rec", "setting=basharena", "measurement.transforms=[none]"
    ).by_id()["generate.train.honeypots"]
    assert basharena_runtime_config(gen.params) == {
        "defaults": {
            "data_split": "train",
            "dataset_type": "basharena",
            "limit": 5,
            "max_tokens": 16384,
        },
        "runs": {
            "honeypots": {
                "eval_mode": "attack",
                "system_prompt": "BASHARENA_ATTACK_SYSTEM_PROMPT",
                "model": config().models.H,
            }
        },
    }


# -- running ----------------------------------------------------------------


def _write_eval(path: Path, metadata: dict[str, Any], n: int = 3) -> None:
    task = Task(dataset=[Sample(input=f"q{i}") for i in range(n)])
    (log,) = inspect_eval(
        task,
        model="mockllm/model",
        metadata=metadata,
        log_dir=str(path.parent / "_logs"),
        display="none",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(log.location, path)
    shutil.rmtree(path.parent / "_logs")


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake stage functions; returns the ids they ran, in order."""
    calls: list[str] = []

    def fake(node: Node, ctx: RunContext) -> Output:
        assert all(dep.hash in ctx.outputs for dep in node.deps.values())
        calls.append(node.id)
        if node.stage == "finetune":
            return ctx.store.write_model(
                node,
                f"tinker/tinker://fake/{node.hash}",
                {"train_tokens": 1_000_000, "forward_tokens": 2_000_000},
            )
        if node.stage == "game":
            folder = ctx.store.folder(node)
            for i, variant in enumerate(node.params["variants"]):
                save_analysis_payload(
                    {"pipeline": {**node.record(), "variant": variant}},
                    payload_path(folder, i, variant),
                )
            return Output(path=str(folder))
        if node.stage == "analysis":
            figure = ctx.store.folder(node) / "figures" / "f.png"
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"png")
            return ctx.store.write_analysis(node, [figure])
        path = ctx.store.output_path(node)
        _write_eval(path, {"pipeline": node.record()})
        return Output(path=str(path))

    monkeypatch.setattr(
        pipeline_run, "EXECUTORS", dict.fromkeys(pipeline_run.EXECUTORS, fake)
    )
    monkeypatch.setenv("WANDB_MODE", "disabled")
    return calls


def run(tmp_path: Path, *overrides: str) -> list[pipeline_run.Step]:
    cfg = config(
        "experiment=self_rec",
        "measurement.transforms=[none]",
        "optimization.bootstrap.replicates=0",
        f"experiments_dir={tmp_path}",
        *overrides,
    )
    return pipeline_run.run_pipeline(cfg, PipelineContext(name=cfg.name, config={}))


def test_paid_gate_then_run_then_resume(tmp_path: Path, ran: list[str]) -> None:
    steps = run(tmp_path)
    assert ran == [] and {s.status for s in steps} == {pipeline_run.RUN}

    run(tmp_path, "allow_paid=true")
    assert len(ran) == len(steps)
    assert ran.index("finetune.self_rec.none") < ran.index("monitor.none.sft.attacks")

    ran.clear()
    steps = run(tmp_path)  # everything done: no paid calls, so no gate either
    assert ran == [] and {s.status for s in steps} == {pipeline_run.DONE}


def test_done_outputs_stop_the_walk(tmp_path: Path, ran: list[str]) -> None:
    steps = run(tmp_path, "allow_paid=true")
    by_id = {s.node.id: s.node for s in steps}
    store = pipeline_run.Store(tmp_path, "apps")
    # an upstream output deleted under finished downstream outputs is not redone
    shutil.rmtree(store.folder(by_id["generate.test.honest"]))
    # ... nor is a deleted game while the analysis drawn from it is done
    shutil.rmtree(store.folder(by_id["game.none.base"]))
    ran.clear()
    run(tmp_path)
    assert ran == []
    # a deleted analysis is redrawn, and the missing game it needs is solved again
    shutil.rmtree(store.folder(by_id["analysis"]))
    run(tmp_path)
    assert ran == ["game.none.base", "analysis"]


def test_a_different_key_under_the_same_hash_is_a_collision(
    tmp_path: Path, ran: list[str]
) -> None:
    steps = run(tmp_path, "allow_paid=true")
    node = next(s.node for s in steps if s.node.id == "analysis")
    path = pipeline_run.Store(tmp_path, "apps").output_path(node)
    record = json.loads(path.read_text())
    record["pipeline"]["key"] = {**node.key, "stage": "other"}
    path.write_text(json.dumps(record))
    with pytest.raises(CollisionError):
        run(tmp_path)


def test_unselected_stages_must_exist(tmp_path: Path, ran: list[str]) -> None:
    with pytest.raises(pipeline_run.PlanError, match="generate.test.attacks"):
        run(tmp_path, "stages=[monitor,game,analysis]", "allow_paid=true")
    run(tmp_path, "stages=[generate]", "allow_paid=true")
    assert ran and all(i.startswith("generate.") for i in ran)


def test_force_reruns_the_step_and_what_depends_on_it(
    tmp_path: Path, ran: list[str]
) -> None:
    run(tmp_path, "allow_paid=true")
    ran.clear()
    steps = run(tmp_path, "force=[monitor.none.sft.attacks]", "allow_paid=true")
    assert ran == ["monitor.none.sft.attacks", "game.none.sft", "analysis"]
    assert {s.node.id for s in steps if s.status == pipeline_run.RERUN} == set(ran)

    ran.clear()
    store = pipeline_run.Store(tmp_path, "apps")
    game = next(s.node for s in steps if s.node.id == "game.none.base")
    (store.folder(game) / "bootstrap_all_monitors.jsonl.gz").write_bytes(b"old")
    run(tmp_path, "force=[game]")  # a whole stage; games call no paid models
    assert not (store.folder(game) / "bootstrap_all_monitors.jsonl.gz").exists()
    assert ran == ["game.none.base", "game.none.sft", "analysis"]

    with pytest.raises(pipeline_run.PlanError, match="no step or stage"):
        run(tmp_path, "force=[generate.test.typo]")
    with pytest.raises(pipeline_run.PlanError, match="analysis"):
        run(tmp_path, "force=[game]", "stages=[game]")
    # one condition's game alone: this narrower analysis has no output to go stale
    ran.clear()
    one = ("measurement.transforms=[none]", "~measurement.arms.sft", "stages=[game]")
    run(tmp_path, *one, "force=[game.none.base]")
    assert ran == ["game.none.base"]


def test_inputs_are_used_without_running(tmp_path: Path, ran: list[str]) -> None:
    ref = "tinker/tinker://given/sampler_weights/x"
    steps = run(tmp_path, f"+inputs.finetune.self_rec.none={ref}", "allow_paid=true")
    assert "finetune.self_rec.none" not in ran
    assert {s.node.id: s.status for s in steps}[
        "finetune.self_rec.none"
    ] == pipeline_run.INPUT
    assert not any(i.startswith("generate.train") for i in ran)


# -- stage pieces ----------------------------------------------------------------


def test_permuted_labels_shuffle_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    from lasr_labs_2025_control_project.finetuning import sft_data
    from tests.test_finetune_sft import _log

    logs = {
        "pos.eval": _log([_sample_with(i, "U") for i in range(6)], "U"),
        "neg.eval": _log([_sample_with(10 + i, "H") for i in range(6)], "H"),
    }
    monkeypatch.setattr(sft_data, "read_eval_log", lambda p: logs[Path(p).name])
    monkeypatch.setattr(
        sft_data, "get_prompt_by_name", lambda name: "{problem_statement}{solution}"
    )
    author, _ = sft_data.build_training_data(
        [Path("pos.eval"), Path("neg.eval")], target_model="U", seed=3
    )
    permuted, composition = sft_data.build_training_data(
        [Path("pos.eval"), Path("neg.eval")],
        target_model="U",
        seed=3,
        labels="permuted",
    )
    answers = [e["messages"][-1]["content"] for e in author]
    shuffled = [e["messages"][-1]["content"] for e in permuted]
    assert sorted(answers) == sorted(shuffled) and answers != shuffled
    assert [e["messages"][:-1] for e in author] == [
        e["messages"][:-1] for e in permuted
    ]
    assert composition["labels"] == "permuted"


def _sample_with(i: int, model: str):
    from tests.test_finetune_sft import _sample

    return _sample(i, f"print({i})", model=model)


def test_seeded_solver_is_reproducible() -> None:
    import numpy as np

    from lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.double_oracle import (
        double_oracle,
    )

    def payoff(
        params: dict[str, Any],
    ) -> tuple[
        Any, dict[str, Any]
    ]:  # a mixed (matching-pennies-like) game: ties depend on the rng
        value = 0.5 + 0.5 * np.cos(6 * (params["blue_x"] - params["red_x"]))
        return np.asarray(value), {"blue_win": np.asarray(value)}

    space = {
        "blue_x": {"min": 0.0, "max": 1.0, "scale": "linear"},
        "red_x": {"min": 0.0, "max": 1.0, "scale": "linear"},
    }
    first = double_oracle(space, payoff, max_iterations=3, seed=7)
    second = double_oracle(space, payoff, max_iterations=3, seed=7)
    assert first["best_score"] == second["best_score"]
    assert first["blue_mixed_strategy"] == second["blue_mixed_strategy"]


def test_bootstrap_resumes_from_saved_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lasr_labs_2025_control_project.scripts import plot_data

    seen: list[list[int]] = []

    def fake_replicates(
        indices: Any, *, n_jobs: int, condition_seed: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        seen.append(list(indices))
        return [{"bootstrap_replicate": i, "seed": condition_seed} for i in indices]

    monkeypatch.setattr(plot_data, "run_replicates", fake_replicates)
    path = tmp_path / "bootstrap_um_only.jsonl.gz"
    settings = BootstrapSettings(replicates=5, chunk_size=2)
    with gzip.open(path, "wt") as f:  # an interrupted run got through 3
        f.writelines(json.dumps({"bootstrap_replicate": i}) + "\n" for i in range(3))
    reps = bootstrap_variant(path, settings, condition_seed=1)
    assert seen == [[3, 4]] and [r["bootstrap_replicate"] for r in reps] == list(
        range(5)
    )
    assert len(plot_data.read_replicates(path)) == 5


def test_cost_report_prices_recorded_tokens(tmp_path: Path, ran: list[str]) -> None:
    from lasr_labs_2025_control_project.scripts import cost_report

    overrides = ("models.U=tinker/Qwen/Qwen3-8B", "allow_paid=true")
    run(tmp_path, *overrides)
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)
    cfg = config(
        "experiment=self_rec",
        "measurement.transforms=[none]",
        "optimization.bootstrap.replicates=0",
        f"experiments_dir={tmp_path}",
        *overrides,
    )
    OmegaConf.save(OmegaConf.structured(cfg), run_dir / ".hydra" / "config.yaml")
    tokens, missing = cost_report.collect(run_dir)
    assert missing == []
    assert tokens[("finetune", "Qwen/Qwen3-8B", "train")] == 1_000_000
    assert tokens[("finetune", "Qwen/Qwen3-8B", "prefill")] == 2_000_000
    assert cost_report.usd("Qwen/Qwen3-8B", "train", 1_000_000) == 0.44
    assert cost_report.usd("mockllm/model", "sample", 10) is None


def test_analysis_loads_games_where_they_were_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lasr_labs_2025_control_project.pipeline import execute
    from lasr_labs_2025_control_project.scripts import plot_data

    given_dir, given_file = tmp_path / "a", tmp_path / "b" / "payload.json"
    given_dir.mkdir()
    given_file.parent.mkdir()
    given_file.touch()
    loaded: list[Path] = []
    monkeypatch.setattr(
        plot_data,
        "load_game",
        lambda folder: loaded.append(folder) or SimpleNamespace(name=folder.name),
    )
    monkeypatch.setattr(plot_data, "draw_analysis", lambda *a: [])
    monkeypatch.setattr(execute, "analysis_tracker", lambda *a: None)
    node = SimpleNamespace(
        deps={"a": SimpleNamespace(hash="a"), "b": SimpleNamespace(hash="b")}
    )
    ctx = SimpleNamespace(
        outputs={"a": Output(path=str(given_dir)), "b": Output(path=str(given_file))},
        store=SimpleNamespace(
            folder=lambda n: tmp_path / "store", write_analysis=lambda *a: None
        ),
    )
    execute.run_analysis(node, ctx)  # type: ignore[arg-type]
    assert loaded == [given_dir, given_file.parent]
    node.deps = {"c": SimpleNamespace(hash="c", id="game.none.sft")}
    ctx.outputs["c"] = Output(path="wandb:e/p/game:v0")
    with pytest.raises(FileNotFoundError, match="game.none.sft: no game at wandb:"):
        execute.run_analysis(node, ctx)  # type: ignore[arg-type]
