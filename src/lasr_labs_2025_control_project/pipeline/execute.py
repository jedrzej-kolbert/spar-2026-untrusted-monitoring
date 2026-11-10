"""Running one node: build the stage's run settings and tracker from the node.

Each ``run_<stage>`` reads only the node (its params, seed and dependencies'
outputs) and the parts of the config that don't change outputs (concurrency,
``allow_unlinked``), so a node's output is fully determined by its key.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from control_arena.eval_mode import EvalMode

from lasr_labs_2025_control_project.pipeline.graph import ConfigError, Node
from lasr_labs_2025_control_project.pipeline.schema import Config
from lasr_labs_2025_control_project.pipeline.store import Output, Store
from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    MonitorConfig,
    TransformInfo,
    TransformType,
)
from lasr_labs_2025_control_project.tracking import (
    PipelineContext,
    ResolvedInput,
    StageRun,
    prompt_digest,
    resolve_input,
)
from lasr_labs_2025_control_project.utils.basharena_views import ActionViewOptions
from lasr_labs_2025_control_project.utils.config_loader import (
    DatasetType,
    DataSplit,
    GeneratorRunConfig,
    MonitorRunConfig,
    RunGenerator,
    RunLabel,
    RunMonitors,
    TransformRunConfig,
    get_prompt_by_name,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASHARENA_RUNTIME = REPO_ROOT / "basharena_runtime"
#: measurement.monitors name -> the stage's monitor slot
MONITOR_SLOTS = {
    "untrusted_collusion": "untrusted_collusion_monitor",
    "trusted_suspiciousness": "trusted_backdoor_monitor",
    "untrusted_suspiciousness": "untrusted_backdoor_monitor",
}


@dataclass
class RunContext:
    cfg: Config
    store: Store
    pipeline: PipelineContext
    #: node hash -> output, for every node that is available
    outputs: dict[str, Output] = field(default_factory=dict)
    #: node hash -> where an imported output came from (pipeline.ingest)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: node hash -> the W&B artifact a step logged in this invocation
    artifacts: dict[str, Any] = field(default_factory=dict)

    def tracker(
        self,
        node: Node,
        *,
        config: dict[str, Any],
        tags: list[str],
        inputs: list[ResolvedInput] = (),  # type: ignore[assignment]
        models: list[str] = (),  # type: ignore[assignment]
    ) -> StageRun:
        source = self.sources.get(node.hash)
        if source is not None:
            config = {**config, "source": source}
            tags = [*tags, "imported"]
        return StageRun(
            node.stage,
            node.name,
            config=config,
            tags=[node.stage, *tags],
            inputs=list(inputs),
            models=list(models),
            allow_unlinked=self.cfg.allow_unlinked,
            pipeline=self.pipeline,
            node=node.record(),
        )

    def input(self, node: Node, role: str) -> ResolvedInput:
        output = self.outputs[node.deps[role].hash]
        assert output.path is not None, f"{node.deps[role].id} has no file output"
        return resolve_input(output.path)

    def condition_tags(self, node: Node) -> list[str]:
        return [
            self.cfg.setting.name,
            self.cfg.measurement.collusion_strategy,
            node.info.get("transform", ""),
            node.info.get("arm", ""),
        ]


def model_info(model: str, system_prompt: str, base_prompt: str) -> ModelInfo:
    return ModelInfo(
        model=model,
        system_prompt=get_prompt_by_name(system_prompt),
        base_prompt=get_prompt_by_name(base_prompt),
    )


# -- generate ----------------------------------------------------------------


def generate_tracker(node: Node, ctx: RunContext) -> StageRun:
    p = node.params
    try:
        digest = prompt_digest(get_prompt_by_name(p["system_prompt"]))
    except ValueError:  # BashArena's policy prompts live in its runtime
        digest = None
    return ctx.tracker(
        node,
        config={
            "label": p["label"],
            "setting": p["setting"],
            "split": p["split"],
            "model": p["model"],
            "prompt": p["system_prompt"],
            "prompt_digest": digest,
        },
        tags=[p["label"], p["setting"], p["split"]],
    )


def run_generate(node: Node, ctx: RunContext) -> Output:
    p = node.params
    out = ctx.store.output_path(node)
    tracker = generate_tracker(node, ctx)
    if p["setting"] == "basharena":
        _generate_basharena(node, out, tracker, ctx)
        return Output(path=str(out))

    from lasr_labs_2025_control_project.scripts.generate_solutions_inspect import (
        run_generation,
    )

    system_prompt = get_prompt_by_name(p["system_prompt"])
    run = GeneratorRunConfig(
        label=p["label"],
        save_path=out,
        limit=p["limit"],
        generator=RunGenerator(
            model_info=ModelInfo(
                model=p["model"],
                system_prompt=system_prompt,
                base_prompt="{problem_statement}",
            ),
            eval_mode=EvalMode(p["eval_mode"]),
            data_split=DataSplit(p["split"]),
            dataset_type=DatasetType(p["setting"]),
            max_tokens=p["max_tokens"],
        ),
    )
    run_generation(
        run,
        tracker,
        problem_ids=p["problem_ids"],
        seed=node.seed,
        temperature=p["temperature"],
        max_connections=ctx.cfg.generation.max_connections,
    )
    return Output(path=str(out))


def basharena_runtime_config(params: dict[str, Any]) -> dict[str, Any]:
    """The generator config ``basharena_runtime.generate`` reads, for one node."""
    if params["limit"] is None:
        raise ConfigError("BashArena generation needs a limit")
    return {
        "defaults": {
            "data_split": params["split"],
            "dataset_type": "basharena",
            "limit": params["limit"],
            "max_tokens": params["max_tokens"],
        },
        "runs": {
            params["label"]: {
                "eval_mode": params["eval_mode"],
                "system_prompt": params["system_prompt"],
                "model": params["model"],
            }
        },
    }


def _generate_basharena(
    node: Node, out: Path, tracker: StageRun, ctx: RunContext
) -> None:
    """Generate in basharena_runtime/ (its own environment and control-arena)."""
    from inspect_ai.log import read_eval_log

    p = node.params
    label = p["label"]
    runtime_config = basharena_runtime_config(p)
    staging = out.parent / "runtime"
    shutil.rmtree(staging, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tracker, tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "generator.yaml"
        config_path.write_text(yaml.safe_dump(runtime_config))
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-m",
                "basharena_runtime.generate",
                "--config",
                str(config_path),
                "--output",
                str(staging.resolve()),
                "--concurrency",
                str(ctx.cfg.setting.generation_concurrency),
                "--metadata-json",
                json.dumps(tracker.eval_metadata()),
                "--seed",
                str(node.seed),
                "--allow-paid",
            ],
            cwd=BASHARENA_RUNTIME,
            check=True,
        )
        shutil.copy(staging / f"{label}.eval", out)
        log = read_eval_log(str(out))
        tracker.finish_eval(
            log, out, limit=p["limit"], metadata={"label": label, "split": p["split"]}
        )
    shutil.rmtree(staging, ignore_errors=True)


# -- transform ---------------------------------------------------------------


def transform_tracker(node: Node, ctx: RunContext, source: ResolvedInput) -> StageRun:
    spec = node.params["transform"]
    digest = (
        prompt_digest(get_prompt_by_name(spec["system_prompt"]))
        if spec["type"] == "llm"
        else None
    )
    return ctx.tracker(
        node,
        config={
            "label": node.info["label"],
            "transform": spec["type"],
            "prompt_digest": digest,
        },
        tags=[node.info["label"], spec["type"]],
        inputs=[source],
    )


def run_transform(node: Node, ctx: RunContext) -> Output:
    from lasr_labs_2025_control_project.scripts.transform_solutions_inspect import (
        run_transform as transform,
    )

    p, spec = node.params, node.params["transform"]
    source = ctx.input(node, "input")
    info = (
        model_info(spec["model"], spec["system_prompt"], spec["base_prompt"])
        if spec["type"] == "llm"
        else None
    )
    out = ctx.store.output_path(node)
    run = TransformRunConfig(
        label=node.info["label"],
        save_path=out,
        limit=None,
        eval_path=source.local_path,
        transform=TransformInfo(
            type=TransformType(spec["type"]),
            model_info=info,
            remove_comments=spec["remove_comments"],
        ),
    )
    tracker = transform_tracker(node, ctx, source)
    transform(
        run,
        tracker,
        seed=node.seed if spec["type"] == "llm" else None,
        temperature=p.get("temperature"),
        max_connections=ctx.cfg.generation.transform_max_connections,
    )
    return Output(path=str(out))


# -- finetune ----------------------------------------------------------------


def training_data(
    node: Node, train: list[ResolvedInput]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A fine-tune's examples and their composition record (deterministic)."""
    from lasr_labs_2025_control_project.finetuning.sft_data import build_training_data

    p, hyper = node.params, node.params["training"]
    action = p["view"] == "action"
    return build_training_data(
        [i.local_path for i in train],
        target_model=p["target_model"],
        transformed=p["transformed"],
        keep_comments=hyper["keep_comments"],
        system_prompt=p["system_prompt"],
        user_prompt=p["user_prompt"],
        seed=node.seed,
        upsample_pos_factor=hyper["upsample_pos_factor"],
        upsample_neg_factor=hyper["upsample_neg_factor"],
        view=p["view"],
        action_options=(
            ActionViewOptions(**p["action_view"], seed=node.seed) if action else None
        ),
        labels=p["labels"],
    )


def finetune_tracker(
    node: Node,
    ctx: RunContext,
    inputs: list[ResolvedInput],
    composition: dict[str, Any],
) -> StageRun:
    p = node.params
    return ctx.tracker(
        node,
        config={
            "backend": p["backend"],
            "base_model": p["base_model"],
            "checkpoint_name": p["checkpoint_name"],
            "max_examples": p["max_examples"],
            "seed": node.seed,
            **p["training"],
            "training_data": composition,
        },
        tags=[
            p["backend"],
            *(f"pos-{m}" for m in composition.get("positive_eval_modes", [])),
            *(f"neg-{m}" for m in composition.get("negative_eval_modes", [])),
            *(["permuted-labels"] if p["labels"] == "permuted" else []),
        ],
        inputs=inputs,
    )


def run_finetune(node: Node, ctx: RunContext) -> Output:
    p = node.params
    train = [ctx.input(node, "positives"), ctx.input(node, "negatives")]
    examples, composition = training_data(node, train)
    minimum = ctx.cfg.elicitation.min_examples_per_label
    if min(composition["n_positive"], composition["n_negative"]) < minimum:
        raise ConfigError(
            f"{node.id}: {composition['n_positive']} positive and "
            f"{composition['n_negative']} negative usable examples, fewer than "
            f"elicitation.min_examples_per_label={minimum}; raise the limits"
        )
    evals = [
        ctx.input(node, r) for r in ("eval_attacks", "eval_honeypots") if r in node.deps
    ]
    tracker = finetune_tracker(node, ctx, train + evals, composition)
    usage: dict[str, int] = {}
    if p["backend"] == "tinker":
        model = _finetune_tinker(node, examples, evals, tracker, usage)
    else:
        model = _finetune_openai(node, examples, tracker, usage)
    return ctx.store.write_model(node, model, usage)


def _finetune_tinker(
    node: Node,
    examples: list[dict[str, Any]],
    evals: list[ResolvedInput],
    tracker: StageRun,
    usage: dict[str, int],
) -> str:
    from lasr_labs_2025_control_project.finetuning.sft_data import load_eval_examples
    from lasr_labs_2025_control_project.finetuning.tinker_lora_finetune import finetune

    p, hyper = node.params, node.params["training"]
    eval_examples = (
        load_eval_examples(
            evals[0].local_path,
            evals[1].local_path,
            "transformed_solution" if p["transformed"] else "base_solution",
        )
        if evals
        else None
    )
    with tracker:
        path = finetune(
            examples,
            base_model=p["base_model"],
            rank=hyper["rank"],
            lr=hyper["lr"],
            epochs=hyper["epochs"],
            batch_size=hyper["batch_size"],
            checkpoint_name=p["checkpoint_name"],
            max_examples=p["max_examples"],
            seed=node.seed,
            val_frac=hyper["val_frac"],
            checkpoint_every_steps=hyper["checkpoint_every_steps"],
            eval_examples=eval_examples,
            tracker=tracker,
            usage=usage,
        )
        model = f"tinker/{path}"
        tracker.log_model_ref(
            model, {"backend": "tinker", "base_model": p["base_model"]}
        )
    return model


def _finetune_openai(
    node: Node,
    examples: list[dict[str, Any]],
    tracker: StageRun,
    usage: dict[str, int],
) -> str:
    from lasr_labs_2025_control_project.finetuning.openai_finetune import (
        submit,
        validate_jsonl,
    )

    p = node.params
    if p["max_examples"] is not None:
        examples = examples[: p["max_examples"]]
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train_messages.jsonl"
        train_path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
        )
        validate_jsonl(train_path)
        with tracker:
            model: Optional[str] = submit(
                train_path,
                suffix=p["checkpoint_name"].replace("_", "-"),
                base_model=p["base_model"],
                n_epochs=p["training"]["n_epochs"],
                validation_jsonl=None,
                wait=True,
                tracker=tracker,
                usage=usage,
            )
    assert model is not None
    return model


# -- monitor -----------------------------------------------------------------


def monitor_models(node: Node, ctx: RunContext) -> dict[str, str]:
    """Each monitor's model, with a fine-tuned collusion monitor resolved."""
    models = {}
    for name, entry in node.params["monitors"].items():
        model = entry["model"]
        if name == "untrusted_collusion" and "collusion_model" in node.deps:
            model = ctx.outputs[node.deps["collusion_model"].hash].model
        assert model is not None
        models[name] = model
    return models


def monitor_tracker(node: Node, ctx: RunContext, source: ResolvedInput) -> StageRun:
    p = node.params
    models = monitor_models(node, ctx)
    return ctx.tracker(
        node,
        config={
            "label": p["label"],
            "monitors": {
                name: {**entry, "model": models[name]}
                for name, entry in p["monitors"].items()
            },
            "prompt_digests": {
                name: prompt_digest(get_prompt_by_name(entry["system_prompt"]))
                for name, entry in p["monitors"].items()
            },
        },
        tags=[p["label"], *ctx.condition_tags(node)],
        inputs=[source],
        models=list(models.values()),
    )


def run_monitor(node: Node, ctx: RunContext) -> Output:
    from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
        run_monitor as monitor,
    )

    p = node.params
    source = ctx.input(node, "input")
    configs: dict[str, Optional[MonitorConfig]] = dict.fromkeys(
        (*MONITOR_SLOTS.values(), "trusted_collusion_monitor")
    )
    for name, model in monitor_models(node, ctx).items():
        entry = p["monitors"][name]
        configs[MONITOR_SLOTS[name]] = MonitorConfig(
            model_info=model_info(model, entry["system_prompt"], entry["base_prompt"]),
            transformed=entry["transformed"],
            remove_comments=entry["remove_comments"],
        )
    out = ctx.store.output_path(node)
    action = p["action_view"]
    run = MonitorRunConfig(
        label=RunLabel(p["label"]),
        save_path=out,
        limit=p["limit"],
        eval_path=source.local_path,
        monitors=RunMonitors(**configs),
        view=p["view"],
        action_options=ActionViewOptions(**action) if action else None,
    )
    tracker = monitor_tracker(node, ctx, source)
    monitor(
        run,
        tracker,
        seed=node.seed,
        temperature=p["temperature"],
        max_connections=ctx.cfg.generation.max_connections,
    )
    return Output(path=str(out))


# -- game ----------------------------------------------------------------------


def game_tracker(
    node: Node, ctx: RunContext, inputs: dict[str, ResolvedInput]
) -> StageRun:
    p = node.params
    return ctx.tracker(
        node,
        config={
            "condition": p["condition"],
            "variants": p["variants"],
            "game_parameters": p["game"],
            "regeneration_cutoffs": p["regeneration_cutoffs"],
            "solver": p["solver"],
            "max_iterations": p["max_iterations"],
            "bootstrap_replicates": p["bootstrap"]["replicates"],
            "bootstrap_ci": p["bootstrap"]["ci"],
        },
        tags=list(p["condition"].values()),
        inputs=list(inputs.values()),
    )


def analysis_record(node: Node, ctx: RunContext) -> dict[str, Any]:
    """What an analysis payload stores about the step that produced it."""
    models = ctx.cfg.models
    return {
        **node.record(),
        "experiment": ctx.cfg.name,
        "models": {"U": models.U, "T": models.T, "H": models.H},
    }


def run_game(node: Node, ctx: RunContext) -> Output:
    from lasr_labs_2025_control_project.scripts.plot_data import (
        BootstrapSettings,
        SolveSettings,
        solve_game,
    )

    p = node.params
    inputs = {label: ctx.input(node, label) for label in node.deps}
    bootstrap = ctx.cfg.optimization.bootstrap
    tracker = game_tracker(node, ctx, inputs)
    folder = ctx.store.folder(node)
    game = solve_game(
        name=node.name,
        inputs={label: i.local_path for label, i in inputs.items()},
        variants=p["variants"],
        solve=SolveSettings(
            game=p["game"],
            regeneration_cutoffs=p["regeneration_cutoffs"],
            solver=p["solver"],
            max_iterations=p["max_iterations"],
        ),
        bootstrap=BootstrapSettings(
            replicates=p["bootstrap"]["replicates"],
            ci=p["bootstrap"]["ci"],
            n_jobs=bootstrap.n_jobs,
            chunk_size=bootstrap.chunk_size,
        ),
        seed=node.seed,
        out_dir=folder,
        tracker=tracker,
        record=analysis_record(node, ctx),
        min_samples=p["min_samples_per_label"],
    )
    if game.artifact is not None:
        ctx.artifacts[node.hash] = game.artifact
    return Output(path=str(folder))


# -- analysis ----------------------------------------------------------------


def step_artifact(node: Node, ctx: RunContext) -> Optional[ResolvedInput]:
    """The W&B artifact holding a step's output, as an input (None if unknown):
    logged in this invocation, or found on W&B by the step's key."""
    from lasr_labs_2025_control_project.pipeline.store import ARTIFACT_TYPES
    from lasr_labs_2025_control_project.tracking import (
        artifact_input,
        available,
        find_pipeline_run,
    )

    artifact = ctx.artifacts.get(node.hash)
    if artifact is None and ctx.cfg.reuse == "wandb" and available():
        run = find_pipeline_run(node.hash, node.key)
        wanted = ARTIFACT_TYPES.get(node.stage)
        if run is not None:
            artifact = next(
                (a for a in run.logged_artifacts() if a.type == wanted), None
            )
    return artifact_input(artifact) if artifact is not None else None


def analysis_tracker(node: Node, ctx: RunContext) -> StageRun:
    inputs = [i for dep in node.deps.values() if (i := step_artifact(dep, ctx))]
    return ctx.tracker(
        node,
        config={
            "experiment": node.params["experiment"],
            "conditions": sorted(dep.name for dep in node.deps.values()),
        },
        tags=[ctx.cfg.setting.name, ctx.cfg.measurement.collusion_strategy],
        inputs=inputs,
    )


def run_analysis(node: Node, ctx: RunContext) -> Output:
    from lasr_labs_2025_control_project.scripts.plot_data import (
        draw_analysis,
        load_game,
    )

    games = {}
    for dep in node.deps.values():
        game = load_game(ctx.store.folder(dep))
        games[game.name] = game
    folder = ctx.store.folder(node)
    folder.mkdir(parents=True, exist_ok=True)
    figures = draw_analysis(games, folder, analysis_tracker(node, ctx))
    return ctx.store.write_analysis(node, figures)


EXECUTORS = {
    "generate": run_generate,
    "transform": run_transform,
    "finetune": run_finetune,
    "monitor": run_monitor,
    "game": run_game,
    "analysis": run_analysis,
}
