"""Import existing outputs into the pipeline's store and W&B, as pipeline steps.

    # .eval files from anywhere (e.g. one experiment's folder from the team
    # Drive), with the pipeline experiment they belong to:
    python -m lasr_labs_2025_control_project.pipeline.ingest <dir> experiment=paper/apps_causal [overrides]

    # a pipeline run made on this machine (re-log it to the current project):
    python -m lasr_labs_2025_control_project.pipeline.ingest experiments/runs/<experiment>/<time>

Add ``--apply`` to write; without it the command only prints what it would do.

Each imported output becomes the output of the step of that experiment that
would have produced it, with the same key: it is copied into the step's
folder, logged as one W&B run of the step's stage with the config, count trail
and artifact a pipeline run logs, and linked to its inputs. Imported runs are
tagged ``imported`` and record their ``source`` (path, md5, original date).
The pipeline then treats imported steps as done, and ``reuse=wandb`` finds
them, exactly like steps it ran itself; running the pipeline on the
experiment afterwards does the analyses, which cost nothing.

Matching a directory's files: a file's stage, models, prompts (by text),
eval mode, split (from problem-id parity), transform, monitors and limits are
read from its contents and must equal what the experiment's config gives that
step; a transform or monitor must also contain exactly its matched input's
solutions. Seeds aren't recorded in old files, so an imported step's key has
the declared config's seed. Files that are incomplete, or match no step (the
pipeline would never log them), are listed and skipped; so are steps already
in the W&B project, which are downloaded instead.

For a run folder, every finished step is re-logged, fine-tunes with the
training curves and summary recovered from the run's local W&B files, and
analyses from their saved payloads and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from lasr_labs_2025_control_project.pipeline.execute import (
    RunContext,
    analyse_tracker,
    finetune_tracker,
    generate_tracker,
    monitor_tracker,
    training_data,
    transform_tracker,
)
from lasr_labs_2025_control_project.pipeline.graph import (
    Graph,
    Node,
    build_graph,
    provider_free,
)
from lasr_labs_2025_control_project.pipeline.schema import Config, register
from lasr_labs_2025_control_project.pipeline.store import EVAL_STAGES, Output, Store
from lasr_labs_2025_control_project.tracking import (
    PipelineContext,
    ResolvedInput,
    find_pipeline_run,
    input_counts,
    mode,
    resolve_input,
)
from lasr_labs_2025_control_project.tracking.local_runs import (
    find_local_run,
    read_local_run,
)
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name

REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "conf"
#: store key -> name in measurement.monitors
_MONITOR_KEYS = {
    "untrusted_collusion_score_metadata": "untrusted_collusion",
    "trusted_suspiciousness_score_metadata": "trusted_suspiciousness",
    "untrusted_suspiciousness_score_metadata": "untrusted_suspiciousness",
}
_SCORE_KEYS = (
    "untrusted_collusion_score",
    "trusted_suspiciousness_score",
    "untrusted_suspiciousness_score",
)
_WILDCARD = "<fine-tuned>"


# -- reading files ---------------------------------------------------------------


@dataclass
class EvalFile:
    path: Path
    log: Any
    stage: str
    fields: dict[str, Any]
    #: sample key -> (base_solution, transformed_solution)
    content: dict[str, tuple[Any, Any]]
    md5: str

    @property
    def created(self) -> str:
        return str(self.log.eval.created)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _setting(log: Any) -> Optional[str]:
    scorers = set((log.samples[0].scores or {}).keys())
    if {"main_task_success", "side_task_success"} <= scorers:
        return "basharena"
    if "apps_scorer" in scorers or log.eval.task.startswith("backdoored"):
        return "apps"
    if scorers & {"bigcodebench_scorer", "verify"} or "bigcodebench" in log.eval.task:
        return "bigcodebench"
    return None


def _number(sample_id: Any) -> Optional[int]:
    text = str(sample_id).split("#", 1)[0]
    digits = "".join(ch for ch in text.rsplit("/", 1)[-1] if ch.isdigit())
    return int(digits) if digits else None


def _split(samples: list[Any]) -> Optional[str]:
    """Train uses odd problem ids and test even ones, in every setting."""
    parities = {n % 2 for s in samples if (n := _number(s.id)) is not None}
    return {frozenset({1}): "train", frozenset({0}): "test"}.get(frozenset(parities))


def _key(sample: Any) -> str:
    data = {**sample.metadata, **sample.store}
    return str(data.get("trajectory_id", sample.id))


def read_eval_file(path: Path) -> tuple[Optional[EvalFile], str]:
    """The file's stage and the fields that identify its step; or a reason to skip it."""
    from inspect_ai.log import read_eval_log

    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if log.status != "success":
        return None, f"status {log.status!r} (incomplete run)"
    if not log.samples:
        return None, "no samples"
    first = {**log.samples[0].metadata, **log.samples[0].store}
    content = {}
    for sample in log.samples:
        data = {**sample.metadata, **sample.store}
        content[_key(sample)] = (
            data.get("base_solution"),
            data.get("transformed_solution"),
        )
    if any(key in first for key in _SCORE_KEYS):
        monitors = {}
        for key, name in _MONITOR_KEYS.items():
            info = first.get(key)
            if not isinstance(info, dict):
                continue
            model_info = info.get("model_info") or {}
            monitors[name] = {
                "model": model_info.get("model"),
                "system_prompt": model_info.get("system_prompt"),
                "base_prompt": model_info.get("base_prompt"),
                "transformed": bool(info.get("transformed")),
                "remove_comments": info.get("remove_comments"),
            }
        # the honest label scores collusion accuracy "N" (not applicable)
        accuracy = {
            getattr((s.scores or {}).get("untrusted_collusion_accuracy"), "value", None)
            for s in log.samples
        }
        fields = {
            "monitors": monitors,
            "view": "action"
            if first.get("solution_view") == "action"
            else "trajectory",
            "limit": log.eval.config.limit,
            "honest_label": accuracy == {"N"}
            if "untrusted_collusion" in monitors
            else None,
        }
        return EvalFile(path, log, "monitor", fields, content, _md5(path)), ""
    if first.get("transform_info") and "transformed_solution" in first:
        info = first["transform_info"]
        model_info = info.get("model_info") or {}
        fields = {
            "type": str(info.get("type")),
            "model": model_info.get("model"),
            "system_prompt": model_info.get("system_prompt"),
            "base_prompt": model_info.get("base_prompt"),
            "remove_comments": bool(info.get("remove_comments")),
        }
        return EvalFile(path, log, "transform", fields, content, _md5(path)), ""
    if "base_solution" in first:
        generator = first.get("generator_info") or {}
        model_info = generator.get("model_info") or {}
        fields = {
            "setting": _setting(log),
            "split": _split(log.samples),
            "eval_mode": str(generator.get("eval_mode") or first.get("eval_mode")),
            # generator_info records the model without its provider prefix
            "model": model_info.get("model") or provider_free(str(log.eval.model)),
            "system_prompt": model_info.get("system_prompt"),
            "limit": log.eval.config.limit,
            "max_tokens": log.plan.config.max_tokens,
        }
        return EvalFile(path, log, "generate", fields, content, _md5(path)), ""
    return None, "not a generation, transform or monitor log"


# -- what a step's output must look like ---------------------------------------


def _prompt(name: Optional[str]) -> Optional[str]:
    """A prompt's text; None (not compared) for prompts outside the registry."""
    if name is None:
        return None
    try:
        return get_prompt_by_name(name)
    except ValueError:
        return None


def expected_fields(node: Node, ctx: RunContext) -> dict[str, Any]:
    p = node.params
    if node.stage == "generate":
        return {
            "setting": p["setting"],
            "split": p["split"],
            "eval_mode": p["eval_mode"],
            "model": provider_free(p["model"]),
            "system_prompt": _prompt(p["system_prompt"]),
            "limit": p["limit"],
            "max_tokens": p["max_tokens"],
        }
    if node.stage == "transform":
        spec = p["transform"]
        llm = spec["type"] == "llm"
        return {
            "type": spec["type"],
            "model": provider_free(spec["model"]) if llm else None,
            "system_prompt": _prompt(spec["system_prompt"]) if llm else None,
            "base_prompt": _prompt(spec["base_prompt"]) if llm else None,
            # the minifier strips comments itself and records False regardless
            "remove_comments": spec["remove_comments"] if llm else None,
        }
    monitors = {}
    for name, entry in p["monitors"].items():
        model = entry["model"]
        if name == "untrusted_collusion" and "collusion_model" in node.deps:
            output = ctx.outputs.get(node.deps["collusion_model"].hash)
            model = output.model if output and output.model else _WILDCARD
        monitors[name] = {
            "model": model,
            "system_prompt": _prompt(entry["system_prompt"]),
            "base_prompt": _prompt(entry["base_prompt"]),
            "transformed": entry["transformed"],
            # not recorded for the collusion monitor
            "remove_comments": None
            if name == "untrusted_collusion"
            else entry["remove_comments"],
        }
    return {
        "monitors": monitors,
        "view": p["view"],
        "limit": p["limit"],
        "honest_label": (p["label"] == "honest")
        if "untrusted_collusion" in monitors
        else None,
    }


def _is_finetuned(model: Any) -> bool:
    return isinstance(model, str) and ("tinker://" in model or "ft:" in model)


def differences(expected: Any, observed: Any, path: str = "") -> list[str]:
    """Fields where a file differs from the step; None in ``expected`` means not compared."""
    if expected is None:
        return []
    if expected == _WILDCARD:
        return [] if _is_finetuned(observed) else [path]
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            return [path or "keys"]
        out: list[str] = []
        for key in expected:
            out += differences(
                expected[key], observed[key], f"{path}.{key}".lstrip(".")
            )
        return out
    if isinstance(expected, str) and isinstance(observed, str):
        same = expected.strip() == observed.strip()
    else:
        same = expected == observed
    return [] if same else [path]


def reads_input(file: EvalFile, source: EvalFile) -> bool:
    """Whether ``file`` was made from exactly ``source``'s solutions."""
    if not set(file.content) <= set(source.content):
        return False
    action = file.fields.get("view") == "action"
    for key, (base, transformed) in file.content.items():
        src_base, src_transformed = source.content[key]
        if action:
            continue  # per-action logs rewrite base_solution per action
        if base != src_base or (
            src_transformed is not None and transformed != src_transformed
        ):
            return False
    return True


# -- matching a directory --------------------------------------------------------


@dataclass
class Match:
    #: node hash -> file
    files: dict[str, EvalFile] = field(default_factory=dict)
    #: node hash -> fine-tuned model id read from a monitor log
    models: dict[str, str] = field(default_factory=dict)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def match_directory(graph: Graph, directory: Path, ctx: RunContext) -> Match:
    match = Match()
    files: list[EvalFile] = []
    for path in sorted(directory.rglob("*.eval")):
        file, reason = read_eval_file(path)
        if file is None:
            match.skipped.append((path, reason))
        else:
            files.append(file)
    used: set[int] = set()
    closest: dict[int, tuple[int, str]] = {}
    for node in graph.nodes:
        if node.stage not in EVAL_STAGES or node.external is not None:
            continue
        source = (
            match.files.get(node.deps["input"].hash) if "input" in node.deps else None
        )
        if "input" in node.deps and source is None:
            continue  # its input wasn't imported, so neither is it
        expected = expected_fields(node, ctx)
        candidates = []
        for index, file in enumerate(files):
            if file.stage != node.stage:
                continue
            diff = differences(expected, file.fields)
            if source is not None and not reads_input(file, source):
                diff.append("input")
            if not diff:
                candidates.append(index)
            elif len(diff) < closest.get(index, (99, ""))[0]:
                closest[index] = (
                    len(diff),
                    f"closest step {node.id} differs in {', '.join(diff)}",
                )
        distinct = {files[i].md5 for i in candidates}
        if len(distinct) > 1:
            raise ValueError(
                f"{node.id}: several different files match: "
                + ", ".join(str(files[i].path) for i in candidates)
            )
        if candidates:
            match.files[node.hash] = files[candidates[0]]
            used.update(candidates)
            collusion = node.deps.get("collusion_model")
            if collusion is not None and collusion.hash not in ctx.outputs:
                # The fine-tune's output is the model this log was scored with;
                # the arm's other monitor logs must then name the same model.
                model = files[candidates[0]].fields["monitors"]["untrusted_collusion"][
                    "model"
                ]
                match.models[collusion.hash] = model
                ctx.outputs[collusion.hash] = Output(model=model)
    for index, file in enumerate(files):
        if index not in used:
            reason = closest.get(
                index, (0, f"no {file.stage} step in this experiment")
            )[1]
            match.skipped.append((file.path, reason))
    return match


# -- logging ---------------------------------------------------------------------


def _inputs(node: Node, ctx: RunContext, roles: tuple[str, ...]) -> list[ResolvedInput]:
    return [
        resolve_input(str(ctx.outputs[node.deps[r].hash].path))
        for r in roles
        if r in node.deps and node.deps[r].hash in ctx.outputs
    ]


def log_eval(node: Node, ctx: RunContext, log: Any, source_path: Path) -> Output:
    """Log an Inspect output as the node's step, and keep it in the node's folder."""
    from inspect_ai.log import write_eval_log

    inputs = _inputs(node, ctx, ("input",))
    if node.stage == "generate":
        tracker = generate_tracker(node, ctx)
    elif node.stage == "transform":
        tracker = transform_tracker(node, ctx, inputs[0])
    else:
        tracker = monitor_tracker(node, ctx, inputs[0])
    out = ctx.store.output_path(node)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tracker:
        log.eval.metadata = {**(log.eval.metadata or {}), **tracker.eval_metadata()}
        write_eval_log(log, str(out))
        extra = (
            input_counts(inputs[0].local_path, log.eval.dataset.samples or 0)
            if inputs
            else None
        )
        tracker.finish_eval(
            log,
            out,
            limit=node.params.get("limit"),
            metadata={
                "label": node.info.get("label"),
                "imported_from": str(source_path),
            },
            extra_counts=extra,
        )
    return Output(path=str(out))


def log_finetune(
    node: Node, ctx: RunContext, model: str, usage: dict[str, int]
) -> Output:
    """Log a fine-tune's model as the node's step (training curves if recoverable)."""
    train = _inputs(node, ctx, ("positives", "negatives"))
    composition = training_data(node, train)[1] if len(train) == 2 else {}
    tracker = finetune_tracker(
        node,
        ctx,
        train + _inputs(node, ctx, ("eval_attacks", "eval_honeypots")),
        composition,
    )
    local = find_local_run(node.hash)
    with tracker:
        if local is not None:
            run = read_local_run(local)
            tracker.define_curve("train/loss", "train/step")
            tracker.define_curve("checkpoint/roc_auc", "train/step")
            for row in run.history:
                tracker.log({k: v for k, v in row.items() if not k.startswith("_")})
            tracker.summary(
                {k: v for k, v in run.summary.items() if not k.startswith("_")}
            )
        tracker.log_model_ref(
            model,
            {
                "backend": node.params["backend"],
                "base_model": node.params["base_model"],
            },
        )
        tracker.finish(complete=True)
    return ctx.store.write_model(node, model, usage)


def log_analysis(node: Node, ctx: RunContext) -> Output:
    """Re-log a saved analysis (curves, strategies, figures, files) as the node's step."""
    from lasr_labs_2025_control_project.scripts.plot_data import (
        bootstrap_path,
        log_payload_overview,
        log_strategies,
        log_variant,
        payload_path,
        read_replicates,
    )
    from lasr_labs_2025_control_project.utils.numerical_pipeline.results_io import (
        load_analysis_payload,
    )

    inputs = {
        label: resolve_input(str(ctx.outputs[dep.hash].path))
        for label, dep in node.deps.items()
    }
    tracker = analyse_tracker(node, ctx, inputs)
    folder = ctx.store.folder(node)
    files: list[Path] = []
    with tracker:
        for index, variant in enumerate(node.params["variants"]):
            path = payload_path(folder, index, variant)
            payload = load_analysis_payload(path)
            files.append(path)
            replicates = read_replicates(bootstrap_path(folder, variant))
            if replicates:
                files.append(bootstrap_path(folder, variant))
            log_variant(
                tracker, variant, payload, replicates, node.params["bootstrap"]["ci"]
            )
            log_strategies(tracker, variant, payload)
            if index == 0:
                log_payload_overview(tracker, payload)
        for image in sorted((folder / "figures").glob("*.png")):
            tracker.log_image(f"figures/{image.stem}", image)
        tracker.log_artifact(
            files, type="analysis", metadata={"variants": node.params["variants"]}
        )
        tracker.finish(complete=True)
    return Output(path=str(folder))


def already_logged(node: Node, md5: Optional[str]) -> Optional[str]:
    """Why to skip uploading: the W&B project already has this step (None if not)."""
    if mode() != "online":
        return None
    run = find_pipeline_run(node.hash, node.key)
    if run is None:
        return None
    theirs = (run.config.get("source") or {}).get("md5")
    if md5 and theirs and theirs != md5:
        return f"W&B already has a different file for this step (run {run.id}); skipped"
    return f"already in W&B (run {run.id})"


# -- the two sources ---------------------------------------------------------------


@dataclass
class Plan:
    #: (node, how its output is provided, source record)
    steps: list[tuple[Node, str, dict[str, Any]]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def plan_directory(
    graph: Graph, directory: Path, ctx: RunContext
) -> tuple[Plan, Match]:
    match = match_directory(graph, directory, ctx)
    plan = Plan(skipped=[(str(p), why) for p, why in match.skipped])
    for node in graph.nodes:
        file = match.files.get(node.hash)
        if file is not None:
            plan.steps.append(
                (
                    node,
                    "file",
                    {"path": str(file.path), "md5": file.md5, "created": file.created},
                )
            )
        elif node.hash in match.models:
            plan.steps.append((node, "model", {"model": match.models[node.hash]}))
    return plan, match


def plan_run(graph: Graph, ctx: RunContext, run_dir: Path) -> Plan:
    plan = Plan()
    for node in graph.nodes:
        if node.external is not None:
            continue
        try:
            output = ctx.store.done(node)
        except Exception as exc:  # a collision is reported, never re-logged
            plan.skipped.append((node.id, f"{type(exc).__name__}: {exc}"))
            continue
        if output is None:
            plan.skipped.append((node.id, "no finished output"))
            continue
        path = ctx.store.output_path(node)
        record: dict[str, Any] = {"path": str(path), "run": str(run_dir)}
        if node.stage in EVAL_STAGES:
            from inspect_ai.log import read_eval_log

            record["md5"] = _md5(path)
            record["created"] = str(
                read_eval_log(str(path), header_only=True).eval.created
            )
        else:
            local = find_local_run(node.hash)
            record["created"] = read_local_run(local).started if local else None
        plan.steps.append((node, "store", record))
    return plan


def apply(plan: Plan, ctx: RunContext, match: Optional[Match] = None) -> None:
    for node, how, record in plan.steps:
        ctx.sources[node.hash] = record
        reason = already_logged(node, record.get("md5"))
        if reason is not None:
            output = ctx.store.done(node) or ctx.store.fetch(node)
            if output is not None:
                ctx.outputs[node.hash] = output
            print(f"  {node.id:45} skip: {reason}")
            continue
        if how == "file":
            assert match is not None
            ctx.outputs[node.hash] = log_eval(
                node, ctx, match.files[node.hash].log, Path(record["path"])
            )
        elif how == "model":
            ctx.outputs[node.hash] = log_finetune(node, ctx, record["model"], {})
        elif node.stage in EVAL_STAGES:
            from inspect_ai.log import read_eval_log

            path = ctx.store.output_path(node)
            ctx.outputs[node.hash] = log_eval(node, ctx, read_eval_log(str(path)), path)
        elif node.stage == "finetune":
            done = ctx.store.done(node)
            assert done is not None and done.model is not None
            usage = _model_usage(ctx.store.output_path(node))
            ctx.outputs[node.hash] = log_finetune(node, ctx, done.model, usage)
        else:
            ctx.outputs[node.hash] = log_analysis(node, ctx)
        print(f"  {node.id:45} logged")


def _model_usage(path: Path) -> dict[str, int]:
    import json

    return dict(json.loads(path.read_text()).get("usage") or {})


# -- entrypoint --------------------------------------------------------------------


def load_config(source: Path, overrides: list[str]) -> tuple[Config, Path, bool]:
    """(config, folder holding .hydra/, whether ``source`` is a pipeline run folder)."""
    from hydra import compose, initialize_config_dir

    register()
    saved = source / ".hydra" / "config.yaml"
    if saved.exists():
        merged = OmegaConf.merge(
            OmegaConf.structured(Config),
            OmegaConf.load(saved),
            OmegaConf.from_dotlist(overrides),
        )
        return OmegaConf.to_object(merged), source, True  # type: ignore[return-value]
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        raw = compose(config_name="config", overrides=overrides)
    cfg: Config = OmegaConf.to_object(raw)  # type: ignore[assignment]
    run_dir = (
        Path(cfg.experiments_dir)
        / "runs"
        / cfg.name
        / datetime.now().strftime("import_%Y-%m-%d_%H-%M-%S")
    )
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(raw, run_dir / ".hydra" / "config.yaml")
    OmegaConf.save(
        OmegaConf.create([*overrides, f"+ingest={source}"]),
        run_dir / ".hydra" / "overrides.yaml",
    )
    return cfg, run_dir, False


def main(argv: Optional[list[str]] = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "source", type=Path, help="a directory of .eval files, or a pipeline run folder"
    )
    parser.add_argument(
        "overrides", nargs="*", help="pipeline config overrides (key=value)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write and log (default: only print)"
    )
    args = parser.parse_args(argv)

    cfg, run_dir, is_run = load_config(args.source, args.overrides)
    graph = build_graph(cfg)
    store = Store(cfg.experiments_dir, cfg.setting.name)
    hydra_dir = run_dir / ".hydra"
    pipeline = PipelineContext(
        name=cfg.name,
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),  # type: ignore[arg-type]
        config_files=[
            p
            for p in (hydra_dir / "config.yaml", hydra_dir / "overrides.yaml")
            if p.exists()
        ],
        tags=[cfg.name, *cfg.wandb_tags],
    )
    os.environ.setdefault("WANDB_RUN_GROUP", f"{cfg.name}-{run_dir.name}")
    ctx = RunContext(cfg, store, pipeline)
    for node in graph.nodes:  # outputs given in `inputs` are available as-is
        if node.external is not None:
            ctx.outputs[node.hash] = Output(
                model=node.external if node.stage == "finetune" else None,
                path=None if node.stage == "finetune" else node.external,
            )

    match = None
    if is_run:
        plan = plan_run(graph, ctx, args.source)
    else:
        plan, match = plan_directory(graph, args.source, ctx)
    print(f"Import into {cfg.name} ({len(plan.steps)} step(s)):")
    for node, how, record in plan.steps:
        print(f"  {node.id:45} {how:6} {record.get('path') or record.get('model')}")
    if plan.skipped:
        print(f"\nSkipped ({len(plan.skipped)}):")
        for what, why in plan.skipped:
            print(f"  {what}: {why}")
    if not args.apply:
        print("\nNothing was written. Re-run with --apply to import.")
        if not is_run:
            shutil.rmtree(run_dir, ignore_errors=True)
        return
    print()
    apply(plan, ctx, match)


if __name__ == "__main__":
    main()
