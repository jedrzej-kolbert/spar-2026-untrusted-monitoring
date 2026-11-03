"""The stage graph: one node per output, keyed by everything that determines it.

A node's *key* is its stage, its own parameters and the hashes of the nodes it
reads; its *hash* (12 hex characters of the key's SHA-256) names its output
folder. Two experiments that need the same output therefore build the same
node and share the folder, and changing a setting re-runs only the nodes whose
key it enters. The full key is stored with each output and compared before
reuse, so a hash collision is reported, never silently reused.

Node ids (``generate.test.attacks``, ``finetune.self_rec.minified``, ...) are
the addresses printed in the plan and used by ``inputs`` overrides.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from lasr_labs_2025_control_project.pipeline.schema import (
    MONITORS,
    NO_TRANSFORM,
    SOLVERS,
    STAGES,
    VARIANTS,
    Config,
    ElicitationArm,
    MonitorArm,
)
from lasr_labs_2025_control_project.tracking import names

HASH_LENGTH = 12
LABELS = ("attacks", "honeypots", "honest")
SFT_PREFIX = "sft:"
#: training label for H's honest-mode solutions (label controls)
HONEST_HONEYPOTS = "honest_honeypots"


class ConfigError(ValueError):
    """The config describes an experiment that can't be built."""


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


@dataclass(eq=False)
class Node:
    stage: str
    id: str
    #: readable output name (tracking.names); W&B artifact and folder prefix
    name: str
    params: dict[str, Any]
    deps: dict[str, "Node"] = field(default_factory=dict)
    #: calls paid models
    paid: bool = False
    #: details for running and tagging that don't change the output
    info: dict[str, Any] = field(default_factory=dict)
    #: an existing output (path, wandb: ref or model id) standing in for this node
    external: Optional[str] = None

    @cached_property
    def key(self) -> dict[str, Any]:
        if self.external is not None:
            return {"stage": self.stage, "external": self.external}
        return {
            "stage": self.stage,
            "params": json.loads(canonical(self.params)),
            "deps": {role: dep.hash for role, dep in sorted(self.deps.items())},
        }

    @cached_property
    def hash(self) -> str:
        return digest(self.key)[:HASH_LENGTH]

    @property
    def seed(self) -> int:
        """This node's seed: derived from its key, which holds the run seed."""
        return int(digest({"seed_for": self.hash})[:8], 16)

    def record(self) -> dict[str, Any]:
        """What outputs carry to identify their node."""
        return {"id": self.id, "hash": self.hash, "key": self.key}


@dataclass
class Graph:
    nodes: list[Node]  # topological order
    targets: list[Node]  # the analysis node

    def by_id(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}


def flatten_inputs(inputs: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested ``inputs`` (as written in YAML or ``+inputs.a.b=...``) -> {"a.b": ref}."""
    flat: dict[str, str] = {}
    for key, value in inputs.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_inputs(value, f"{path}."))
        elif isinstance(value, str) and value:
            flat[path] = value
        else:
            raise ConfigError(f"inputs.{path}: expected a path, wandb: ref or model id")
    return flat


def provider_free(model: str) -> str:
    """``tinker/Qwen/Qwen3-8B`` -> ``Qwen/Qwen3-8B``: the id generation logs record."""
    return model.split("/", 1)[1] if "/" in model else model


def finetune_backend(model: str) -> str:
    if model.startswith("tinker/"):
        return "tinker"
    if model.startswith("openai/"):
        return "openai"
    raise ConfigError(f"can't fine-tune {model!r}: only tinker/ and openai/ models")


def checkpoint_name(arm: str, transform: str) -> str:
    return f"{arm}_{'base' if transform == NO_TRANSFORM else transform}"


def external_name(ref: str) -> str:
    """A readable stand-in name for an output supplied through ``inputs``."""
    if ref.startswith("wandb:"):
        return ref.rsplit("/", 1)[-1].split(":", 1)[0]
    path = Path(ref)
    if path.suffix == ".eval" and path.exists():
        from lasr_labs_2025_control_project.tracking import read_header

        name = read_header(path).get("name")
        if name:
            return str(name)
        return names.fit(f"untracked-{names.clean(path.stem)}")
    return names.short_model(ref)


class GraphBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.overrides = flatten_inputs(cfg.inputs)
        self.used_overrides: set[str] = set()
        self._by_id: dict[str, Node] = {}
        self._by_hash: dict[str, Node] = {}
        self.games: list[Node] = []

    # -- bookkeeping -------------------------------------------------------

    def add(self, node: Node) -> Node:
        ref = self.overrides.get(node.id)
        if ref is not None:
            self.used_overrides.add(node.id)
            node = Node(
                node.stage,
                node.id,
                external_name(ref),
                {},
                info=node.info,
                external=ref,
            )
        existing = self._by_id.get(node.id)
        if existing is not None:
            if existing.hash != node.hash:
                raise ConfigError(
                    f"two different outputs would share the id {node.id!r}; give "
                    "them the same prompts or split the experiment in two"
                )
            return existing
        shared = self._by_hash.get(node.hash)
        if shared is not None:
            self._by_id[node.id] = shared
            return shared
        self._by_id[node.id] = self._by_hash[node.hash] = node
        return node

    @property
    def scope(self) -> Optional[str]:
        m = self.cfg.measurement
        return None if m.share_generations else m.collusion_strategy

    # -- nodes ---------------------------------------------------------------

    def generation(
        self,
        split: str,
        label: str,
        role: str,
        eval_mode: str,
        prompt: str,
        limit: Optional[int],
    ) -> Node:
        setting, models = self.cfg.setting, self.cfg.models
        model = getattr(models, role)
        if limit is None:
            limit = setting.default_generation_limit
        params = {
            "setting": setting.name,
            "split": split,
            "label": label,
            "model": model,
            "eval_mode": eval_mode,
            "system_prompt": prompt,
            "problem_ids": setting.problem_ids,
            "limit": limit,
            "max_tokens": models.generation_max_tokens,
            "temperature": self.cfg.generation.temperature,
            "seed": self.cfg.seed,
            "scope": self.scope,
        }
        return self.add(
            Node(
                "generate",
                f"generate.{split}.{label}",
                names.generation_name(setting.name, model, split, label, prompt),
                params,
                paid=True,
                info={"label": label, "split": split, "role": role},
            )
        )

    def transform(self, split: str, transform: str, source: Node) -> Node:
        if transform == NO_TRANSFORM:
            return source
        spec = self.cfg.transforms[transform]
        params: dict[str, Any] = {"transform": dataclasses.asdict(spec)}
        if spec.type == "llm":
            params["temperature"] = self.cfg.generation.temperature
            params["seed"] = self.cfg.seed
            segment = names.stage_segment("para", spec.model, spec.system_prompt)
        else:
            segment = names.stage_segment("min")
        label = source.info["label"]
        return self.add(
            Node(
                "transform",
                f"transform.{split}.{transform}.{label}",
                names.extend(source.name, segment),
                params,
                deps={"input": source},
                paid=spec.type == "llm",
                info={**source.info, "transform": transform},
            )
        )

    def test_input(self, label: str, transform: str) -> Node:
        """A test-split generation of ``label``, after ``transform``."""
        m, prompts = self.cfg.measurement, self.cfg.setting.prompts
        limit = m.limits.generate
        if label == "attacks":
            prompt = m.attack_prompt or prompts.attack[m.collusion_strategy]
            gen = self.generation("test", "attacks", "U", "attack", prompt, limit)
        elif label == "honest":
            gen = self.generation(
                "test", "honest", "U", "honest", prompts.honest, limit
            )
        else:
            prompt = m.honeypot_prompt or prompts.honeypot
            gen = self.generation("test", "honeypots", "H", "attack", prompt, limit)
        return self.transform("test", transform, gen)

    def finetune(self, arm_name: str, transform: str) -> Node:
        cfg = self.cfg
        arm: Optional[ElicitationArm] = cfg.elicitation.arms.get(arm_name)
        if arm is None:
            raise ConfigError(
                f"monitor arm uses sft:{arm_name}, which isn't in elicitation.arms"
            )
        prompts, limit = cfg.setting.prompts, cfg.elicitation.limits.generate
        if arm.eval_mode == "attack":
            pos_prompt = (
                arm.attack_prompt or prompts.attack[cfg.measurement.collusion_strategy]
            )
            neg_prompt = arm.honeypot_prompt or prompts.honeypot
            pos_label, neg_label = "attacks", "honeypots"
        else:
            pos_prompt = arm.attack_prompt or prompts.honest
            neg_prompt = arm.honeypot_prompt or prompts.honest
            pos_label, neg_label = "honest", HONEST_HONEYPOTS
        deps = {
            "positives": self.transform(
                "train",
                transform,
                self.generation(
                    "train", pos_label, "U", arm.eval_mode, pos_prompt, limit
                ),
            ),
            "negatives": self.transform(
                "train",
                transform,
                self.generation(
                    "train", neg_label, "H", arm.eval_mode, neg_prompt, limit
                ),
            ),
        }
        training = cfg.elicitation.training
        backend = finetune_backend(cfg.models.U)
        hyper = {
            "upsample_pos_factor": training.upsample_pos_factor,
            "upsample_neg_factor": training.upsample_neg_factor,
            "keep_comments": training.keep_comments,
        }
        if backend == "tinker":
            hyper.update(
                rank=training.rank,
                lr=training.lr,
                epochs=training.epochs,
                batch_size=training.batch_size,
                val_frac=training.val_frac,
                checkpoint_every_steps=training.checkpoint_every_steps,
            )
            if training.checkpoint_auc:
                deps["eval_attacks"] = self.test_input("attacks", transform)
                deps["eval_honeypots"] = self.test_input("honeypots", transform)
        else:
            hyper["n_epochs"] = training.openai_n_epochs
        base_model = provider_free(cfg.models.U)
        name = checkpoint_name(arm_name, transform)
        params = {
            "backend": backend,
            "base_model": base_model,
            "target_model": base_model,
            "checkpoint_name": name,
            "labels": arm.labels,
            "transformed": transform != NO_TRANSFORM,
            "system_prompt": prompts.sft_system,
            "user_prompt": prompts.sft_user,
            "view": cfg.setting.view,
            "action_view": dataclasses.asdict(cfg.setting.action_view),
            "max_examples": cfg.elicitation.limits.examples,
            "training": hyper,
            "seed": cfg.seed,
        }
        return self.add(
            Node(
                "finetune",
                f"finetune.{arm_name}.{transform}",
                names.extend(deps["positives"].name, f"ft-{names.clean(name)}"),
                params,
                deps=deps,
                paid=True,
                info={"arm": arm_name, "transform": transform, "checkpoint_name": name},
            )
        )

    def monitor(
        self, transform: str, arm_name: str, arm: MonitorArm, label: str
    ) -> Node:
        cfg, m = self.cfg, self.cfg.measurement
        prompts, setting = cfg.setting.prompts, cfg.setting
        source_label = arm.attacks_label if label == "attacks" else label
        source = self.test_input(source_label, transform)
        deps: dict[str, Node] = {"input": source}

        if arm.model.startswith(SFT_PREFIX):
            ft = self.finetune(arm.model[len(SFT_PREFIX) :], transform)
            deps["collusion_model"] = ft
            collusion_model = "<collusion_model>"  # the fine-tune's output
            collusion_alias = (
                names.short_model(ft.external)
                if ft.external
                else names.clean(f"ft_{ft.info['checkpoint_name']}")
            )
        else:
            collusion_model = getattr(cfg.models, arm.model, None) or arm.model
            collusion_alias = names.short_model(collusion_model)
        collusion_prompt = (
            arm.collusion_prompt or prompts.collusion[m.collusion_strategy]
        )

        transformed = transform != NO_TRANSFORM
        action = setting.view == "action"
        untrusted_view = {
            "transformed": transformed and not action,
            "remove_comments": not transformed and not action,
        }
        entries = {
            "untrusted_collusion": {
                "model": collusion_model,
                "system_prompt": collusion_prompt,
                "base_prompt": prompts.collusion_user,
                **untrusted_view,
            },
            # the trusted monitor always reads the original code, comments stripped
            "trusted_suspiciousness": {
                "model": cfg.models.T,
                "system_prompt": prompts.trusted_suspiciousness,
                "base_prompt": prompts.suspiciousness_user,
                "transformed": False,
                "remove_comments": not action,
            },
            "untrusted_suspiciousness": {
                "model": cfg.models.U,
                "system_prompt": prompts.suspiciousness[transform],
                "base_prompt": prompts.suspiciousness_user,
                **untrusted_view,
            },
        }
        monitors = {name: entries[name] for name in m.monitors}
        params = {
            "label": label,
            "view": setting.view,
            "action_view": (
                {"max_history_chars": setting.action_view.max_history_chars}
                if action
                else None
            ),
            "monitors": monitors,
            "limit": m.limits.monitor,
            "temperature": cfg.generation.temperature,
            "seed": cfg.seed,
        }
        if "untrusted_collusion" in monitors:
            segment_model, segment_prompt = collusion_alias, collusion_prompt
        else:
            first = monitors[m.monitors[0]]
            segment_model = names.short_model(first["model"])
            segment_prompt = first["system_prompt"]
        segment = f"mon-{segment_model}-{names.short_prompt(segment_prompt)}"
        return self.add(
            Node(
                "monitor",
                f"monitor.{transform}.{arm_name}.{label}",
                names.extend(source.name, segment),
                params,
                deps=deps,
                paid=True,
                info={"label": label, "transform": transform, "arm": arm_name},
            )
        )

    def game(self, transform: str, arm_name: str, arm: MonitorArm) -> Node:
        """One condition's solved game (and its bootstrap)."""
        cfg, o = self.cfg, self.cfg.optimization
        condition = {
            "setting": cfg.setting.name,
            "collusion_strategy": cfg.measurement.collusion_strategy,
            "transform": transform,
            "monitor_arm": arm_name,
        }
        params = {
            "condition": condition,
            "variants": list(o.variants),
            "game": dataclasses.asdict(o.game),
            "regeneration_cutoffs": list(o.regeneration_cutoffs),
            "solver": o.solver,
            "max_iterations": o.max_iterations,
            "bootstrap": {"replicates": o.bootstrap.replicates, "ci": o.bootstrap.ci},
            "min_samples_per_label": cfg.measurement.min_samples_per_label,
            "seed": cfg.seed,
        }
        node = self.add(
            Node(
                "game",
                f"game.{transform}.{arm_name}",
                names.condition_name(**condition),
                params,
                deps={
                    label: self.monitor(transform, arm_name, arm, label)
                    for label in LABELS
                },
                info={"condition": condition},
            )
        )
        self.games.append(node)
        return node

    def analysis(self) -> Node:
        """The experiment's figures, drawn once from all its games."""
        cfg = self.cfg
        games = {node.id.removeprefix("game."): node for node in self.games}
        return self.add(
            Node(
                "analysis",
                "analysis",
                names.fit(
                    f"analysis.{names.clean(cfg.setting.name)}.{names.clean(cfg.name)}"
                ),
                {"experiment": cfg.name},
                deps=games,
                info={"experiment": cfg.name},
            )
        )


def validate(cfg: Config) -> None:
    m, s, o = cfg.measurement, cfg.setting, cfg.optimization

    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ConfigError(message)

    check(set(cfg.stages) <= set(STAGES), f"stages must be from {STAGES}")
    check(cfg.reuse in ("local", "wandb"), "reuse must be 'local' or 'wandb'")
    check(o.solver in SOLVERS, f"optimization.solver must be one of {SOLVERS}")
    check(bool(o.variants), "optimization.variants is empty")
    check(
        set(m.monitors) <= set(MONITORS),
        f"measurement.monitors must be from {MONITORS}",
    )
    for variant in o.variants:
        check(variant in VARIANTS, f"unknown analysis variant {variant!r}")
        missing = set(VARIANTS[variant]) - set(m.monitors)
        check(not missing, f"variant {variant} needs monitors {sorted(missing)}")
    check(
        m.collusion_strategy in s.prompts.collusion,
        f"setting {s.name} has no collusion prompt for {m.collusion_strategy!r}",
    )
    check(
        m.attack_prompt is not None or m.collusion_strategy in s.prompts.attack,
        f"setting {s.name} has no attack prompt for {m.collusion_strategy!r}",
    )
    arms = {k: v for k, v in m.arms.items() if v is not None}
    check(
        bool(arms),
        "measurement.arms is empty: add a monitor arm (e.g. base: {model: U})",
    )
    for transform in m.transforms:
        check(
            transform in s.transforms,
            f"setting {s.name} doesn't support transform {transform!r}",
        )
        check(
            transform == NO_TRANSFORM or transform in cfg.transforms,
            f"no transform named {transform!r} in transforms",
        )
        check(
            transform in s.prompts.suspiciousness,
            f"setting {s.name} has no suspiciousness prompt for {transform!r}",
        )
    for name, arm in arms.items():
        check(
            arm.attacks_label in ("attacks", "honest"), f"arm {name}: bad attacks_label"
        )
        for transform in arm.transforms or []:
            check(
                transform in m.transforms,
                f"arm {name}: transform {transform!r} isn't in measurement.transforms",
            )
    for name, earm in cfg.elicitation.arms.items():
        if earm is None:
            continue
        check(
            earm.eval_mode in ("attack", "honest"),
            f"elicitation arm {name}: bad eval_mode",
        )
        check(
            earm.labels in ("author", "permuted"), f"elicitation arm {name}: bad labels"
        )
        check(
            s.name != "basharena" or earm.eval_mode == "attack",
            f"elicitation arm {name}: BashArena has no honest-mode honeypots",
        )
    check(
        s.view in ("trajectory", "action"), "setting.view must be trajectory or action"
    )
    check(
        s.view == "trajectory" or not cfg.elicitation.training.checkpoint_auc,
        "training.checkpoint_auc isn't available for per-action views",
    )


def build_graph(cfg: Config) -> Graph:
    validate(cfg)
    builder = GraphBuilder(cfg)
    m = cfg.measurement
    for arm_name, arm in m.arms.items():
        if arm is None:
            continue
        for transform in arm.transforms or m.transforms:
            builder.game(transform, arm_name, arm)
    unused = sorted(set(builder.overrides) - builder.used_overrides)
    if unused:
        raise ConfigError(f"inputs names nodes this experiment doesn't have: {unused}")
    analysis = builder.analysis()
    return Graph(list(topological([analysis])), [analysis])


def topological(targets: list[Node]) -> Iterator[Node]:
    seen: set[int] = set()

    def visit(node: Node) -> Iterator[Node]:
        if id(node) in seen:
            return
        seen.add(id(node))
        for dep in node.deps.values():
            yield from visit(dep)
        yield node

    for target in targets:
        yield from visit(target)
