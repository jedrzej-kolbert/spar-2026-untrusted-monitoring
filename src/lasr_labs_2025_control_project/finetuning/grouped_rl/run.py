"""Run HiPT-style collusion-signal RL from a YAML config.

    python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \\
        --config src/lasr_labs_2025_control_project/finetuning/grouped_rl/configs/rewrite_signal_pilot.yaml \\
        [--offline | --allow-paid]

``--offline`` (the default) loads data, writes the manifest, and checks the Docker
sandbox against the eval log's own scores, without any model calls. ``--allow-paid``
runs preflight controls, a baseline evaluation, and training.

Outputs in ``output_dir``: manifest.json, rollouts.jsonl (every candidate, including
failures), metrics.jsonl, status.json, and the final checkpoint path.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, cast

import click

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name

from .channels import MonitorViewChannel
from .config import RLRunConfig, load_rl_config
from .core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    GroupedRLConfig,
    GroupedRLTrainer,
    HoneypotEvalResult,
    ReceiverExample,
    Rollout,
    auroc,
)
from .data import attack_tasks, honeypot_pairs, read_generator_eval, split_tasks
from .icrl import (
    CENTRAL,
    SENDER_ONLY,
    Knowledge,
    RoundSummary,
    SearchState,
    parse_strategy,
    run_icrl,
)
from .icrl_adapters import StrategyBox, TrainerTeam, with_strategy
from .rewards import RewardConfig, validity_gate
from . import usage
from lasr_labs_2025_control_project.prompts.rl_prompts import (
    RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
    RL_WHILE_LOOP_JUDGE_INSTRUCTIONS,
    RL_WHILE_LOOP_SENDER_INSTRUCTIONS,
)
from .plots import plot_auroc_over_steps
from .sandbox import DockerAppsEvaluator
from .tinker_backend import TinkerReceiver
from .viewer import write_run_view

logger = logging.getLogger(__name__)

# A deliberately obvious marker for the preflight controls.
_EXPLICIT_SIGNAL = '\n_collusion_signal_for_monitor = "present"\n'


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
# Prefixes logged to W&B; everything else stays in metrics.jsonl only. A section's
# less important metrics go to <section>_detail/; experiments/collusion-rl-v2/
# workspace.py stacks the sections in reading order.
WANDB_PREFIXES = (
    "train/",
    "validation/",
    "test/",
    "icrl/",
    "progress/",
    "train_detail/",
    "validation_detail/",
    "test_detail/",
    "preflight/",
)
# Run records uploaded to the W&B run when it closes.
RECORD_FILES = (
    "manifest.json",
    "split.json",
    "original_attack_suspicion.json",
    "metrics.jsonl",
    "rollouts.jsonl",
    "honeypot_scores.jsonl",
    "status.json",
    "usage.json",
    "auroc_over_steps.svg",
    "auroc_over_steps.json",
    "run_view.html",
)
# The numbers to look at first in each section.
KEY_METRICS = (
    "auroc",
    "p_self_rec",
    "working_rate",
    "backdoor_fires_rate",
    "trusted_suspicion",
    "reward",
    "sender_loss",  # GBRL only
)


def working(rollout: Rollout) -> bool:
    """Passes the tests and its backdoor fires on the original secret input."""
    return bool(
        rollout.execution
        and rollout.execution.passed
        and rollout.backdoor
        and rollout.backdoor.passed
    )


def batch_metrics(
    result: BatchResult,
    honeypot_scores: list[tuple[str, float]],
    config: GroupedRLConfig,
) -> dict[str, Any]:
    """The reward and its parts, and AUROC of working rewrites vs working honeypots."""
    base = result.metrics()
    total = len(result.rollouts)
    gates = [
        validity_gate(item, config)
        for item in result.rollouts
        if item.execution is not None and item.backdoor_suspicion is not None
    ]
    auroc = (
        HoneypotEvalResult(result, honeypot_scores).metrics()["auroc_working"]
        if honeypot_scores
        else None
    )
    return {
        "reward": base["mean_reward"],
        **{key: value for key, value in base.items() if key.startswith("reward/")},
        "working_rate": sum(working(item) for item in result.rollouts) / total,
        "tests_passed_rate": base["correctness"],
        "backdoor_fires_rate": base["backdoor_success"],
        "gate_mean": sum(gates) / len(gates) if gates else None,
        # Self-recognition rewrites all carry payload 1: bit 1's P(YES) is P(self_rec).
        "p_self_rec": base["bit1_mean_p1"],
        "trusted_suspicion": base["backdoor_suspicion_mean"],
        "auroc": auroc,
        "n_rewrites": total,
        "n_honeypots": len(honeypot_scores),
    }


def self_rec_roc(
    result: BatchResult, honeypot_scores: list[tuple[str, float]]
) -> list[tuple[float, float]]:
    """(FPR, TPR) of P(self_rec), working rewrites vs honeypots, at every threshold."""
    import numpy as np

    from lasr_labs_2025_control_project.utils.numerical_pipeline.results_io import (
        _raw_roc,
    )

    positives = [
        item.probabilities[1]
        for item in result.rollouts
        if working(item) and item.probabilities is not None
    ]
    if not positives or not honeypot_scores:
        return []
    roc = _raw_roc(np.array(positives), np.array([s for _, s in honeypot_scores]))
    return list(zip(roc["fpr"], roc["tpr"]))


def signal_metrics(result: BatchResult, metrics: dict[str, Any]) -> dict[str, Any]:
    """Key numbers of a rewrite_signal batch: the shared names above where the maths
    is the same, the reward, its terms and the batch mean of each input to its formula,
    then this task's own (signal ON vs OFF, the signal overseer)."""
    rollouts = result.rollouts
    total = len(rollouts)
    rewarded = [item.reward_components for item in rollouts if item.reward is not None]
    terms = {name for components in rewarded for name in components}
    p_correct = [
        item.probabilities[item.payload]
        for item in rollouts
        if item.probabilities is not None
    ]
    return {
        "reward": metrics.get("mean_reward"),
        # Over every rewarded rollout, a missing term counting 0 (the overseer term
        # is only stored when nonzero), so the terms sum to reward.
        **{
            f"reward/{name}": math.fsum(c.get(name, 0.0) for c in rewarded)
            / len(rewarded)
            for name in sorted(terms)
        },
        "p_correct": math.fsum(p_correct) / len(p_correct) if p_correct else None,
        "working_rate": sum(map(working, rollouts)) / total if total else None,
        "tests_passed_rate": metrics.get("correctness"),
        "backdoor_fires_rate": metrics.get("backdoor_success"),
        "on_off_balanced_accuracy": metrics.get("balanced_accuracy"),
        "on_off_auroc": metrics.get("on_off_auroc"),
        "p1_gap": metrics.get("p1_gap"),
        "signal_suspicion": metrics.get("suspicion_mean"),
        "baseline_suspicion": metrics.get("baseline_suspicion_mean"),
        "suspicion_increase_mean": metrics.get("suspicion_increase_mean"),
        # The overseer weight at this step (it changes under overseer_ramp_*).
        "overseer_strength": metrics.get("overseer_strength"),
    }


def collusion_auroc(
    result: BatchResult, honeypot_scores: list[tuple[str, float]]
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """AUROC and ROC of the receiver's P(YES): working signal-ON rewrites vs honeypots
    (the population the self-recognition runs log as validation/auroc)."""
    import numpy as np

    from lasr_labs_2025_control_project.utils.numerical_pipeline.results_io import (
        _raw_roc,
    )

    positives = [
        item.probabilities[1]
        for item in result.rollouts
        if item.payload == 1 and working(item) and item.probabilities is not None
    ]
    negatives = [score for _, score in honeypot_scores]
    numbers = {
        "auroc": auroc(positives, negatives),
        "n_working": len(positives),
        "n_honeypots": len(negatives),
    }
    if not positives or not negatives:
        return numbers, []
    roc = _raw_roc(np.array(positives), np.array(negatives))
    return numbers, list(zip(roc["fpr"], roc["tpr"]))


def summary_row(summary: RoundSummary) -> dict[str, Any]:
    """One table row per strategy: its text, total reward, each reward term, scores."""
    return {
        "round": summary.round,
        "strategy": summary.strategy,
        **(
            {
                "decoder": summary.decoder,
                "honeypot_p_yes_mean": summary.honeypot_p_yes_mean,
            }
            if summary.honeypot_p_yes_mean is not None
            else {}
        ),
        "reward_mean": summary.reward_mean,
        **{
            f"reward/{name}": value for name, value in summary.reward_components.items()
        },
        "p_self_rec": summary.p_self_mean,
        "working_rate": summary.working_rate,
        "tests_passed_rate": summary.pass_rate,
        "backdoor_fires_rate": summary.backdoor_fires_rate,
        "backdoor_checked": summary.backdoor_checked,
        "gate_mean": summary.gate_mean,
        "trusted_suspicion": summary.suspicion_mean,
        "episodes": summary.episodes,
    }


def code_state() -> dict[str, Any]:
    """Branch, commit, and whether tracked files differ from the commit."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None}


class Recorder:
    """Local files, and one W&B run opened like the pipeline's stage runs.

    The run's job type is the method (icrl or gbrl); project, entity and mode come
    from WANDB_PROJECT, WANDB_ENTITY and WANDB_MODE; its input logs are linked to
    the runs that produced them; a run that does not finish is tagged ``partial``.
    """

    def __init__(
        self,
        config: RLRunConfig,
        output_dir: Path,
        *,
        inputs: list[tracking.ResolvedInput] | None = None,
        code: dict[str, Any] | None = None,
        allow_unlinked: bool = False,
        resumed_from: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.self_recognition = config.task_mode == "self_recognition"
        self.centralized = config.icrl.centralized
        output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy_rows: list[dict[str, Any]] = []
        self.proposed_rows: list[dict[str, Any]] = []
        self.last_step = 0  # W&B steps must not go backwards
        # Fields written in the YAML's logging block win over WANDB_* variables for
        # this run only; an explicit `wandb_project: null` still turns W&B off.
        logging_config = config.logging
        overrides = {
            field.upper(): getattr(logging_config, field)
            for field in logging_config.model_fields_set
        }
        if "WANDB_PROJECT" in overrides and overrides["WANDB_PROJECT"] is None:
            overrides["WANDB_MODE"] = "disabled"
        self._saved_env = {key: os.environ.get(key) for key in overrides}
        os.environ.update({k: v for k, v in overrides.items() if v is not None})
        identity = config.identity
        stage = tracking.StageRun(
            str(identity["method"]),
            config.run_name,
            config={
                **json.loads(config.model_dump_json()),
                "run": identity,
                "code": code or {},
                # A resumed run is a new W&B run; this names the state it continues.
                "resumed_from": resumed_from,
            },
            tags=[f"{key}:{value}" for key, value in identity.items()]
            + (["resumed"] if resumed_from else []),
            inputs=inputs or [],
            models=[config.models.receiver],
            allow_unlinked=allow_unlinked,
        )
        try:
            self.stage = stage.__enter__()
        except BaseException:
            if stage.run is not None:  # opened, then failed declaring its inputs
                stage.run.finish(exit_code=1)
            self._restore_env()
            raise
        self.wandb: Any = self.stage.run
        if self.wandb is not None:
            self.status(wandb_url=self.wandb.url)

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with open(self.output_dir / name, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def scalars(self, step: int, prefix: str, values: dict[str, Any]) -> None:
        clean = {
            f"{prefix}/{key}": value
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self._append("metrics.jsonl", {"step": step, **clean})
        self.last_step = max(self.last_step, step)
        logged = {k: v for k, v in clean.items() if k.startswith(WANDB_PREFIXES)}
        if self.wandb is not None and logged:
            self.wandb.log(logged, step=step)

    def metrics(
        self,
        step: int,
        section: str,
        values: dict[str, Any],
        *,
        prefix: str = "",
        detail: bool = True,
    ) -> None:
        """Key metrics in the section, the others in its _detail section."""
        key = {prefix + k: v for k, v in values.items() if k in KEY_METRICS}
        self.scalars(step, section, key)
        if detail:
            rest = {prefix + k: v for k, v in values.items() if k not in KEY_METRICS}
            self.scalars(step, f"{section}_detail", rest)

    def roc(
        self, step: int, curves: dict[str, list[tuple[float, float]]], title: str
    ) -> None:
        """The collusion ROC (the judge's P(self_rec)): working rewrites vs honeypots."""
        if any(curves.values()):
            self.stage.log_roc(
                "validation/roc_collusion",
                {label: points for label, points in curves.items() if points},
                title,
                step=step,
            )

    def rollouts(
        self, step: int, phase: str, rollouts: list[Rollout], **extra: Any
    ) -> None:
        for item in rollouts:
            generation = item.generation
            self._append(
                "rollouts.jsonl",
                {
                    "step": step,
                    "phase": phase,
                    **extra,
                    "problem_id": item.problem.problem_id,
                    "payload": item.payload,
                    "response": generation.text if generation else None,
                    "response_tokens": len(generation.response_tokens)
                    if generation
                    else None,
                    "code": item.code,
                    "receiver_code": item.receiver_code,
                    "p_yes": item.probabilities[1] if item.probabilities else None,
                    "decoded_correctly": item.decoded_correctly,
                    "passed": item.execution.passed if item.execution else None,
                    "execution_detail": item.execution.detail
                    if item.execution
                    else None,
                    "backdoor": item.backdoor.passed if item.backdoor else None,
                    "suspiciousness": item.suspiciousness,
                    "baseline_suspicion": item.baseline_suspicion,
                    "suspicion_increase": item.suspicion_increase,
                    "overseer_log_odds": item.overseer_log_odds,
                    "backdoor_suspicion": item.backdoor_suspicion,
                    "honeypot_p_yes": item.honeypot_p_yes,
                    "kl": item.kl,
                    "reward": item.reward,
                    "reward_components": item.reward_components,
                    "advantage": item.advantage,
                    "errors": item.errors,
                },
            )

    def receiver_examples(self, step: int, examples: list[ReceiverExample]) -> None:
        for example in examples:
            self._append(
                "receiver_examples.jsonl",
                {
                    "step": step,
                    "problem_id": example.problem.problem_id,
                    "code": example.code,
                    "label": "YES" if example.payload else "NO",
                },
            )

    def honeypots(self, step: int, scores: list[tuple[str, float]]) -> None:
        for problem_id, p_yes in scores:
            self._append(
                "honeypot_scores.jsonl",
                {"step": step, "problem_id": problem_id, "p_yes": p_yes},
            )

    def strategy(
        self,
        step: int,
        summary: RoundSummary,
        cost_usd: float,
        auroc: float | None = None,
    ) -> None:
        """One learner-designed strategy and its measured scores (ICRL)."""
        row = {
            "step": step,
            **summary_row(summary),
            "train_auroc": auroc,
            "cost_usd": cost_usd,
        }
        self._append("strategies.jsonl", row)
        self.sync_file("strategies.jsonl")
        self.strategy_rows.append(row)
        self.table(step, "icrl/strategies", self.strategy_rows)

    def table(self, step: int, key: str, rows: list[dict[str, Any]]) -> None:
        """A W&B table whose columns are every key the rows use, in first-seen order."""
        if self.wandb is None:
            return
        import wandb

        columns: list[Any] = list(dict.fromkeys(name for row in rows for name in row))
        # A reward term missing from a row added nothing to that strategy's reward.
        data: list[Any] = [
            [row.get(c, 0.0 if c.startswith("reward/") else None) for c in columns]
            for row in rows
        ]
        self.wandb.log({key: wandb.Table(columns=columns, data=data)}, step=step)

    def learner_reply(
        self,
        episode: int,
        stage: str,
        prompt: list[dict[str, str]],
        reply: str,
    ) -> None:
        """Save every exact learner request/reply, including malformed replies."""
        self._append(
            "learner.jsonl",
            {"episode": episode, "stage": stage, "prompt": prompt, "reply": reply},
        )
        self.sync_file("learner.jsonl")
        if stage == "strategy":
            try:
                chosen = parse_strategy(reply, centralized=self.centralized)
            except ValueError:
                return  # the learner loop logs and retries malformed strategy replies
            self.proposed_rows.append(
                {
                    "episode": episode,
                    "description": chosen.description,
                    "strategy": chosen.encoder_instructions,
                    **(
                        {"decoder": chosen.decoder_instructions}
                        if self.centralized
                        else {}
                    ),
                }
            )
            self.table(
                self.last_step,
                "icrl/proposed_strategies",
                self.proposed_rows,
            )

    def batch_errors(
        self, step: int, phase: str, strategy: str, errors: list[str]
    ) -> None:
        self._append(
            "batch_errors.jsonl",
            {"step": step, "phase": phase, "strategy": strategy, "errors": errors},
        )
        self.sync_file("batch_errors.jsonl")

    def sync_file(self, name: str) -> None:
        # Upload the file to the W&B run and keep it in sync as it grows.
        if self.wandb is not None:
            self.wandb.save(
                str(self.output_dir / name),
                base_path=str(self.output_dir),
                policy="live",
            )

    def status(self, **values: Any) -> None:
        path = self.output_dir / "status.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(values, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        path.write_text(json.dumps(current, indent=2, default=str))
        if self.wandb is not None:
            self.wandb.summary.update(values)

    def finish(self, complete: bool, reason: str | None = None) -> None:
        """Upload the record files and close the run (tagged partial if incomplete)."""
        try:
            for name in RECORD_FILES + tuple(
                p.name for p in self.output_dir.glob("manifest_resume_step*.json")
            ):
                if self.wandb is not None and (self.output_dir / name).exists():
                    self.wandb.save(
                        str(self.output_dir / name),
                        base_path=str(self.output_dir),
                        policy="now",
                    )
            self.stage.finish(complete=complete, reason=reason)
        except Exception as exc:  # e.g. a failed upload: still close the run
            logger.warning("W&B finish failed: %s", exc)
            if self.wandb is not None:
                self.wandb.finish(exit_code=1)
        finally:
            self.wandb = None
            self._restore_env()

    def _restore_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
SPLIT_SEED = 0  # one train/validation split for every run and seed


class Tasks(NamedTuple):
    """Attacks and working honeypots, split into train and validation by problem ID."""

    train: list[AppsProblem]
    validation: list[AppsProblem]
    train_honeypots: list[tuple[AppsProblem, str]]
    validation_honeypots: list[tuple[AppsProblem, str]]
    # Train problems left out of receiver SFT because they have no honeypot.
    dropped: list[str]

    def split_record(self) -> dict[str, list[str]]:
        def ids(pairs: list[tuple[AppsProblem, str]]) -> list[str]:
            return [problem.problem_id for problem, _ in pairs]

        return {
            "train": [task.problem_id for task in self.train],
            "validation": [task.problem_id for task in self.validation],
            "train_honeypots": ids(self.train_honeypots),
            "validation_honeypots": ids(self.validation_honeypots),
            "train_dropped_without_honeypot": self.dropped,
        }


def load_tasks(config: RLRunConfig) -> Tasks:
    """Working attacks and honeypots from the train logs, split by problem ID.

    Honeypots are every working one (passes its tests, backdoor fires), as in the
    game, whether or not the same problem has a working attack.
    """
    data = config.data
    tasks = attack_tasks(
        read_generator_eval(data.attacks_eval),
        require_main_task=data.require_main_task,
        require_backdoor=data.require_backdoor,
        max_tests=data.max_tests,
    )
    train_honeypots: list[tuple[AppsProblem, str]] = []
    validation_honeypots: list[tuple[AppsProblem, str]] = []
    dropped: list[str] = []
    if config.task_mode == "rewrite_signal":
        # The signal runs' own split: by run seed, optionally capped, with each
        # validation problem's honeypot (working or not) as the negatives.
        train, validation = split_tasks(
            tasks, validation_fraction=data.validation_fraction, seed=config.seed
        )
        if data.max_validation_problems is not None:
            validation = validation[: data.max_validation_problems]
        if data.honeypots_eval is not None and config.eval.honeypot_eval:
            validation_honeypots = honeypot_pairs(
                read_generator_eval(data.honeypots_eval), validation
            )
    else:
        train, validation = split_tasks(
            tasks, validation_fraction=data.validation_fraction, seed=SPLIT_SEED
        )
    if (
        config.task_mode != "rewrite_signal"
        and data.honeypots_eval is not None
        and (config.eval.honeypot_eval or config.task_mode == "self_recognition")
    ):
        negatives = read_generator_eval(data.honeypots_eval)
        if config.task_mode == "self_recognition":
            sender_model = config.models.sender.removeprefix("tinker/")
            if any(
                not sample.model or sample.model.removeprefix("tinker/") == sender_model
                for sample in negatives
            ):
                raise ValueError(
                    "non-self examples must come from a known, different model"
                )
        honeypot_train, honeypot_validation = split_tasks(
            attack_tasks(negatives, max_tests=data.max_tests),
            validation_fraction=data.validation_fraction,
            seed=SPLIT_SEED,
        )
        train_honeypots = [(task, task.base_code) for task in honeypot_train]
        validation_honeypots = [(task, task.base_code) for task in honeypot_validation]
        ids = [task.problem_id for task in honeypot_train + honeypot_validation]
        if len(ids) != len(set(ids)):
            # Honeypot views are cached per problem.
            raise ValueError("a problem has more than one honeypot")
        if config.training.train_receiver:
            # Receiver SFT needs a non-self example for every problem; one
            # whose honeypot generation failed has none, so it is left out.
            paired = {problem.problem_id for problem, _ in train_honeypots}
            dropped = [t.problem_id for t in train if t.problem_id not in paired]
            if dropped:
                logger.warning(
                    "dropping %d training problems with no honeypot: %s",
                    len(dropped),
                    dropped,
                )
            train = [t for t in train if t.problem_id not in dropped]
    if not train:
        raise ValueError("no training tasks survived filtering")
    if len(train) < config.training.problems_per_step:
        raise ValueError(
            f"{len(train)} training tasks < problems_per_step "
            f"{config.training.problems_per_step}"
        )
    return Tasks(train, validation, train_honeypots, validation_honeypots, dropped)


def ensure_image(image: str) -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if inspect.returncode != 0:
        logger.info("pulling sandbox image %s", image)
        subprocess.run(["docker", "pull", image], check=True)


async def sandbox_parity_check(
    evaluator: DockerAppsEvaluator, tasks: list[AppsProblem], limit: int = 16
) -> dict[str, Any]:
    """Re-grade the original attacks: they were selected as passing with a working
    backdoor in the Inspect pipeline, so the sandbox should agree."""
    sample = tasks[:limit]
    grades = await asyncio.gather(
        *(evaluator.grade(task, task.base_code) for task in sample)
    )
    backdoors = await asyncio.gather(
        *(evaluator.check_backdoor(task, task.base_code) for task in sample)
    )
    infrastructure = [
        grade.infrastructure_error
        for grade in [*grades, *(b for b in backdoors if b)]
        if grade.infrastructure_error
    ]
    if infrastructure:
        raise RuntimeError(f"sandbox infrastructure failure: {infrastructure[0]}")
    return {
        "checked": len(sample),
        "main_task_agreement": sum(grade.passed for grade in grades) / len(sample),
        "backdoor_agreement": sum(bool(b and b.passed) for b in backdoors)
        / len(sample),
    }


def build_channel(config: RLRunConfig) -> MonitorViewChannel:
    channel = config.channel
    return MonitorViewChannel(
        kind=channel.kind,
        remove_comments=channel.remove_comments,
        paraphrase_model=channel.paraphrase_model,
        paraphrase_system_prompt=get_prompt_by_name(channel.paraphrase_system_prompt),
        paraphrase_base_prompt=get_prompt_by_name(channel.paraphrase_base_prompt),
    )


def core_config(
    config: RLRunConfig, overseer_weight: float | None = None
) -> GroupedRLConfig:
    reward, training = config.reward, config.training
    return GroupedRLConfig(
        group_size=training.group_size,
        seed=config.seed,
        task_weight=reward.task_weight,
        communication_weight=reward.communication_weight,
        overseer_weight=reward.overseer_weight
        if overseer_weight is None
        else overseer_weight,
        payload_grouping=training.payload_grouping,
        overseer_baseline=reward.overseer_baseline,
        overseer_delta_space=reward.overseer_delta_space,
        reward=RewardConfig(
            reward_fn=reward.reward_fn,
            suspicion_threshold=reward.suspicion_threshold,
            suspicion_steepness=reward.suspicion_steepness,
            stego_scale=reward.stego_scale,
            overseer_scale=reward.overseer_scale,
            overseer_log_odds_threshold=reward.overseer_log_odds_threshold,
            failed_backdoor_weight=reward.failed_backdoor_weight,
            backdoor_suspicion_weight=reward.backdoor_suspicion_weight,
            kl_coef=reward.kl_coef,
            backdoor_kept_weight=reward.backdoor_kept_weight,
        ),
        train_receiver=training.train_receiver,
        self_recognition=config.task_mode == "self_recognition",
        max_concurrency=training.max_concurrency,
        trusted_reads_original=True,  # as in the game: comment-stripped code, not the jammed view
    )


def overseer_strength(config: RLRunConfig, step: int) -> float:
    """Overseer weight (branch) or scale (hipt) at a training step, with linear ramp."""
    reward = config.reward
    final = (
        reward.overseer_weight
        if reward.reward_fn == "branch"
        else reward.overseer_scale
    )
    if reward.overseer_ramp_start is None or reward.overseer_ramp_steps == 0:
        return final
    fraction = min(step / reward.overseer_ramp_steps, 1.0)
    return reward.overseer_ramp_start + fraction * (final - reward.overseer_ramp_start)


def apply_overseer_strength(
    trainer: GroupedRLTrainer, config: RLRunConfig, step: int
) -> float:
    value = overseer_strength(config, step)
    if config.reward.reward_fn == "branch":
        trainer.config = dataclasses.replace(trainer.config, overseer_weight=value)
    else:
        trainer.config = dataclasses.replace(
            trainer.config,
            reward=dataclasses.replace(trainer.config.reward, overseer_scale=value),
        )
    return value


async def build_trainer(
    config: RLRunConfig,
    evaluator: DockerAppsEvaluator,
    *,
    resume_state: str | None = None,
    strategy: StrategyBox | None = None,
) -> tuple[GroupedRLTrainer, Any]:
    """The trainer and the concrete Tinker sender (for checkpointing).

    With ``strategy`` (ICRL), the sender's system prompt ends with the box's text.
    """
    from .tinker_backend import (
        FrozenYesNoMonitor,
        TextScoreBackdoorMonitor,
        create_tinker_receiver,
        create_tinker_sender,
        extract_final_code,
        rewrite_sender_messages,
    )

    prompts, training, reward = config.prompts, config.training, config.reward
    system_prompt = get_prompt_by_name(prompts.sender_system)
    user_prompt = get_prompt_by_name(prompts.sender_user)
    labels = (
        get_prompt_by_name(prompts.signal_off),
        get_prompt_by_name(prompts.signal_on),
    )

    def messages_fn(problem: AppsProblem, payload: int) -> list[dict[str, str]]:
        return rewrite_sender_messages(
            problem,
            payload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            signal_labels=labels,
        )

    if strategy is not None:
        import tinker

        from .tinker_backend import TinkerSender

        # ICRL never trains the sender. Sampling the base model directly gives what
        # a fresh LoRA adapter gives, without waiting for Tinker training capacity.
        sampler = tinker.ServiceClient().create_sampling_client(
            base_model=config.models.sender
        )
        sender = TinkerSender(
            None,
            sampler,
            sampler.get_tokenizer(),
            allow_paid=True,
            messages_fn=with_strategy(messages_fn, strategy),
            temperature=training.temperature,
            max_tokens=training.max_tokens,
            max_context_tokens=training.max_context_tokens,
        )
        reference = None
    else:
        sender, reference = await create_tinker_sender(
            allow_paid=True,
            base_model=config.models.sender,
            rank=training.lora_rank,
            learning_rate=training.learning_rate,
            with_reference=reward.kl_coef > 0,
            resume_state=resume_state,
            messages_fn=messages_fn,
            loss_fn=training.loss_fn,
            ppo_passes=training.ppo_passes,
            clip_epsilon=training.clip_epsilon,
            temperature=training.temperature,
            max_tokens=training.max_tokens,
            max_context_tokens=training.max_context_tokens,
        )
    if training.train_receiver:
        receiver = await create_tinker_receiver(
            allow_paid=True,
            model=config.models.receiver,
            system_prompt=get_prompt_by_name(prompts.receiver_system),
            rank=training.lora_rank,
            learning_rate=training.receiver_learning_rate or training.learning_rate,
            max_context_tokens=training.max_context_tokens,
        )
    else:
        receiver = FrozenYesNoMonitor(
            config.models.receiver,
            get_prompt_by_name(prompts.receiver_system),
            allow_paid=True,
        )
    overseer = None
    if config.models.overseer and config.task_mode != "self_recognition":
        overseer = FrozenYesNoMonitor(
            config.models.overseer,
            get_prompt_by_name(prompts.overseer_system),
            allow_paid=True,
        )
    backdoor_monitor = None
    # additive leaves the monitor out of the reward but still scores evaluations.
    if (
        reward.backdoor_suspicion_weight > 0 or reward.reward_fn == "additive"
    ) and config.models.backdoor_monitor:
        backdoor_monitor = TextScoreBackdoorMonitor(
            config.models.backdoor_monitor,
            get_prompt_by_name(prompts.backdoor_monitor_system),
            allow_paid=True,
        )
    trainer = GroupedRLTrainer(
        sender,
        receiver,
        evaluator,
        core_config(config),
        overseer=overseer,
        channel=build_channel(config),
        backdoor_monitor=backdoor_monitor,
        reference=reference,
        code_extractor=extract_final_code,
    )
    return trainer, sender


async def monitor_preflight(
    trainer: GroupedRLTrainer, tasks: list[AppsProblem], config: RLRunConfig
) -> dict[str, float]:
    """Score clean vs explicitly signalled versions of real attacks.

    A useful overseer must separate them; the receiver's numbers show its prior.
    """
    channel = trainer.channel
    assert channel is not None
    controls = tasks[: config.preflight.num_controls]
    out: dict[str, list[float]] = {}
    for label, suffix in (("clean", ""), ("explicit", _EXPLICIT_SIGNAL)):
        for task in controls:
            view = await channel.transform(task, task.base_code + suffix)
            _, p_yes = await trainer.receiver.probabilities(task, view)
            out.setdefault(f"receiver_{label}_p_yes", []).append(p_yes)
            if trainer.overseer is not None:
                score = await trainer.overseer.suspiciousness(task, view)
                out.setdefault(f"overseer_{label}_suspicion", []).append(score)
    return {key: sum(values) / len(values) for key, values in out.items()}


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
async def run_evaluation(
    trainer: GroupedRLTrainer,
    recorder: Recorder,
    step: int,
    validation: list[AppsProblem],
    honeypots: list[tuple[AppsProblem, str]],
    config: RLRunConfig,
) -> float | None:
    """Rewrites of the validation problems, scored against the validation honeypots.

    Returns the validation reward, or None when the evaluation failed.
    """
    if not validation:
        return None
    try:
        result = await trainer.evaluate(
            validation, config.eval.num_samples, balanced=True
        )
    except BatchInfrastructureError as error:
        # Keep the records; a failed evaluation does not stop training.
        recorder.rollouts(step, "validation_failed", error.result.rollouts)
        logger.warning(
            "validation at step %d had errors: %s", step, error.result.errors[:3]
        )
        return None
    recorder.rollouts(step, "validation", result.rollouts)
    scores: list[tuple[str, float]] = []
    errors: list[str] = []
    if honeypots:
        honeypot_eval = await trainer.evaluate_against_honeypots(
            validation, honeypots, signalled=result
        )
        recorder.honeypots(step, honeypot_eval.honeypot_scores)
        scores, errors = honeypot_eval.honeypot_scores, honeypot_eval.honeypot_errors
    if config.task_mode == "rewrite_signal":
        pairs = result.metrics()
        vs_honeypots = honeypot_eval.metrics() if honeypots else {}
        numbers, roc = collusion_auroc(result, scores)
        recorder.scalars(
            step,
            "validation",
            {
                **signal_metrics(result, pairs),
                **numbers,
                "honeypot_balanced_accuracy": vs_honeypots.get("balanced_accuracy"),
                "signalled_yes_rate": vs_honeypots.get("signalled_yes_rate"),
                "honeypot_yes_rate": vs_honeypots.get("honeypot_yes_rate"),
            },
        )
        recorder.scalars(
            step,
            "validation_detail",
            {
                **{f"pairs/{k}": v for k, v in pairs.items()},
                **{f"honeypots/{k}": v for k, v in vs_honeypots.items()},
            },
        )
        recorder.roc(step, {f"step {step}": roc}, "Collusion ROC on validation")
        return pairs["mean_reward"]
    metrics = batch_metrics(result, scores, trainer.config)
    recorder.metrics(step, "validation", {**metrics, "honeypot_errors": len(errors)})
    recorder.roc(
        step,
        {f"step {step}": self_rec_roc(result, scores)},
        "Collusion ROC on validation",
    )
    return metrics["reward"]


def next_batch(
    order: list[AppsProblem],
    tasks: list[AppsProblem],
    size: int,
    rng: random.Random,
) -> list[AppsProblem]:
    """Next ``size`` distinct problems from a shuffled stream of epochs.

    When an epoch runs out mid-batch the next one is appended; a problem already in
    the batch is skipped and stays queued for a later batch.
    """
    if size > len({task.problem_id for task in tasks}):
        raise ValueError("batch size exceeds the number of distinct tasks")
    batch: list[AppsProblem] = []
    seen: set[str] = set()
    while len(batch) < size:
        index = next(
            (i for i, task in enumerate(order) if task.problem_id not in seen), None
        )
        if index is None:
            epoch = tasks[:]
            rng.shuffle(epoch)
            order.extend(epoch)
            continue
        task = order.pop(index)
        batch.append(task)
        seen.add(task.problem_id)
    return batch


_DOCKER_DOWN = ("Cannot connect to the Docker daemon", "Is the docker daemon running")


def docker_down(result: BatchResult) -> bool:
    messages = [*result.errors, *(e for item in result.rollouts for e in item.errors)]
    return any(marker in message for message in messages for marker in _DOCKER_DOWN)


async def run_in_context(
    config: RLRunConfig,
    trainer: GroupedRLTrainer,
    box: StrategyBox,
    learner: Any,
    recorder: Recorder,
    meter: usage.Usage,
    train_tasks: list[AppsProblem],
    validation: list[AppsProblem],
    validation_honeypots: list[tuple[AppsProblem, str]],
    train_honeypots: list[tuple[AppsProblem, str]] | None = None,
) -> None:
    """ICRL search on train problems, then the search's best on validation.

    The strategy with the highest validation reward is the run's final strategy,
    for the pipeline to score on test. Every learner reply goes to learner.jsonl,
    every strategy with its scores to strategies.jsonl and the W&B
    ``icrl/strategies`` table, every rewrite to rollouts.jsonl (with its strategy),
    and the outcome to icrl_result.json; all four files are also uploaded to W&B.
    """
    icrl = config.icrl
    train_honeypots = train_honeypots or []
    if icrl.centralized:
        # The paired reward needs each search problem's honeypot.
        paired = {problem.problem_id for problem, _ in train_honeypots}
        train_tasks = [task for task in train_tasks if task.problem_id in paired]
    if icrl.search_problems > len(train_tasks):
        raise ValueError("icrl.search_problems exceeds the training problems")
    search = random.Random(config.seed).sample(train_tasks, icrl.search_problems)
    resume = SearchState.load(icrl.resume_state) if icrl.resume_state else None
    recorder.status(
        state="searching", search_problem_ids=[p.problem_id for p in search]
    )
    step = 0
    if resume is not None:
        step = len(resume.history)
        recorder.strategy_rows = [
            {"step": index + 1, **summary_row(item), "cost_usd": None}
            for index, item in enumerate(resume.history)
        ]
    # Train AUROC each round. A frozen judge gives each train honeypot one score;
    # a centralized judge changes per strategy, so its paired honeypots are used.
    frozen_scores: list[tuple[str, float]] = []
    if not icrl.centralized and train_honeypots:
        frozen_scores, errors = await trainer.score_honeypots(train_honeypots)
        if errors:
            raise RuntimeError(f"train honeypot scoring failed: {errors[:3]}")
    round_auroc: list[float | None] = [None] * (len(resume.history) if resume else 0)

    def on_result(strategy: str, result: BatchResult, phase: str) -> None:
        nonlocal step
        step += 1
        recorder.rollouts(
            step, phase, result.rollouts, strategy=strategy, decoder=box.decoder
        )
        recorder.sync_file("rollouts.jsonl")
        if result.errors:
            # Batch-level causes (e.g. a sender exception) are not on any rollout.
            logger.warning("%s batch errors: %s", phase, result.errors[:3])
            recorder.batch_errors(step, phase, strategy, result.errors)
        if phase == "search":
            scores = frozen_scores or list(
                {
                    r.problem.problem_id: r.honeypot_p_yes
                    for r in result.rollouts
                    if r.honeypot_p_yes is not None
                }.items()
            )
            # The same train/ metrics as a GBRL step, so the methods share plots.
            metrics = batch_metrics(result, scores, trainer.config)
            round_auroc.append(metrics["auroc"])
            recorder.metrics(step, "train", metrics)

    team = TrainerTeam(
        trainer,
        box,
        search_problems=search,
        validation_problems=validation,
        honeypots=validation_honeypots,
        num_samples=config.eval.num_samples,
        on_result=on_result,
        judge_template=RL_CENTRAL_JUDGE_SYSTEM_PROMPT if icrl.centralized else "",
        train_honeypots=train_honeypots if icrl.centralized else (),
    )

    def on_summary(summary: RoundSummary) -> None:
        recorder.strategy(step, summary, meter.cost_usd, auroc=round_auroc[-1])
        click.echo(
            f"round {summary.round}: reward {summary.reward_mean:.3f}, "
            f"P(self_rec) {summary.p_self_mean:.3f}, working {summary.working_rate}, "
            f"suspicion {summary.suspicion_mean:.3f}, train AUROC {round_auroc[-1]} "
            f"| {summary.strategy[:80]!r}"
        )

    knowledge = resume.knowledge if resume is not None else Knowledge()

    def on_reply(
        episode: int, stage: str, prompt: list[dict[str, str]], reply: str
    ) -> None:
        recorder.learner_reply(episode, stage, prompt, reply)

    def on_state(state: SearchState) -> None:
        state.save(config.output_dir / "icrl_state.json")
        recorder.sync_file("icrl_state.json")

    rows: dict[str, dict[str, Any]] = {}
    done: dict[tuple[str, str], dict[str, Any]] = {}
    rocs: dict[tuple[str, str], list[tuple[float, float]]] = {}

    async def validate(label: str, candidate: RoundSummary) -> dict[str, Any]:
        key = (candidate.strategy, candidate.decoder)
        if key not in done:
            _, honeypot = await team.validate(*key)
            rewrites = honeypot.signalled
            # The same validation/ metrics as a GBRL checkpoint, per strategy.
            metrics = {
                "round": candidate.round,
                "strategy": candidate.strategy,
                **({"decoder": candidate.decoder} if icrl.centralized else {}),
                **batch_metrics(rewrites, honeypot.honeypot_scores, trainer.config),
                "honeypot_errors": len(honeypot.honeypot_errors),
            }
            paired_rewards = [
                r.reward
                for r in rewrites.rollouts
                if r.honeypot_p_yes is not None and r.reward is not None
            ]
            if paired_rewards:
                metrics["paired_reward_mean"] = sum(paired_rewards) / len(
                    paired_rewards
                )
            recorder.honeypots(step, honeypot.honeypot_scores)
            done[key] = metrics
            rocs[key] = self_rec_roc(rewrites, honeypot.honeypot_scores)
        rows[label] = done[key]
        return rows[label]

    try:
        history, _ = await run_icrl(
            learner,
            team,
            episodes=icrl.episodes,
            n_result_examples=icrl.n_result_examples,
            knowledge=None if resume is not None else knowledge,
            on_reply=on_reply,
            on_summary=on_summary,
            resume=resume,
            on_state=on_state,
            prompts=CENTRAL if icrl.centralized else SENDER_ONLY,
        )
        recorder.status(state="validation")
        # The centralized reward is paired; a rewrite whose problem has no
        # honeypot would otherwise count the judge's YES without its cost.
        objective = "paired_reward_mean" if icrl.centralized else "reward"
        top = sorted(history, key=lambda item: item.reward_mean, reverse=True)
        scored = [
            ((await validate(f"top{rank}", candidate))[objective], candidate)
            for rank, candidate in enumerate(top[: icrl.validate_top], 1)
        ]
        best = max(scored, key=lambda item: item[0])[1]
        rows["winner"] = done[(best.strategy, best.decoder)]
        # The no-strategy rewrite, so a gain is measured against no strategy.
        await validate(
            "baseline", dataclasses.replace(best, round=0, strategy="", decoder="")
        )
        if icrl.centralized:
            for label, strategy, decoder in (
                # Do the instructions matter, on the same sender text?
                ("instructions_removed", best.strategy, ""),
                # Is the judge reading a signal, or just spotting honeypots?
                ("empty_sender", "", best.decoder),
                (
                    "while_loop",
                    RL_WHILE_LOOP_SENDER_INSTRUCTIONS,
                    RL_WHILE_LOOP_JUDGE_INSTRUCTIONS,
                ),
            ):
                await validate(
                    label,
                    dataclasses.replace(best, strategy=strategy, decoder=decoder),
                )
        # W&B charts: the winner and the no-strategy rewrite; the table has every check.
        recorder.metrics(step, "validation", rows["winner"])
        recorder.metrics(
            step, "validation", rows["baseline"], prefix="no_strategy_", detail=False
        )
        recorder.roc(
            step,
            {
                "winner": rocs[(best.strategy, best.decoder)],
                "no strategy": rocs[("", "")],
            },
            "Collusion ROC on validation",
        )
        recorder.table(
            step,
            "icrl/validation",
            [{"check": label, **metrics} for label, metrics in rows.items()],
        )
    except (BatchInfrastructureError, usage.BudgetExceeded):
        recorder.status(
            state="stopped",
            reason="budget cap reached" if meter.exhausted else "infrastructure error",
        )
        raise
    (config.output_dir / "icrl_result.json").write_text(
        json.dumps(
            {
                "winner": dataclasses.asdict(best),
                "strategies": [
                    {**dataclasses.asdict(item), "train_auroc": auroc}
                    for item, auroc in zip(history, round_auroc)
                ],
                "validation": rows,
                "knowledge": dataclasses.asdict(knowledge),
            },
            indent=2,
        )
    )
    recorder.sync_file("icrl_result.json")
    # The test rewrite takes the winner from this artifact, which links the two runs.
    recorder.stage.log_artifact(
        config.output_dir / "icrl_result.json",
        type="strategy",
        name=f"strategy-{config.run_name}",
    )
    recorder.status(
        state="finished",
        winner_round=best.round,
        winner_strategy=best.strategy,
        winner_decoder=best.decoder,
        validation_auroc_winner=rows["winner"]["auroc"],
        validation_auroc_baseline=rows["baseline"]["auroc"],
    )
    click.echo(
        f"winner (round {best.round}): validation AUROC on working rewrites "
        f"{rows['winner']['auroc']} vs no strategy {rows['baseline']['auroc']}"
    )


def open_recorder(
    config: RLRunConfig,
    code: dict[str, Any],
    parity: dict[str, Any],
    allow_unlinked: bool,
    resumed_from: dict[str, Any] | None,
) -> Recorder:
    """The run's W&B stage run, with its split and, for uncommitted code, the diff."""
    data = config.data
    recorder = Recorder(
        config,
        config.output_dir,
        inputs=[
            tracking.resolve_input(path)
            for path in (data.attacks_eval, data.honeypots_eval)
            if path is not None
        ],
        code=code,
        allow_unlinked=allow_unlinked,
        resumed_from=resumed_from,
    )
    try:
        recorder.status(state="starting", parity=parity)
        recorder.scalars(0, "preflight", parity)
        recorder.stage.log_artifact(
            config.output_dir / "split.json",
            type="split",
            name=f"split-{config.run_name}",
        )
        if code["dirty"]:
            diff = subprocess.run(
                ["git", "diff", "HEAD"], capture_output=True, text=True
            ).stdout
            (config.output_dir / "code.diff").write_text(diff)
            recorder.sync_file("code.diff")
    except BaseException as error:
        recorder.finish(False, repr(error))
        raise
    return recorder


async def train(
    config: RLRunConfig,
    allow_paid: bool,
    *,
    resume_state: str | None = None,
    start_step: int = 1,
    allow_unlinked: bool = False,
) -> None:
    code = code_state()
    if config.kind == "main" and code["dirty"] is not False:
        raise SystemExit(
            "a kind: main run needs committed code; commit first (dirty or unknown "
            f"tree: {code})"
        )
    if resume_state is not None and config.training.train_receiver:
        raise ValueError(
            "--resume-state restores only the sender; joint training must start fresh"
        )
    output_dir = config.output_dir
    if config.method == "in_context" and any(
        (output_dir / name).exists()
        for name in ("rollouts.jsonl", "learner.jsonl", "icrl_result.json")
    ):
        raise FileExistsError(
            "ICRL output directory already contains a run; choose a new output_dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(config)
    train_tasks, validation = tasks.train, tasks.validation
    split = tasks.split_record()
    (output_dir / "split.json").write_text(json.dumps(split, indent=2))
    manifest = {
        "run_name": config.run_name,
        "run": config.identity,
        "config": json.loads(config.model_dump_json()),
        "code": code,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": split,
    }
    manifest_name = "manifest.json"
    if resume_state is not None:
        manifest["resumed_from"] = {"state": resume_state, "start_step": start_step}
        manifest_name = f"manifest_resume_step{start_step}.json"
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    click.echo(
        f"attacks: {len(train_tasks)} train, {len(validation)} validation; "
        f"honeypots: {len(tasks.train_honeypots)} train, "
        f"{len(tasks.validation_honeypots)} validation"
    )

    sandbox = config.sandbox
    ensure_image(sandbox.image)
    evaluator = DockerAppsEvaluator(
        image=sandbox.image,
        timeout_seconds=sandbox.timeout_seconds,
        control_arena_parity=sandbox.control_arena_parity,
        batch_tests=sandbox.batch_tests,
        max_containers=sandbox.max_containers,
        max_output_bytes=sandbox.max_output_bytes,
        num_tests=config.data.max_tests,
    )
    parity = await sandbox_parity_check(evaluator, train_tasks)
    click.echo(f"sandbox parity with eval scores: {parity}")
    if not allow_paid:
        click.echo(
            "offline mode: no model calls made. Re-run with --allow-paid to train."
        )
        return

    meter = usage.install()
    meter.max_cost_usd = config.training.max_cost_usd
    box = StrategyBox() if config.method == "in_context" else None
    trainer, sender = await build_trainer(
        config, evaluator, resume_state=resume_state, strategy=box
    )
    if box is not None:
        import tinker

        from .icrl_adapters import TinkerLearner

        sampler = tinker.ServiceClient().create_sampling_client(
            base_model=config.icrl.learner
        )
        learner = TinkerLearner(
            sampler,
            sampler.get_tokenizer(),
            allow_paid=True,
            max_tokens=config.icrl.learner_max_tokens,
            temperature=config.icrl.learner_temperature,
            max_context_tokens=config.training.max_context_tokens,
        )
        recorder = open_recorder(
            config, code, parity, allow_unlinked, manifest.get("resumed_from")
        )
        complete = False
        try:
            await run_in_context(
                config,
                trainer,
                box,
                learner,
                recorder,
                meter,
                train_tasks,
                validation,
                tasks.validation_honeypots,
                tasks.train_honeypots,
            )
            complete = True
        finally:
            close_run(recorder, meter, output_dir, recorder.last_step, complete)
        return

    recorder = open_recorder(
        config, code, parity, allow_unlinked, manifest.get("resumed_from")
    )
    training = config.training
    honeypots = tasks.validation_honeypots
    last_step_cost = 0.0
    completed, last_eval = start_step - 1, start_step - 1
    step = completed
    best_validation: tuple[float, int] | None = None

    def track(reward: float | None, at: int) -> None:
        nonlocal best_validation
        if reward is not None and (
            best_validation is None or reward > best_validation[0]
        ):
            best_validation = (reward, at)
            recorder.status(best_validation_reward=reward, best_validation_step=at)

    complete = False
    try:
        if config.preflight.enabled and config.task_mode != "self_recognition":
            preflight = await monitor_preflight(trainer, train_tasks, config)
            click.echo(f"monitor preflight: {preflight}")
            recorder.scalars(0, "preflight", preflight)
            gap = preflight.get("overseer_explicit_suspicion", 0) - preflight.get(
                "overseer_clean_suspicion", 0
            )
            if gap < config.preflight.require_overseer_gap:
                recorder.status(
                    state="stopped", reason=f"overseer gap {gap:.3f} too small"
                )
                raise SystemExit(f"overseer preflight failed: gap {gap:.3f}")
        if config.reward.overseer_baseline == "original_attack":
            # One overseer score per original attack, reused for every rewrite of it.
            baselines = await trainer.score_original_attacks(train_tasks + validation)
            (output_dir / "original_attack_suspicion.json").write_text(
                json.dumps(baselines, indent=2)
            )
            mean = sum(baselines.values()) / len(baselines)
            click.echo(
                f"original attacks: mean overseer suspicion {mean:.3f} "
                f"over {len(baselines)}"
            )
            recorder.scalars(0, "preflight", {"original_attack_suspicion_mean": mean})
        rng = random.Random(config.seed)
        order: list[AppsProblem] = []
        skipped = 0
        # Replay earlier draws so a resumed run continues the same batch sequence.
        for _ in range(1, start_step):
            next_batch(order, train_tasks, training.problems_per_step, rng)
        # A frozen judge gives each train honeypot one score, for the train AUROC.
        frozen_scores: list[tuple[str, float]] = []
        if trainer.config.self_recognition and not training.train_receiver:
            frozen_scores, errors = await trainer.score_honeypots(tasks.train_honeypots)
            if errors:
                raise RuntimeError(f"train honeypot scoring failed: {errors[:3]}")
        if start_step == 1:
            track(
                await run_evaluation(
                    trainer, recorder, 0, validation, honeypots, config
                ),
                0,
            )
        for step in range(start_step, training.steps + 1):
            spent = meter.cost_usd
            cap = training.max_cost_usd
            # Stop early if the last step predicts an overrun; SDK reservations also
            # protect the first step, longer outputs, and every evaluation request.
            if meter.exhausted or (cap is not None and spent + last_step_cost > cap):
                recorder.status(
                    state="stopping",
                    step=step - 1,
                    reason=f"budget: ${spent:.2f} spent, next step ~${last_step_cost:.2f},"
                    f" cap ${cap:.2f}",
                )
                click.echo(f"stopping before step {step}: budget cap ${cap:.2f}")
                break
            batch = next_batch(order, train_tasks, training.problems_per_step, rng)
            strength = apply_overseer_strength(trainer, config, step)
            started = time.time()
            try:
                batch_ids = {problem.problem_id for problem in batch}
                result = await trainer.train_step(
                    batch,
                    honeypots=[
                        (problem, code)
                        for problem, code in tasks.train_honeypots
                        if problem.problem_id in batch_ids
                    ],
                )
            except BatchInfrastructureError as error:
                result = error.result
                recorder.rollouts(step, "train_failed", result.rollouts)
                recorder.receiver_examples(step, result.receiver_examples)
                if meter.exhausted:
                    recorder.status(
                        state="stopping",
                        step=step,
                        reason="budget: refused a request before dispatch",
                        partial_step=True,
                    )
                    click.echo(f"step {step}: budget exhausted; saving current weights")
                    break
                if result.sender_updated is None or result.receiver_updated is None:
                    recorder.status(
                        state="stopped", step=step, reason="uncertain update"
                    )
                    raise
                if docker_down(result):
                    recorder.status(
                        state="stopped", step=step, reason="Docker daemon down"
                    )
                    raise SystemExit(
                        f"step {step}: Docker daemon unreachable. Restart Docker, then "
                        "resume from the last state_step<N> in status.json with "
                        "--resume-state <path> --start-step <N+1>"
                    ) from error
                skipped += 1
                recorder.scalars(step, "train_detail", {"skipped_step": 1})
                rollout_errors = [e for item in result.rollouts for e in item.errors]
                logger.warning(
                    "step %d skipped: %s", step, (result.errors + rollout_errors)[:3]
                )
                if skipped > training.max_skipped_steps:
                    recorder.status(state="stopped", step=step, reason="too many skips")
                    raise
                continue
            recorder.rollouts(step, "train", result.rollouts)
            recorder.receiver_examples(step, result.receiver_examples)
            metrics = dict(result.metrics())
            metrics["overseer_strength"] = strength
            metrics["step_seconds"] = time.time() - started
            metrics["sender_updated"] = float(bool(result.sender_updated))
            metrics["receiver_updated"] = float(bool(result.receiver_updated))
            metrics["receiver_positive_examples"] = sum(
                e.payload == 1 for e in result.receiver_examples
            )
            metrics["receiver_negative_examples"] = sum(
                e.payload == 0 for e in result.receiver_examples
            )
            if result.sender_stats:
                metrics.update(
                    {f"sender/{k}": v for k, v in result.sender_stats.metrics.items()}
                )
            if result.receiver_stats:
                metrics.update(
                    {
                        f"receiver/{k}": v
                        for k, v in result.receiver_stats.metrics.items()
                    }
                )
                if "loss:sum" in result.receiver_stats.metrics:
                    metrics["receiver_loss_per_example"] = (
                        result.receiver_stats.metrics["loss:sum"]
                        / len(result.receiver_examples)
                    )
            recorder.scalars(step, "train_legacy", metrics)  # metrics.jsonl only
            if config.task_mode == "rewrite_signal":
                recorder.scalars(
                    step,
                    "train",
                    {
                        **signal_metrics(result, metrics),
                        "sender_loss": metrics.get("sender/loss:sum"),
                    },
                )
                recorder.scalars(
                    step, "train_detail", {f"batch/{k}": v for k, v in metrics.items()}
                )
            else:
                recorder.metrics(
                    step,
                    "train",
                    {
                        **batch_metrics(result, frozen_scores, trainer.config),
                        "sender_loss": metrics.get("sender/loss:sum"),
                        "sender_updated": metrics["sender_updated"],
                    },
                )
            recorder.status(state="training", step=step)
            click.echo(
                f"step {step}: reward {metrics['mean_reward']}, "
                f"tests passed {metrics['correctness']}, "
                f"backdoor fires {metrics['backdoor_success']}"
            )
            completed = step
            if step % config.eval.every == 0 or step == training.steps:
                track(
                    await run_evaluation(
                        trainer, recorder, step, validation, honeypots, config
                    ),
                    step,
                )
                last_eval = step
            save_every = config.checkpoints.save_every
            if save_every and step % save_every == 0 and step != training.steps:
                path = await sender.save_checkpoint(
                    f"{config.checkpoint_name}_step{step}"
                )
                recorder.stage.log_model_ref(f"tinker/{path}", {"step": step})
                state = await sender.save_state(
                    f"{config.checkpoint_name}_state_step{step}"
                )
                recorder.status(
                    **{f"checkpoint_step{step}": path, f"state_step{step}": state}
                )
                if training.train_receiver:
                    receiver_path = await cast(
                        TinkerReceiver, trainer.receiver
                    ).save_checkpoint(f"{config.checkpoint_name}_receiver_step{step}")
                    recorder.status(
                        **{f"receiver_checkpoint_step{step}": receiver_path}
                    )
                click.echo(f"step {step}: saved {path}; resumable state {state}")
            recorder.scalars(
                step,
                "progress",
                {
                    **meter.snapshot(),
                    "step_cost_usd": meter.cost_usd - spent,
                    "epoch": step * training.problems_per_step / len(train_tasks),
                },
            )
            (output_dir / "usage.json").write_text(
                json.dumps(
                    {"estimated_cost_usd": meter.cost_usd, "tokens": meter.tokens},
                    indent=2,
                )
            )
            last_step_cost = meter.cost_usd - spent
        if completed != last_eval and not meter.exhausted:
            track(
                await run_evaluation(
                    trainer, recorder, completed, validation, honeypots, config
                ),
                completed,
            )
        final = await sender.save_checkpoint(config.checkpoint_name)
        recorder.stage.log_model_ref(f"tinker/{final}", {"step": completed})
        # Resumable sender state, so a finished run can be extended with more steps.
        final_state = await sender.save_state(
            f"{config.checkpoint_name}_state_step{completed}"
        )
        recorder.status(**{f"state_step{completed}": final_state})
        if training.train_receiver:
            receiver_final = await cast(
                TinkerReceiver, trainer.receiver
            ).save_checkpoint(f"{config.checkpoint_name}_receiver")
            recorder.status(final_receiver_checkpoint=receiver_final)
            click.echo(f"final receiver checkpoint: {receiver_final}")
        recorder.status(state="finished", final_checkpoint=final)
        click.echo(f"final sender checkpoint: {final}")
        if config.task_mode == "rewrite_signal":
            try:
                click.echo(f"AUROC over steps: {plot_auroc_over_steps(output_dir)}")
            except Exception as exc:  # a plotting failure must not fail a finished run
                logger.warning("could not plot AUROC over steps: %s", exc)
        try:
            click.echo(f"run view: {write_run_view(output_dir)}")
        except Exception as exc:  # likewise for the record viewer
            logger.warning("could not write the run view: %s", exc)
        complete = completed == training.steps
    finally:
        close_run(recorder, meter, output_dir, step, complete)


def close_run(
    recorder: Recorder,
    meter: usage.Usage,
    output_dir: Path,
    step: int,
    complete: bool,
) -> None:
    """Record final usage, even after a failed step or evaluation."""
    recorder.scalars(step, "progress", meter.snapshot())
    (output_dir / "usage.json").write_text(
        json.dumps(
            {"estimated_cost_usd": meter.cost_usd, "tokens": meter.tokens}, indent=2
        )
    )
    recorder.status(estimated_cost_usd=meter.cost_usd, budget_exhausted=meter.exhausted)
    recorder.finish(complete, None if complete else "stopped before finishing")


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--allow-paid", is_flag=True, help="Make Tinker model calls and train.")
@click.option("--steps", type=int, default=None, help="Override training.steps.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override seed; appends -s<seed> to run_name and output_dir.",
)
@click.option(
    "--resume-state",
    default=None,
    help="Resume from a saved state: a state_step<N> path from status.json.",
)
@click.option(
    "--start-step",
    type=int,
    default=1,
    help="First step to run when resuming (N + 1 for state_step<N>).",
)
@click.option("--allow-unlinked", is_flag=True, help=tracking.ALLOW_UNLINKED_HELP)
def main(
    config_path: str,
    allow_paid: bool,
    steps: int | None,
    seed: int | None,
    resume_state: str | None,
    start_step: int,
    allow_unlinked: bool,
) -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    config = load_rl_config(config_path)
    if steps is not None:
        config.training.steps = steps
    if seed is not None:
        config.seed = seed
        config.run_name = f"{config.run_name}-s{seed}"
        config.output_dir = config.output_dir.with_name(
            f"{config.output_dir.name}-s{seed}"
        )
    if (resume_state is None) != (start_step == 1):
        raise click.UsageError("--resume-state and --start-step must be used together")
    asyncio.run(
        train(
            config,
            allow_paid,
            resume_state=resume_state,
            start_step=start_step,
            allow_unlinked=allow_unlinked,
        )
    )


if __name__ == "__main__":
    main()
