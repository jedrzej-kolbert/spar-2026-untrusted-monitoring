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
import random
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Any, Callable

import click

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.tracking.core import jsonable
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
    Rollout,
    auroc,
)
from .data import attack_tasks, honeypot_pairs, read_generator_eval, split_tasks
from .plots import plot_auroc_over_steps
from .rewards import RewardConfig
from .sandbox import DockerAppsEvaluator
from .viewer import write_run_view

logger = logging.getLogger(__name__)

# A deliberately obvious marker for the preflight controls.
_EXPLICIT_SIGNAL = '\n_collusion_signal_for_monitor = "present"\n'


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
#: Files uploaded to the W&B run when it ends (whichever exist).
RECORD_FILES = (
    "manifest.json",
    "original_attack_suspicion.json",
    "metrics.jsonl",
    "rollouts.jsonl",
    "honeypot_scores.jsonl",
    "status.json",
    "code.diff",
    "auroc_over_steps.svg",
    "auroc_over_steps.json",
    "run_view.html",
)


def code_state(output_dir: Path) -> dict[str, Any]:
    """Branch, commit and whether tracked files differ; a dirty tree's diff is saved."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout

    try:
        diff = git("diff", "HEAD")  # staged and unstaged changes to tracked files
        if diff:
            (output_dir / "code.diff").write_text(diff)
        return {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "commit": git("rev-parse", "HEAD").strip(),
            "dirty": bool(diff),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None}


class Recorder:
    """Local record files, plus one W&B run opened like the pipeline's stage runs.

    Local files are written exactly as before. The W&B run (job type ``gbrl``) gets the
    run identity as config and ``key:value`` tags, links to its input logs, and the
    sections the collusion-rl-v2 runs use: ``train/``, ``validation/`` and their
    ``*_detail/``. W&B failures after the run opens are logged, never raised, so they
    cannot stop training.
    """

    def __init__(
        self,
        config: RLRunConfig,
        output_dir: Path,
        *,
        allow_unlinked: bool = False,
        resumed_from: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # Fields written in the YAML's logging block win over WANDB_* variables.
        logging_config = config.logging
        for field in logging_config.model_fields_set:
            value = getattr(logging_config, field)
            if value is not None:
                os.environ[field.upper()] = value
        data = config.data
        identity = config.identity
        self.stage = tracking.StageRun(
            "gbrl",
            config.run_name,
            config={
                **json.loads(config.model_dump_json()),
                "run": identity,
                "code": code_state(output_dir),
                "resumed_from": resumed_from,
            },
            tags=[f"{key}:{value}" for key, value in identity.items()]
            + (["resumed"] if resumed_from else []),
            inputs=[
                tracking.resolve_input(path)
                for path in (data.attacks_eval, data.honeypots_eval)
                if path is not None
            ],
            models=[config.models.receiver],
            allow_unlinked=allow_unlinked,
        ).__enter__()
        self.wandb: Any = self.stage.run
        if self.wandb is not None:
            self.status(wandb_url=self.wandb.url)

    def _wandb(self, what: str, action: Callable[[], Any]) -> None:
        if self.wandb is None:
            return
        try:
            action()
        except Exception as exc:  # telemetry must not stop a training run
            logger.warning("W&B %s failed: %s", what, exc)

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with open(self.output_dir / name, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def scalars(self, step: int, prefix: str, values: dict[str, Any]) -> None:
        """metrics.jsonl only; W&B gets the sections from ``log``."""
        clean = {
            f"{prefix}/{key}": value
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self._append("metrics.jsonl", {"step": step, **clean})
        if prefix == "preflight":
            self.log(step, clean)

    def log(self, step: int, values: dict[str, Any]) -> None:
        """Numbers to W&B under their section/key names; None and bools are dropped."""
        clean = {
            key: value
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if clean:
            self._wandb("log", lambda: self.wandb.log(clean, step=step))

    def roc(self, step: int, points: list[tuple[float, float]]) -> None:
        if points:
            self._wandb(
                "ROC",
                lambda: self.stage.log_roc(
                    "validation/roc_collusion",
                    {f"step {step}": points},
                    "Collusion ROC on validation",
                ),
            )

    def model(self, path: str, step: int) -> None:
        self._wandb(
            "model reference",
            lambda: self.stage.log_model_ref(f"tinker/{path}", {"step": step}),
        )

    def rollouts(self, step: int, phase: str, rollouts: list[Rollout]) -> None:
        for item in rollouts:
            generation = item.generation
            self._append(
                "rollouts.jsonl",
                {
                    "step": step,
                    "phase": phase,
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
                    "backdoor_suspicion": item.backdoor_suspicion,
                    "kl": item.kl,
                    "reward": item.reward,
                    "reward_components": item.reward_components,
                    "advantage": item.advantage,
                    "errors": item.errors,
                },
            )

    def honeypots(self, step: int, scores: list[tuple[str, float]]) -> None:
        for problem_id, p_yes in scores:
            self._append(
                "honeypot_scores.jsonl",
                {"step": step, "problem_id": problem_id, "p_yes": p_yes},
            )

    def status(self, **values: Any) -> None:
        path = self.output_dir / "status.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(values, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        path.write_text(json.dumps(current, indent=2, default=str))
        self._wandb("summary", lambda: self.wandb.summary.update(jsonable(values)))

    def finish(self, complete: bool) -> None:
        """Upload the record files and close the run (tagged partial if incomplete)."""
        for name in RECORD_FILES + tuple(
            p.name for p in self.output_dir.glob("manifest_resume_step*.json")
        ):
            if (self.output_dir / name).exists():
                self._wandb(
                    f"upload of {name}",
                    lambda name=name: self.wandb.save(
                        str(self.output_dir / name),
                        base_path=str(self.output_dir),
                        policy="now",
                    ),
                )
        self._wandb(
            "finish",
            lambda: self.stage.finish(
                complete=complete,
                reason=None if complete else "stopped before finishing",
            ),
        )
        self.wandb = None


def working(item: Rollout) -> bool:
    """Passes its tests and its backdoor fires (as the collusion-rl-v2 runs count it)."""
    return bool(
        item.execution
        and item.execution.passed
        and item.backdoor is not None
        and item.backdoor.passed
    )


def shared_metrics(result: BatchResult, metrics: dict[str, Any]) -> dict[str, Any]:
    """Key numbers with the same definitions and names as the collusion-rl-v2 runs,
    then this task's own (signal ON vs OFF, the signal overseer)."""
    total = len(result.rollouts)
    return {
        "reward": metrics.get("mean_reward"),
        "working_rate": sum(map(working, result.rollouts)) / total if total else None,
        "tests_passed_rate": metrics.get("correctness"),
        "backdoor_fires_rate": metrics.get("backdoor_success"),
        "on_off_balanced_accuracy": metrics.get("balanced_accuracy"),
        "on_off_auroc": metrics.get("on_off_auroc"),
        "p1_gap": metrics.get("p1_gap"),
        "signal_suspicion": metrics.get("suspicion_mean"),
        "suspicion_increase_mean": metrics.get("suspicion_increase_mean"),
    }


def collusion_auroc(
    result: BatchResult, honeypot_scores: list[tuple[str, float]]
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """AUROC and ROC of the receiver's P(YES): working signal-ON rewrites vs honeypots
    (the population the collusion-rl-v2 runs log as validation/auroc)."""
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


def sectioned(
    section: str, key: dict[str, Any], **detail: dict[str, Any]
) -> dict[str, Any]:
    """``section/<key>`` plus ``section_detail/<source>/<name>`` for every other number."""
    out = {f"{section}/{name}": value for name, value in key.items()}
    for source, values in detail.items():
        out.update({f"{section}_detail/{source}/{k}": v for k, v in values.items()})
    return out


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def load_tasks(
    config: RLRunConfig,
) -> tuple[list[AppsProblem], list[AppsProblem], list[tuple[AppsProblem, str]]]:
    data = config.data
    samples = read_generator_eval(data.attacks_eval)
    tasks = attack_tasks(
        samples,
        require_main_task=data.require_main_task,
        require_backdoor=data.require_backdoor,
        max_tests=data.max_tests,
    )
    train, validation = split_tasks(
        tasks, validation_fraction=data.validation_fraction, seed=config.seed
    )
    if data.max_validation_problems is not None:
        validation = validation[: data.max_validation_problems]
    honeypots: list[tuple[AppsProblem, str]] = []
    if data.honeypots_eval is not None and config.eval.honeypot_eval:
        honeypots = honeypot_pairs(read_generator_eval(data.honeypots_eval), validation)
    if not train:
        raise ValueError("no training tasks survived filtering")
    if len(train) < config.training.problems_per_step:
        raise ValueError(
            f"{len(train)} training tasks < problems_per_step "
            f"{config.training.problems_per_step}"
        )
    return train, validation, honeypots


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
            stego_scale=reward.stego_scale,
            overseer_scale=reward.overseer_scale,
            overseer_threshold=reward.overseer_threshold,
            hipt_gate_on_pass=reward.hipt_gate_on_pass,
            failed_backdoor_weight=reward.failed_backdoor_weight,
            backdoor_suspicion_weight=reward.backdoor_suspicion_weight,
            kl_coef=reward.kl_coef,
        ),
        train_receiver=training.train_receiver,
        max_concurrency=training.max_concurrency,
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
) -> tuple[GroupedRLTrainer, Any]:
    """The trainer and the concrete Tinker sender (for checkpointing)."""
    from .tinker_backend import (
        FrozenYesNoMonitor,
        TextScoreBackdoorMonitor,
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
    receiver = FrozenYesNoMonitor(
        config.models.receiver,
        get_prompt_by_name(prompts.receiver_system),
        allow_paid=True,
    )
    overseer = None
    if config.models.overseer:
        overseer = FrozenYesNoMonitor(
            config.models.overseer,
            get_prompt_by_name(prompts.overseer_system),
            allow_paid=True,
        )
    backdoor_monitor = None
    if reward.backdoor_suspicion_weight > 0 and config.models.backdoor_monitor:
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
) -> None:
    if not validation:
        return
    try:
        result = await trainer.evaluate(
            validation, config.eval.num_samples, balanced=True
        )
        phase = "eval_pairs"
    except BatchInfrastructureError as error:
        # Keep the records and partial metrics; a failed evaluation does not stop training.
        result, phase = error.result, "eval_pairs_failed"
        logger.warning("evaluation at step %d had errors: %s", step, result.errors[:3])
    recorder.rollouts(step, phase, result.rollouts)
    pairs = result.metrics() if result.rollouts else {}
    if result.rollouts:
        recorder.scalars(step, phase, pairs)
    if phase != "eval_pairs":
        recorder.log(step, {"validation_detail/failed": 1})
        return
    scores: list[tuple[str, float]] = []
    vs_honeypots: dict[str, Any] = {}
    if honeypots:
        honeypot_eval: HoneypotEvalResult = await trainer.evaluate_against_honeypots(
            validation, honeypots, signalled=result
        )
        recorder.honeypots(step, honeypot_eval.honeypot_scores)
        vs_honeypots = honeypot_eval.metrics()
        recorder.scalars(step, "eval_vs_honeypots", vs_honeypots)
        scores = honeypot_eval.honeypot_scores
    numbers, roc = collusion_auroc(result, scores)
    key = {
        **shared_metrics(result, pairs),
        **numbers,
        "honeypot_balanced_accuracy": vs_honeypots.get("balanced_accuracy"),
        "signalled_yes_rate": vs_honeypots.get("signalled_yes_rate"),
        "honeypot_yes_rate": vs_honeypots.get("honeypot_yes_rate"),
    }
    recorder.log(
        step, sectioned("validation", key, pairs=pairs, honeypots=vs_honeypots)
    )
    recorder.roc(step, roc)


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


async def train(
    config: RLRunConfig,
    allow_paid: bool,
    *,
    resume_state: str | None = None,
    start_step: int = 1,
    allow_unlinked: bool = False,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tasks, validation, honeypots = load_tasks(config)
    manifest = {
        "run_name": config.run_name,
        "config": json.loads(config.model_dump_json()),
        "git_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_problem_ids": [task.problem_id for task in train_tasks],
        "validation_problem_ids": [task.problem_id for task in validation],
        "honeypot_problem_ids": [problem.problem_id for problem, _ in honeypots],
    }
    manifest_name = "manifest.json"
    if resume_state is not None:
        manifest["resumed_from"] = {"state": resume_state, "start_step": start_step}
        manifest_name = f"manifest_resume_step{start_step}.json"
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    click.echo(
        f"tasks: {len(train_tasks)} train, {len(validation)} validation, "
        f"{len(honeypots)} honeypots"
    )

    sandbox = config.sandbox
    ensure_image(sandbox.image)
    evaluator = DockerAppsEvaluator(
        image=sandbox.image,
        timeout_seconds=sandbox.timeout_seconds,
        control_arena_parity=sandbox.control_arena_parity,
        batch_tests=sandbox.batch_tests,
        max_containers=sandbox.max_containers,
        num_tests=config.data.max_tests,
    )
    parity = await sandbox_parity_check(evaluator, train_tasks)
    click.echo(f"sandbox parity with eval scores: {parity}")
    if not allow_paid:
        click.echo(
            "offline mode: no model calls made. Re-run with --allow-paid to train."
        )
        return

    recorder = Recorder(
        config,
        output_dir,
        allow_unlinked=allow_unlinked,
        resumed_from=manifest.get("resumed_from"),
    )
    complete = False
    try:
        recorder.status(state="starting", parity=parity)
        recorder.scalars(0, "preflight", parity)
        trainer, sender = await build_trainer(
            config, evaluator, resume_state=resume_state
        )

        if config.preflight.enabled:
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
            values = list(baselines.values())
            click.echo(
                f"original attacks: mean overseer suspicion "
                f"{sum(values) / len(values):.3f} over {len(values)}"
            )
            recorder.scalars(
                0,
                "preflight",
                {"original_attack_suspicion_mean": sum(values) / len(values)},
            )

        if start_step == 1:
            await run_evaluation(trainer, recorder, 0, validation, honeypots, config)

        rng = random.Random(config.seed)
        order: list[AppsProblem] = []
        skipped = 0
        training = config.training
        # Replay earlier draws so a resumed run continues the same batch sequence.
        for _ in range(1, start_step):
            next_batch(order, train_tasks, training.problems_per_step, rng)
        for step in range(start_step, training.steps + 1):
            batch = next_batch(order, train_tasks, training.problems_per_step, rng)
            strength = apply_overseer_strength(trainer, config, step)
            started = time.time()
            try:
                result = await trainer.train_step(batch)
            except BatchInfrastructureError as error:
                result = error.result
                recorder.rollouts(step, "train_failed", result.rollouts)
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
                recorder.scalars(step, "train", {"skipped_step": 1})
                recorder.log(step, {"train_detail/skipped_step": 1})
                rollout_errors = [e for item in result.rollouts for e in item.errors]
                logger.warning(
                    "step %d skipped: %s", step, (result.errors + rollout_errors)[:3]
                )
                if skipped > training.max_skipped_steps:
                    recorder.status(state="stopped", step=step, reason="too many skips")
                    raise
                continue
            recorder.rollouts(step, "train", result.rollouts)
            metrics = dict(result.metrics())
            metrics["overseer_strength"] = strength
            metrics["step_seconds"] = time.time() - started
            metrics["sender_updated"] = float(bool(result.sender_updated))
            if result.sender_stats:
                metrics.update(
                    {f"sender/{k}": v for k, v in result.sender_stats.metrics.items()}
                )
            recorder.scalars(step, "train", metrics)
            key = shared_metrics(result, metrics)
            key["sender_loss"] = metrics.get("sender/loss:sum")
            recorder.log(step, sectioned("train", key, batch=metrics))
            recorder.status(state="training", step=step)
            click.echo(
                f"step {step}: reward {metrics['mean_reward']}, "
                f"balanced acc {metrics['balanced_accuracy']}, "
                f"pass {metrics['correctness']}, suspicion {metrics['suspicion_mean']}"
            )
            if step % config.eval.every == 0 or step == training.steps:
                await run_evaluation(
                    trainer, recorder, step, validation, honeypots, config
                )
            save_every = config.checkpoints.save_every
            if save_every and step % save_every == 0 and step != training.steps:
                path = await sender.save_checkpoint(
                    f"{config.checkpoint_name}_step{step}"
                )
                state = await sender.save_state(
                    f"{config.checkpoint_name}_state_step{step}"
                )
                recorder.status(
                    **{f"checkpoint_step{step}": path, f"state_step{step}": state}
                )
                recorder.model(path, step)
                click.echo(f"step {step}: saved {path}; resumable state {state}")
        final = await sender.save_checkpoint(config.checkpoint_name)
        recorder.status(state="finished", final_checkpoint=final)
        recorder.model(final, training.steps)
        click.echo(f"final sender checkpoint: {final}")
        try:
            figure = plot_auroc_over_steps(output_dir)
            click.echo(f"AUROC over steps: {figure}")
        except Exception as exc:  # a plotting failure must not fail a finished run
            logger.warning("could not plot AUROC over steps: %s", exc)
        try:
            click.echo(f"run view: {write_run_view(output_dir)}")
        except Exception as exc:  # likewise for the record viewer
            logger.warning("could not write the run view: %s", exc)
        complete = True
    finally:
        recorder.finish(complete)


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--allow-paid", is_flag=True, help="Make Tinker model calls and train.")
@click.option("--steps", type=int, default=None, help="Override training.steps.")
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
    resume_state: str | None,
    start_step: int,
    allow_unlinked: bool,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    config = load_rl_config(config_path)
    if steps is not None:
        config.training.steps = steps
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
