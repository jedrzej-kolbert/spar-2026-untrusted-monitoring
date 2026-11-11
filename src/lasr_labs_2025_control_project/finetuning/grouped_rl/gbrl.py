"""GBRL: gradient-based RL of the sender (and optionally the receiver) with
Tinker LoRA, evaluated on validation every ``eval.every`` steps."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from typing import TYPE_CHECKING, Any, cast

import click

from .config import RLRunConfig
from .core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    GroupedRLTrainer,
)
from . import usage
from .plots import plot_auroc_over_steps
from .tinker_backend import TinkerReceiver
from .viewer import write_run_view
from .records import (
    Recorder,
    batch_metrics,
    close_run,
    collusion_auroc,
    self_rec_roc,
    signal_metrics,
)

if TYPE_CHECKING:
    from .run import Tasks

logger = logging.getLogger(__name__)


# A deliberately obvious marker for the preflight controls.
_EXPLICIT_SIGNAL = '\n_collusion_signal_for_monitor = "present"\n'


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


async def run_gbrl(
    config: RLRunConfig,
    trainer: GroupedRLTrainer,
    sender: Any,
    recorder: Recorder,
    meter: usage.Usage,
    tasks: Tasks,
    start_step: int,
) -> None:
    """Train for ``training.steps`` steps from ``start_step``, then save the final checkpoint."""
    train_tasks, validation = tasks.train, tasks.validation
    output_dir = config.output_dir
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
