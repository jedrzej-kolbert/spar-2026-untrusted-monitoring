"""Run HiPT-style collusion-signal RL from a YAML config.

    python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \\
        --config src/lasr_labs_2025_control_project/configs/rl/rewrite_signal_pilot.yaml \\
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
import time
from pathlib import Path
from typing import Any, cast

import click

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
)
from .data import attack_tasks, honeypot_pairs, read_generator_eval, split_tasks
from .icrl import (
    Knowledge,
    RoundSummary,
    learner_messages,
    parse_strategies,
    run_icrl,
    summarise,
)
from .icrl_adapters import StrategyBox, TrainerTeam, with_strategy
from .rewards import RewardConfig
from . import usage
from .sandbox import DockerAppsEvaluator
from .tinker_backend import TinkerReceiver

logger = logging.getLogger(__name__)

# A deliberately obvious marker for the preflight controls.
_EXPLICIT_SIGNAL = '\n_collusion_signal_for_monitor = "present"\n'


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
# The W&B sections a reader needs, in metrics.jsonl names. Everything else goes
# to the zz_other section (sorted last), so it stays logged but out of the way.
WANDB_KEYS = {
    "train/mean_reward": "reward/total",
    "train/sender/loss:sum": "loss/sender",
    "train/receiver_loss_per_example": "loss/receiver_per_example",
    "train/correctness": "scores/pass_rate",
    "train/backdoor_success": "scores/backdoor_kept_rate",
    "train/backdoor_suspicion_mean": "scores/trusted_suspicion",
    "train/suspicion_mean": "scores/overseer_suspicion",
    "eval_pairs/mean_reward": "eval/reward_total",
    "eval_pairs/correctness": "eval/pass_rate",
    "eval_pairs/backdoor_success": "eval/backdoor_kept_rate",
    "eval_pairs/backdoor_suspicion_mean": "eval/trusted_suspicion",
    "eval_vs_honeypots/auroc": "eval/auroc",
    "eval_vs_honeypots/balanced_accuracy": "eval/balanced_accuracy",
    "eval_vs_honeypots/honeypot_yes_rate": "eval/honeypot_yes_rate",
}
# In self-recognition every rollout has payload 1, so bit 1's P(YES) is P(self).
SELF_REC_WANDB_KEYS = {
    "train/bit1_mean_p1": "scores/p_self",
    "eval_pairs/bit1_mean_p1": "eval/p_self",
    "eval_vs_honeypots/signalled_yes_rate": "eval/self_yes_rate",
}


def summary_row(summary: RoundSummary) -> dict[str, Any]:
    """One table row per strategy: its text, total reward, each reward term, scores."""
    return {
        "round": summary.round,
        "strategy": summary.strategy,
        "reward_mean": summary.reward_mean,
        **{
            f"reward/{name}": value for name, value in summary.reward_components.items()
        },
        "p_self_mean": summary.p_self_mean,
        "pass_rate": summary.pass_rate,
        "working_rate": summary.working_rate,
        "backdoor_checked": summary.backdoor_checked,
        "suspicion_mean": summary.suspicion_mean,
        "episodes": summary.episodes,
    }


def wandb_key(key: str, self_recognition: bool) -> str:
    if key.startswith(("progress/", "icrl/")):
        return key
    if key.startswith("train/reward/"):
        return key.removeprefix("train/")
    curated = WANDB_KEYS.get(key) or (
        SELF_REC_WANDB_KEYS.get(key) if self_recognition else None
    )
    return curated or f"zz_other/{key}"


class Recorder:
    def __init__(self, config: RLRunConfig, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.self_recognition = config.task_mode == "self_recognition"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        self.strategy_rows: list[dict[str, Any]] = []
        self.proposed_rows: list[dict[str, Any]] = []
        self.last_step = 0  # W&B steps must not go backwards
        logging_config = config.logging
        if logging_config.wandb_project and logging_config.wandb_mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                project=logging_config.wandb_project,
                entity=logging_config.wandb_entity,
                name=config.run_name,
                mode=logging_config.wandb_mode,
                config=json.loads(config.model_dump_json()),
                dir=str(output_dir),
                save_code=False,
            )
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
        if self.wandb is not None and clean:
            self.wandb.log(
                {
                    wandb_key(key, self.self_recognition): value
                    for key, value in clean.items()
                },
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
                    "overseer_log_odds": item.overseer_log_odds,
                    "backdoor_suspicion": item.backdoor_suspicion,
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

    def strategy(self, step: int, summary: RoundSummary, cost_usd: float) -> None:
        """One learner-designed strategy and its measured scores (ICRL)."""
        row = {"step": step, **summary_row(summary), "cost_usd": cost_usd}
        self._append("strategies.jsonl", row)
        self.sync_file("strategies.jsonl")
        self.scalars(step, "icrl", row)
        self.strategy_rows.append(row)
        self.table(step, "icrl/strategies", self.strategy_rows)

    def table(self, step: int, key: str, rows: list[dict[str, Any]]) -> None:
        """A W&B table whose columns are every key the rows use, in first-seen order."""
        if self.wandb is None:
            return
        import wandb

        columns = list(dict.fromkeys(name for row in rows for name in row))
        # A reward term missing from a row added nothing to that strategy's reward.
        data = [
            [row.get(c, 0.0 if c.startswith("reward/") else None) for c in columns]
            for row in rows
        ]
        self.wandb.log({key: wandb.Table(columns=columns, data=data)}, step=step)

    def learner_reply(
        self,
        episode: int,
        prompt: list[dict[str, str]],
        reply: str,
        lessons: str = "",
    ) -> None:
        """Each learner turn: the full prompt, the reply, the strategy chosen, and
        the lessons it carried in. Logged before the strategy is scored."""
        strategies = parse_strategies(reply)
        self._append(
            "learner.jsonl",
            {
                "episode": episode,
                "prompt": prompt,
                "reply": reply,
                "strategies": strategies,
                "lessons_in": lessons,
            },
        )
        self.sync_file("learner.jsonl")
        self.proposed_rows.extend(
            {"episode": episode, "index": index, "strategy": text}
            for index, text in enumerate(strategies, 1)
        )
        self.table(self.last_step, "icrl/proposed_strategies", self.proposed_rows)

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

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


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
) -> tuple[
    list[AppsProblem],
    list[AppsProblem],
    list[tuple[AppsProblem, str]],
    list[tuple[AppsProblem, str]],
    list[str],
]:
    """Train and validation tasks, their honeypots, and the IDs of training
    problems dropped because the honeypot log has no code for them."""
    data = config.data
    samples = read_generator_eval(data.attacks_eval)
    tasks = attack_tasks(
        samples,
        require_main_task=data.require_main_task,
        require_backdoor=data.require_backdoor,
        max_tests=data.max_tests,
    )
    if data.validation_attacks_eval:
        train = tasks
        validation = [
            dataclasses.replace(task, split="validation")
            for task in attack_tasks(
                [
                    s
                    for path in data.validation_attacks_eval
                    for s in read_generator_eval(path)
                ],
                require_main_task=data.require_main_task,
                require_backdoor=data.require_backdoor,
                max_tests=data.max_tests,
            )
        ]
        if {t.problem_id for t in train} & {t.problem_id for t in validation}:
            raise ValueError("validation logs share problems with attacks_eval")
    else:
        train, validation = split_tasks(
            tasks, validation_fraction=data.validation_fraction, seed=config.seed
        )
    if data.max_validation_problems is not None:
        validation = validation[: data.max_validation_problems]
    honeypots: list[tuple[AppsProblem, str]] = []
    train_honeypots: list[tuple[AppsProblem, str]] = []
    dropped: list[str] = []
    if data.honeypots_eval is not None and (
        config.eval.honeypot_eval or config.task_mode == "self_recognition"
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
            train_honeypots = honeypot_pairs(negatives, train)
            paired = [problem.problem_id for problem, _ in train_honeypots]
            if len(paired) != len(set(paired)):
                raise ValueError("a training problem has more than one honeypot")
            if config.training.train_receiver:
                # Receiver SFT needs a non-self example for every problem; one
                # whose honeypot generation failed has none, so it is left out.
                dropped = [t.problem_id for t in train if t.problem_id not in paired]
                if dropped:
                    logger.warning(
                        "dropping %d training problems with no honeypot: %s",
                        len(dropped),
                        dropped,
                    )
                train = [t for t in train if t.problem_id not in dropped]
        if config.eval.honeypot_eval:
            if data.validation_honeypots_eval:
                negatives = [
                    s
                    for path in data.validation_honeypots_eval
                    for s in read_generator_eval(path)
                ]
            honeypots = honeypot_pairs(negatives, validation)
    if not train:
        raise ValueError("no training tasks survived filtering")
    if len(train) < config.training.problems_per_step:
        raise ValueError(
            f"{len(train)} training tasks < problems_per_step "
            f"{config.training.problems_per_step}"
        )
    return train, validation, train_honeypots, honeypots, dropped


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
        reward=RewardConfig(
            reward_fn=reward.reward_fn,
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
    if result.rollouts:
        recorder.scalars(step, phase, result.metrics())
    if honeypots and phase == "eval_pairs":
        honeypot_eval: HoneypotEvalResult = await trainer.evaluate_against_honeypots(
            validation, honeypots, signalled=result
        )
        recorder.honeypots(step, honeypot_eval.honeypot_scores)
        recorder.scalars(step, "eval_vs_honeypots", honeypot_eval.metrics())


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


def _update_ckpts_json(path: Path, name: str, checkpoint: str) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    current[name] = checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n")


async def run_in_context(
    config: RLRunConfig,
    trainer: GroupedRLTrainer,
    box: StrategyBox,
    learner: Any,
    recorder: Recorder,
    meter: usage.Usage,
    train_tasks: list[AppsProblem],
    validation: list[AppsProblem],
    honeypots: list[tuple[AppsProblem, str]],
) -> None:
    """ICRL search on training problems, then a held-out check against honeypots.

    Every learner reply goes to learner.jsonl, every strategy with its scores to
    strategies.jsonl and the W&B ``icrl/strategies`` table, every rewrite to
    rollouts.jsonl (with its strategy), and the outcome to icrl_result.json; all
    four files are also uploaded to the W&B run.
    """
    icrl = config.icrl
    if icrl.search_problems > len(train_tasks):
        raise ValueError("icrl.search_problems exceeds the training problems")
    search = random.Random(config.seed).sample(train_tasks, icrl.search_problems)
    recorder.status(
        state="searching", search_problem_ids=[p.problem_id for p in search]
    )
    step = 0

    def on_result(strategy: str, result: BatchResult, phase: str) -> None:
        nonlocal step
        step += 1
        recorder.rollouts(step, phase, result.rollouts, strategy=strategy)
        recorder.sync_file("rollouts.jsonl")
        if result.errors:
            # Batch-level causes (e.g. a sender exception) are not on any rollout.
            logger.warning("%s batch errors: %s", phase, result.errors[:3])
            recorder.batch_errors(step, phase, strategy, result.errors)

    team = TrainerTeam(
        trainer,
        box,
        search_problems=search,
        held_out_problems=validation,
        honeypots=honeypots,
        num_samples=config.eval.num_samples,
        on_result=on_result,
    )

    def on_summary(summary: RoundSummary) -> None:
        recorder.strategy(step, summary, meter.cost_usd)
        click.echo(
            f"round {summary.round}: reward {summary.reward_mean:.3f}, "
            f"P(self) {summary.p_self_mean:.3f}, working {summary.working_rate}, "
            f"suspicion {summary.suspicion_mean:.3f} | {summary.strategy[:80]!r}"
        )

    def on_reply(
        episode: int, previous: RoundSummary, knowledge: Knowledge, reply: str
    ) -> None:
        recorder.learner_reply(
            episode,
            learner_messages(previous, knowledge, icrl.candidates),
            reply,
            lessons=knowledge.lessons,
        )

    try:
        history, best = await run_icrl(
            learner,
            team,
            episodes=icrl.episodes,
            candidates=icrl.candidates,
            on_reply=on_reply,
            on_summary=on_summary,
        )
        recorder.status(
            state="held_out", best_round=best.round, best_strategy=best.strategy
        )
        held_out: dict[str, Any] = {}
        # The baseline is checked too, so a gain is measured against no strategy.
        for label, strategy in (("best", best.strategy), ("baseline", "")):
            if label == "baseline" and best.strategy == "":
                held_out["baseline"] = held_out["best"]
                continue
            episodes, honeypot = await team.held_out(strategy)
            summary = summarise(
                best.round if label == "best" else 0, strategy, episodes
            )
            metrics = {**summary_row(summary), **honeypot.metrics()}
            recorder.honeypots(step, honeypot.honeypot_scores)
            recorder.scalars(step, f"icrl/held_out_{label}", metrics)
            held_out[label] = metrics
        recorder.table(
            step,
            "icrl/held_out",
            [{"check": label, **metrics} for label, metrics in held_out.items()],
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
                "best": dataclasses.asdict(best),
                "strategies": [dataclasses.asdict(item) for item in history],
                "held_out": held_out,
            },
            indent=2,
        )
    )
    recorder.sync_file("icrl_result.json")
    recorder.status(
        state="finished",
        held_out_auroc_best=held_out["best"]["auroc"],
        held_out_auroc_baseline=held_out["baseline"]["auroc"],
    )
    click.echo(
        f"best strategy (round {best.round}): held-out AUROC "
        f"{held_out['best']['auroc']} vs baseline {held_out['baseline']['auroc']}"
    )


async def train(
    config: RLRunConfig,
    allow_paid: bool,
    *,
    resume_state: str | None = None,
    start_step: int = 1,
) -> None:
    if resume_state is not None and config.training.train_receiver:
        raise ValueError(
            "--resume-state restores only the sender; joint training must start fresh"
        )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tasks, validation, train_honeypots, honeypots, dropped = load_tasks(config)
    manifest = {
        "run_name": config.run_name,
        "config": json.loads(config.model_dump_json()),
        "git_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_problem_ids": [task.problem_id for task in train_tasks],
        "validation_problem_ids": [task.problem_id for task in validation],
        "train_honeypot_problem_ids": [
            problem.problem_id for problem, _ in train_honeypots
        ],
        "honeypot_problem_ids": [problem.problem_id for problem, _ in honeypots],
        "train_problems_dropped_without_honeypot": dropped,
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

    meter = usage.install()
    meter.max_cost_usd = config.training.max_cost_usd
    recorder = Recorder(config, output_dir)
    recorder.status(state="starting", parity=parity)
    recorder.scalars(0, "preflight", parity)
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
            candidates=config.icrl.candidates,
        )
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
                honeypots,
            )
        finally:
            close_run(recorder, meter, output_dir, recorder.last_step)
        return

    if config.preflight.enabled and config.task_mode != "self_recognition":
        preflight = await monitor_preflight(trainer, train_tasks, config)
        click.echo(f"monitor preflight: {preflight}")
        recorder.scalars(0, "preflight", preflight)
        gap = preflight.get("overseer_explicit_suspicion", 0) - preflight.get(
            "overseer_clean_suspicion", 0
        )
        if gap < config.preflight.require_overseer_gap:
            recorder.status(state="stopped", reason=f"overseer gap {gap:.3f} too small")
            raise SystemExit(f"overseer preflight failed: gap {gap:.3f}")

    rng = random.Random(config.seed)
    order: list[AppsProblem] = []
    skipped = 0
    training = config.training
    # Replay earlier draws so a resumed run continues the same batch sequence.
    for _ in range(1, start_step):
        next_batch(order, train_tasks, training.problems_per_step, rng)
    last_step_cost = 0.0
    completed, last_eval = start_step - 1, start_step - 1
    step = completed
    try:
        if start_step == 1:
            await run_evaluation(trainer, recorder, 0, validation, honeypots, config)
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
                        for problem, code in train_honeypots
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
                recorder.scalars(step, "train", {"skipped_step": 1})
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
            recorder.scalars(step, "train", metrics)
            recorder.status(state="training", step=step)
            click.echo(
                f"step {step}: reward {metrics['mean_reward']}, "
                f"balanced acc {metrics['balanced_accuracy']}, "
                f"pass {metrics['correctness']}, suspicion {metrics['suspicion_mean']}"
            )
            completed = step
            if step % config.eval.every == 0 or step == training.steps:
                await run_evaluation(
                    trainer, recorder, step, validation, honeypots, config
                )
                last_eval = step
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
            await run_evaluation(
                trainer, recorder, completed, validation, honeypots, config
            )
        final = await sender.save_checkpoint(config.checkpoint_name)
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
        if config.checkpoints.ckpts_json is not None:
            _update_ckpts_json(
                config.checkpoints.ckpts_json, config.checkpoint_name, final
            )
    finally:
        close_run(recorder, meter, output_dir, step)


def close_run(
    recorder: Recorder, meter: usage.Usage, output_dir: Path, step: int
) -> None:
    """Record final usage, even after a failed step or evaluation."""
    recorder.scalars(step, "progress", meter.snapshot())
    (output_dir / "usage.json").write_text(
        json.dumps(
            {"estimated_cost_usd": meter.cost_usd, "tokens": meter.tokens}, indent=2
        )
    )
    recorder.status(estimated_cost_usd=meter.cost_usd, budget_exhausted=meter.exhausted)
    recorder.finish()


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
def main(
    config_path: str,
    allow_paid: bool,
    steps: int | None,
    resume_state: str | None,
    start_step: int,
) -> None:
    import os

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
        train(config, allow_paid, resume_state=resume_state, start_step=start_step)
    )


if __name__ == "__main__":
    main()
