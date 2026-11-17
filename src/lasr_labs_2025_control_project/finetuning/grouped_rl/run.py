"""Run collusion RL from a YAML config (example config: ``rl-steps.md``).

    python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \\
        --config <config.yaml> [--allow-paid]

Without ``--allow-paid`` it loads data, writes the manifest, and checks the Docker
sandbox against the eval log's own scores, without any model calls. With it, the
run trains (``method: gradient``, ``gbrl.py``) or searches (``method: in_context``,
``icrl_run.py``); both write their records through ``records.py``.

Outputs in ``output_dir``: manifest.json, rollouts.jsonl (every candidate, including
failures), metrics.jsonl, status.json, and the final checkpoint path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from typing import Any, NamedTuple

import click

from lasr_labs_2025_control_project import tracking
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name

from .channels import MonitorViewChannel
from .config import RLRunConfig, load_rl_config
from .core import (
    AppsProblem,
    GroupedRLConfig,
    GroupedRLTrainer,
)
from .data import attack_tasks, honeypot_pairs, read_generator_eval, split_tasks
from .icrl_adapters import StrategyBox, with_strategy
from .rewards import RewardConfig
from . import usage
from .sandbox import DockerAppsEvaluator
from .gbrl import run_gbrl
from .icrl_run import run_in_context
from .records import close_run, code_state, open_recorder

logger = logging.getLogger(__name__)


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
    else:  # self_recognition, which RLRunConfig requires to have honeypots_eval
        train, validation = split_tasks(
            tasks, validation_fraction=data.validation_fraction, seed=SPLIT_SEED
        )
        assert data.honeypots_eval is not None
        negatives = read_generator_eval(data.honeypots_eval)
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
        # self-recognition only, as in the game: comment-stripped code, not the jammed view
        trusted_reads_original=config.task_mode == "self_recognition",
    )


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
    await run_gbrl(config, trainer, sender, recorder, meter, tasks, start_step)


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
