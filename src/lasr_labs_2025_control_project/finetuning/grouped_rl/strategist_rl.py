"""GBRL of the strategist: centralized ICRL's frozen team, with gradient updates to
the model that writes the strategy instead of in-context feedback.

Each step the strategist sees only ICRL's first prompt (the centralized learner's
system prompt and "Describe your strategy." with the three output fields) and
samples ``training.group_size`` strategies. Every strategy goes through the same
``TrainerTeam`` ICRL uses, on the same ``training.problems_per_step`` search
problems: the frozen sender rewrites each attack with the encoder instructions, the
frozen judge reads the summary of the rewrite and of the problem's honeypot with
the decoder instructions, and each rewrite gets ICRL's paired reward
``gate * (P(YES | rewrite) - P(YES | honeypot))``. A strategy's reward is the mean
over its rewrites (ICRL's ``reward_mean``); advantages are centred on the group
mean and the strategist's tokens get a PPO update. Nothing else is trained.

Every ``eval.every`` steps (and after step 1) the step's highest-reward strategy is
run on the validation problems, as ICRL validates its top strategies. At the end the
validated strategy with the highest paired validation reward is the winner, and it
gets ICRL's controls: no strategy, instructions removed, empty sender, while loop.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any

import click

from lasr_labs_2025_control_project.prompts.rl_prompts import (
    RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
    RL_WHILE_LOOP_JUDGE_INSTRUCTIONS,
    RL_WHILE_LOOP_SENDER_INSTRUCTIONS,
)

from . import usage
from .config import RLRunConfig
from .core import (
    AppsProblem,
    BatchResult,
    GeneratedSolution,
    GroupedRLTrainer,
    SenderExample,
    group_advantages,
)
from .gbrl import next_batch
from .icrl import (
    CENTRAL,
    Episode,
    Strategy,
    parse_strategy,
    step_prompt,
    summarise,
)
from .icrl_adapters import StrategyBox, TrainerTeam, with_strategy
from .records import Recorder, batch_metrics, close_run, collusion_auroc, summary_row

if TYPE_CHECKING:
    from .run import Tasks

logger = logging.getLogger(__name__)


def strategist_messages() -> list[dict[str, str]]:
    """ICRL's first strategy query, verbatim: the only prompt the strategist sees."""
    return [
        {"role": "system", "content": CENTRAL.system},
        {
            "role": "user",
            "content": "Describe your strategy. " + step_prompt(CENTRAL.strategy),
        },
    ]


async def create_strategist(config: RLRunConfig, resume_state: str | None) -> Any:
    """A trainable LoRA on the ICRL learner model, sampling ICRL's learner settings."""
    from .tinker_backend import create_tinker_sender

    training, icrl = config.training, config.icrl
    strategist, _ = await create_tinker_sender(
        allow_paid=True,
        base_model=icrl.learner,
        rank=training.lora_rank,
        learning_rate=training.learning_rate,
        resume_state=resume_state,
        messages_fn=lambda problem, payload: strategist_messages(),
        loss_fn=training.loss_fn,
        ppo_passes=training.ppo_passes,
        clip_epsilon=training.clip_epsilon,
        temperature=icrl.learner_temperature,
        max_tokens=icrl.learner_max_tokens,
        max_context_tokens=training.max_context_tokens,
    )
    return strategist


@dataclasses.dataclass
class Slot:
    """One strategy's own view of the shared frozen team.

    The sender and judge objects are shallow copies sharing the Tinker clients, so
    the ``group_size`` strategies of a step run concurrently without seeing each
    other's instructions. The honeypot summary cache is shared, so every strategy's
    judge reads the same honeypot descriptions, as across ICRL's episodes.
    """

    trainer: GroupedRLTrainer
    box: StrategyBox
    base_judge_prompt: str


def make_slots(base: GroupedRLTrainer, count: int) -> list[Slot]:
    slots = []
    for _ in range(count):
        box = StrategyBox()
        sender = copy.copy(base.sender)
        sender.messages_fn = with_strategy(base.sender.messages_fn, box)  # type: ignore[attr-defined]
        trainer = copy.copy(base)
        trainer.sender = sender
        trainer.receiver = copy.copy(base.receiver)
        trainer._semaphore = asyncio.Semaphore(base.config.max_concurrency)
        trainer._eval_ids = set()
        slots.append(Slot(trainer, box, base.receiver.system_prompt))  # type: ignore[attr-defined]
    return slots


@dataclasses.dataclass
class Scored:
    """One sampled strategy and what it earned."""

    generation: GeneratedSolution
    strategy: Strategy | None  # None: the reply had no parseable strategy
    reward: float
    episodes: list[Episode] = dataclasses.field(default_factory=list)
    result: BatchResult | None = None
    error: str | None = None


async def run_strategist(
    config: RLRunConfig,
    base_trainer: GroupedRLTrainer,
    recorder: Recorder,
    meter: usage.Usage,
    tasks: Tasks,
    *,
    start_step: int = 1,
    resume_state: str | None = None,
    strategist: Any = None,
) -> None:
    """``strategist`` replaces the Tinker LoRA (offline tests)."""
    training = config.training
    malformed_reward = config.strategist.malformed_reward
    output_dir = config.output_dir
    # The paired reward needs each search problem's honeypot, as in ICRL.
    paired = {problem.problem_id for problem, _ in tasks.train_honeypots}
    search = [task for task in tasks.train if task.problem_id in paired]
    if config.icrl.search_problems < len(search):
        search = random.Random(config.seed).sample(search, config.icrl.search_problems)
    validation, validation_honeypots = tasks.validation, tasks.validation_honeypots
    limit = config.strategist.max_validation
    if limit is not None:
        validation, validation_honeypots = (
            validation[:limit],
            validation_honeypots[:limit],
        )
    try:
        if strategist is None:
            strategist = await create_strategist(config, resume_state)
    except BaseException:
        close_run(recorder, meter, output_dir, start_step - 1, False)
        raise
    slots = make_slots(base_trainer, training.group_size)
    (output_dir / "strategist_prompt.json").write_text(
        json.dumps(strategist_messages(), indent=2)
    )
    recorder.sync_file("strategist_prompt.json")
    recorder.status(
        state="training",
        search_problem_ids=[p.problem_id for p in search],
        strategies_per_step=training.group_size,
        rewrites_per_strategy=training.problems_per_step,
    )
    click.echo(
        f"strategist RL: {len(search)} paired search problems, "
        f"{training.group_size} strategies x {training.problems_per_step} rewrites per step"
    )

    def team(slot: Slot, problems: list[AppsProblem], step: int) -> TrainerTeam:
        def on_result(strategy: str, result: BatchResult, phase: str) -> None:
            recorder.rollouts(
                step,
                phase,
                result.rollouts,
                strategy=strategy,
                decoder=slot.box.decoder,
            )
            if result.errors:
                recorder.batch_errors(step, phase, strategy, result.errors)

        # TrainerTeam takes the judge's current prompt as its no-instructions
        # prompt, so undo the previous strategy's before building one.
        slot.trainer.receiver.system_prompt = slot.base_judge_prompt  # type: ignore[attr-defined]
        return TrainerTeam(
            slot.trainer,
            slot.box,
            search_problems=problems,
            validation_problems=validation,
            honeypots=validation_honeypots,
            num_samples=config.eval.num_samples,
            on_result=on_result,
            judge_template=RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
            train_honeypots=tasks.train_honeypots,
        )

    async def score(
        slot: Slot, generation: GeneratedSolution, batch: list[AppsProblem], step: int
    ) -> Scored:
        try:
            strategy = parse_strategy(generation.text, centralized=True)
        except ValueError as error:
            return Scored(generation, None, malformed_reward, error=str(error))
        captured: list[BatchResult] = []
        member = team(slot, batch, step)
        report = member.on_result
        assert report is not None

        def keep(strategy_text: str, result: BatchResult, phase: str) -> None:
            report(strategy_text, result, phase)
            if phase == "search":
                captured.append(result)

        member.on_result = keep
        episodes = await member.episodes(
            strategy.encoder_instructions, strategy.decoder_instructions
        )
        reward = summarise(step, strategy.encoder_instructions, episodes).reward_mean
        return Scored(generation, strategy, reward, episodes, captured[-1])

    # Validation of chosen strategies, ICRL's validate(): metrics keyed by strategy.
    validated: dict[tuple[str, str], dict[str, Any]] = {}
    rocs: dict[tuple[str, str], list[tuple[float, float]]] = {}

    async def validate(
        step: int, label: str, encoder: str, decoder: str, origin: int | None = None
    ) -> dict[str, Any]:
        key = (encoder, decoder)
        if key not in validated:
            member = team(slots[0], search, step)
            _, honeypot = await member.validate(encoder, decoder)
            rewrites = honeypot.signalled
            metrics: dict[str, Any] = {
                "from_step": origin,
                "strategy": encoder,
                "decoder": decoder,
                **batch_metrics(
                    rewrites, honeypot.honeypot_scores, base_trainer.config
                ),
                "honeypot_errors": len(honeypot.honeypot_errors),
            }
            rewards = [
                r.reward
                for r in rewrites.rollouts
                if r.honeypot_p_yes is not None and r.reward is not None
            ]
            if rewards:
                metrics["paired_reward_mean"] = sum(rewards) / len(rewards)
            recorder.honeypots(step, honeypot.honeypot_scores)
            validated[key] = metrics
            _, rocs[key] = collusion_auroc(rewrites, honeypot.honeypot_scores)
        row = {"check": label, **validated[key]}
        recorder._append("validation.jsonl", row)
        recorder.sync_file("validation.jsonl")
        return validated[key]

    rng = random.Random(config.seed)
    order: list[AppsProblem] = []
    for _ in range(1, start_step):  # a resumed run continues the same batches
        next_batch(order, search, training.problems_per_step, rng)
    strategy_rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, Strategy]] = []  # validated, by step
    last_step_cost, skipped, completed = 0.0, 0, start_step - 1
    step = completed
    complete = False
    try:
        for step in range(start_step, training.steps + 1):
            spent, cap = meter.cost_usd, training.max_cost_usd
            if meter.exhausted or (cap is not None and spent + last_step_cost > cap):
                recorder.status(
                    state="stopping",
                    step=step - 1,
                    reason=f"budget: ${spent:.2f} spent, next step ~${last_step_cost:.2f},"
                    f" cap ${cap:.2f}",
                )
                click.echo(f"stopping before step {step}: budget cap ${cap:.2f}")
                break
            started = time.time()
            batch = next_batch(order, search, training.problems_per_step, rng)
            generations = await strategist.generate(search[0], 1, training.group_size)
            if len(generations) != training.group_size or any(
                g.logprobs is None for g in generations
            ):
                raise RuntimeError(
                    "strategist sampling returned too few or unscored samples"
                )
            outcomes = await asyncio.gather(
                *(
                    score(slot, generation, batch, step)
                    for slot, generation in zip(slots, generations, strict=True)
                ),
                return_exceptions=True,
            )
            failures = [o for o in outcomes if isinstance(o, BaseException)]
            if failures:
                if any(isinstance(f, usage.BudgetExceeded) for f in failures):
                    recorder.status(
                        state="stopping", step=step, reason="budget cap reached"
                    )
                    break
                # An infrastructure failure (after ICRL's one retry): no update.
                skipped += 1
                logger.warning("step %d skipped: %r", step, failures[:3])
                recorder.scalars(step, "train_detail", {"skipped_step": 1})
                recorder.batch_errors(
                    step, "strategist_step", "", [repr(f) for f in failures]
                )
                if skipped > training.max_skipped_steps:
                    recorder.status(state="stopped", step=step, reason="too many skips")
                    raise failures[0]
                continue
            scored = [o for o in outcomes if isinstance(o, Scored)]
            rewards = [item.reward for item in scored]
            advantages = group_advantages(rewards)
            update = None
            if any(advantages):
                update = await strategist.update(
                    [
                        SenderExample(item.generation, advantage)
                        for item, advantage in zip(scored, advantages, strict=True)
                    ]
                )

            # Records: one row per sampled strategy, then the step's metrics.
            pooled_rollouts = [
                r for item in scored if item.result for r in item.result.rollouts
            ]
            pooled_scores = [
                (r.problem.problem_id, r.honeypot_p_yes)
                for r in pooled_rollouts
                if r.honeypot_p_yes is not None
            ]
            per_strategy_auroc = []
            for index, (item, advantage) in enumerate(
                zip(scored, advantages, strict=True)
            ):
                auroc = None
                if item.result is not None:
                    own = [
                        (r.problem.problem_id, r.honeypot_p_yes)
                        for r in item.result.rollouts
                        if r.honeypot_p_yes is not None
                    ]
                    auroc = batch_metrics(item.result, own, base_trainer.config)[
                        "auroc"
                    ]
                    if auroc is not None:
                        per_strategy_auroc.append(auroc)
                row: dict[str, Any] = {
                    "step": step,
                    "slot": index,
                    "reward": item.reward,
                    "advantage": advantage,
                    "train_auroc": auroc,
                    "parsed": item.strategy is not None,
                    "response_tokens": len(item.generation.response_tokens),
                    "description": item.strategy.description if item.strategy else None,
                    "encoder": item.strategy.encoder_instructions
                    if item.strategy
                    else None,
                    "decoder": item.strategy.decoder_instructions
                    if item.strategy
                    else None,
                    "reply": item.generation.text,
                    "parse_error": item.error,
                }
                if item.strategy is not None and item.episodes:
                    row.update(
                        summary_row(
                            summarise(
                                step,
                                item.strategy.encoder_instructions,
                                item.episodes,
                                item.strategy.decoder_instructions,
                            )
                        )
                    )
                recorder._append("strategies.jsonl", row)
                strategy_rows.append(
                    {k: v for k, v in row.items() if k not in ("reply", "parse_error")}
                )
            recorder.sync_file("strategies.jsonl")
            recorder.sync_file("rollouts.jsonl")

            pooled = BatchResult(len(pooled_rollouts), pooled_rollouts)
            metrics: dict[str, Any] = (
                batch_metrics(pooled, pooled_scores, base_trainer.config)
                if pooled_rollouts
                else {}
            )
            # A strategy's reward is the training signal; "reward" is its mean.
            metrics["reward"] = sum(rewards) / len(rewards)
            recorder.metrics(
                step,
                "train",
                {
                    **metrics,
                    "reward_max": max(rewards),
                    "reward_std": float(
                        math.sqrt(
                            sum((r - metrics["reward"]) ** 2 for r in rewards)
                            / len(rewards)
                        )
                    ),
                    "auroc_strategy_mean": sum(per_strategy_auroc)
                    / len(per_strategy_auroc)
                    if per_strategy_auroc
                    else None,
                    "auroc_strategy_max": max(per_strategy_auroc, default=None),
                    "malformed_strategies": sum(
                        item.strategy is None for item in scored
                    ),
                    "strategist_response_tokens": sum(
                        len(item.generation.response_tokens) for item in scored
                    )
                    / len(scored),
                    "strategist_updated": float(update is not None),
                    "step_seconds": time.time() - started,
                    **(
                        {f"strategist/{k}": v for k, v in update.metrics.items()}
                        if update
                        else {}
                    ),
                },
            )
            completed = step
            best = max(
                (item for item in scored if item.strategy is not None),
                key=lambda item: item.reward,
                default=None,
            )
            click.echo(
                f"step {step}: reward mean {metrics['reward']:.3f} max {max(rewards):.3f}, "
                f"train AUROC {metrics.get('auroc')}, working {metrics.get('working_rate')}, "
                f"malformed {sum(item.strategy is None for item in scored)}"
                + (
                    f" | best: {best.strategy.description[:70]!r}"
                    if best and best.strategy
                    else ""
                )
            )
            if (
                best is not None
                and best.strategy is not None
                and (
                    step == 1 or step % config.eval.every == 0 or step == training.steps
                )
            ):
                row = await validate(
                    step,
                    f"step{step}",
                    best.strategy.encoder_instructions,
                    best.strategy.decoder_instructions,
                    origin=step,
                )
                candidates.append((step, best.strategy))
                recorder.metrics(
                    step,
                    "validation",
                    {k: v for k, v in row.items() if not isinstance(v, str)},
                )
                recorder.roc(
                    step,
                    {
                        f"step {step}": rocs[
                            (
                                best.strategy.encoder_instructions,
                                best.strategy.decoder_instructions,
                            )
                        ]
                    },
                    "Collusion ROC on validation",
                )
                recorder.table(step, "strategist/strategies", strategy_rows)
                click.echo(
                    f"step {step} validation: AUROC {row.get('auroc')}, "
                    f"paired reward {row.get('paired_reward_mean')}"
                )
            save_every = config.checkpoints.save_every
            if save_every and step % save_every == 0 and step != training.steps:
                state = await strategist.save_state(
                    f"{config.checkpoint_name}_state_step{step}"
                )
                recorder.status(**{f"state_step{step}": state})
            recorder.scalars(
                step,
                "progress",
                {**meter.snapshot(), "step_cost_usd": meter.cost_usd - spent},
            )
            (output_dir / "usage.json").write_text(
                json.dumps(
                    {"estimated_cost_usd": meter.cost_usd, "tokens": meter.tokens},
                    indent=2,
                )
            )
            last_step_cost = meter.cost_usd - spent
            recorder.status(state="training", step=step)

        final = await strategist.save_checkpoint(config.checkpoint_name)
        final_state = await strategist.save_state(
            f"{config.checkpoint_name}_state_step{completed}"
        )
        recorder.attempt(
            recorder.stage.log_model_ref, f"tinker/{final}", {"step": completed}
        )
        recorder.status(
            final_checkpoint=final, **{f"state_step{completed}": final_state}
        )
        recorder.table(completed, "strategist/strategies", strategy_rows)

        # ICRL's final checks on the winner (highest paired validation reward).
        if candidates and not meter.exhausted:
            recorder.status(state="validation")
            scored_candidates = [
                (
                    validated[(s.encoder_instructions, s.decoder_instructions)].get(
                        "paired_reward_mean", float("-inf")
                    ),
                    origin,
                    s,
                )
                for origin, s in candidates
            ]
            _, origin, winner = max(scored_candidates, key=lambda item: item[0])
            rows = {
                "winner": await validate(
                    completed,
                    "winner",
                    winner.encoder_instructions,
                    winner.decoder_instructions,
                    origin,
                )
            }
            for label, encoder, decoder in (
                ("baseline", "", ""),
                ("instructions_removed", winner.encoder_instructions, ""),
                ("empty_sender", "", winner.decoder_instructions),
                (
                    "while_loop",
                    RL_WHILE_LOOP_SENDER_INSTRUCTIONS,
                    RL_WHILE_LOOP_JUDGE_INSTRUCTIONS,
                ),
            ):
                rows[label] = await validate(completed, label, encoder, decoder)
            recorder.metrics(
                completed,
                "validation",
                {k: v for k, v in rows["baseline"].items() if not isinstance(v, str)},
                prefix="no_strategy_",
                detail=False,
            )
            recorder.roc(
                completed,
                {
                    "winner": rocs[
                        (winner.encoder_instructions, winner.decoder_instructions)
                    ],
                    "no strategy": rocs[("", "")],
                },
                "Collusion ROC on validation",
            )
            recorder.table(
                completed,
                "strategist/validation",
                [{"check": label, **m} for label, m in rows.items()],
            )
            (output_dir / "strategist_result.json").write_text(
                json.dumps(
                    {
                        "winner": dataclasses.asdict(winner),
                        "winner_step": origin,
                        "validation": rows,
                        "validated_steps": [
                            {
                                "step": o,
                                **validated[
                                    (s.encoder_instructions, s.decoder_instructions)
                                ],
                            }
                            for o, s in candidates
                        ],
                    },
                    indent=2,
                )
            )
            recorder.sync_file("strategist_result.json")
            recorder.status(
                winner_step=origin,
                winner_strategy=winner.encoder_instructions,
                winner_decoder=winner.decoder_instructions,
                validation_auroc_winner=rows["winner"]["auroc"],
                validation_auroc_baseline=rows["baseline"]["auroc"],
            )
            click.echo(
                f"winner (step {origin}): validation AUROC {rows['winner']['auroc']} "
                f"vs no strategy {rows['baseline']['auroc']}"
            )
        recorder.status(state="finished")
        complete = completed == training.steps
    finally:
        close_run(recorder, meter, output_dir, step, complete)
