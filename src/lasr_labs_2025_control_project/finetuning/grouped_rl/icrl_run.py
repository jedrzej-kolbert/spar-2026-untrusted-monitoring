"""ICRL driver: the learner's strategy search on train problems, then the best
strategies checked on validation; the winner goes to icrl_result.json."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from typing import Any

import click


from .config import RLRunConfig
from .core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    GroupedRLTrainer,
)
from .icrl import (
    CENTRAL,
    SENDER_ONLY,
    Knowledge,
    RoundSummary,
    SearchState,
    run_icrl,
)
from .icrl_adapters import StrategyBox, TrainerTeam
from . import usage
from lasr_labs_2025_control_project.prompts.rl_prompts import (
    RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
    RL_WHILE_LOOP_JUDGE_INSTRUCTIONS,
    RL_WHILE_LOOP_SENDER_INSTRUCTIONS,
)
from .records import Recorder, batch_metrics, self_rec_roc, summary_row

logger = logging.getLogger(__name__)


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
