"""Run records: local JSONL/JSON files and the W&B stage run, plus the metric
helpers shared by the GBRL and ICRL drivers."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


from lasr_labs_2025_control_project import tracking

from .config import RLRunConfig
from .core import (
    BatchResult,
    GroupedRLConfig,
    HoneypotEvalResult,
    ReceiverExample,
    Rollout,
    auroc,
)
from .icrl import (
    RoundSummary,
    parse_strategy,
)
from .rewards import validity_gate
from . import usage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
# Prefixes logged to W&B; everything else stays in metrics.jsonl only. A section's
# less important metrics go to <section>_detail/.
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
