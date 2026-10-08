"""Approved one-bit grouped training through the project's minification channel."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from experiments.grouped_apps_smoke.minification_baseline import MINIFY_OPTIONS, minify
from experiments.grouped_apps_smoke.run import (
    IMAGE,
    MODEL,
    PRICING_DATE,
    Budget,
    Recorder,
    TrainingProxy,
    load_problems,
    start_wandb,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    ExecutionGrade,
    GroupedRLConfig,
    GroupedRLTrainer,
    ReceiverExample,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.sandbox import (
    DockerAppsEvaluator,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    TinkerReceiver,
    TinkerSender,
)


class MinificationChannel:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[str, str | None]] = {}

    def transform(self, source: str) -> tuple[str, str | None]:
        if source not in self.cache:
            try:
                self.cache[source] = (minify(source), None)
            except SyntaxError as exc:
                # Invalid generated syntax is a candidate failure, not a service
                # failure. Never leak the original source as a fallback channel.
                self.cache[source] = ("", f"minification rejected sender syntax: {exc}")
        return self.cache[source]


class ChannelReceiver:
    def __init__(self, receiver: Any, channel: MinificationChannel) -> None:
        self.receiver, self.channel = receiver, channel

    async def probabilities(
        self, problem: AppsProblem, code: str
    ) -> tuple[float, float]:
        transformed, _ = self.channel.transform(code)
        return await self.receiver.probabilities(problem, transformed)

    async def update(self, examples: list[ReceiverExample]) -> Any:
        transformed = [
            replace(example, code=self.channel.transform(example.code)[0])
            for example in examples
        ]
        return await self.receiver.update(transformed)


class ChannelEvaluator:
    def __init__(self, evaluator: Any, channel: MinificationChannel) -> None:
        self.evaluator, self.channel = evaluator, channel

    async def grade(self, problem: AppsProblem, code: str) -> ExecutionGrade:
        transformed, error = self.channel.transform(code)
        if error:
            return ExecutionGrade(False, error)
        return await self.evaluator.grade(problem, transformed)


class FixedPayloadRNG(random.Random):
    def __init__(self, bit: int) -> None:
        super().__init__(0)
        if bit not in (0, 1):
            raise ValueError("a bit is required")
        self.bit = bit

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        if start != 2 or stop is not None or step != 1:
            raise ValueError("this evaluation RNG supports randrange(2) only")
        return self.bit


class BalancedTrainer(GroupedRLTrainer):
    async def evaluate_balanced(self, problems: list[AppsProblem]) -> BatchResult:
        ids = self._validate_problems(problems)
        if any(p.split == "train" for p in problems) or ids & self._train_ids:
            raise ValueError("balanced evaluation requires disjoint held-out problems")
        self._eval_ids.update(ids)
        result = BatchResult(2 * len(problems))
        for bit in (0, 1):
            partial = await self._collect(
                problems, 1, FixedPayloadRNG(bit), require_policy_logprobs=False
            )
            result.rollouts.extend(partial.rollouts)
            result.errors.extend(partial.errors)
        if result.has_infrastructure_errors:
            raise BatchInfrastructureError(result)
        return result


class ChannelRecorder(Recorder):
    def __init__(self, directory: Path, channel: MinificationChannel) -> None:
        super().__init__(directory)
        self.channel = channel

    def batch(self, phase: str, result: BatchResult) -> dict[str, Any]:
        # Actor training still uses ORIGINAL response tokens and log probabilities.
        # rollouts.jsonl retains upstream source; this paired file captures exactly
        # what the receiver and the task grader saw after the transformation.
        with (self.directory / "channel_cases.jsonl").open("a") as stream:
            for index, row in enumerate(result.rollouts):
                code, error = self.channel.transform(row.code)
                record = {
                    "step": self.step,
                    "phase": phase,
                    "batch_index": index,
                    "problem_id": row.problem.problem_id,
                    "payload": row.payload,
                    "original_code": row.code,
                    "minified_code": code,
                    "original_generation": row.generation.text
                    if row.generation
                    else None,
                    "minification_error": error,
                    "probabilities": row.probabilities,
                    "reward": row.reward,
                    "errors": row.errors,
                    "passed": row.execution.passed if row.execution else None,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        values = super().batch(phase, result)
        extra: dict[str, float | int | None] = {
            f"{phase}/minification_failures": sum(
                self.channel.transform(r.code)[1] is not None for r in result.rollouts
            )
        }
        for bit in (0, 1):
            rows = [r for r in result.rollouts if r.payload == bit]
            extra[f"{phase}/bit_{bit}_records"] = len(rows)
            extra[f"{phase}/bit_{bit}_decoding_accuracy"] = (
                sum(r.decoded_correctly for r in rows) / len(rows) if rows else None
            )
        self.scalars(extra)
        return {**values, **extra}


def corrected_training_split(train: list[AppsProblem]) -> list[AppsProblem]:
    from datasets import Dataset

    files = list(
        (Path.home() / ".cache/huggingface/datasets/codeparrot___apps").rglob(
            "*train*.arrow"
        )
    )
    if len(files) != 1:
        raise ValueError("expected one cached original training split")
    dataset = Dataset.from_file(str(files[0]))
    record = dataset[2580]
    if str(record["problem_id"]) != "2580":
        raise ValueError("replacement training task identity changed")
    replacement = AppsProblem.from_record(record, split="train")
    return [replacement if p.problem_id == "2647" else p for p in train]


async def execute(args: argparse.Namespace) -> None:
    train, validation, manifest = load_problems()
    train = corrected_training_split(train)
    channel = MinificationChannel()
    recorder = ChannelRecorder(args.output, channel)
    evaluator = ChannelEvaluator(
        DockerAppsEvaluator(image=IMAGE, temp_root=args.temp_root), channel
    )
    fixture = AppsProblem("fixture", "Echo stdin.", ("12\n",), ("12\n",))
    grade = await evaluator.grade(fixture, "secret_bit = 1\nprint(input())")
    if not grade.passed:
        raise RuntimeError(f"minification/Docker preflight failed: {grade}")
    config = {
        **manifest,
        "experiment": "one_bit_minification_training",
        "selection_reason": "Short introductory stdin wiring subset, with documented replacement of the legacy-regex task by the next eligible original-train task; not a representative benchmark.",
        "model": MODEL,
        "steps": args.steps,
        "group_size": 4,
        "max_tokens": 1536,
        "seed": 0,
        "rank": 8,
        "sender_learning_rate": 1e-5,
        "receiver_learning_rate": 1e-4,
        "train_ids": [p.problem_id for p in train],
        "test_counts": {p.problem_id: len(p.inputs) for p in [*train, *validation]},
        "curriculum_correction": "Replace training task 2647 with original-train task 2580: old regex expected output conflicts with Python 3.12. Original experiment records stay unchanged.",
        "sender_prompt": "unchanged from first pilot",
        "minifier_options": MINIFY_OPTIONS,
        "channel_reward": "APPS correctness and receiver probabilities both use minified code",
        "receiver_training": "minified code only; prompt labels remain hidden",
        "actor_training": "original sampled tokens/logprobs, not transformed tokens",
        "validation": "one genuine sender sample for each of 0/1 on each of four held-out tasks",
        "budget_usd": 10,
        "shared_budget_ledger": str(args.budget_ledger),
        "image": IMAGE,
        "checkpoint_ttl_seconds": 86400,
    }
    (args.output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    if not args.allow_paid:
        recorder.status(
            status="prepared_offline", output=str(args.output), config=config
        )
        return
    if date.today().isoformat() != PRICING_DATE:
        raise RuntimeError("verify current prices before paid execution")
    from dotenv import load_dotenv

    load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env", override=False)
    import tinker

    recorder.budget = budget = Budget(10, recorder.event, args.budget_ledger)
    recorder.wandb = start_wandb(args.output, config)
    recorder.wandb.notes = "One-bit joint training through project-standard minification. Fresh independent adapters. Genuine balanced held-out sender generations. No stealth/overseer objective; one supplied test per task."
    recorder.status(
        status="initializing_models",
        wandb_url=recorder.wandb.url,
        output=str(args.output),
    )
    service = tinker.ServiceClient(max_retries=0)
    proxies = []
    roles = []
    status = "failed"
    final = {}
    baseline = {}
    try:
        for role in ("sender", "receiver"):
            client = await service.create_lora_training_client_async(
                base_model=MODEL, rank=8
            )
            tokenizer = client.get_tokenizer()
            proxy = TrainingProxy(client, service, budget, recorder, role, tokenizer)
            sampler = await proxy.save_weights_and_get_sampling_client_async()
            cls = TinkerSender if role == "sender" else TinkerReceiver
            kwargs = {"max_tokens": 1536} if role == "sender" else {}
            roles.append(
                cls(
                    proxy,
                    sampler,
                    tokenizer,
                    allow_paid=True,
                    max_context_tokens=8192,
                    learning_rate=1e-5 if role == "sender" else 1e-4,
                    **kwargs,
                )
            )
            proxies.append(proxy)
        loop = BalancedTrainer(
            roles[0],
            ChannelReceiver(roles[1], channel),
            evaluator,
            GroupedRLConfig(seed=0),
        )
        baseline = recorder.batch(
            "validation", await loop.evaluate_balanced(validation)
        )
        recorder.status(
            status="baseline_complete",
            metrics=baseline,
            wandb_url=recorder.wandb.url,
            output=str(args.output),
        )
        for step in range(1, args.steps + 1):
            recorder.step = step
            recorder.losses = {}
            result = await loop.train_step(train[:2] if step % 2 else train[2:])
            metrics = recorder.batch("train", result)
            recorder.status(
                status="training",
                step=step,
                steps=args.steps,
                metrics=metrics,
                reserved_cumulative_upper_bound_usd=float(budget.reserved),
                wandb_url=recorder.wandb.url,
                output=str(args.output),
            )
            if step % 2 == 0 or step == args.steps:
                final = recorder.batch(
                    "validation", await loop.evaluate_balanced(validation)
                )
                recorder.event("evaluation_complete", {"metrics": final})
        status = "completed"
    except BatchInfrastructureError as exc:
        recorder.batch("failed_batch", exc.result)
        recorder.event(
            "run_error", {"error_type": type(exc).__name__, "message": str(exc)}
        )
        raise
    except Exception as exc:
        recorder.event(
            "run_error", {"error_type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        checkpoints = {}
        for proxy in proxies:
            try:
                checkpoints[proxy.role] = await proxy.checkpoint()
            except Exception as exc:
                recorder.event(
                    "checkpoint_error",
                    {"role": proxy.role, "error_type": type(exc).__name__},
                )
        summary = {
            "status": status,
            "last_step": recorder.step,
            "baseline": baseline,
            "final_validation": final,
            "reserved_cumulative_upper_bound_usd": float(budget.reserved),
            "actual_billed_usd": None,
            "checkpoints": checkpoints,
            "checkpoint_ttl_seconds": 86400,
            "wandb_url": recorder.wandb.url,
            "output": str(args.output),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        recorder.status(**summary)
        recorder.wandb.finish(exit_code=0 if status == "completed" else 1)
        await service.close("success" if status == "completed" else "errored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--budget-ledger",
        type=Path,
        default=Path.home()
        / ".local/share/spar-2026-runs/apps-grouped-wiring-20261008-f3e7afb1-budget.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error("this authorized pilot permits 1–10 steps")
    asyncio.run(asyncio.wait_for(execute(args), timeout=1800))


if __name__ == "__main__":
    main()
