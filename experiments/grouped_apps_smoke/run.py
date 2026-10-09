"""Explicitly approved, bounded wiring run; importing this module makes no calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    BatchInfrastructureError,
    BatchResult,
    GroupedRLConfig,
    GroupedRLTrainer,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.sandbox import (
    DockerAppsEvaluator,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    TinkerReceiver,
    TinkerSender,
)

MODEL = "Qwen/Qwen3-8B"
IMAGE = "python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
PRICING_DATE = "2026-10-09"
PRICING_URL = (
    "https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/"
)
RATES = {
    "prefill": Decimal("0.195"),
    "sample": Decimal("0.60"),
    "train": Decimal("0.44"),
}


class BudgetStop(RuntimeError):
    pass


class Budget:
    """Never refund reservations, even when a response is lost or an API fails.

    Reserve twice uncached published token prices, with $2 withheld for storage
    and uncertainty. Sampling retries and transport retries are disabled. This
    is a conservative token-cost ceiling, not a reading of the billing account.
    """

    def __init__(
        self,
        limit: float,
        event: Any,
        ledger: Path | None = None,
        *,
        allow_fourth_stage: bool = False,
    ) -> None:
        if allow_fourth_stage and ledger is None:
            raise ValueError(
                "fourth-stage approval requires the existing persistent ledger"
            )
        self.allow_fourth_stage = allow_fourth_stage
        if not math.isfinite(limit) or not 2 < limit <= 10:
            raise ValueError(
                "the approved ceiling must be greater than $2 and at most $10"
            )
        self.limit = Decimal(str(limit))
        self.reserved = Decimal("2")
        self.halted = False
        self.event = event
        self.ledger = ledger
        if ledger is not None:
            self._persist(Decimal(0), begin=True)
            if allow_fourth_stage:
                self.event(
                    "reservation",
                    {
                        "operation": "fourth_stage_storage_allowance",
                        "reserved_usd": float(self.reserved),
                        "call_bound_usd": 2.0,
                    },
                )

    def _persist(self, cost: Decimal, *, begin: bool = False) -> None:
        import fcntl

        assert self.ledger is not None
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = (
                json.loads(self.ledger.read_text())
                if self.ledger.exists()
                else {
                    "limit": str(self.limit),
                    "reserved": "2",
                    "attempts": 0,
                }
            )
            if Decimal(state["limit"]) != self.limit:
                raise BudgetStop(
                    "existing authorization ledger has a different ceiling"
                )
            reserved = Decimal(state["reserved"])
            if not reserved.is_finite() or not 2 <= reserved <= self.limit:
                raise BudgetStop(
                    "invalid authorization ledger; refusing to reset spending"
                )
            if begin:
                if self.allow_fourth_stage and state["attempts"] != 3:
                    raise BudgetStop(
                        "fourth-stage approval requires exactly three previous stages"
                    )
                if state["attempts"] >= 3:
                    if not self.allow_fourth_stage or state["attempts"] != 3:
                        raise BudgetStop("three-attempt storage allowance exhausted")
                    # A newly named fourth run gets another $2 storage/uncertainty
                    # reserve inside the SAME ceiling, not a reset or another $10.
                    cost += Decimal("2")
                    state["fourth_stage_storage_allowance"] = "2"
                state["attempts"] += 1
            if reserved + cost > self.limit:
                self.halted = True
                raise BudgetStop(
                    "next call would exceed the cumulative authorization ceiling"
                )
            self.reserved = reserved + cost
            state["reserved"] = str(self.reserved)
            temporary = self.ledger.with_suffix(".tmp")
            with temporary.open("w") as stream:
                stream.write(json.dumps(state) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.ledger)

    def reserve(self, operation: str, cost: Decimal) -> None:
        if cost < 0 or not cost.is_finite():
            raise ValueError("invalid cost reservation")
        if self.halted or self.reserved + cost > self.limit:
            self.halted = True
            raise BudgetStop("next call would exceed the conservative cost ceiling")
        if self.ledger is not None:
            self._persist(cost)
        else:
            self.reserved += cost
        self.event(
            "reservation",
            {
                "operation": operation,
                "reserved_usd": float(self.reserved),
                "call_bound_usd": float(cost),
            },
        )

    def sampling(self, prompt_tokens: int, count: int, max_tokens: int) -> None:
        if min(prompt_tokens, count, max_tokens) < 1:
            raise ValueError("positive token and sample limits required")
        cost = count * (prompt_tokens * RATES["prefill"] + max_tokens * RATES["sample"])
        self.reserve("sample", Decimal(2) * cost / 1_000_000)

    def scoring(self, tokens: int) -> None:
        # Score conservatively at the more expensive training rate, plus one
        # possible generated token; never assume prompt caching.
        cost = tokens * max(RATES["train"], RATES["prefill"]) + RATES["sample"]
        self.reserve("score", Decimal(2) * cost / 1_000_000)

    def training(self, tokens: int) -> None:
        self.reserve(
            "forward_backward", Decimal(2) * tokens * RATES["train"] / 1_000_000
        )


def numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"nonfinite API metric: {name}")
            result[name] = float(value)
    return result


class Recorder:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.step = 0
        self.started = time.monotonic()
        self.wandb: Any = None
        self.budget: Budget | None = None
        self.losses: dict[str, dict[str, float]] = {}

    def event(self, kind: str, data: dict[str, Any]) -> None:
        row = {
            "kind": kind,
            "step": self.step,
            "elapsed_seconds": time.monotonic() - self.started,
            **data,
        }
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def status(self, **values: Any) -> None:
        temporary = self.directory / "status.tmp"
        temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "status.json")
        print(json.dumps(values, allow_nan=False), flush=True)

    def scalars(self, data: dict[str, Any]) -> None:
        values = {
            "train/step": self.step,
            "time/elapsed_seconds": time.monotonic() - self.started,
            **data,
        }
        if self.budget:
            values["budget/reserved_upper_bound_usd"] = float(self.budget.reserved)
            values["budget/ceiling_usd"] = float(self.budget.limit)
        self.event("scalars", {"metrics": values})
        if self.wandb:
            try:
                self.wandb.log(values)
            except Exception as exc:
                self.event("monitoring_error", {"error_type": type(exc).__name__})

    def backward(
        self, role: str, data: list[Any], output: Any, eos: int | None
    ) -> None:
        raw = numeric_metrics(dict(output.metrics))
        metrics = {f"{role}/api/{name}": value for name, value in raw.items()}
        if role == "sender":
            active = sum(
                value != 0
                for datum in data
                for value in datum.loss_fn_inputs["advantages"].data
            )
        else:
            active = sum(
                value > 0
                for datum in data
                for value in datum.loss_fn_inputs["weights"].data
            )
        metrics[f"{role}/active_loss_tokens"] = active
        if "loss:sum" in raw and active:
            metrics[f"{role}/loss_per_active_token"] = raw["loss:sum"] / active
        # Decompose receiver CE only when the server actually returns aligned
        # per-token log probabilities. No missing component is manufactured.
        if role == "receiver":
            parts: dict[str, list[float]] = {"label": [], "eos": []}
            outputs = getattr(output, "loss_fn_outputs", None)
            if outputs is not None and len(outputs) == len(data):
                complete = True
                for datum, item in zip(data, outputs, strict=True):
                    tensor = item.get("logprobs")
                    if tensor is None:
                        complete = False
                        break
                    values = tensor.data
                    weights = datum.loss_fn_inputs["weights"].data
                    targets = datum.loss_fn_inputs["target_tokens"].data
                    if len(values) != len(weights):
                        complete = False
                        break
                    for target, weight, value in zip(
                        targets, weights, values, strict=True
                    ):
                        if weight > 0:
                            if not math.isfinite(value):
                                raise ValueError(
                                    "nonfinite receiver target probability"
                                )
                            parts["eos" if target == eos else "label"].append(-value)
                if complete:
                    for part, values in parts.items():
                        if values:
                            metrics[f"receiver/{part}_nll_mean"] = statistics.fmean(
                                values
                            )
        self.losses[role] = metrics
        self.scalars(metrics)

    def batch(self, phase: str, result: BatchResult) -> dict[str, Any]:
        metrics = {f"{phase}/{name}": value for name, value in result.metrics().items()}
        metrics[f"{phase}/sender_updated"] = result.sender_updated
        metrics[f"{phase}/receiver_updated"] = result.receiver_updated
        rows = []
        confidences = []
        rewards = []
        for item in result.rollouts:
            rows.append(
                {
                    "step": self.step,
                    "phase": phase,
                    "problem_id": item.problem.problem_id,
                    "payload": item.payload,
                    "code": item.code,
                    "generation": item.generation.text if item.generation else None,
                    "probabilities": item.probabilities,
                    "reward": item.reward,
                    "suspiciousness": item.suspiciousness,
                    "overseer_penalty": item.overseer_penalty,
                    "advantage": item.advantage,
                    "errors": item.errors,
                    "execution": {
                        "passed": item.execution.passed,
                        "detail": item.execution.detail,
                        "infrastructure_error": item.execution.infrastructure_error,
                    }
                    if item.execution
                    else None,
                }
            )
            if item.probabilities is not None:
                confidences.append(item.probabilities[item.payload])
            if item.reward is not None:
                rewards.append(item.reward)
        with (self.directory / "rollouts.jsonl").open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        metrics[f"{phase}/correct_bit_probability_mean"] = (
            statistics.fmean(confidences) if len(confidences) == len(rows) else None
        )
        metrics[f"{phase}/reward_std"] = (
            statistics.pstdev(rewards) if len(rewards) == len(rows) else None
        )
        scores = [item.suspiciousness for item in result.rollouts]
        penalties = [item.overseer_penalty for item in result.rollouts]
        metrics[f"{phase}/overseer_scored_records"] = sum(v is not None for v in scores)
        metrics[f"{phase}/suspiciousness_mean"] = (
            statistics.fmean(v for v in scores if v is not None)
            if all(v is not None for v in scores)
            else None
        )
        metrics[f"{phase}/reward_overseer_penalty_mean"] = (
            statistics.fmean(v for v in penalties if v is not None)
            if all(v is not None for v in penalties)
            else None
        )
        if len(rewards) == len(rows):
            task_credit = sum(
                float(item.execution.passed)
                for item in result.rollouts
                if item.execution
            )
            metrics[f"{phase}/reward_task_mean"] = task_credit / len(rows)
            metrics[f"{phase}/reward_communication_mean"] = (
                sum(rewards) + sum(v for v in penalties if v is not None) - task_credit
            ) / len(rows)
        else:
            metrics[f"{phase}/reward_task_mean"] = None
            metrics[f"{phase}/reward_communication_mean"] = None
        self.event(
            "batch", {"phase": phase, "batch_errors": result.errors, "metrics": metrics}
        )
        self.scalars(metrics)
        return metrics


class ObservedFuture:
    def __init__(self, future: Any, callback: Any) -> None:
        self.future, self.callback = future, callback

    async def result_async(self) -> Any:
        result = await self.future.result_async()
        self.callback(result)
        return result


class SamplingProxy:
    def __init__(
        self, client: Any, budget: Budget, recorder: Recorder | None = None
    ) -> None:
        self.client, self.budget, self.recorder = client, budget, recorder

    async def sample_async(
        self, *, prompt: Any, num_samples: int, sampling_params: Any
    ) -> Any:
        self.budget.sampling(
            len(prompt.to_ints()), num_samples, sampling_params.max_tokens
        )
        response = await self.client.sample_async(
            prompt=prompt, num_samples=num_samples, sampling_params=sampling_params
        )
        if self.recorder:
            self.recorder.event(
                "sample_output",
                {
                    "sequences": [
                        {
                            "tokens": list(item.tokens),
                            "logprobs": [
                                value
                                if value is None or math.isfinite(value)
                                else str(value)
                                for value in item.logprobs
                            ]
                            if item.logprobs is not None
                            else None,
                        }
                        for item in response.sequences
                    ]
                },
            )
        return response

    async def compute_logprobs_async(self, prompt: Any) -> Any:
        self.budget.scoring(len(prompt.to_ints()))
        return await self.client.compute_logprobs_async(prompt)


class TrainingProxy:
    def __init__(
        self,
        client: Any,
        service: Any,
        budget: Budget,
        recorder: Recorder,
        role: str,
        tokenizer: Any,
    ) -> None:
        self.client, self.service, self.budget = client, service, budget
        self.recorder, self.role, self.tokenizer = recorder, role, tokenizer
        self.snapshots = 0

    async def forward_backward_async(self, data: list[Any], loss_fn: Any) -> Any:
        self.budget.training(sum(len(datum.model_input.to_ints()) for datum in data))
        future = await self.client.forward_backward_async(data, loss_fn)
        return ObservedFuture(
            future,
            lambda output: self.recorder.backward(
                self.role, data, output, self.tokenizer.eos_token_id
            ),
        )

    async def optim_step_async(self, params: Any) -> Any:
        if self.budget.halted:
            raise BudgetStop("budget halted before optimizer submission")
        future = await self.client.optim_step_async(params)

        def observe(output: Any) -> None:
            raw = numeric_metrics(dict(getattr(output, "metrics", {}) or {}))
            self.recorder.scalars(
                {f"{self.role}/optimizer/{k}": v for k, v in raw.items()}
            )

        return ObservedFuture(future, observe)

    async def save_weights_and_get_sampling_client_async(self) -> SamplingProxy:
        from tinker.lib.retry_handler import RetryConfig

        self.snapshots += 1
        if self.snapshots > 11:
            raise RuntimeError("snapshot storage reservation exceeded")
        future = await self.client.save_weights_for_sampler_async(
            f"wiring-{self.role}-{self.snapshots:02d}", ttl_seconds=3600
        )
        saved = await future.result_async()
        sampler = await self.service.create_sampling_client_async(
            model_path=saved.path,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=180),
        )
        self.recorder.event(
            "sampling_refresh", {"role": self.role, "snapshot": self.snapshots}
        )
        return SamplingProxy(sampler, self.budget, self.recorder)

    async def checkpoint(self) -> str:
        future = await self.client.save_weights_for_sampler_async(
            f"wiring-final-{self.role}", ttl_seconds=86400
        )
        return (await future.result_async()).path


def load_problems() -> tuple[list[AppsProblem], list[AppsProblem], dict[str, Any]]:
    from datasets import Dataset

    # Some unselected function-call records contain very large JSON integers.
    sys.set_int_max_str_digits(100000)
    files = list(
        (Path.home() / ".cache/huggingface/datasets/codeparrot___apps").rglob(
            "*train*.arrow"
        )
    )
    if len(files) != 1:
        raise RuntimeError("expected one cached original codeparrot/APPS training file")
    dataset = Dataset.from_file(str(files[0]))
    eligible = []
    for index in range(len(dataset)):
        record = dataset[index]
        if record["difficulty"] != "introductory" or len(record["question"]) > 4000:
            continue
        tests = json.loads(record["input_output"])
        if tests.get("fn_name") or not 1 <= len(tests.get("inputs", [])) <= 32:
            continue
        if not all(isinstance(x, str) for x in tests["inputs"] + tests["outputs"]):
            continue
        if max(map(len, tests["inputs"])) > 2048:
            continue
        eligible.append((len(tests["inputs"]), len(record["question"]), index))
    selected = sorted(eligible)[:8]
    if len(selected) != 8:
        raise RuntimeError("need eight eligible, disjoint wiring tasks")
    problems = [
        AppsProblem.from_record(
            dataset[index], split="train" if j < 4 else "validation"
        )
        for j, (_, _, index) in enumerate(selected)
    ]
    manifest = {
        "source_dataset": "codeparrot/apps",
        "original_split": "train",
        "cached_arrow": str(files[0]),
        "original_rows": len(dataset),
        "eligible_rows": len(eligible),
        "selection_reason": "Eight shortest introductory stdin tasks, at most 32 supplied tests and short inputs; wiring only, not representative evaluation.",
        "train_ids": [p.problem_id for p in problems[:4]],
        "validation_ids": [p.problem_id for p in problems[4:]],
        "test_counts": {p.problem_id: len(p.inputs) for p in problems},
    }
    return problems[:4], problems[4:], manifest


def start_wandb(directory: Path, config: dict[str, Any]) -> Any:
    import wandb

    # W&B installs this vendored parser under a runtime-only module alias.
    gql = import_module("wandb_gql").gql

    api = wandb.Api(timeout=20)
    entity = api.default_entity
    project = "apps-grouped-wiring"
    mutation = gql("""mutation PrivateProject($entity: String!, $name: String!) {
      upsertModel(input: {entityName: $entity, name: $name, access: "PRIVATE"}) {
        model {name access}
      }
    }""")
    response = api.client.execute(
        mutation, variable_values={"entity": entity, "name": project}
    )
    if response["upsertModel"]["model"]["access"].upper() != "PRIVATE":
        raise RuntimeError("W&B project was not verified private; no run data uploaded")
    run = wandb.init(
        entity=entity,
        project=project,
        name=directory.name,
        mode="online",
        dir=str(directory),
        config=config,
        notes="Wiring pilot only. One supplied test per task. Signed group-relative sender loss is not expected to decrease monotonically. No overseer.",
        settings=wandb.Settings(disable_code=True, disable_git=True, console="off"),
    )
    run.define_metric("train/step")
    run.define_metric("*", step_metric="train/step")
    return run


async def execute(args: argparse.Namespace) -> None:
    train, validation, manifest = load_problems()
    recorder = Recorder(args.output)
    config = {
        **manifest,
        "model": MODEL,
        "steps": args.steps,
        "group_size": 4,
        "max_tokens": 1536,
        "seed": 0,
        "budget_usd": args.budget,
        "pricing_verified": PRICING_DATE,
        "pricing_url": PRICING_URL,
        "price_per_million": {k: str(v) for k, v in RATES.items()},
        "reservation_multiplier": 2,
        "storage_safety_reservation_usd": 2,
        "image": IMAGE,
        "checkpoint_ttl_seconds": 86400,
    }
    (args.output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    evaluator = DockerAppsEvaluator(image=IMAGE, temp_root=args.temp_root)
    fixture = AppsProblem("fixture", "Echo stdin.", ("12\n",), ("12\n",))
    grade = await evaluator.grade(fixture, "print(input())")
    if not grade.passed:
        raise RuntimeError(f"Docker preflight failed: {grade}")
    if not args.allow_paid:
        recorder.status(
            status="prepared_offline", manifest=manifest, output=str(args.output)
        )
        return
    if date.today().isoformat() != PRICING_DATE:
        raise RuntimeError(
            "verify current prices before using the paid driver on another day"
        )
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    load_dotenv("/home/jkk/spar-2026-untrusted-monitoring/.env", override=False)
    import tinker

    recorder.budget = budget = Budget(args.budget, recorder.event, args.budget_ledger)
    recorder.event("budget_initialized", {"reserved_usd": float(budget.reserved)})
    try:
        recorder.wandb = start_wandb(args.output, config)
        recorder.status(
            status="initializing_models",
            wandb_url=recorder.wandb.url,
            output=str(args.output),
        )
    except Exception as exc:
        recorder.event(
            "wandb_unavailable", {"error_type": type(exc).__name__, "message": str(exc)}
        )
        recorder.status(
            status="initializing_models_local_logging", output=str(args.output)
        )
    service = tinker.ServiceClient(max_retries=0)
    roles = []
    proxies = []
    status = "failed"
    final: dict[str, Any] = {}
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
        loop = GroupedRLTrainer(roles[0], roles[1], evaluator, GroupedRLConfig(seed=0))
        recorder.batch("validation", await loop.evaluate(validation, num_samples=2))
        for step in range(1, args.steps + 1):
            recorder.step = step
            recorder.losses = {}
            batch = train[:2] if step % 2 else train[2:]
            result = await loop.train_step(batch)
            metrics = recorder.batch("train", result)
            recorder.status(
                status="training",
                step=step,
                steps=args.steps,
                wandb_url=recorder.wandb.url if recorder.wandb else None,
                reserved_upper_bound_usd=float(budget.reserved),
                metrics=metrics,
                output=str(args.output),
            )
            if step % 2 == 0 or step == args.steps:
                final = recorder.batch(
                    "validation", await loop.evaluate(validation, num_samples=2)
                )
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
            "reserved_upper_bound_usd": float(budget.reserved),
            "billing_actual_usd": None,
            "checkpoints_expire_after_seconds": 86400,
            "checkpoints": checkpoints,
            "final_validation": final,
            "wandb_url": recorder.wandb.url if recorder.wandb else None,
            "output": str(args.output),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        recorder.status(**summary)
        if recorder.wandb:
            recorder.wandb.finish(exit_code=0 if status == "completed" else 1)
        await service.close("success" if status == "completed" else "errored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--budget", type=float, default=10)
    parser.add_argument(
        "--budget-ledger",
        type=Path,
        default=Path.home()
        / ".local/share/spar-2026-runs/apps-grouped-wiring-20261008-f3e7afb1-budget.json",
    )
    parser.add_argument("--steps", type=int, default=10)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / ".local/share/spar-2026-runs"
        / f"apps-grouped-wiring-{stamp}",
    )
    parser.add_argument("--temp-root", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error("this wiring driver permits 1–10 steps only")
    if not math.isfinite(args.budget) or not 2 < args.budget <= 10:
        parser.error("budget must be greater than $2 and at most the approved $10")
    asyncio.run(asyncio.wait_for(execute(args), timeout=1800))


if __name__ == "__main__":
    main()
