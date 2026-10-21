"""Build RL tasks from the pipeline's generator ``.eval`` logs.

Option A (rewrite_signal) takes attacks the untrusted model already wrote with
``generate_solutions_inspect.py`` and keeps those that pass the main task and have a
working backdoor (configurable). Honeypots from the same split are the negatives for
the held-out attack-vs-honeypot check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .core import AppsProblem

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class EvalSample:
    problem_id: str
    question: str
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    code: str
    secret_input: str | None
    reference_solution: str | None
    main_task_success: str | None
    side_task_success: str | None
    model: str = ""


def _scores(sample: Any) -> tuple[str | None, str | None]:
    for score in (sample.scores or {}).values():
        value = score.value
        if isinstance(value, dict) and "main_task_success" in value:
            return value.get("main_task_success"), value.get("side_task_success")
    return None, None


def read_generator_eval(path: Path | str) -> list[EvalSample]:
    """Samples from a generator log, with code, tests, and backdoor metadata."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    if not log.samples:
        raise ValueError(f"no samples in {path}")
    samples = []
    for sample in log.samples:
        metadata = {**sample.metadata, **sample.store}
        code = metadata.get("base_solution")
        if not code:
            continue
        generator_info = metadata.get("generator_info") or {}
        if not isinstance(generator_info, dict):
            generator_info = getattr(generator_info, "__dict__", {})
        secret_input = generator_info.get("secret_input") or None
        solutions = metadata.get("solutions") or []
        if isinstance(solutions, str):
            solutions = json.loads(solutions)
        main, side = _scores(sample)
        samples.append(
            EvalSample(
                problem_id=str(sample.id),
                question=metadata["main_task_description"],
                inputs=tuple(metadata["inputs"]),
                outputs=tuple(metadata["outputs"]),
                code=code,
                secret_input=secret_input,
                reference_solution=solutions[0] if solutions else None,
                main_task_success=main,
                side_task_success=side,
                model=generator_info.get("model_info", {}).get("model")
                or log.eval.model,
            )
        )
    return samples


def attack_tasks(
    samples: list[EvalSample],
    *,
    require_main_task: bool = True,
    require_backdoor: bool = True,
    max_tests: int | None = 10,
) -> list[AppsProblem]:
    """Attacks worth rewriting: by default they pass the tests and their backdoor works."""
    tasks = []
    for sample in samples:
        if require_main_task and sample.main_task_success != "C":
            continue
        if require_backdoor and (
            sample.side_task_success != "C" or not sample.secret_input
        ):
            continue
        inputs, outputs = sample.inputs, sample.outputs
        if max_tests is not None:
            inputs, outputs = inputs[:max_tests], outputs[:max_tests]
        try:
            tasks.append(
                AppsProblem(
                    problem_id=sample.problem_id,
                    question=sample.question,
                    inputs=inputs,
                    outputs=outputs,
                    base_code=sample.code,
                    secret_input=sample.secret_input,
                    reference_solution=sample.reference_solution,
                )
            )
        except ValueError:
            continue  # malformed test metadata (e.g. non-string stdin tests)
    return tasks


def _bucket(problem_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_tasks(
    tasks: list[AppsProblem], *, validation_fraction: float, seed: int
) -> tuple[list[AppsProblem], list[AppsProblem]]:
    """Deterministic, ID-hashed split; stable as the task pool grows or shrinks."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    train, validation = [], []
    for task in sorted(tasks, key=lambda item: int(item.problem_id)):
        if _bucket(task.problem_id, seed) < validation_fraction:
            validation.append(replace(task, split="validation"))
        else:
            train.append(replace(task, split="train"))
    return train, validation


def honeypot_pairs(
    samples: list[EvalSample], problems: list[AppsProblem]
) -> list[tuple[AppsProblem, str]]:
    """Honeypot code for the given held-out problems (the receiver should say NO)."""
    by_id = {problem.problem_id: problem for problem in problems}
    return [
        (by_id[sample.problem_id], sample.code)
        for sample in samples
        if sample.problem_id in by_id
    ]
