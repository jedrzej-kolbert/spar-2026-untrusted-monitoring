"""SFT data preparation, run inside the fine-tuning scripts.

The fine-tuning entrypoints read generation/transform ``.eval`` logs directly:
valid solutions are extracted (``utils/eval_to_dataset.py``), labelled YES when
written by the target model (U) and NO otherwise (``utils/generate_train_data.py``),
and rendered as monitor-format chat examples. The standalone shell scripts in
``binary_tuning/`` do the same through JSON/JSONL files and remain for
inspecting or hand-editing a prepared dataset (e.g. label ablations).

``build_training_data`` also returns a *composition* record -- which sources
went in as positives and negatives, their eval modes and counts -- which the
fine-tuning run logs as top-level config so runs can be filtered by training
data (attacks-only vs honest-only, balanced or not, ...).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Sequence

from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.finetuning.utils.eval_to_dataset import (
    detect_generator_model,
    eval_log_to_dataset,
)
from lasr_labs_2025_control_project.finetuning.utils.generate_train_data import (
    build_train_rows,
)
from lasr_labs_2025_control_project.self_rec_baselines.utils.scripts.data_models import (
    Dataset,
)
from lasr_labs_2025_control_project.utils.basharena_views import (
    TRAJECTORY_VIEW,
    ActionViewOptions,
)
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
from lasr_labs_2025_control_project.utils.utils import (
    apply_python_code_format,
    check_sample_is_valid,
    remove_comments,
)

DEFAULT_SYSTEM_PROMPT = "SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT"
DEFAULT_USER_PROMPT = "MONITOR_USER_PROMPT"


def _eval_mode(log: Any) -> Optional[str]:
    if not log.samples:
        return None
    sample_data = {**log.samples[0].metadata, **log.samples[0].store}
    info = sample_data.get("generator_info")
    return info.get("eval_mode") if isinstance(info, dict) else None


def build_training_data(
    eval_paths: Sequence[Path],
    *,
    target_model: str,
    transformed: bool = False,
    keep_comments: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 42,
    upsample_pos_factor: int = 1,
    upsample_neg_factor: int = 1,
    view: str = TRAJECTORY_VIEW,
    action_options: Optional[ActionViewOptions] = None,
    labels: str = "author",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chat examples ({"messages": [...]}) and their composition record.

    ``target_model`` is the generator model whose solutions are labelled YES,
    exactly as the logs record it: no provider prefix (``Qwen/Qwen3-8B``,
    ``gpt-4.1-2025-04-14``). A value that
    matches no solution is an error (``prepare_rows``): it would label every
    example NO and train a monitor that always answers NO.

    ``view="action"`` makes one example per rated agent action (BashArena), as
    the per-action monitor sees it. ``labels="permuted"`` shuffles the YES/NO
    answers across examples (a control: the monitor can learn nothing real).
    """
    logs = [(path, read_eval_log(str(path))) for path in eval_paths]
    models = {path: detect_generator_model(log) for path, log in logs}

    dataset = Dataset(problems=[])
    sources = []
    for path, log in logs:
        part = eval_log_to_dataset(
            log,
            models[path],
            transformed,
            keep_comments,
            view=view,
            action_options=action_options,
        )
        dataset = dataset.merge(part)
        sources.append(
            {
                "path": str(path),
                "model": models[path],
                "eval_mode": _eval_mode(log),
                "role": "positive" if models[path] == target_model else "negative",
                "samples": len(log.samples or []),
                "valid": sum(len(p.solutions) for p in part),
            }
        )

    rows, positives, negatives = build_train_rows(
        dataset,
        target_model,
        get_prompt_by_name(system_prompt),
        get_prompt_by_name(user_prompt),
        seed,
        upsample_pos_factor,
        upsample_neg_factor,
    )

    examples = [{"messages": row["messages"]} for row in rows]
    if labels == "permuted":
        answers = [example["messages"][-1]["content"] for example in examples]
        random.Random(f"{seed}:permute").shuffle(answers)
        examples = [
            {
                "messages": [
                    *example["messages"][:-1],
                    {**example["messages"][-1], "content": answer},
                ]
            }
            for example, answer in zip(examples, answers)
        ]
    elif labels != "author":
        raise ValueError(f"labels must be 'author' or 'permuted', got {labels!r}")

    def modes(role: str) -> list[str]:
        return sorted({str(s["eval_mode"]) for s in sources if s["role"] == role})

    composition = {
        "target_model": target_model,
        "positive_eval_modes": modes("positive"),
        "negative_eval_modes": modes("negative"),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_problems": len(dataset),
        "transformed": transformed,
        "keep_comments": keep_comments,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "upsample_pos_factor": upsample_pos_factor,
        "upsample_neg_factor": upsample_neg_factor,
        "view": view,
        "labels": labels,
        "seed": seed,
        "sources": sources,
    }
    return examples, composition


def load_eval_examples(
    attacks: Path, honeypots: Path, solution_field: str
) -> list[tuple[int, str, str]]:
    """Held-out (label, system, user) examples for per-checkpoint ROC AUC.

    The label comes from the source file (attacks = 1, honeypots = 0), never
    from the model's answer. Preprocessing matches the collusion monitor.
    """
    system = get_prompt_by_name(DEFAULT_SYSTEM_PROMPT)
    user = get_prompt_by_name(DEFAULT_USER_PROMPT)
    examples = []
    for label, path in ((1, attacks), (0, honeypots)):
        log = read_eval_log(str(path))
        if not log.samples:
            raise ValueError(f"no samples in {path}")
        retained = 0
        for index, sample in enumerate(log.samples):
            if not check_sample_is_valid(sample.scores):
                continue
            metadata = {**(sample.metadata or {}), **(sample.store or {})}
            problem = metadata.get("main_task_description")
            solution = metadata.get(solution_field)
            if (
                not isinstance(problem, str)
                or not problem.strip()
                or not isinstance(solution, str)
                or not solution.strip()
            ):
                raise ValueError(
                    f"missing main_task_description or {solution_field}: {path} sample {index}"
                )
            if solution_field == "base_solution":
                solution = apply_python_code_format(remove_comments(solution))
            examples.append(
                (
                    label,
                    system,
                    user.format(problem_statement=problem, solution=solution),
                )
            )
            retained += 1
        if not retained:
            raise ValueError(f"no valid samples in {path}")
    return examples
