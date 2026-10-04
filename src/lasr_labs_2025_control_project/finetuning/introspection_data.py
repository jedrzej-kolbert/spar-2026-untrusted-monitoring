"""Introspection (self-prediction) data for Tinker LoRA training, after Binder et al.

"Looking Inward: Language Models Can Learn About Themselves by Introspection"
(arXiv:2410.13787, code: github.com/felixbinder/introspection_self_prediction).
Their released prompts (``dataset_release/{train,test}/<task>_<property>.jsonl``)
carry no labels: the label for a hypothetical question ("what is the second
character of the response you gave?") is a property of *the model's own*
object-level answer. So labels are model-specific and are made here by sampling
the object-level prompt from ``--label-model`` at temperature 0 and extracting
the property, exactly as their ``evals/response_property.py`` does.

Two subcommands:

``build``     write ``{"messages": [user, assistant]}`` JSONL for
              ``finetuning/tinker_lora_finetune.py`` (train with --val-frac 0:
              its held-out check is for YES/NO self-recognition labels).
              ``--label-model`` defaults to the trained model (self-prediction);
              point it at another model for the paper's cross-prediction
              control, which separates "learned about itself" from "learned the
              task format / something any finetune on these prompts teaches".

``accuracy``  self-prediction accuracy of a model on held-out prompts: the
              model's hypothetical answer is compared to *its own* object-level
              answer, so a finetuned checkpoint is scored against its (possibly
              shifted) behaviour. Also reports the paper's baseline of always
              guessing the most common object-level property per file. If the
              trained checkpoint does not beat the base model here, training did
              not instil introspection and a downstream APPS result says nothing
              about introspection either way.

Usage:
  python -m lasr_labs_2025_control_project.finetuning.introspection_data build \\
      --data-dir introspection_self_prediction/dataset_release/train \\
      --model deepseek-ai/DeepSeek-V3.1 --out train.jsonl [--per-file 400]
  python -m lasr_labs_2025_control_project.finetuning.introspection_data accuracy \\
      --data-dir introspection_self_prediction/dataset_release/test \\
      --model tinker://<run>/sampler_weights/<name> --out acc.json [--per-file 100]
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import click

from lasr_labs_2025_control_project.utils import tinker_provider as tp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Object-level answers are a few words / a letter; hypotheticals a word or less.
OBJECT_MAX_TOKENS = 64
META_MAX_TOKENS = 16


def _index(i: int) -> Callable[[str], Optional[str]]:
    return lambda r: r[i] if len(r) > i else None


def _word(i: int) -> Callable[[str], Optional[str]]:
    return lambda r: r.split()[i] if len(r.split()) > i else None


# Property names as they appear in the release file names / behavioral_property.
# Semantics follow evals/response_property.py in the Binder et al. repo.
_PROPERTIES: dict[str, Callable[[str], Optional[str]]] = {
    "identity": lambda r: r or None,
    "first_character": _index(0),
    "second_character": _index(1),
    "third_character": _index(2),
    "first_word": _word(0),
    "second_word": _word(1),
    "third_word": _word(2),
    "starts_with_vowel": lambda r: str(r[0].lower() in "aeiou").lower() if r else None,
    "among_a_or_c": lambda r: str(r.lower() in ("a", "c")).lower(),
    "among_b_or_d": lambda r: str(r.lower() in ("b", "d")).lower(),
}


def extract_property(
    behavioral_property: str, response: str, target: Optional[str] = None
) -> Optional[str]:
    """Property of an object-level response, or None if it is undefined.

    The response is stripped first (the paper strips via its cleaning step and
    in ``identity``/``matches_target``); a leading space would otherwise become
    the "first character".
    """
    response = response.strip()
    if behavioral_property == "ethical_stance":
        if not target or not response:
            return None
        return str(response.lower() == target.strip().lower()).lower()
    fn = _PROPERTIES.get(behavioral_property)
    if fn is None:
        raise ValueError(f"unknown behavioral property: {behavioral_property!r}")
    return fn(response)


def load_rows(
    data_dir: Path, per_file: Optional[int], seed: int
) -> list[dict[str, Any]]:
    """Read every release file in ``data_dir``, keeping ``per_file`` rows of each.

    Rows are shuffled within a file before truncation so a cap does not keep only
    the first block of a file. Each row gets ``task`` = the file stem.
    """
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"no .jsonl files in {data_dir}")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            file_rows = [json.loads(line) for line in f if line.strip()]
        rng.shuffle(file_rows)
        if per_file is not None:
            file_rows = file_rows[:per_file]
        for row in file_rows:
            row["task"] = path.stem
        rows.extend(file_rows)
    return rows


async def sample_text(
    model_ref: str, messages: list[dict[str, str]], max_tokens: int
) -> str:
    """Greedy completion from a base model id or a ``tinker://`` checkpoint."""
    from tinker import types

    prompt_ids = tp._apply_chat_template(
        model_ref, messages, add_generation_prompt=True
    )
    resp = await tp._sampling_client(model_ref).sample_async(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    return tp._tokenizer(model_ref).decode(
        list(resp.sequences[0].tokens), skip_special_tokens=True
    )


async def _gather_limited(coros: list[Any], concurrency: int) -> list[Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))


async def object_properties(
    model_ref: str, rows: list[dict[str, Any]], concurrency: int
) -> list[Optional[str]]:
    responses = await _gather_limited(
        [
            sample_text(model_ref, row["object_level_prompt"], OBJECT_MAX_TOKENS)
            for row in rows
        ],
        concurrency,
    )
    return [
        extract_property(
            row["behavioral_property"], resp, row.get("option_matching_ethical_stance")
        )
        for row, resp in zip(rows, responses)
    ]


def training_examples(
    rows: list[dict[str, Any]], labels: list[Optional[str]]
) -> list[dict[str, Any]]:
    """Pair each hypothetical prompt with its label; drop undefined labels."""
    return [
        {
            "messages": [
                *row["hypothetical_prompt"],
                {"role": "assistant", "content": label},
            ],
            "task": row["task"],
        }
        for row, label in zip(rows, labels)
        if label is not None
    ]


def score_accuracy(
    rows: list[dict[str, Any]],
    object_labels: list[Optional[str]],
    meta_answers: list[str],
) -> dict[str, Any]:
    """Per-task and overall self-prediction accuracy plus the mode baseline.

    Rows whose object-level property is undefined are excluded (as in the
    paper). Matching is case-insensitive on the stripped hypothetical answer.
    """
    by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row, obj, meta in zip(rows, object_labels, meta_answers):
        if obj is None:
            continue
        by_task[row["task"]].append((obj, meta.strip()))
    per_task = {}
    hits = mode_hits = total = 0
    for task, pairs in sorted(by_task.items()):
        n = len(pairs)
        task_hits = sum(obj.lower() == meta.lower() for obj, meta in pairs)
        mode_count = Counter(obj for obj, _ in pairs).most_common(1)[0][1]
        per_task[task] = {
            "n": n,
            "accuracy": task_hits / n,
            "mode_baseline": mode_count / n,
        }
        hits += task_hits
        mode_hits += mode_count
        total += n
    if not total:
        raise ValueError("no rows with a defined object-level property")
    return {
        "n": total,
        "accuracy": hits / total,
        "mode_baseline": mode_hits / total,
        "per_task": per_task,
    }


@click.group()
def cli() -> None:
    from dotenv import load_dotenv

    load_dotenv()


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--model", required=True, help="Model whose behaviour labels the data")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--per-file", type=int, default=400, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--concurrency", type=click.IntRange(1, 128), default=32, show_default=True
)
def build(
    data_dir: Path, model: str, out: Path, per_file: int, seed: int, concurrency: int
) -> None:
    """Write self-prediction training JSONL labelled by MODEL's own answers."""
    model = tp.strip_prefix(model)
    rows = load_rows(data_dir, per_file, seed)
    logger.info("sampling %d object-level prompts from %s", len(rows), model)
    labels = asyncio.run(object_properties(model, rows, concurrency))
    examples = training_examples(rows, labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(
        "wrote %d examples (%d dropped: undefined property) to %s",
        len(examples),
        len(rows) - len(examples),
        out,
    )


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--model", required=True, help="Base model id or tinker:// checkpoint")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--per-file", type=int, default=100, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--concurrency", type=click.IntRange(1, 128), default=32, show_default=True
)
def accuracy(
    data_dir: Path, model: str, out: Path, per_file: int, seed: int, concurrency: int
) -> None:
    """Self-prediction accuracy of MODEL against its own object-level answers."""
    model = tp.strip_prefix(model)
    rows = load_rows(data_dir, per_file, seed)

    async def run() -> tuple[list[Optional[str]], list[str]]:
        obj = await object_properties(model, rows, concurrency)
        meta = await _gather_limited(
            [
                sample_text(model, row["hypothetical_prompt"], META_MAX_TOKENS)
                for row in rows
            ],
            concurrency,
        )
        return obj, meta

    obj, meta = asyncio.run(run())
    result = {"model": model, **score_accuracy(rows, obj, meta)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "%s: self-prediction accuracy %.3f vs mode baseline %.3f (n=%d) -> %s",
        model,
        result["accuracy"],
        result["mode_baseline"],
        result["n"],
        out,
    )


if __name__ == "__main__":
    cli()
