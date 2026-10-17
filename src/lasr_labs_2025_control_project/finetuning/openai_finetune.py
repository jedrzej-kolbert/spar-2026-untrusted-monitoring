"""Submit and poll an OpenAI supervised fine-tune for a collusion monitor.

The closed-model SFT arms (`SFT_ARMS` in experiments/open_models/make_configs.py)
are OpenAI finetunes of U. This is the closed-model counterpart to
`tinker_lora_finetune.py`: it prepares training data from `.eval` logs
(`--train-eval`, repeatable; see `sft_data.py`) or takes a prepared JSONL
(`--train-jsonl`), creates the job, polls to completion, and prints the
resulting model id to paste into `SFT_ARMS` (or into a checkpoints JSON).

    python -m lasr_labs_2025_control_project.finetuning.openai_finetune \\
        --train-eval .../generator/attacks.eval \\
        --train-eval .../generator/honeypots.eval \\
        --suffix causal-base

The printed id looks like `ft:gpt-4.1-2025-04-14:lasr-labs:causal-base:XXXX`.
Checkpoint ids (`:ckpt-step-N`) are also listed on completion -- the paper's
arms used intermediate checkpoints, so prefer an explicit checkpoint id over
the final model when reproducing.

Note the finetune lives in the API org that owns the key. Anyone reproducing
the eval later needs a key in that same org, and checkpoints do not live
forever -- record the ids and, if the arm matters, the prepared JSONL.
"""

import csv
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import click

from lasr_labs_2025_control_project.finetuning.sft_data import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    build_training_data,
)
from lasr_labs_2025_control_project.tracking import (
    ALLOW_UNLINKED_HELP,
    StageRun,
    input_name,
    names,
    resolve_input,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "gpt-4.1-2025-04-14"
POLL_SECONDS = 30
# Terminal job states in the OpenAI fine-tuning API.
DONE_STATES = {"succeeded", "failed", "cancelled"}


def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (put it in the repo .env)")
    return OpenAI(api_key=api_key)


def validate_jsonl(path: Path) -> dict[str, int]:
    """Check the prepared file is chat-format and not degenerately labelled.

    The label footgun this guards against: a wrong --target-model in
    generate_train_data.py yields a file where every assistant turn is NO,
    which trains a monitor that always answers NO and looks fine until the
    eval numbers come back flat.
    """
    counts: dict[str, int] = {}
    n = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path}:{line_no}: expected chat rows ending in an "
                    f"assistant message, got {row!r:.120}"
                )
            answer = messages[-1]["content"].strip()
            counts[answer] = counts.get(answer, 0) + 1
            n += 1
    if n == 0:
        raise ValueError(f"{path} is empty")
    if len(counts) < 2:
        raise ValueError(
            f"{path}: every one of {n} examples has the same label {list(counts)!r}. "
            "This is the --target-model footgun -- regenerate the prepared data "
            "with the correct --target-model."
        )
    return counts


def _log_training_metrics(client: Any, job: Any, tracker: StageRun) -> None:
    """Per-step metrics from the job's result file (best effort; logging only)."""
    try:
        content = client.files.content(job.result_files[0]).text
    except Exception as exc:
        logger.warning("could not fetch fine-tuning metrics: %s", exc)
        return
    curves: dict[str, list[tuple[float, float]]] = {
        "train_loss": [],
        "train_accuracy": [],
    }
    for row in csv.DictReader(io.StringIO(content)):
        for key, points in curves.items():
            try:
                points.append((float(row["step"]), float(row[key])))
            except (KeyError, TypeError, ValueError):
                pass
    tracker.log_line(
        "train/loss",
        curves["train_loss"],
        x="Training step",
        y="Training loss",
        title="Training loss",
    )
    tracker.log_line(
        "train/accuracy",
        curves["train_accuracy"],
        x="Training step",
        y="Training token accuracy",
        title="Training accuracy",
    )


@click.command()
@click.option(
    "--train-eval",
    "train_evals",
    multiple=True,
    help="Generation/transform .eval log (path or wandb: ref); repeat per source. "
    "Solutions by --target-model are labelled YES, all others NO.",
)
@click.option(
    "--train-jsonl",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Use an already prepared chat JSONL instead of --train-eval.",
)
@click.option(
    "--target-model",
    default=None,
    help="Generator model whose solutions are YES, as the .eval logs record it "
    "(no provider prefix, e.g. Qwen/Qwen3-8B). Required with --train-eval.",
)
@click.option("--keep-comments", is_flag=True, help="Keep comments in base solutions.")
@click.option("--transformed", is_flag=True, help="Train on transformed solutions.")
@click.option("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, show_default=True)
@click.option("--user-prompt", default=DEFAULT_USER_PROMPT, show_default=True)
@click.option("--upsample-pos-factor", type=int, default=1, show_default=True)
@click.option("--upsample-neg-factor", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--suffix",
    required=True,
    help="Finetune suffix, i.e. the SFT_ARMS arm name (e.g. causal-base)",
)
@click.option("--base-model", default=DEFAULT_BASE_MODEL, show_default=True)
@click.option("--n-epochs", type=int, default=None, help="Default: OpenAI's auto")
@click.option(
    "--validation-jsonl",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--wait/--no-wait", default=True, help="Poll until the job finishes")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the training data and print what would be submitted, then stop",
)
@click.option("--allow-unlinked", is_flag=True, help=ALLOW_UNLINKED_HELP)
def main(
    train_evals: tuple[str, ...],
    train_jsonl: Optional[Path],
    target_model: Optional[str],
    keep_comments: bool,
    transformed: bool,
    system_prompt: str,
    user_prompt: str,
    upsample_pos_factor: int,
    upsample_neg_factor: int,
    seed: int,
    suffix: str,
    base_model: str,
    n_epochs: Optional[int],
    validation_jsonl: Optional[Path],
    wait: bool,
    dry_run: bool,
    allow_unlinked: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if bool(train_evals) == bool(train_jsonl):
        raise click.UsageError("pass --train-eval (one or more) or --train-jsonl")
    if train_evals and not target_model:
        raise click.UsageError(
            "--train-eval needs --target-model (the YES-labelled model)"
        )

    train_inputs = [resolve_input(p) for p in (train_evals or [str(train_jsonl)])]
    with tempfile.TemporaryDirectory() as tmp:
        composition: dict[str, Any]
        if train_evals:
            examples, composition = build_training_data(
                [i.local_path for i in train_inputs],
                target_model=str(target_model),
                transformed=transformed,
                keep_comments=keep_comments,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=seed,
                upsample_pos_factor=upsample_pos_factor,
                upsample_neg_factor=upsample_neg_factor,
            )
            train_path = Path(tmp) / "train_messages.jsonl"
            train_path.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
            )
        else:
            assert train_jsonl is not None
            train_path, composition = train_jsonl, {"prepared_jsonl": True}

        counts = validate_jsonl(train_path)
        logger.info("%s examples, label counts %s", sum(counts.values()), counts)
        if dry_run:
            logger.info(
                "Dry run: would fine-tune %s with suffix %r", base_model, suffix
            )
            return

        tracker = StageRun(
            "finetune",
            names.extend(
                input_name(train_inputs[0], "train"), f"ft-{names.clean(suffix)}"
            ),
            config={
                "backend": "openai",
                "base_model": base_model,
                "suffix": suffix,
                "n_epochs": n_epochs,
                "training_data": composition,
            },
            tags=[
                "finetune",
                "openai",
                *(f"pos-{m}" for m in composition.get("positive_eval_modes", [])),
                *(f"neg-{m}" for m in composition.get("negative_eval_modes", [])),
            ],
            inputs=train_inputs,
            allow_unlinked=allow_unlinked,
        )
        with tracker:
            _submit(
                train_path,
                suffix=suffix,
                base_model=base_model,
                n_epochs=n_epochs,
                validation_jsonl=validation_jsonl,
                wait=wait,
                tracker=tracker,
            )


def _submit(
    train_path: Path,
    *,
    suffix: str,
    base_model: str,
    n_epochs: Optional[int],
    validation_jsonl: Optional[Path],
    wait: bool,
    tracker: StageRun,
) -> None:
    client = _client()
    logger.info("Uploading %s", train_path)
    train_file = client.files.create(file=train_path.open("rb"), purpose="fine-tune")
    validation_file = None
    if validation_jsonl:
        validation_file = client.files.create(
            file=validation_jsonl.open("rb"), purpose="fine-tune"
        )

    kwargs: dict = {
        "training_file": train_file.id,
        "model": base_model,
        "suffix": suffix,
    }
    if validation_file:
        kwargs["validation_file"] = validation_file.id
    if n_epochs:
        kwargs["method"] = {
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": n_epochs}},
        }

    job = client.fine_tuning.jobs.create(**kwargs)
    logger.info("Created fine-tuning job %s (%s)", job.id, job.status)
    tracker.summary({"job_id": job.id})
    if not wait:
        logger.info(
            "Not waiting; poll with: openai api fine_tuning.jobs.get -i %s", job.id
        )
        tracker.finish(complete=False, reason="--no-wait: model id not recorded")
        return

    while job.status not in DONE_STATES:
        time.sleep(POLL_SECONDS)
        job = client.fine_tuning.jobs.retrieve(job.id)
        logger.info(
            "job %s: %s (trained tokens: %s)", job.id, job.status, job.trained_tokens
        )

    if job.status != "succeeded":
        raise SystemExit(f"Fine-tuning job {job.id} ended as {job.status}: {job.error}")

    logger.info("Fine-tuned model: %s", job.fine_tuned_model)
    _log_training_metrics(client, job, tracker)
    record = {"backend": "openai", "base_model": base_model, "job_id": job.id}
    tracker.log_model_ref(f"openai/{job.fine_tuned_model}", record)
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id)
    for ckpt in checkpoints.data:
        logger.info(
            "  checkpoint: %s (step %s)",
            ckpt.fine_tuned_model_checkpoint,
            ckpt.step_number,
        )
        tracker.log_model_ref(
            f"openai/{ckpt.fine_tuned_model_checkpoint}",
            {**record, "step": ckpt.step_number},
        )
    tracker.summary({"trained_tokens": job.trained_tokens})
    print(
        json.dumps(
            {suffix.replace("-", "_"): f"openai/{job.fine_tuned_model}"}, indent=1
        )
    )


if __name__ == "__main__":
    main()
