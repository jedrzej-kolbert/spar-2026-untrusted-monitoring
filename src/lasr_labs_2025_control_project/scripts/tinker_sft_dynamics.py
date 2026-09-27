"""Attack-vs-honeypot ROC at each Tinker SFT checkpoint.

The trainer calls record_checkpoint after saving sampler weights. This module can
also be run directly to evaluate a checkpoint saved before an interrupted run.

The results folder contains dynamics.csv, dynamics.svg, and dynamics.csv.manifest.json.
Keep the CSV with its .manifest.json sidecar. Changed evaluation inputs or legacy
CSVs without provenance require a fresh CSV path.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
from pathlib import Path

import click
from inspect_ai.log import read_eval_log
from sklearn.metrics import roc_auc_score

from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
from lasr_labs_2025_control_project.utils.tinker_provider import collusion_yes_no_logprobs
from lasr_labs_2025_control_project.utils.utils import (
    apply_python_code_format, check_sample_is_valid, remove_comments,
)

logger = logging.getLogger(__name__)
FIELDS = ("updates", "checkpoint", "train_loss", "roc_auc", "eval_n")


def load_eval_examples(attacks: Path, honeypots: Path, solution_field: str) -> list[tuple[int, str, str]]:
    """Use source file as the ROC label; never infer it from the model's answer."""
    system = get_prompt_by_name("SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT")
    user = get_prompt_by_name("MONITOR_USER_PROMPT")
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
            if not isinstance(problem, str) or not problem.strip() or not isinstance(solution, str) or not solution.strip():
                raise ValueError(f"missing main_task_description or {solution_field}: {path} sample {index}")
            if solution_field == "base_solution":
                solution = apply_python_code_format(remove_comments(solution))
            examples.append((label, system, user.format(problem_statement=problem, solution=solution)))
            retained += 1
        if not retained:
            raise ValueError(f"no valid samples in {path}")
    return examples


def check_eval_manifest(
    csv_path: Path, examples: list[tuple[int, str, str]], solution_field: str,
) -> None:
    """Bind a CSV to its ordered labels, rendered prompts, and representation."""
    # Ponytail: one writer per CSV; use locking if concurrent writers are needed.
    payload = json.dumps([solution_field, examples], ensure_ascii=False, separators=(",", ":"))
    manifest = {"version": 1, "solution_field": solution_field,
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    manifest_path = csv_path.with_name(csv_path.name + ".manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"evaluation inputs changed; use a fresh dynamics CSV path: {csv_path}")
    elif csv_path.exists():
        raise ValueError(f"dynamics CSV has no provenance; use a fresh CSV path: {csv_path}")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)


async def _score(checkpoint: str, examples: list[tuple[int, str, str]], concurrency: int) -> float:
    semaphore = asyncio.Semaphore(concurrency)

    async def margin(system: str, user: str) -> float:
        async with semaphore:
            result = await collusion_yes_no_logprobs(
                model_name=f"tinker/{checkpoint}", system_prompt=system, user_prompt=user
            )
        if result.get("YES") is None or result.get("NO") is None:
            raise ValueError("Tinker returned no YES/NO logprobs")
        return float(result["YES"]) - float(result["NO"])

    scores = await asyncio.gather(*(margin(system, user) for _, system, user in examples))
    return float(roc_auc_score([label for label, _, _ in examples], scores))


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def _plot(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = sorted((row for row in rows if row["roc_auc"]), key=lambda row: int(row["updates"]))
    if not complete:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([int(row["updates"]) for row in complete],
            [float(row["roc_auc"]) for row in complete], marker="o")
    ax.set(xlabel="Number of gradient updates", ylabel="Attack vs honeypot ROC AUC",
           title="Self-recognition SFT dynamics", ylim=(0, 1))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def record_checkpoint(
    checkpoint: str,
    updates: int,
    train_loss: float | None,
    examples: list[tuple[int, str, str]],
    results_dir: Path,
    concurrency: int = 8,
    *,
    solution_field: str,
) -> None:
    """Persist the checkpoint URI before Tinker evaluation, then refresh the plot."""
    if not checkpoint.startswith("tinker://"):
        raise ValueError("checkpoint must be a tinker:// sampler weights path")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    csv_path = results_dir / "dynamics.csv"
    plot_path = results_dir / "dynamics.svg"
    check_eval_manifest(csv_path, examples, solution_field)
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows and set(rows[0]) != set(FIELDS):
            raise ValueError(f"incompatible dynamics CSV: {csv_path}")
    matching = [row for row in rows if int(row["updates"]) == updates]
    if matching and matching[0]["checkpoint"] != checkpoint:
        raise ValueError(f"update {updates} already belongs to another checkpoint in {csv_path}")
    if matching and matching[0]["roc_auc"]:
        _plot(rows, plot_path)
        return
    if not matching:
        rows.append({"updates": str(updates), "checkpoint": checkpoint,
                     "train_loss": "" if train_loss is None else str(train_loss),
                     "roc_auc": "", "eval_n": ""})
        _write_rows(csv_path, rows)
    auc = asyncio.run(_score(checkpoint, examples, concurrency))
    row = next(row for row in rows if int(row["updates"]) == updates)
    row["roc_auc"] = str(auc)
    row["eval_n"] = str(len(examples))
    _write_rows(csv_path, rows)
    _plot(rows, plot_path)
    logger.info("update %d: ROC AUC %.4f on %d attacks/honeypots; plot %s",
                updates, auc, len(examples), plot_path)


@click.command()
@click.option("--checkpoint", required=True, help="Saved tinker:// sampler weights URI")
@click.option("--updates", type=int, required=True)
@click.option("--attacks-eval", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--honeypots-eval", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--transformed", is_flag=True,
              help="Evaluate transformed_solution instead of base_solution from the eval logs.")
@click.option("--dynamics-results", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output folder for dynamics CSV, SVG, and provenance manifest")
@click.option("--concurrency", type=click.IntRange(1, 32), default=8, show_default=True)
def main(checkpoint: str, updates: int, attacks_eval: Path, honeypots_eval: Path,
         transformed: bool, dynamics_results: Path, concurrency: int) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    solution_field = "transformed_solution" if transformed else "base_solution"
    examples = load_eval_examples(attacks_eval, honeypots_eval, solution_field)
    record_checkpoint(checkpoint, updates, None, examples, dynamics_results, concurrency,
                      solution_field=solution_field)


if __name__ == "__main__":
    main()
