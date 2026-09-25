"""Attack-vs-honeypot ROC at each Tinker SFT checkpoint.

The trainer calls record_checkpoint after saving sampler weights. This module can
also be run directly to evaluate a checkpoint saved before an interrupted run.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path

import click
from inspect_ai.log import read_eval_log
from sklearn.metrics import roc_auc_score

from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
from lasr_labs_2025_control_project.utils.tinker_provider import collusion_yes_no_logprobs

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
        for index, sample in enumerate(log.samples):
            metadata = {**(sample.metadata or {}), **(sample.store or {})}
            problem = metadata.get("main_task_description")
            solution = metadata.get(solution_field)
            if not isinstance(problem, str) or not problem.strip() or not isinstance(solution, str) or not solution.strip():
                raise ValueError(f"missing main_task_description or {solution_field}: {path} sample {index}")
            examples.append((label, system, user.format(problem_statement=problem, solution=solution)))
    return examples


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
    csv_path: Path,
    plot_path: Path,
    concurrency: int = 8,
) -> None:
    """Persist the checkpoint URI before Tinker evaluation, then refresh the plot."""
    if not checkpoint.startswith("tinker://"):
        raise ValueError("checkpoint must be a tinker:// sampler weights path")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
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
@click.option("--solution-field", type=click.Choice(["base_solution", "transformed_solution"]),
              default="base_solution", show_default=True)
@click.option("--dynamics-csv", type=click.Path(path_type=Path), required=True)
@click.option("--dynamics-plot", type=click.Path(path_type=Path), required=True)
@click.option("--concurrency", type=click.IntRange(1, 32), default=8, show_default=True)
def main(checkpoint: str, updates: int, attacks_eval: Path, honeypots_eval: Path,
         solution_field: str, dynamics_csv: Path, dynamics_plot: Path, concurrency: int) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    examples = load_eval_examples(attacks_eval, honeypots_eval, solution_field)
    record_checkpoint(checkpoint, updates, None, examples, dynamics_csv, dynamics_plot, concurrency)


if __name__ == "__main__":
    main()
