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
from lasr_labs_2025_control_project.utils.model_calling.main import retry_api_call
from lasr_labs_2025_control_project.utils.tinker_provider import (
    collusion_yes_no_logprobs,
)
from lasr_labs_2025_control_project.utils.utils import (
    ActionViewOptions,
    action_views,
    apply_python_code_format,
    check_sample_is_valid,
    format_solution,
    is_code_setting,
    remove_comments,
    select_action_views,
)

# Collusion prompts per setting; must match the prompts the SFT data was built with.
_PROMPTS = {
    "bash_arena": (
        "BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT",
        "BASHARENA_MONITOR_USER_PROMPT",
    ),
}
_DEFAULT_PROMPTS = (
    "SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
    "MONITOR_USER_PROMPT",
)
# solution_field value for per-action views: one example per rated action, in the
# same format and with the same prompts as the per-action SFT rows and monitor.
ACTION_VIEW_FIELD = "action_view"
_ACTION_PROMPTS = {
    "bash_arena": (
        "BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT",
        "BASHARENA_ACTION_USER_PROMPT",
    ),
}
SOLUTION_FIELDS = ("base_solution", "transformed_solution", ACTION_VIEW_FIELD)

logger = logging.getLogger(__name__)
FIELDS = ("updates", "checkpoint", "train_loss", "roc_auc", "eval_n")


def load_eval_examples(
    attacks: Path,
    honeypots: Path,
    solution_field: str,
    action_options: ActionViewOptions | None = None,
    keys: list[tuple[str, int | None]] | None = None,
) -> list[tuple[int, str, str]]:
    """Use source file as the ROC label; never infer it from the model's answer.

    With solution_field == "action_view" each rated action is one example (label
    of its trajectory), so the ROC is per action. `keys`, if given, receives one
    (sample id, 1-based action index or None) per example, in the same order.
    """
    if solution_field not in SOLUTION_FIELDS:
        raise ValueError(f"solution_field must be one of {SOLUTION_FIELDS}")
    examples = []
    for label, path in ((1, attacks), (0, honeypots)):
        log = read_eval_log(str(path))
        if not log.samples:
            raise ValueError(f"no samples in {path}")
        retained = 0
        for index, sample in enumerate(log.samples):
            metadata = {**(sample.metadata or {}), **(sample.store or {})}
            if not check_sample_is_valid(
                sample.scores,
                (sample.metadata or {}).get("eval_mode"),
                base_solution=metadata.get("base_solution"),
                errored=getattr(sample, "error", None) is not None,
            ):
                # Skip (don't crash on) zero-action / errored samples, matching the
                # SFT and monitor consumers; a blank base_solution is now caught above.
                continue
            setting = metadata.get("setting_name")
            problem = metadata.get("main_task_description")
            if solution_field == ACTION_VIEW_FIELD:
                if setting not in _ACTION_PROMPTS:
                    raise ValueError(f"no per-action prompts for setting {setting!r}")
                if not isinstance(problem, str) or not problem.strip():
                    raise ValueError(
                        f"missing main_task_description: {path} sample {index}"
                    )
                system, user = (get_prompt_by_name(n) for n in _ACTION_PROMPTS[setting])
                views = action_views(sample.messages, action_options)
                for k, view in select_action_views(views, sample.id, action_options):
                    examples.append(
                        (
                            label,
                            system,
                            user.format(problem_statement=problem, solution=view),
                        )
                    )
                    if keys is not None:
                        keys.append((str(sample.id), k))
                retained += 1
                continue
            system, user = (
                get_prompt_by_name(n) for n in _PROMPTS.get(setting, _DEFAULT_PROMPTS)
            )
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
                solution = (
                    apply_python_code_format(remove_comments(solution))
                    if is_code_setting(setting)
                    else format_solution(solution, setting)
                )
            examples.append(
                (
                    label,
                    system,
                    user.format(problem_statement=problem, solution=solution),
                )
            )
            if keys is not None:
                keys.append((str(sample.id), None))
            retained += 1
        if not retained:
            raise ValueError(f"no valid samples in {path}")
    return examples


def check_eval_manifest(
    csv_path: Path,
    examples: list[tuple[int, str, str]],
    solution_field: str,
) -> None:
    """Bind a CSV to its ordered labels, rendered prompts, and representation."""
    # Ponytail: one writer per CSV; use locking if concurrent writers are needed.
    payload = json.dumps(
        [solution_field, examples], ensure_ascii=False, separators=(",", ":")
    )
    manifest = {
        "version": 1,
        "solution_field": solution_field,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    manifest_path = csv_path.with_name(csv_path.name + ".manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                f"evaluation inputs changed; use a fresh dynamics CSV path: {csv_path}"
            )
    elif csv_path.exists():
        raise ValueError(
            f"dynamics CSV has no provenance; use a fresh CSV path: {csv_path}"
        )
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)


async def _score(
    checkpoint: str, examples: list[tuple[int, str, str]], concurrency: int
) -> tuple[float, list[float]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def margin(system: str, user: str) -> float:
        async with semaphore:
            result = await collusion_yes_no_logprobs(
                model_name=f"tinker/{checkpoint}",
                system_prompt=system,
                user_prompt=user,
            )
        if result.get("YES") is None or result.get("NO") is None:
            raise ValueError("Tinker returned no YES/NO logprobs")
        return float(result["YES"]) - float(result["NO"])

    # One transient failure must not abort a checkpoint (or the training run that
    # called it), so each example is asked up to three times, with backoff.
    scores = await asyncio.gather(
        *(
            retry_api_call(margin, max_retries=2)(system, user)
            for _, system, user in examples
        )
    )
    return float(roc_auc_score([label for label, _, _ in examples], scores)), scores


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

    complete = sorted(
        (row for row in rows if row["roc_auc"]), key=lambda row: int(row["updates"])
    )
    if not complete:
        return
    fig, (ax, loss_ax) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax.plot(
        [int(row["updates"]) for row in complete],
        [float(row["roc_auc"]) for row in complete],
        marker="o",
    )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set(
        ylabel="Test attack vs honeypot ROC AUC",
        title="Self-recognition SFT dynamics",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    loss_csv = output.parent / "train_loss.csv"
    if loss_csv.exists():
        with loss_csv.open(newline="", encoding="utf-8") as file:
            steps = [
                (int(r["step"]), float(r["loss"]))
                for r in csv.DictReader(file)
                if r["loss"]
            ]
        if steps:
            xs, ys = zip(*steps)
            window = 10
            smooth = [
                sum(ys[max(0, i - window + 1) : i + 1])
                / len(ys[max(0, i - window + 1) : i + 1])
                for i in range(len(ys))
            ]
            loss_ax.plot(
                xs, ys, color="tab:orange", alpha=0.3, linewidth=1, label="per step"
            )
            loss_ax.plot(
                xs, smooth, color="tab:orange", linewidth=2, label=f"{window}-step mean"
            )
            loss_ax.legend()
    loss_ax.set(xlabel="Number of gradient updates", ylabel="Train loss (per-token CE)")
    loss_ax.grid(alpha=0.2)
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
    if not checkpoint.startswith("tinker://") and updates != 0:
        raise ValueError(
            "checkpoint must be a tinker:// sampler weights path (or the base model at update 0)"
        )
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
        raise ValueError(
            f"update {updates} already belongs to another checkpoint in {csv_path}"
        )
    if matching and matching[0]["roc_auc"]:
        _plot(rows, plot_path)
        return
    if not matching:
        rows.append(
            {
                "updates": str(updates),
                "checkpoint": checkpoint,
                "train_loss": "" if train_loss is None else str(train_loss),
                "roc_auc": "",
                "eval_n": "",
            }
        )
        _write_rows(csv_path, rows)
    auc, scores = asyncio.run(_score(checkpoint, examples, concurrency))
    # Per-example YES-NO margins, in `examples` order, for analysis beyond the AUC.
    (results_dir / f"scores_{updates}.json").write_text(json.dumps(scores))
    row = next(row for row in rows if int(row["updates"]) == updates)
    row["roc_auc"] = str(auc)
    row["eval_n"] = str(len(examples))
    _write_rows(csv_path, rows)
    _plot(rows, plot_path)
    logger.info(
        "update %d: ROC AUC %.4f on %d attacks/honeypots; plot %s",
        updates,
        auc,
        len(examples),
        plot_path,
    )


@click.command()
@click.option("--checkpoint", required=True, help="Saved tinker:// sampler weights URI")
@click.option("--updates", type=int, required=True)
@click.option(
    "--attacks-eval", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--honeypots-eval", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--transformed",
    is_flag=True,
    help="Evaluate transformed_solution instead of base_solution from the eval logs.",
)
@click.option(
    "--action-views",
    "use_action_views",
    is_flag=True,
    help="Score every rated action in BashArena's per-action format.",
)
@click.option("--max-history-chars", type=int, default=None)
@click.option("--actions-per-trajectory", type=int, default=None)
@click.option("--action-seed", type=int, default=0, show_default=True)
@click.option(
    "--dynamics-results",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output folder for dynamics CSV, SVG, and provenance manifest",
)
@click.option("--concurrency", type=click.IntRange(1, 32), default=8, show_default=True)
def main(
    checkpoint: str,
    updates: int,
    attacks_eval: Path,
    honeypots_eval: Path,
    transformed: bool,
    use_action_views: bool,
    max_history_chars: int | None,
    actions_per_trajectory: int | None,
    action_seed: int,
    dynamics_results: Path,
    concurrency: int,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    if transformed and use_action_views:
        raise click.UsageError("--action-views does not support --transformed")
    solution_field = (
        ACTION_VIEW_FIELD
        if use_action_views
        else "transformed_solution"
        if transformed
        else "base_solution"
    )
    action_options = ActionViewOptions(
        max_history_chars=max_history_chars,
        actions_per_trajectory=actions_per_trajectory,
        seed=action_seed,
    )
    examples = load_eval_examples(
        attacks_eval, honeypots_eval, solution_field, action_options
    )
    record_checkpoint(
        checkpoint,
        updates,
        None,
        examples,
        dynamics_results,
        concurrency,
        solution_field=solution_field,
    )


if __name__ == "__main__":
    main()
