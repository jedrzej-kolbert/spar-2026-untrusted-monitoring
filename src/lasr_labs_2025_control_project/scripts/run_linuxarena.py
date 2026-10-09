"""Run the generated LinuxArena pipeline, with paid stages explicitly opted in."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from lasr_labs_2025_control_project.utils.linuxarena_pipeline import (
    POPULATION_ROLES,
    REPO_ROOT,
    append_status,
    assemble_flat_rows,
    bridge_options,
    ensure_plan,
    export_manifest,
    generation_arguments,
    generation_manifest,
    latest_status,
    missing_scores,
    monitoring_arguments,
    pending_job,
    raw_model,
    read_jsonl,
    select_cases,
    statistical_tables,
    validate_config,
    write_json,
)

MODULE = "lasr_labs_2025_control_project.scripts.run_linuxarena"


def resources(root: Path) -> dict[str, float]:
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return {
        "available_ram_gib": int(memory["MemAvailable"].split()[0]) / 2**20,
        "free_disk_gib": shutil.disk_usage(root).free / 2**30,
    }


def recover_status(path: Path) -> list[dict[str, Any]]:
    """Include partial logs left by interrupted attempts, without deleting history."""
    records = read_jsonl(path)
    for record in records:
        if record.get("destination"):
            discovered = {
                str(p.resolve()) for p in Path(record["destination"]).glob("*.eval")
            }
            record["eval_files"] = sorted(
                set(record.get("eval_files", [])) | discovered
            )
    return records


def save_manifests(root: Path) -> None:
    generation = recover_status(root / "generation_status.jsonl")
    monitors = recover_status(root / "monitor_status.jsonl")
    sources = generation_manifest(generation)
    write_json(root / "manifest.json", sources)
    write_json(root / "export_manifest.json", export_manifest(sources, monitors))
    write_json(
        root / "run_ledger.json", {"generation": generation, "monitoring": monitors}
    )


def run_command(command: list[str], env_file: Path, dry_run: bool) -> int:
    print(json.dumps({"command": command}), flush=True)
    if dry_run:
        return 0
    from dotenv import load_dotenv

    load_dotenv(env_file)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def ct_command(root: Path, env_file: Path, arguments: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "ct",
        "--options",
        str(root / "bridge_options.json"),
        "--env-file",
        str(env_file),
        "--allow-paid",
        *arguments,
    ]


def check_ct_environment(config: dict[str, Any], root: Path) -> None:
    if version("inspect-ai") != "0.3.268":
        raise RuntimeError(
            "Use Control Tower's Python (inspect-ai==0.3.268) for generate, monitor and export"
        )
    model_info = import_module("tinker_cookbook.model_info")
    from lasr_labs_2025_control_project.utils.linuxarena_tinker import resolve_renderer

    options = bridge_options(config, root)
    for route, models in options["models"].items():
        for model, settings in models.items():
            resolve_renderer(model, settings, route == "True", model_info)
    write_json(root / "bridge_options.json", options)


def selected_cases(config: dict[str, Any], root: Path, dry_run: bool) -> list[str]:
    frozen = (root / "plan.json").exists()
    if frozen:
        cases = json.loads((root / "plan.json").read_text())["cases"]
    else:
        environments = import_module("control_tower.environments.all_environments")
        environment = environments.get_environment(config["generation"]["environment"])
        tasks = environments.get_main_tasks_for_env(environment)
        cases = select_cases(tasks, config["generation"])
        side_tasks = environments.get_side_tasks_for_env(environment)
        if config["generation"]["side_task"] not in side_tasks:
            raise ValueError("Configured side_task is not in the environment")
    # A dry run must not freeze the triple and task plan for the real run.
    if frozen or not dry_run:
        ensure_plan(root, config, cases)
    return cases


def drive(
    config: dict[str, Any], root: Path, args: argparse.Namespace, monitoring: bool
) -> int:
    check_ct_environment(config, root)
    status_path = root / (
        "monitor_status.jsonl" if monitoring else "generation_status.jsonl"
    )
    records = recover_status(status_path)
    states = latest_status(records)
    save_manifests(root)
    if monitoring:
        sources = json.loads((root / "manifest.json").read_text())
        if not sources:
            raise ValueError("No generation logs to monitor; inspect run_ledger.json")
        jobs = [
            (hashlib.sha256(source["path"].encode()).hexdigest(), source)
            for source in sources
        ]
    else:
        jobs = [
            (
                f"{population}:{case}",
                {
                    "case": case,
                    "population": population,
                    "actor_id": raw_model(config["models"][role]),
                },
            )
            for case in selected_cases(config, root, args.dry_run)
            for population, role in POPULATION_ROLES.items()
        ]
    launched = 0
    for job_id, job in jobs:
        rescore = (
            monitoring
            and args.rescore_complete
            and states.get(job_id, {}).get("state") == "complete"
        )
        if not rescore and not pending_job(states.get(job_id), args.retry_failed):
            continue
        if args.limit is not None and launched >= args.limit:
            break
        if monitoring and not Path(job["path"]).is_file():
            raise FileNotFoundError(
                f"Manifest source is missing, not skipped: {job['path']}"
            )
        destination = (
            root
            / ("monitored" if monitoring else "logs")
            / job["population"]
            / job_id.replace(":", "_")
            / str(uuid4())
        )
        arguments = (
            monitoring_arguments(config, Path(job["path"]), destination)
            if monitoring
            else generation_arguments(
                config, job["case"], job["population"], destination
            )
        )
        command = ct_command(root, args.env_file, arguments)
        if args.dry_run:
            print(json.dumps({"job_id": job_id, "command": command}), flush=True)
            launched += 1
            continue
        if not monitoring:
            health = resources(root)
            limits = config["resources"]
            if (
                health["available_ram_gib"] < limits["min_available_ram_gib"]
                or health["free_disk_gib"] < limits["min_free_disk_gib"]
            ):
                print(
                    json.dumps({"state": "paused-resource-headroom", **health}),
                    flush=True,
                )
                return 2
        launched += 1
        destination.mkdir(parents=True)
        record = {
            **job,
            "job_id": job_id,
            "destination": str(destination),
            "state": "starting",
            "eval_files": [],
        }
        if monitoring:
            record["source"] = job["path"]
        append_status(status_path, record)
        try:
            with (destination / "console.log").open("a") as console:
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    check=False,
                    env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"},
                )
            record["exit_code"] = process.returncode
        except OSError as exc:
            record.update(exit_code=1, error=str(exc))
        record["eval_files"] = sorted(
            str(p.resolve()) for p in destination.glob("*.eval")
        )
        record["state"] = (
            "complete"
            if record["exit_code"] == 0 and record["eval_files"]
            else "failed"
        )
        append_status(status_path, record)
        save_manifests(root)
        print(json.dumps(record), flush=True)
        if record["state"] == "failed":
            return record["exit_code"] or 1
    save_manifests(root)
    print(
        json.dumps(
            {
                "launched": launched,
                "states": {
                    key: value["state"]
                    for key, value in latest_status(read_jsonl(status_path)).items()
                },
            }
        ),
        flush=True,
    )
    return 0


def export(config: dict[str, Any], root: Path, args: argparse.Namespace) -> int:
    if version("inspect-ai") != "0.3.268":
        raise RuntimeError(
            "Export requires Control Tower's Python (inspect-ai==0.3.268)"
        )
    save_manifests(root)
    command = [
        sys.executable,
        "-m",
        "lasr_labs_2025_control_project.scripts.export_linuxarena_actions",
        "--manifest",
        str(root / "export_manifest.json"),
        "--target-actor",
        raw_model(config["models"]["U"]),
        "--output-dir",
        str(root / "exports"),
        "--cot-level",
        "none",
        "--split-seed",
        str(config["export"]["split_seed"]),
    ]
    if not json.loads((root / "export_manifest.json").read_text()):
        raise ValueError(
            "No eval logs to export; run_ledger.json retains failed runs without logs"
        )
    return run_command(command, args.env_file, args.dry_run)


def finetune(config: dict[str, Any], root: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "lasr_labs_2025_control_project.finetuning.tinker_lora_finetune",
        "--train-jsonl",
        str(root / "exports/sft_attack_train.jsonl"),
        "--base-model",
        raw_model(config["models"]["U"]),
        "--checkpoint-name",
        "linuxarena_self_rec",
        "--out-path-file",
        str(root / "sft/checkpoint.txt"),
        "--val-frac",
        "0",
    ]
    for key, value in config["finetuning"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    (root / "sft").mkdir(exist_ok=True)
    return run_command(command, args.env_file, args.dry_run)


def score(config: dict[str, Any], root: Path, args: argparse.Namespace) -> int:
    checkpoint = (root / "sft/checkpoint.txt").read_text().strip()
    if not checkpoint.startswith("tinker://"):
        raise ValueError(
            "Expected a tinker:// sampler checkpoint in sft/checkpoint.txt"
        )
    from lasr_labs_2025_control_project.scripts.score_linuxarena_actions import (
        validate_resume,
    )

    arms = (("base", config["models"]["U"]), ("sft", "tinker/" + checkpoint))
    rows = read_jsonl(root / "exports/sft_attack_test.jsonl")
    for arm, model in arms:
        validate_resume(rows, model, root / f"sft/scores_test_{arm}.jsonl")
    for arm, model in arms:
        command = [
            sys.executable,
            "-m",
            "lasr_labs_2025_control_project.scripts.score_linuxarena_actions",
            "--rows-jsonl",
            str(root / "exports/sft_attack_test.jsonl"),
            "--model",
            model,
            "--output-jsonl",
            str(root / f"sft/scores_test_{arm}.jsonl"),
            "--concurrency",
            str(config["scoring"]["concurrency"]),
        ]
        result = run_command(command, args.env_file, args.dry_run)
        if result:
            return result
    return 0


def plot(config: dict[str, Any], root: Path) -> None:
    # Use the exact optimizer and figure entry point behind scripts/plot_data.py.
    import matplotlib

    matplotlib.use("Agg")
    from lasr_labs_2025_control_project.utils.numerical_pipeline import (
        build_analysis_payload,
        build_optimization_config,
        optimize_parameters,
        plot_data,
        save_analysis_payload,
    )
    from lasr_labs_2025_control_project.scripts.score_linuxarena_actions import (
        summarize,
    )

    output = root / "sft/plot_all"
    output.mkdir(parents=True, exist_ok=True)
    actions = read_jsonl(root / "exports/actions.jsonl")
    results = {}
    reports = {}
    tables = {}
    for arm in ("base", "sft"):
        scores = read_jsonl(root / f"sft/scores_test_{arm}.jsonl")
        rows = assemble_flat_rows(actions, scores)
        with (output / f"flat_{arm}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        reports[arm] = {**missing_scores(rows), "recognition": summarize(scores)}
        tables[arm] = rows
    write_json(output / "coverage.json", reports)
    # Check both arms before any fitting; never fall back to a perfect honest U.
    checked = {arm: statistical_tables(rows) for arm, rows in tables.items()}
    settings = config["plot"]
    for arm, (attacks, honeypots, benign) in checked.items():
        optimization = build_optimization_config(
            attacks_df=attacks,
            honeypots_df=honeypots,
            benign_df=benign,
            **{
                key: value for key, value in settings.items() if key != "max_iterations"
            },
        )
        result = optimize_parameters(
            base_config=optimization, max_iterations=settings["max_iterations"]
        )
        results[f"LinuxArena {arm}"] = result
        save_analysis_payload(
            build_analysis_payload(result, include_history=True),
            output / f"analysis_data_{arm}.json.gz",
        )
    plot_data(results, destination_folder=output)
    print(json.dumps(reports, indent=2))


def main() -> None:
    if sys.argv[1:2] == ["ct"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--options", type=Path, required=True)
        parser.add_argument("--env-file", type=Path, required=True)
        parser.add_argument("--allow-paid", action="store_true", required=True)
        args, remaining = parser.parse_known_args(sys.argv[2:])
        from lasr_labs_2025_control_project.utils.linuxarena_tinker import launch

        launch(args.options, args.env_file, remaining)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Required for sampling, logprobs or training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running CT or Tinker",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry failed generation/monitor jobs",
    )
    parser.add_argument(
        "--rescore-complete",
        action="store_true",
        help="Explicitly pay for a new monitor pass over completed jobs, retaining old attempts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum generation/monitor jobs launched this invocation",
    )
    parser.add_argument(
        "stage", choices=("generate", "monitor", "export", "finetune", "score", "plot")
    )
    args = parser.parse_args()
    if args.stage in {"generate", "monitor", "finetune", "score"} and not (
        args.allow_paid or args.dry_run
    ):
        parser.error(
            f"{args.stage} is cost-bearing; pass --allow-paid only after approving this run"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    root = (REPO_ROOT / config["experiment_folder"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.env_file = args.env_file.resolve()
    with (root / ".pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "Another LinuxArena stage is running in this experiment_folder"
            )
        if (root / "plan.json").exists():
            plan = json.loads((root / "plan.json").read_text())
            ensure_plan(root, config, plan["cases"])
        elif args.stage != "generate":
            parser.error("Run generate first to freeze the triple and task plan")
        if args.stage in {"generate", "monitor"}:
            result = drive(config, root, args, args.stage == "monitor")
        elif args.stage == "export":
            result = export(config, root, args)
        elif args.stage == "finetune":
            result = finetune(config, root, args)
        elif args.stage == "score":
            result = score(config, root, args)
        else:
            if args.dry_run:
                parser.error("plot is offline; --dry-run is not needed")
            plot(config, root)
            result = 0
    sys.exit(result)


if __name__ == "__main__":
    main()
