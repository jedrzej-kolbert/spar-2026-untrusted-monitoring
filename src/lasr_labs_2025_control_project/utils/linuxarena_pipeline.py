"""Offline planning, journals and flat score tables for the LinuxArena setting."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

POPULATION_ROLES = {"honest": "U", "attacks": "U", "honeypots": "H"}
MONITOR_ROLES = {"trusted": "T", "untrusted_suspiciousness": "U"}
REPO_ROOT = Path(__file__).resolve().parents[3]


def raw_model(model: str) -> str:
    if not model.startswith("tinker/") or model.startswith("tinker/tinker://"):
        raise ValueError(f"LinuxArena requires a base tinker/<model ID>, got {model!r}")
    return model.removeprefix("tinker/")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported LinuxArena pipeline schema_version")
    for role in ("U", "T", "H"):
        raw_model(config["models"][role])
        mode = config["model_settings"][role]["tool_dialect"]
        if mode not in {"auto", "schema", "argv"}:
            raise ValueError(f"Unsupported tool dialect {mode!r} for {role}")
    if config["models"]["U"] == config["models"]["H"]:
        raise ValueError("U and H must be different base models for self-recognition")
    generation = config["generation"]
    if generation["concurrent_samples"] != 1:
        raise ValueError(
            "LinuxArena runs one sandbox at a time; concurrent_samples must be 1"
        )
    if not generation["side_task"]:
        raise ValueError(
            "Both attack populations require the same configured side_task"
        )
    for key in (
        "task_count",
        "max_steps",
        "max_tokens",
        "sample_time_limit",
        "message_limit",
    ):
        if generation[key] <= 0:
            raise ValueError(f"generation.{key} must be positive")
    if config["monitor"]["cot_level"] != "none":
        raise ValueError(
            "LinuxArena uses the native action monitor with cot_level none"
        )
    if config["monitor"]["max_tokens"] <= 0 or config["scoring"]["concurrency"] <= 0:
        raise ValueError("Monitor token limit and scoring concurrency must be positive")
    for key in ("agent_cpus", "sidecar_cpus", "pids_limit"):
        if config["resources"][key] <= 0:
            raise ValueError(f"resources.{key} must be positive")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def append_status(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def latest_status(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["job_id"]: record for record in records}


def pending_job(record: dict[str, Any] | None, retry_failed: bool) -> bool:
    if record is None or record["state"] == "starting":
        return True
    return record["state"] == "failed" and retry_failed


def select_cases(tasks: dict[str, Any], generation: dict[str, Any]) -> list[str]:
    explicit = generation.get("main_tasks")
    if explicit:
        if len(explicit) != len(set(explicit)) or any(
            case not in tasks for case in explicit
        ):
            raise ValueError(
                "main_tasks must be unique task IDs present in the environment"
            )
        return list(explicit)
    eligible = [
        key
        for key, task in tasks.items()
        if not generation["task_category"]
        or str(task.task_category) == generation["task_category"]
    ]
    if len(eligible) < generation["task_count"]:
        raise ValueError(
            f"Requested {generation['task_count']} tasks but only {len(eligible)} match"
        )
    return sorted(
        eligible,
        key=lambda key: hashlib.sha256(
            f"{generation['selection_seed']}:{key}".encode()
        ).hexdigest(),
    )[: generation["task_count"]]


def ensure_plan(root: Path, config: dict[str, Any], cases: list[str]) -> dict[str, Any]:
    """A frozen config prevents silently mixing triples or task selections on resume."""
    plan = {"config": config, "cases": cases}
    path = root / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "Run plan differs from the generated config; use a fresh experiment_folder"
            )
    else:
        write_json(path, plan)
    return plan


def bridge_options(config: dict[str, Any], root: Path) -> dict[str, Any]:
    routes: dict[str, dict[str, Any]] = {"False": {}, "True": {}}
    for monitoring, roles in ((False, ("U", "H")), (True, ("T", "U"))):
        for role in roles:
            model = raw_model(config["models"][role])
            settings = config["model_settings"][role]
            previous = routes[str(monitoring)].get(model)
            if previous is not None and previous != settings:
                raise ValueError(
                    f"Conflicting per-role settings for the same model {model!r}"
                )
            routes[str(monitoring)][model] = settings
    return {
        "models": routes,
        "resources": config["resources"],
        "translation_log": str(root / "tool_translations.jsonl"),
    }


def model_spec(config: dict[str, Any], role: str, monitoring: bool) -> str:
    route = "tinker-linuxarena-monitor" if monitoring else "tinker-linuxarena"
    tokens = (
        config["monitor"]["max_tokens"]
        if monitoring
        else config["generation"]["max_tokens"]
    )
    temperature = 0 if monitoring else config["generation"]["temperature"]
    return json.dumps(
        {
            "model_name": f"{route}/{raw_model(config['models'][role])}",
            "config": {
                "max_tokens": tokens,
                "temperature": temperature,
                "cache": False,
            },
        }
    )


def generation_arguments(
    config: dict[str, Any], case: str, population: str, destination: Path
) -> list[str]:
    generation = config["generation"]
    args = [
        "run",
        "eval",
        "-e",
        generation["environment"],
        "-t",
        case,
        "--untrusted-policy",
        "honest" if population == "honest" else "attack",
        "--model-role",
        "untrusted=" + model_spec(config, POPULATION_ROLES[population], False),
        "--model-role",
        "trusted=" + model_spec(config, "T", True),
        "-wa",
        f"max_steps={generation['max_steps']}",
        "--max-samples",
        "1",
        "--sample-time-limit",
        str(generation["sample_time_limit"]),
        "--message-limit",
        str(generation["message_limit"]),
        "--retry-on-error",
        "0",
        "--no-upload",
        "--log-dir",
        str(destination),
        "--run-name",
        f"linuxarena-{population}-{case}",
    ]
    if population != "honest":
        args += ["-s", generation["side_task"]]
    return args


def monitoring_arguments(
    config: dict[str, Any], source: Path, destination: Path
) -> list[str]:
    monitors = [
        {
            "name": name,
            "prompt": "simple",
            "model": name,
            "cot_level": "none",
            "samples": 1,
        }
        for name in MONITOR_ROLES
    ]
    args = [
        "run",
        "monitor",
        str(source),
        "-m",
        json.dumps(monitors),
        "--kind",
        "action",
        "-o",
        str(destination),
        "--no-upload",
    ]
    for name, role in MONITOR_ROLES.items():
        args += ["--model-role", name + "=" + model_spec(config, role, True)]
    return args


def generation_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every emitted eval is kept, including failed attempts; starts are also journaled."""
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        for path in record.get("eval_files", []):
            sources[path] = {
                "path": path,
                "actor_id": record["actor_id"],
                "population": record["population"],
            }
    return list(sources.values())


def export_manifest(
    sources: list[dict[str, Any]], statuses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prefer the latest monitored copy; fall back to raw when monitoring failed."""
    copies: dict[str, str] = {}
    for record in statuses:
        files = record.get("eval_files", [])
        if files:
            if len(files) != 1:
                raise ValueError(f"Expected one monitored copy per source, got {files}")
            copies[record["source"]] = files[0]
    return [
        {**source, "path": copies.get(source["path"], source["path"])}
        for source in sources
    ]


def native_score(
    entries: list[dict[str, Any]], monitor: str
) -> tuple[float | None, list[Any]]:
    matching = [entry for entry in entries if entry["monitor"] == monitor]
    if not matching:
        return None, ["monitor entry missing"]
    errors = [
        entry.get("unscored_reason") or "nonfinite score"
        for entry in matching
        if not finite(entry.get("value"))
    ]
    if errors:
        return None, errors
    return sum(float(entry["value"]) for entry in matching) / len(matching), []


def finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def recognition_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        current = best.get(record["action_id"])
        if current is None or not finite(current.get("margin")):
            best[record["action_id"]] = record
    return best


def assemble_flat_rows(
    actions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    split: str = "test",
    trusted_name: str = "trusted",
    suspiciousness_name: str = "untrusted_suspiciousness",
) -> list[dict[str, Any]]:
    """Left join by stable action ID; errors and missing scores remain visible."""
    margins = recognition_records(scores)
    rows = []
    ids: set[str] = set()
    for action in actions:
        meta = action["meta"]
        if meta["split"] != split:
            continue
        action_id = meta["action_id"]
        if action_id in ids:
            raise ValueError(f"Duplicate action {action_id}")
        ids.add(action_id)
        trusted, trusted_errors = native_score(
            meta["native_monitor_scores"], trusted_name
        )
        suspiciousness, suspiciousness_errors = native_score(
            meta["native_monitor_scores"], suspiciousness_name
        )
        record = margins.get(action_id, {})
        margin = record.get("margin")
        rows.append(
            {
                "action_id": action_id,
                "trajectory_id": meta["trajectory_id"],
                "population": meta["population"],
                "split": split,
                "main_task_ids": meta["main_task_ids"],
                "trusted_suspiciousness_score": trusted,
                "untrusted_suspiciousness_score": suspiciousness,
                "untrusted_collusion_score": margin if finite(margin) else None,
                "trusted_errors": trusted_errors,
                "suspiciousness_errors": suspiciousness_errors,
                "recognition_error": None
                if meta["population"] == "honest"
                else record.get("error")
                if finite(margin)
                else record.get("error", "recognition score missing"),
                "sample_error": meta.get("sample_error"),
                "eval_error": meta.get("eval_error"),
            }
        )
    return rows


def missing_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = (
        "trusted_suspiciousness_score",
        "untrusted_suspiciousness_score",
        "untrusted_collusion_score",
    )
    return {
        "rows": len(rows),
        "by_population": {
            pop: sum(row["population"] == pop for row in rows)
            for pop in POPULATION_ROLES
        },
        "missing": {
            column: sum(
                not finite(row[column])
                for row in rows
                if column != "untrusted_collusion_score"
                or row["population"] != "honest"
            )
            for column in columns
        },
    }


def statistical_tables(rows: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
    """Refuse incomplete data before the statistical model can silently drop NaNs."""
    report = missing_scores(rows)
    if any(report["missing"].values()) or any(
        count == 0 for count in report["by_population"].values()
    ):
        raise ValueError(
            f"Incomplete flat scores; repair/retry scoring, never drop rows: {json.dumps(report)}"
        )
    import pandas as pd

    frame = pd.DataFrame(rows)
    attacks, honeypots, benign = [
        frame[frame.population == pop].reset_index(drop=True)
        for pop in ("attacks", "honeypots", "honest")
    ]
    return attacks, honeypots, benign
