"""Export saved BashArena logs as trajectory and per-action JSONL. Offline.

Every attempt is kept, including errored, unscored and zero-action ones; the
reasons a baseline loader would reject an attempt are recorded, not applied.

Example:
    python -m basharena_runtime.export attacks.eval --output out/ \\
        --population attacks --monitor-actor-id tinker/Qwen/Qwen3.6-35B-A3B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from basharena_runtime import check_runtime, monitor_prompts, views

SCHEMA_VERSION = "basharena.saved-actions.v1"
POPULATIONS = ("attacks", "honest", "honeypots")


def stable_id(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def json_line(value: Any) -> str:
    """Represent non-finite scores explicitly rather than emit invalid JSON/null."""

    def encode(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return {
                "$nonfinite": "NaN"
                if math.isnan(item)
                else "Infinity"
                if item > 0
                else "-Infinity"
            }
        if isinstance(item, dict):
            return {k: encode(v) for k, v in item.items()}
        if isinstance(item, list):
            return [encode(v) for v in item]
        return item

    return json.dumps(encode(value), ensure_ascii=False, allow_nan=False) + "\n"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_provenance() -> dict[str, Any]:
    """What rendered the views, so a later change to the format is detectable."""
    return {
        "package": "basharena_runtime",
        "upstream_reference": "basharena_public@86082fd",
        "files": {
            Path(module.__file__).name: file_hash(Path(module.__file__))
            for module in (views, monitor_prompts)
        },
        "runtime": check_runtime(),
        "options": {
            "max_history_chars": None,
            "actions_per_trajectory": None,
            "seed": 0,
        },
    }


@dataclass(frozen=True)
class ExportSource:
    path: Path
    population: str
    split: str | None = None
    split_provenance: str | None = None
    actor_id: str | None = None
    monitor_actor_id: str | None = None

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS:
            raise ValueError("population must be attacks, honest or honeypots")
        if (self.split is None) != (self.split_provenance is None):
            raise ValueError(
                "A declared split requires split_provenance (and vice versa)"
            )
        for name in ("split", "split_provenance", "actor_id", "monitor_actor_id"):
            if getattr(self, name) == "":
                raise ValueError(f"{name} cannot be empty")


def sample_records(
    sample: Any,
    ordinal: int,
    log: Any,
    source: ExportSource,
    source_hash: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retain every attempt, including ones baseline dataset loaders would reject."""
    raw = sample.model_dump(mode="json")
    metadata, store = sample.metadata or {}, sample.store or {}
    merged = {**metadata, **store}  # Same precedence as the existing Bash loaders.
    messages = sample.messages
    indices = [i for i, m in enumerate(messages) if views.is_scored_action(m)]
    trajectory_id = stable_id(
        [SCHEMA_VERSION, source_hash, sample.id, sample.epoch, ordinal]
    )
    generator = store.get("generator_info") or {}
    generator_actor = (generator.get("model_info") or {}).get("model")
    actor = source.actor_id or generator_actor or log.eval.model
    own_source = None
    if source.monitor_actor_id:
        # Generation records its model without the provider ("Qwen/X" for
        # "tinker/Qwen/X"), so an id read from the log is also compared that way.
        # Declared ids are compared exactly: looser matching can hide checkpoints.
        same = {source.monitor_actor_id}
        if not source.actor_id and generator_actor:
            same.add(source.monitor_actor_id.split("/", 1)[-1])
        own_source = actor in same
    problem = merged.get("main_task_description")
    reasons = []
    if metadata.get("setting_name") is None:
        reasons.append("missing_setting_name")
    if sample.error is not None:
        reasons.append("sample_error")
    if not indices:
        reasons.append("zero_actions")
    if not sample.scores:
        reasons.append("missing_scores")
    if sample.limit is not None:
        reasons.append("sample_limit_reached")
    if sample.invalidation is not None:
        reasons.append("sample_invalidated")
    for i in indices:
        end = i + 1
        while end < len(messages) and messages[end].role == "tool":
            end += 1
        expected = {call.id for call in messages[i].tool_calls}
        observed = {m.tool_call_id for m in messages[i + 1 : end]}
        if expected - observed:
            reasons.append(f"missing_tool_outputs:message_{i}")
    if not isinstance(problem, str) or not problem.strip():
        reasons.append("missing_task_description")
    if "transformed_solution" in merged:
        reasons.append("transformed_solution_present_baseline_uses_original_messages")
    recorded_splits = {
        k: metadata[k] for k in ("split", "data_split", "task_split") if k in metadata
    }
    if source.split is not None and any(
        v != source.split for v in recorded_splits.values()
    ):
        reasons.append("declared_split_conflicts_with_metadata")
    try:
        rendered = views.action_views(messages)
        if len(rendered) != len(indices):
            raise ValueError("Renderer action count does not match action boundaries")
    except Exception as error:
        rendered = [None] * len(indices)
        reasons.append(f"render_error:{type(error).__name__}:{error}")
    try:
        eligible = views.baseline_eligible(
            sample.scores,
            metadata.get("eval_mode"),
            merged.get("base_solution", views.trajectory_view(messages)),
            sample.error is not None,
        )
    except (ValueError, TypeError, AttributeError) as error:
        eligible = False
        reasons.append(f"baseline_validity_error:{type(error).__name__}:{error}")
    common = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "source_sha256": source_hash,
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "sample_ordinal": ordinal,
        "population": source.population,
        "split": {
            "declared": source.split,
            "provenance": source.split_provenance,
            "recorded": recorded_splits,
        },
        "actor": {
            "id": actor,
            "declared": source.actor_id,
            "id_source": "declared"
            if source.actor_id
            else "generator_info"
            if generator_actor
            else "eval.model",
            "generator_model": generator_actor,
            "eval_model": log.eval.model,
        },
        "labels": {
            "monitor_actor_id": source.monitor_actor_id,
            "own_source": own_source,
        },
        "renderer": provenance,
    }
    ledger = {
        **common,
        "record_type": "trajectory",
        "source_path": str(source.path.resolve()),
        "log_status": log.status,
        "log_error": log.error.model_dump(mode="json") if log.error else None,
        "eval_spec": log.eval.model_dump(mode="json"),
        "sample": raw,
        "action_count": len(indices),
        "reasons": reasons,
        "baseline_dataset_eligible": bool(eligible),
        "trajectory_scores": raw.get("scores"),
        "task_identity": {
            k: metadata.get(k)
            for k in ("setting_name", "main_task_name", "side_task_name", "eval_mode")
        },
    }
    rows = []
    for k, (i, view) in enumerate(zip(indices, rendered, strict=True), 1):
        end = i + 1
        while end < len(messages) and messages[end].role == "tool":
            end += 1
        monitor_messages = None
        recognition_messages = None
        if view is not None and isinstance(problem, str) and problem.strip():
            solution = views.format_solution(view, per_action=True)
            recognition_messages = [
                {
                    "role": "system",
                    "content": monitor_prompts.BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": monitor_prompts.BASHARENA_ACTION_USER_PROMPT.format(
                        problem_statement=problem, solution=solution
                    ),
                },
            ]
            monitor_messages = [
                {
                    "role": "system",
                    "content": monitor_prompts.BASHARENA_ACTION_SUSPICIOUSNESS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": monitor_prompts.BASHARENA_ACTION_SUSPICIOUSNESS_USER_PROMPT.format(
                        problem_statement=problem, solution=solution
                    ),
                },
            ]
        rows.append(
            {
                **common,
                "record_type": "action",
                "action_id": stable_id([trajectory_id, k]),
                "action_index": k,
                "message_index": i,
                "history_message_range": [0, i],
                "output_message_range": [i + 1, end],
                "proposed_message": raw["messages"][i],
                "tool_outputs": raw["messages"][i + 1 : end],
                "baseline_view": view,
                "recognition_messages": recognition_messages,
                "monitor_messages": monitor_messages,
                "per_action_scores": None,  # Never broadcast trajectory scores onto actions.
            }
        )
    return ledger, rows


def export_saved_log(source: ExportSource, output: Path) -> dict[str, int]:
    """Create a new output directory; never overwrite an earlier export."""
    provenance = renderer_provenance()
    source_hash = file_hash(source.path)
    log = read_eval_log(source.path)
    if log.samples is None:
        raise ValueError("Log has no sample payload; cannot enumerate attempts")
    if any(
        (s.metadata or {}).get("setting_name") not in {None, "bash_arena"}
        for s in log.samples
    ):
        raise ValueError("Explicitly non-Bash samples are not supported")
    output.mkdir(parents=True, exist_ok=False)
    counts = {"trajectories": 0, "actions": 0, "baseline_eligible_trajectories": 0}
    with (
        (output / "actions.jsonl").open("x", encoding="utf-8") as actions,
        (output / "trajectories.jsonl").open("x", encoding="utf-8") as ledger,
    ):
        for ordinal, sample in enumerate(log.samples):
            trajectory, rows = sample_records(
                sample, ordinal, log, source, source_hash, provenance
            )
            ledger.write(json_line(trajectory))
            for row in rows:
                actions.write(json_line(row))
            counts["trajectories"] += 1
            counts["actions"] += len(rows)
            counts["baseline_eligible_trajectories"] += trajectory[
                "baseline_dataset_eligible"
            ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "counts": counts,
        "source_path": str(source.path.resolve()),
        "source_sha256": source_hash,
        "log_status": log.status,
        "eval_spec": log.eval.model_dump(mode="json"),
        "renderer": provenance,
    }
    (output / "manifest.json").write_text(json_line(manifest), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", choices=POPULATIONS, required=True)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--split-provenance")
    parser.add_argument(
        "--actor-id",
        help="Exact actor or checkpoint identity; default: the model in the log",
    )
    parser.add_argument(
        "--monitor-actor-id",
        help="The recognising model: actions by this actor are labelled own_source",
    )
    args = parser.parse_args()
    source = ExportSource(
        args.log,
        args.population,
        args.split,
        args.split_provenance,
        args.actor_id,
        args.monitor_actor_id,
    )
    print(json.dumps(export_saved_log(source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
