"""Browse an RL run's JSONL records as one self-contained HTML page.

    python -m lasr_labs_2025_control_project.finetuning.grouped_rl.viewer \\
        --run-dir experiments/open-source-model/apps/rl/<run>

Writes ``run_view.html`` into the run directory (open it in any browser; no server or
network needed). Runs write it themselves when they finish. The page has a tab for
each record file:

- Rollouts (``rollouts.jsonl``): each problem at each step, with the original attack
  beside its signal-OFF and signal-ON rewrites and their scores.
- Metrics (``metrics.jsonl``): every logged number, by step.
- Honeypots (``honeypot_scores.jsonl``): the receiver's P(YES) on each honeypot.

Rollouts don't store the original attacks or honeypots, so these are read from the
``.eval`` logs named in the run's manifest. If a log is missing the page still builds
and says what is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .data import read_generator_eval

VIEW_FILE = "run_view.html"
_TEMPLATE = Path(__file__).with_name("viewer_template.html")
_PLACEHOLDER = "__RUN_DATA__"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of a JSONL file (none if missing) and how many lines were unreadable.

    A run that was killed can leave a partial last line. NaN and Infinity become
    None, because the page's JSON parser rejects them.
    """
    if not path.exists():
        return [], 0
    rows, bad = [], 0
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line, parse_constant=lambda _: None))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _sample(row: dict[str, Any]) -> dict[str, Any]:
    """A rollout for the page, dropping a response that only repeats the code."""
    sample = {key: value for key, value in row.items() if key != "response"}
    response = row.get("response") or ""
    unwrapped = response.replace("<final_code>", "").replace("</final_code>", "")
    if unwrapped.strip() != (row.get("code") or "").strip():
        sample["response"] = response
    return sample


def group_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rollouts of one problem at one step and phase, split into OFF and ON."""
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("phase", ""), row.get("step", 0), str(row.get("problem_id")))
        group = groups.setdefault(
            key,
            {
                "phase": key[0],
                "step": key[1],
                "problem_id": key[2],
                "off": [],
                "on": [],
            },
        )
        group["on" if row.get("payload") else "off"].append(_sample(row))
    for group in groups.values():
        off = {(item.get("code") or "").strip() for item in group["off"]}
        on = {(item.get("code") or "").strip() for item in group["on"]}
        # The same test as the identical_pair_fraction metric.
        group["identical"] = bool((off & on) - {""})
    return sorted(
        groups.values(), key=lambda g: (g["step"], g["phase"], g["problem_id"])
    )


def _eval_path(value: str | None) -> Path | None:
    """A manifest's data path, relative to where the run started or to the repo."""
    if not value:
        return None
    path = Path(value)
    return path if path.exists() or path.is_absolute() else _REPO_ROOT / path


def _read_code(
    value: str | None, label: str, wanted: set[str], notes: list[str]
) -> dict[str, Any]:
    """Samples for the wanted problems from an eval log, noting if it can't be read."""
    path = _eval_path(value)
    if path is None or not path.exists():
        notes.append(
            f"{label} unavailable: log not found ({value or 'not configured'})."
        )
        return {}
    try:
        samples = read_generator_eval(path)
    except Exception as exc:  # a bad log must not lose the rest of the page
        notes.append(f"{label} unavailable: could not read {path} ({exc}).")
        return {}
    return {
        sample.problem_id: sample for sample in samples if sample.problem_id in wanted
    }


def build_view_data(run_dir: Path) -> dict[str, Any]:
    """Everything the page shows, as JSON-serialisable data."""
    notes: list[str] = []
    rollouts, bad_rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    metrics, bad_metrics = read_jsonl(run_dir / "metrics.jsonl")
    honeypots, bad_honeypots = read_jsonl(run_dir / "honeypot_scores.jsonl")
    bad = bad_rollouts + bad_metrics + bad_honeypots
    if bad:
        notes.append(f"Skipped {bad} unreadable line(s) in the run's JSONL files.")
    if not rollouts and not metrics and not honeypots:
        raise ValueError(f"no RL records in {run_dir}")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        notes.append("No manifest.json, so original attacks can't be located.")
    config = manifest.get("config", {})
    data = config.get("data", {})

    rollout_ids = {str(row.get("problem_id")) for row in rollouts}
    honeypot_ids = {str(row.get("problem_id")) for row in honeypots}
    attacks = _read_code(
        data.get("attacks_eval"), "Original attacks", rollout_ids | honeypot_ids, notes
    )
    honeypot_code = (
        _read_code(data.get("honeypots_eval"), "Honeypot code", honeypot_ids, notes)
        if honeypots
        else {}
    )
    suspicion_path = run_dir / "original_attack_suspicion.json"
    suspicion = (
        json.loads(suspicion_path.read_text()) if suspicion_path.exists() else {}
    )
    problems = {
        problem_id: {
            "question": sample.question,
            "original": sample.code,
            "original_suspicion": suspicion.get(problem_id),
            "honeypot": honeypot_code[problem_id].code
            if problem_id in honeypot_code
            else None,
        }
        for problem_id, sample in attacks.items()
    }
    for problem_id, sample in honeypot_code.items():
        problems.setdefault(
            problem_id,
            {
                "question": sample.question,
                "original": None,
                "original_suspicion": suspicion.get(problem_id),
                "honeypot": sample.code,
            },
        )
    return {
        "run_name": manifest.get("run_name") or run_dir.name,
        "started": manifest.get("started"),
        "models": config.get("models", {}),
        "reward": config.get("reward", {}),
        "notes": notes,
        "groups": group_rollouts(rollouts),
        "metrics": metrics,
        "honeypots": honeypots,
        "problems": problems,
    }


def render_view(data: dict[str, Any]) -> str:
    """The page with ``data`` embedded.

    Model output is untrusted: ``<`` is escaped so it can't close the script tag, and
    the page only ever inserts it as text.
    """
    payload = json.dumps(data, default=str).replace("<", "\\u003c")
    template = _TEMPLATE.read_text()
    assert template.count(_PLACEHOLDER) == 1
    return template.replace(_PLACEHOLDER, payload)


def write_run_view(run_dir: Path) -> Path:
    """Write the run's ``run_view.html``; returns its path."""
    path = run_dir / VIEW_FILE
    path.write_text(render_view(build_view_data(run_dir)))
    return path


@click.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True)
def main(run_dir: str) -> None:
    click.echo(f"wrote {write_run_view(Path(run_dir))}")


if __name__ == "__main__":
    main()
