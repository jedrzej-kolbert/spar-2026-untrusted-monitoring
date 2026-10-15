"""Bootstrap the full analysis for many conditions, resumably, saving every replicate.

The analyse stage (plot_data.py) already bootstraps each condition it runs
(`--bootstrap-replicates`); this runner is for large overnight sweeps across
whole config trees. The replicate logic lives in
utils/numerical_pipeline/bootstrap.py: each replicate's complete analysis
payload (safety/usefulness sweep, blue/red Nash mixed strategies, empirical ROC
curves, joint-surface samples) is persisted as one JSON line, so CI bands and
strategy diagnostics can be computed later without re-running anything.

Conditions are enumerated the same way as the analysis driver: from monitor
configs whose output triples exist on disk (see
helpers/list_runnable_monitor_configs.py), so the bootstrap can never silently
run on a different set of conditions than the point estimates.

    PYTHONPATH=src python src/lasr_labs_2025_control_project/scripts/bootstrap.py \
        --trees apps_open --replicates 1000 --out-root bootstrap_results/open-source-model

Output layout (one directory per condition):

    <out-root>/<setting>/<experiment>/<arm>[__<config-stem>]/
        replicates.jsonl.gz   # one analysis payload per line, gzip members appended per chunk
        manifest.json         # commit, params, seeds, timing, replicate count

Resumable: on restart, completed replicates (lines already in replicates.jsonl.gz)
are skipped. Seeds are deterministic per (condition, replicate index).
"""

import argparse
import gzip
import json
import logging
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from lasr_labs_2025_control_project.scripts.helpers.list_runnable_monitor_configs import (
    runnable_configs,
)
from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    RunLabel,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    PAPER_GAME_PARAMETERS,
    run_replicates,
)

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _existing_replicates(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        logger.warning("Could not read %s; starting this condition over", path)
        path.unlink()
        return 0


def bootstrap_condition(
    key: str,
    cfg_path: Path,
    *,
    out_dir: Path,
    replicates: int,
    n_jobs: int,
    chunk_size: int,
) -> None:
    normalized = load_and_normalize_config(cfg_path, config_type="monitor")
    runs = {run.label: run for run in normalized.runs}
    attacks_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.ATTACKS].save_path)
    honeypots_df = build_jsonl_dataframe_from_eval_log(
        runs[RunLabel.HONEYPOTS].save_path
    )
    benign_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.HONEST].save_path)
    disable_tm = normalized.defaults.get("disable_trusted_monitor", False)
    disable_um = normalized.defaults.get("disable_untrusted_monitor", False)

    out_dir.mkdir(parents=True, exist_ok=True)
    lines_path = out_dir / "replicates.jsonl.gz"
    done = _existing_replicates(lines_path)
    if done >= replicates:
        logger.info("[%s] already complete (%d replicates)", key, done)
        return
    if done:
        logger.info("[%s] resuming at replicate %d", key, done)

    # Stable across processes and restarts (unlike builtin hash()).
    condition_seed = zlib.crc32(key.encode("utf-8"))
    manifest = {
        "condition": key,
        "config": str(cfg_path),
        "commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "replicates_target": replicates,
        "condition_seed": condition_seed,
        "game_parameters": dict(PAPER_GAME_PARAMETERS),
        "sample_counts": {
            "attacks": len(attacks_df),
            "honeypots": len(honeypots_df),
            "benign": len(benign_df),
        },
        "disable_trusted_monitor": disable_tm,
        "disable_untrusted_monitor": disable_um,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    start = time.time()
    for chunk_start in range(done, replicates, chunk_size):
        chunk = range(chunk_start, min(chunk_start + chunk_size, replicates))
        payloads = run_replicates(
            chunk,
            n_jobs=n_jobs,
            condition_seed=condition_seed,
            attacks_df=attacks_df,
            honeypots_df=honeypots_df,
            benign_df=benign_df,
            disable_trusted_monitor=disable_tm,
            disable_untrusted_monitor=disable_um,
        )
        # Appending opens a new gzip member; readers see one continuous stream.
        failed = [p["bootstrap_replicate"] for p in payloads if "error" in p]
        if failed:
            logger.warning(
                "[%s] %d failed replicate(s) in chunk: %s", key, len(failed), failed
            )
        with gzip.open(lines_path, "at", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload) + "\n")
        elapsed = time.time() - start
        completed = chunk.stop - done
        logger.info(
            "[%s] %d/%d replicates (%.1fs/replicate wall)",
            key,
            chunk.stop,
            replicates,
            elapsed / max(completed, 1),
        )

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["replicates_done"] = replicates
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition-key substrings to filter to",
    )
    args = parser.parse_args()

    todo: list[tuple[str, Path]] = []
    for tree in args.trees:
        for key, cfg_path in runnable_configs(tree):
            # tm_only shares a condition key with none_base; disambiguate by
            # config stem so both keep separate output directories.
            out_key = key
            if "tm_only" in cfg_path.stem:
                out_key = f"{key}__tm_only"
            if args.conditions and not any(s in out_key for s in args.conditions):
                continue
            todo.append((out_key, cfg_path))

    logger.info(
        "Bootstrapping %d conditions x %d replicates", len(todo), args.replicates
    )
    for out_key, cfg_path in todo:
        bootstrap_condition(
            out_key,
            cfg_path,
            out_dir=args.out_root / out_key,
            replicates=args.replicates,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
        )
    logger.info("ALL_BOOTSTRAPS_DONE")


if __name__ == "__main__":
    main()
