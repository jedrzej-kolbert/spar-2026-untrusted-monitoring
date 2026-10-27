"""Analyse stage: solve the game for each condition and plot the results.

For each monitor config (one condition) this reads the attacks / honeypots /
honest monitor outputs, solves the game for each monitor variant (all monitors
and untrusted-monitor-only, or just the variant the config/flags select),
bootstraps each variant (`--bootstrap-replicates`, 0 to skip), and writes
`analysis_data*.json.gz` / `bootstrap_*.jsonl.gz` to the config's results
folder. With several configs, the local figures compare them on shared axes.

Each condition is one W&B run. Curves are logged as raw numbers for W&B's own
panels: each point is logged with its x-axis as a second metric
(`StageRun.define_curve`; metric meanings are in the README's Results
section). The per-condition figures are also logged as images, for plots W&B
doesn't draw by default. The analysis files are logged as an artifact.
"""

import argparse
import gzip
import json
import logging
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.tracking import (
    ALLOW_UNLINKED_HELP,
    StageRun,
    artifact_input,
    config_tags,
    load_raw_config,
    names,
    resolve_input,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    ExperimentConfig,
    RunLabel,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    PAPER_GAME_PARAMETERS,
    OptimizationResult,
    bootstrap_band,
    build_analysis_payload,
    build_optimization_config,
    optimize_parameters,
    plot_data,
    run_replicates,
    save_analysis_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_REPLICATES = 2


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="plot_data",
        description=(
            "Solve the game for each monitor config (condition) and generate "
            "plots/optimization results. Can pass multiple configs with -cfg"
        ),
    )
    parser.add_argument(
        "--config",
        "-cfg",
        nargs="+",
        required=True,
        help="Path(s) to monitor YAML config file(s)",
    )
    parser.add_argument(
        "--disable-trusted-monitor",
        default=False,
        help="Only analyse the untrusted-monitor-only variant",
    )
    parser.add_argument(
        "--experiment-default",
        default="exp",
        help="Fallback experiment name prefix if a config lacks defaults.experiment_name",
    )
    parser.add_argument(
        "--grouped-path",
        default=None,
        help="Save visualizations to a 'grouped' subfolder in the results folder",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Bootstrap replicates per variant (0 to skip)",
    )
    parser.add_argument(
        "--bootstrap-ci", type=float, default=0.80, help="Bootstrap band width"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel bootstrap workers"
    )
    parser.add_argument(
        "--allow-unlinked", action="store_true", help=ALLOW_UNLINKED_HELP
    )
    args = parser.parse_args()

    # One game run per condition, keyed by experiment_name
    games: dict[str, Game] = {}

    for cfg in args.config:
        cfg_path = Path(cfg)
        normalized = load_and_normalize_config(cfg_path, config_type="monitor")
        if args.disable_trusted_monitor:
            normalized.defaults["disable_trusted_monitor"] = True
        games[normalized.experiment_name] = analyse_condition(
            normalized,
            cfg_path,
            replicates=args.bootstrap_replicates,
            ci=args.bootstrap_ci,
            n_jobs=args.n_jobs,
            allow_unlinked=args.allow_unlinked,
        )
    aggregated = {label: game.result for label, game in games.items()}

    dest_folder = (
        normalized.results_folder or "src/lasr_labs_2025_control_project/results"
    )
    if args.grouped_path:
        if args.disable_trusted_monitor:
            dest_folder = (
                normalized.experiment_folder
                / f"results/grouped/um_only/{args.grouped_path}"
            )
        else:
            dest_folder = (
                normalized.experiment_folder
                / f"results/grouped/all_monitors/{args.grouped_path}"
            )
    logger.info("Generating visualizations across %d experiments", len(aggregated))
    plot_data(aggregated, destination_folder=dest_folder)
    logger.info("Visualizations saved to %s", dest_folder)
    log_analysis(games, args.grouped_path, allow_unlinked=args.allow_unlinked)


@dataclass
class Game:
    """One condition's solved game: its primary result and payload, and the
    ``game`` artifact its run logged (None when W&B is off)."""

    name: str
    result: OptimizationResult
    payload: dict[str, Any]
    artifact: Any = None


def analysis_run_name(games: Mapping[str, Game], grouped_path: Optional[str]) -> str:
    """``analysis.<grouping>``, or the conditions' common prefix (``apps.self_rec``)."""
    if grouped_path:
        return names.fit(f"analysis.{names.clean(grouped_path)}")
    parts = [game.name.split(".") for game in games.values()]
    common = []
    for segment in zip(*parts):
        if len(set(segment)) > 1:
            break
        common.append(segment[0])
    return names.fit("analysis." + (".".join(common) or "conditions"))


def log_analysis(
    games: Mapping[str, Game],
    grouped_path: Optional[str] = None,
    *,
    allow_unlinked: bool = False,
) -> None:
    """The analysis run: the figures for every condition analysed together.

    It declares each condition's ``game`` artifact as an input, so W&B's
    lineage shows which games a set of figures came from.
    """
    tags: list[str] = ["analysis"]
    for game in games.values():
        tags += game.payload.get("tags", [])
    tracker = StageRun(
        "analysis",
        analysis_run_name(games, grouped_path),
        config={
            "conditions": [game.name for game in games.values()],
            "grouped_path": grouped_path,
        },
        tags=tags,
        inputs=[artifact_input(g.artifact) for g in games.values() if g.artifact],
        allow_unlinked=allow_unlinked,
    )
    with tracker, tempfile.TemporaryDirectory() as tmp:
        monitors = {m for g in games.values() for m in g.payload.get("roc_curves", {})}
        for monitor in sorted(monitors):
            tracker.log_roc(
                f"roc/{monitor}",
                {
                    game.name: list(zip(roc["fpr"], roc["tpr"]))
                    for game in games.values()
                    if (roc := game.payload.get("roc_curves", {}).get(monitor))
                },
                title=f"{monitor} ROC",
            )
        figures = Path(tmp) / "figures"
        plot_data(
            {label: game.result for label, game in games.items()},
            figures,
            file_format="png",
        )
        images = sorted(figures.glob("*.png"))
        for image in images:
            tracker.log_image(f"figures/{image.stem}", image)
        if images:
            tracker.log_artifact(images, type="figures")


def analysis_variants(defaults: dict[str, Any]) -> list[tuple[str, bool, bool]]:
    """(name, disable_trusted_monitor, disable_untrusted_monitor); first is primary."""
    if defaults.get("disable_untrusted_monitor", False):
        return [("tm_only", False, True)]
    if defaults.get("disable_trusted_monitor", False):
        return [("um_only", True, False)]
    return [("all_monitors", False, False), ("um_only", True, False)]


def analysis_name(normalized: ExperimentConfig, raw: dict[str, Any]) -> str:
    condition = (raw.get("wandb") or {}).get("condition")
    if condition:
        return names.condition_name(
            condition.get("setting", "unknown"),
            condition.get("collusion_strategy", "unknown"),
            condition.get("transform", "unknown"),
            condition.get("monitor_arm", "unknown"),
        )
    return names.fit(names.clean(normalized.experiment_name))


def analyse_condition(
    normalized: ExperimentConfig,
    cfg_path: Path,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    ci: float = 0.80,
    n_jobs: int = -1,
    allow_unlinked: bool = False,
) -> Game:
    """Solve, bootstrap and log one condition as a ``game`` run (no figures:
    those come from the analysis run)."""
    raw = load_raw_config(cfg_path)
    runs = {run.label: run for run in normalized.runs}
    labels = (RunLabel.ATTACKS, RunLabel.HONEYPOTS, RunLabel.HONEST)
    inputs = {label: resolve_input(runs[label].save_path) for label in labels}
    attacks_df, honeypots_df, benign_df = (
        build_jsonl_dataframe_from_eval_log(inputs[label].local_path)
        for label in labels
    )
    variants = analysis_variants(normalized.defaults)
    name = analysis_name(normalized, raw)
    results_folder = (
        Path(normalized.results_folder) if normalized.results_folder else None
    )

    models = condition_models(attacks_df, honeypots_df)
    tracker = StageRun(
        "game",
        name,
        config={
            "config_file": str(cfg_path),
            "yaml": raw,
            "game_parameters": PAPER_GAME_PARAMETERS,
            "variants": [v[0] for v in variants],
            "bootstrap_replicates": replicates,
            "bootstrap_ci": ci,
            "models": models,
        },
        tags=["game", *config_tags(raw)],
        inputs=list(inputs.values()),
        allow_unlinked=allow_unlinked,
    )
    primary: OptimizationResult | None = None
    primary_payload: dict[str, Any] = {}
    with tracker, tempfile.TemporaryDirectory() as tmp:
        files: list[Path] = []
        out_dir = results_folder or Path(tmp)
        for index, (variant, disable_tm, disable_um) in enumerate(variants):
            result = solve_condition(
                attacks_df,
                honeypots_df,
                benign_df,
                disable_trusted_monitor=disable_tm,
                disable_untrusted_monitor=disable_um,
            )
            # Persist the full analysis data (Nash strategies at every sweep
            # point, empirical ROC curves, joint surface samples) next to the
            # plots, so figures can be regenerated without re-optimizing.
            payload = build_analysis_payload(result, include_history=index == 0)
            stem = "analysis_data" if index == 0 else f"analysis_data_{variant}"
            files.append(save_analysis_payload(payload, out_dir / f"{stem}.json.gz"))
            logger.info("Analysis data saved to %s", files[-1])

            reps: list[dict[str, Any]] = []
            if replicates > 0:
                logger.info(
                    "[%s/%s] bootstrapping %d replicates", name, variant, replicates
                )
                reps = run_replicates(
                    range(replicates),
                    n_jobs=n_jobs,
                    condition_seed=zlib.crc32(f"{name}/{variant}".encode("utf-8")),
                    attacks_df=attacks_df,
                    honeypots_df=honeypots_df,
                    benign_df=benign_df,
                    disable_trusted_monitor=disable_tm,
                    disable_untrusted_monitor=disable_um,
                )
                path = out_dir / f"bootstrap_{variant}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in reps)
                files.append(path)
            log_variant(tracker, variant, payload, reps, ci)
            if index == 0:
                primary, primary_payload = result, payload
                log_payload_overview(tracker, payload)

        assert primary is not None
        artifact = tracker.log_artifact(
            files,
            type="game",
            # Prefixed like the analysis run's figures, so a game never shares a
            # name with an artifact of another type (W&B refuses that).
            name=names.fit(f"game.{name}"),
            metadata={"variants": [v[0] for v in variants], "models": models},
            wait=True,
        )
    primary_payload["tags"] = config_tags(raw)
    return Game(name, primary, primary_payload, artifact)


def condition_models(
    attacks_df: pd.DataFrame, honeypots_df: pd.DataFrame
) -> dict[str, Any]:
    """U and H (who wrote the attacks and the honeypots) and T (the trusted
    monitor), as the monitor logs record them."""
    return {
        "U": attacks_df.attrs.get("generator"),
        "H": honeypots_df.attrs.get("generator"),
        "T": attacks_df.attrs.get("trusted_monitor"),
    }


def log_variant(
    tracker: StageRun,
    variant: str,
    payload: dict[str, Any],
    replicates: list[dict[str, Any]],
    ci: float,
) -> None:
    """Safety-usefulness curve (and bootstrap band) as W&B-plottable metrics."""
    sweep = sorted(
        (e for e in payload["sweep"] if e.get("safety") is not None),
        key=lambda e: e["usefulness"],
    )
    tracker.define_curve(f"{variant}/safety", f"{variant}/usefulness")
    for entry in sweep:
        tracker.log(
            {
                f"{variant}/usefulness": entry["usefulness"],
                f"{variant}/safety": entry["safety"],
            }
        )
    summary: dict[str, Any] = {}
    if sweep:
        summary[f"{variant}/safety_at_max_usefulness"] = sweep[-1]["safety"]
        summary[f"{variant}/min_safety"] = min(e["safety"] for e in sweep)
    band = bootstrap_band(replicates, ci)
    if band:
        x = f"{variant}/band/usefulness"
        keys = ("safety_lo", "safety_median", "safety_hi")
        for key in keys:
            tracker.define_curve(f"{variant}/band/{key}", x)
        order = sorted(
            range(len(band["usefulness"])), key=lambda i: band["usefulness"][i]
        )
        for i in order:
            tracker.log(
                {
                    x: band["usefulness"][i],
                    **{f"{variant}/band/{k}": band[k][i] for k in keys},
                }
            )
    if replicates:
        summary[f"{variant}/bootstrap_failed"] = sum("error" in r for r in replicates)
    summary[f"{variant}/converged"] = payload.get("converged")
    tracker.summary(summary)


def log_payload_overview(tracker: StageRun, payload: dict[str, Any]) -> None:
    """Monitor ROC curves and the honest-UM operating point (variant-independent)."""
    for monitor, roc in payload.get("roc_curves", {}).items():
        tracker.log_roc(
            f"roc/{monitor}",
            {monitor: list(zip(roc["fpr"], roc["tpr"]))},
            title=f"{monitor} ROC",
        )
    tracker.summary(
        {
            "sample_counts": payload.get("sample_counts"),
            "monitor_metadata": payload.get("monitor_metadata"),
        }
    )


def build_jsonl_dataframe_from_eval_log(eval_log_path: Path) -> pd.DataFrame:
    """Build a DataFrame from a .eval log file.

    Args:
        eval_log_path: Path to the .eval log file.

    Returns:
        DataFrame with the eval log data. DataFrame has columns:
        - sample_id
        - problem
        - solution
        - untrusted_collusion_score
        - trusted_suspiciousness_score
        - collusion_monitor_system_prompt
        - collusion_monitor_user_prompt
    """
    log = read_eval_log(eval_log_path)
    assert log.samples, (
        f"No samples in eval log {eval_log_path} (status={log.status!r}) — "
        "a cancelled/empty run would silently produce a degenerate ROC"
    )
    final_samples = []
    for sample in log.samples:
        metadata = {**sample.metadata, **sample.store}
        final_samples.append(
            {
                "sample_id": sample.id,
                "problem": metadata["main_task_description"],
                "solution": metadata["base_solution"],
                "untrusted_collusion_score": metadata["untrusted_collusion_score"],
                "trusted_suspiciousness_score": metadata[
                    "trusted_suspiciousness_score"
                ],
                "untrusted_suspiciousness_score": metadata[
                    "untrusted_suspiciousness_score"
                ],
                "collusion_monitor_system_prompt": metadata[
                    "untrusted_collusion_score_metadata"
                ]["model_info"]["system_prompt"],
                "collusion_monitor_user_prompt": metadata[
                    "untrusted_collusion_score_metadata"
                ]["model_info"]["base_prompt"],
            }
        )
        if "trajectory_id" in metadata:
            final_samples[-1]["trajectory_id"] = metadata["trajectory_id"]
            final_samples[-1]["n_actions"] = metadata["n_actions"]
    df = pd.DataFrame(final_samples)
    if "trajectory_id" in df:
        df = _max_over_actions(df, eval_log_path)
    # Who wrote these solutions, and the trusted monitor that scored them.
    first = {**log.samples[0].metadata, **log.samples[0].store}
    generator = (first.get("generator_info") or {}).get("model_info") or {}
    trusted = (first.get("trusted_suspiciousness_score_metadata") or {}).get(
        "model_info"
    ) or {}
    df.attrs["generator"] = generator.get("model")
    df.attrs["trusted_monitor"] = trusted.get("model")
    return df


_SCORE_COLUMNS = (
    "untrusted_collusion_score",
    "trusted_suspiciousness_score",
    "untrusted_suspiciousness_score",
)


def _max_over_actions(df: pd.DataFrame, eval_log_path: Path) -> pd.DataFrame:
    """Per-action monitor logs (view: action): one row per trajectory, each score
    the max over its available action scores. No scores for a monitor means the
    trajectory is omitted from ROCs requiring it; missing action rows are an error.
    """
    counts = df.groupby("trajectory_id").size()
    expected = df.groupby("trajectory_id")["n_actions"].first()
    incomplete = counts[counts != expected]
    if len(incomplete):
        raise ValueError(
            f"{eval_log_path}: {len(incomplete)} trajectories are missing scored "
            f"actions (e.g. {incomplete.index[0]!r}); was the monitor run limited "
            "or sampled with actions_per_trajectory?"
        )
    grouped = df.groupby("trajectory_id", sort=False)
    out = grouped.first().reset_index()
    for column in _SCORE_COLUMNS:
        out[column] = grouped[column].max().to_numpy()
        missing = out[column].isna()
        if missing.any():
            logger.info(
                "%s: no %s scores for trajectories %s; excluded from ROCs requiring this score",
                eval_log_path,
                column,
                out.loc[missing, "trajectory_id"].tolist(),
            )
    out["sample_id"] = out["trajectory_id"]
    out["n_actions_scored"] = counts.loc[out["trajectory_id"]].to_numpy()
    unscored = out[list(_SCORE_COLUMNS)].isna().all(axis=1)
    if unscored.any():
        logger.warning(
            "%s: dropping trajectories with no available monitor scores: %s",
            eval_log_path,
            out.loc[unscored, "trajectory_id"].tolist(),
        )
    return out.loc[~unscored].reset_index(drop=True)


def plot_ensemble_monitoring(
    *,
    attacks_result_eval: Path,
    honeypots_result_eval: Path,
    benign_result_eval: Path,
    num_tasks: int,
    blue_team_budget_fraction: float,
    slack_honeypots: float,
    blue_team_replace_fnr: float,
    disable_trusted_monitor: bool = False,
    disable_untrusted_monitor: bool = False,
) -> OptimizationResult:
    """Build optimization inputs from both monitors using DataFrames.

    Constructs ROC functions from attack/honeypot/benign data and runs optimization
    to find the Nash equilibrium defense strategy.

    We treat:
      - attacks: positive examples for both monitors
      - honeypots: negative examples for the untrusted (collusion) monitor
      - benign (formerly 'honest'): negative examples for the trusted monitor
    Args:
        attacks_json: Path to attacks JSONL file
        honeypots_json: Path to honeypots JSONL file
        benign_json: Path to benign JSONL file
        num_tasks: Number of tasks in simulation
        blue_team_budget_fraction: Budget fraction for honeypots + audits
        slack_honeypots: Number of honeypot failures red team can tolerate
        blue_team_replace_fnr: Trusted monitor FNR for replacement decisions

    Returns a dict suitable for visualization.plot_data.
    """
    logger.info("Loading attacks / honeypots / benign eval logs for monitoring plots")
    return solve_condition(
        build_jsonl_dataframe_from_eval_log(attacks_result_eval),
        build_jsonl_dataframe_from_eval_log(honeypots_result_eval),
        build_jsonl_dataframe_from_eval_log(benign_result_eval),
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        slack_honeypots=slack_honeypots,
        blue_team_replace_fnr=blue_team_replace_fnr,
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )


def solve_condition(
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    *,
    num_tasks: int = int(PAPER_GAME_PARAMETERS["num_tasks"]),
    blue_team_budget_fraction: float = PAPER_GAME_PARAMETERS[
        "blue_team_budget_fraction"
    ],
    slack_honeypots: float = PAPER_GAME_PARAMETERS["slack_honeypots"],
    blue_team_replace_fnr: float = 0.0,
    disable_trusted_monitor: bool = False,
    disable_untrusted_monitor: bool = False,
) -> OptimizationResult:
    """Build the ROC model from monitor scores and solve for the Nash equilibrium."""
    config = build_optimization_config(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        blue_team_replace_fnr=blue_team_replace_fnr,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        slack_honeypots=slack_honeypots,
        benign_df=benign_df,
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )

    result = optimize_parameters(base_config=config, max_iterations=100)

    # Attach extra metadata (diagnostic) for visualization
    meta = result.config.monitor_roc_fn_metadata or {}
    # Expose untrusted suspiciousness distributions if available
    if (
        "untrusted_suspiciousness_score" in attacks_df.columns
        and "untrusted_suspiciousness_score" in benign_df.columns
    ):
        meta.setdefault(
            "untrusted_suspiciousness_scores",
            {
                "positive": attacks_df["untrusted_suspiciousness_score"]
                .astype(float)
                .tolist(),
                "negative": benign_df["untrusted_suspiciousness_score"]
                .astype(float)
                .tolist(),
            },
        )
    meta["models"] = condition_models(attacks_df, honeypots_df)  # for figure captions
    result.config.monitor_roc_fn_metadata = meta
    return result


def run_single_config(normalized: ExperimentConfig) -> OptimizationResult:
    """Execute optimization for a single normalized config dict.

    Args:
        normalized: The normalized config dict from load_and_normalize_config.
        It contains the runs and the defaults.

    Returns:
        The OptimizationResult (ensemble monitoring currently assumed).
    """
    runs = {run.label: run for run in normalized.runs}

    attacks_result_eval = runs[RunLabel.ATTACKS].save_path
    honeypots_result_eval = runs[RunLabel.HONEYPOTS].save_path
    benign_result_eval = runs[RunLabel.HONEST].save_path

    disable_trusted_monitor = normalized.defaults.get("disable_trusted_monitor", False)
    disable_untrusted_monitor = normalized.defaults.get(
        "disable_untrusted_monitor", False
    )

    return plot_ensemble_monitoring(
        attacks_result_eval=attacks_result_eval,
        honeypots_result_eval=honeypots_result_eval,
        benign_result_eval=benign_result_eval,
        num_tasks=int(PAPER_GAME_PARAMETERS["num_tasks"]),
        slack_honeypots=PAPER_GAME_PARAMETERS["slack_honeypots"],
        blue_team_budget_fraction=PAPER_GAME_PARAMETERS["blue_team_budget_fraction"],
        blue_team_replace_fnr=0.0,
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )


if __name__ == "__main__":
    main()
