"""Game and analysis stages: solve the game per condition, then draw the figures.

The pipeline (``lasr_labs_2025_control_project.pipeline``) calls ``solve_game``
once per condition (collusion strategy x transform x monitor arm). It reads the
attacks / honeypots / honest monitor outputs, solves the game for each
analysis variant (all monitors, untrusted-monitor only, trusted-monitor only),
bootstraps each variant, and writes into the condition's folder:

    analysis_data[_<variant>].json.gz   the payload (Nash strategies at every
                                        sweep point, ROC curves, joint surface)
                                        plus what produced it, under "pipeline"
    bootstrap_<variant>.jsonl.gz        one replicate payload per line; an
                                        interrupted bootstrap resumes here
    scores.json.gz                      the monitor scores the game was solved on

That folder is the condition's ``game`` W&B run's artifact. Curves are logged
as raw numbers for W&B's own panels (each point with its x-axis as a second
metric, `StageRun.define_curve`), ROC curves as W&B ROC charts; metric meanings
are in the README's Results section.

Then ``draw_analysis``, once per experiment, draws the figures for all its
conditions from their saved games (``load_game``; nothing is re-solved) and
logs them in the ``analysis`` run, with the games as inputs.

As a script, this compares saved conditions on shared axes:

    python -m lasr_labs_2025_control_project.scripts.plot_data \
        experiments/apps/game/<condition>-<hash> ... --out figures/ [--variant um_only]
"""

import argparse
import gzip
import json
import logging
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.tracking import StageRun, names
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
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    OPTIMIZERS,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.results_io import (
    load_analysis_payload,
)

logger = logging.getLogger(__name__)

LABELS = ("attacks", "honeypots", "honest")
#: analysis variant -> (disable_trusted_monitor, disable_untrusted_monitor)
VARIANT_FLAGS = {
    "all_monitors": (False, False),
    "um_only": (True, False),
    "tm_only": (False, True),
}
_SCORE_COLUMNS = (
    "untrusted_collusion_score",
    "trusted_suspiciousness_score",
    "untrusted_suspiciousness_score",
)


@dataclass
class SolveSettings:
    """How one condition's game is solved (optimization section of the config)."""

    game: Mapping[str, float] = field(
        default_factory=lambda: dict(PAPER_GAME_PARAMETERS)
    )
    regeneration_cutoffs: Optional[Sequence[float]] = None
    solver: str = "double_oracle"
    max_iterations: int = 100


@dataclass
class BootstrapSettings:
    replicates: int = 2
    ci: float = 0.80
    n_jobs: int = -1
    chunk_size: int = 25


def payload_path(folder: Path, variant_index: int, variant: str) -> Path:
    stem = "analysis_data" if variant_index == 0 else f"analysis_data_{variant}"
    return folder / f"{stem}.json.gz"


def bootstrap_path(folder: Path, variant: str) -> Path:
    return folder / f"bootstrap_{variant}.jsonl.gz"


def variant_seeds(seed: int, index: int) -> tuple[int, int]:
    """(solver seed, bootstrap condition seed) for the index-th variant."""
    solver_seed, bootstrap_seed = np.random.SeedSequence([seed, index]).generate_state(
        2
    )
    return int(solver_seed), int(bootstrap_seed)


def read_replicates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    # zlib.error: a chunk cut mid-stream with another appended after it
    except (OSError, EOFError, zlib.error, json.JSONDecodeError):
        logger.warning("%s: interrupted write, kept %d replicates", path, len(rows))
    return rows


def bootstrap_variant(
    path: Path,
    settings: BootstrapSettings,
    *,
    condition_seed: int,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run (or resume) a variant's bootstrap, appending replicates in chunks."""
    replicates = read_replicates(path)[: settings.replicates]
    done = len(replicates)
    if path.exists() and done < settings.replicates:
        logger.info("%s: resuming at replicate %d", path.name, done)
        # rewrite, so a cut-off tail can't hide the chunks appended after it
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(p) + "\n" for p in replicates)
    for start in range(done, settings.replicates, settings.chunk_size):
        chunk = range(start, min(start + settings.chunk_size, settings.replicates))
        payloads = run_replicates(
            chunk, n_jobs=settings.n_jobs, condition_seed=condition_seed, **kwargs
        )
        failed = [p["bootstrap_replicate"] for p in payloads if "error" in p]
        if failed:
            logger.warning("%s: failed replicate(s) %s", path.name, failed)
        # Appending opens a new gzip member; readers see one continuous stream.
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.writelines(json.dumps(p) + "\n" for p in payloads)
        replicates.extend(payloads)
    return replicates


def load_condition(
    inputs: Mapping[str, Path], min_samples: int = 0
) -> dict[str, pd.DataFrame]:
    """The three monitor outputs as dataframes, refusing too-small samples."""
    frames = {
        label: build_jsonl_dataframe_from_eval_log(inputs[label]) for label in LABELS
    }
    small = {label: len(df) for label, df in frames.items() if len(df) < min_samples}
    if small:
        raise ValueError(
            f"too few usable samples to analyse {small} (minimum {min_samples}, "
            "measurement.min_samples_per_label); raise the generation limits"
        )
    return frames


SCORES_FILE = "scores.json.gz"


@dataclass
class Game:
    """One condition's solved game: its primary result and payload, its
    bootstrap replicates, and the ``game`` artifact its run logged (or None)."""

    name: str
    result: OptimizationResult
    payload: dict[str, Any]
    replicates: list[dict[str, Any]] = field(default_factory=list)
    artifact: Any = None


def save_scores(frames: Mapping[str, pd.DataFrame], path: Path) -> Path:
    """The monitor scores a game is solved on, with the models that made them."""
    record = {
        label: {"rows": df.to_dict(orient="records"), "attrs": dict(df.attrs)}
        for label, df in frames.items()
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(record, f, default=str)
    return path


def load_scores(path: Path) -> dict[str, pd.DataFrame]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.load(f)
    frames = {}
    for label, saved in record.items():
        df = pd.DataFrame(saved["rows"])
        df.attrs.update(saved["attrs"])
        frames[label] = df
    return frames


def solve_game(
    *,
    name: str,
    inputs: Mapping[str, Path],
    variants: Sequence[str],
    solve: SolveSettings,
    bootstrap: BootstrapSettings,
    seed: int,
    out_dir: Path,
    tracker: StageRun,
    record: Optional[Mapping[str, Any]] = None,
    min_samples: int = 0,
) -> Game:
    """Solve, bootstrap, save and log one condition (the ``game`` run).

    ``record`` is stored in every payload under "pipeline" (with the inputs,
    settings and seeds), so a saved condition says what produced it.
    """
    frames = load_condition(inputs, min_samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary: Optional[OptimizationResult] = None
    primary_payload: dict[str, Any] = {}
    primary_reps: list[dict[str, Any]] = []
    with tracker:
        files: list[Path] = [save_scores(frames, out_dir / SCORES_FILE)]
        for index, variant in enumerate(variants):
            disable_tm, disable_um = VARIANT_FLAGS[variant]
            solver_seed, bootstrap_seed = variant_seeds(seed, index)
            result = solve_condition(
                frames["attacks"],
                frames["honeypots"],
                frames["honest"],
                settings=solve,
                seed=solver_seed,
                disable_trusted_monitor=disable_tm,
                disable_untrusted_monitor=disable_um,
            )
            # Persist the full analysis data (Nash strategies at every sweep
            # point, empirical ROC curves, joint surface samples), so figures
            # can be drawn without re-optimizing.
            payload = build_analysis_payload(result, include_history=index == 0)
            payload["pipeline"] = {
                **(record or {}),
                "name": name,
                "variant": variant,
                "inputs": {label: str(path) for label, path in inputs.items()},
                "solve": asdict(solve),
                "solver_seed": solver_seed,
                "bootstrap_seed": bootstrap_seed,
            }
            files.append(
                save_analysis_payload(payload, payload_path(out_dir, index, variant))
            )
            logger.info("Analysis data saved to %s", files[-1])

            reps: list[dict[str, Any]] = []
            if bootstrap.replicates > 0:
                logger.info(
                    "[%s/%s] bootstrapping %d replicates",
                    name,
                    variant,
                    bootstrap.replicates,
                )
                reps = bootstrap_variant(
                    bootstrap_path(out_dir, variant),
                    bootstrap,
                    condition_seed=bootstrap_seed,
                    attacks_df=frames["attacks"],
                    honeypots_df=frames["honeypots"],
                    benign_df=frames["honest"],
                    disable_trusted_monitor=disable_tm,
                    disable_untrusted_monitor=disable_um,
                    game=dict(solve.game),
                    regeneration_cutoffs=solve.regeneration_cutoffs,
                    optimizer=OPTIMIZERS[solve.solver],
                    max_iterations=solve.max_iterations,
                )
                files.append(bootstrap_path(out_dir, variant))
            log_variant(tracker, variant, payload, reps, bootstrap.ci)
            log_strategies(tracker, variant, payload)
            if index == 0:
                primary, primary_payload, primary_reps = result, payload, reps
                log_payload_overview(tracker, payload)

        assert primary is not None
        artifact = tracker.log_artifact(
            files,
            type="game",
            # Prefixed like the analysis run's figures, so a game never shares a
            # name with an artifact of another type (W&B refuses that).
            name=names.fit(f"game.{name}"),
            metadata={
                "variants": list(variants),
                "models": primary_payload.get("models"),
            },
            wait=True,
        )
        tracker.finish(complete=True)
    return Game(name, primary, primary_payload, primary_reps, artifact)


def load_game(folder: Path, variant: Optional[str] = None) -> Game:
    """A saved game, ready to plot: its result is rebuilt from the saved sweep
    and scores (the ROC model is refitted, which is cheap; nothing is re-solved)."""
    payload = load_saved_condition(folder, variant)
    record = payload["pipeline"]
    if (folder / SCORES_FILE).exists():
        frames = load_scores(folder / SCORES_FILE)
    else:  # saved before scores were kept with the game
        frames = load_condition({k: Path(v) for k, v in record["inputs"].items()})
    disable_tm, disable_um = VARIANT_FLAGS[record["variant"]]
    game = {**PAPER_GAME_PARAMETERS, **record["solve"]["game"]}
    config = build_optimization_config(
        num_tasks=game["num_tasks"],
        blue_team_budget_fraction=game["blue_team_budget_fraction"],
        slack_honeypots=game["slack_honeypots"],
        human_audit_success_rate=game.get("human_audit_success_rate", 1.0),
        blue_team_replace_fnr=0.0,
        attacks_df=frames["attacks"],
        honeypots_df=frames["honeypots"],
        benign_df=frames["honest"],
        disable_trusted_monitor=disable_tm,
        disable_untrusted_monitor=disable_um,
    )
    result = OptimizationResult(
        results=[dict(entry) for entry in payload["sweep"]],
        config=config,
        converged=bool(payload.get("converged", True)),
    )
    _attach_diagnostics(
        result, frames["attacks"], frames["honeypots"], frames["honest"]
    )
    replicates = read_replicates(bootstrap_path(folder, record["variant"]))
    return Game(record["name"], result, payload, replicates)


def draw_analysis(
    games: Mapping[str, Game], out_dir: Path, tracker: StageRun
) -> list[Path]:
    """The figures for several conditions together (the ``analysis`` run).

    Logs each monitor's ROC curves overlaid as one W&B ROC chart, the figures
    as images, and the figure files as an artifact; returns the files.
    """
    figures = out_dir / "figures"
    with tracker:
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
        plot_data(
            {label: game.result for label, game in games.items()},
            figures,
            file_format="png",
            bootstrap={label: game.replicates for label, game in games.items()},
        )
        images = sorted(figures.glob("*.png"))
        for image in images:
            tracker.log_image(f"figures/{image.stem}", image)
        files = [*images, *sorted(figures.glob("*.jsonl"))]
        tracker.log_artifact(files, type="figures")
        tracker.finish(complete=True)
    return files


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


def log_strategies(tracker: StageRun, variant: str, payload: dict[str, Any]) -> None:
    """Equilibrium strategies across the sweep, in the `strategy/` panel section.

    Each strategy variable is logged under its game-parameter name, taken
    from `best_parameters`: blue's most likely pure strategy and red's best
    response to blue's mix (what safety is computed against). Alongside them,
    how mixed each team's Nash strategy is.
    """
    prefix = f"strategy/{variant}"
    x = f"{prefix}/usefulness"
    points = []
    for entry in sorted(payload["sweep"], key=lambda e: e["usefulness"]):
        blue_mix = entry.get("blue_mixed_strategy") or []
        point = {
            **{
                f"{prefix}/{name}": value
                for name, value in (entry.get("best_parameters") or {}).items()
            },
            f"{prefix}/blue_top_probability": max(
                (prob for _, prob in blue_mix), default=None
            ),
            f"{prefix}/blue_support_size": len(blue_mix),
            f"{prefix}/red_nash_support_size": len(
                entry.get("red_mixed_strategy") or []
            ),
        }
        points.append(
            (entry["usefulness"], {k: v for k, v in point.items() if v is not None})
        )
    for key in dict.fromkeys(k for _, point in points for k in point):
        tracker.define_curve(key, x)
    for usefulness, point in points:
        tracker.log({x: usefulness, **point})


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
        row = {
            "sample_id": sample.id,
            "problem": metadata["main_task_description"],
            "solution": metadata["base_solution"],
        }
        # A monitor left out of measurement.monitors leaves its column out.
        row.update({c: metadata[c] for c in _SCORE_COLUMNS if c in metadata})
        collusion = metadata.get("untrusted_collusion_score_metadata")
        if collusion:
            row["collusion_monitor_system_prompt"] = collusion["model_info"][
                "system_prompt"
            ]
            row["collusion_monitor_user_prompt"] = collusion["model_info"][
                "base_prompt"
            ]
        final_samples.append(row)
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
    columns = [c for c in _SCORE_COLUMNS if c in df]
    for column in columns:
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
    unscored = out[columns].isna().all(axis=1)
    if unscored.any():
        logger.warning(
            "%s: dropping trajectories with no available monitor scores: %s",
            eval_log_path,
            out.loc[unscored, "trajectory_id"].tolist(),
        )
    return out.loc[~unscored].reset_index(drop=True)


def solve_condition(
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    *,
    settings: Optional[SolveSettings] = None,
    seed: Optional[int] = None,
    disable_trusted_monitor: bool = False,
    disable_untrusted_monitor: bool = False,
) -> OptimizationResult:
    """Build the ROC model from monitor scores and solve for the Nash equilibrium."""
    settings = settings or SolveSettings()
    game = {**PAPER_GAME_PARAMETERS, **settings.game}
    config = build_optimization_config(
        num_tasks=game["num_tasks"],
        blue_team_budget_fraction=game["blue_team_budget_fraction"],
        slack_honeypots=game["slack_honeypots"],
        human_audit_success_rate=game.get("human_audit_success_rate", 1.0),
        blue_team_replace_fnr=0.0,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )

    result = optimize_parameters(
        base_config=config,
        max_iterations=settings.max_iterations,
        replace_fnr_values=(
            list(settings.regeneration_cutoffs)
            if settings.regeneration_cutoffs is not None
            else None
        ),
        optimizer=OPTIMIZERS[settings.solver],
        seed=seed,
    )

    _attach_diagnostics(result, attacks_df, honeypots_df, benign_df)
    return result


def _attach_diagnostics(
    result: OptimizationResult,
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    benign_df: pd.DataFrame,
) -> None:
    """Extra metadata the figures read: suspiciousness scores and the models."""
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
    meta["models"] = condition_models(attacks_df, honeypots_df)  # figure captions
    result.config.monitor_roc_fn_metadata = meta


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


def load_saved_condition(folder: Path, variant: Optional[str] = None) -> dict[str, Any]:
    """The payload a pipeline game step saved in ``folder`` (primary variant
    by default), with its ``pipeline`` record of inputs, settings and seeds."""
    if variant is None:
        return load_analysis_payload(folder / "analysis_data.json.gz")
    for path in sorted(folder.glob("analysis_data*.json.gz")):
        payload = load_analysis_payload(path)
        if (payload.get("pipeline") or {}).get("variant") == variant:
            return payload
    raise FileNotFoundError(f"{folder}: no saved analysis for variant {variant!r}")


def saved_conditions(root: Path, setting: Optional[str] = None) -> list[Path]:
    """Every game folder under ``root`` (an experiments dir), optionally one setting's."""
    pattern = (
        f"{setting}/game/*/analysis_data.json.gz"
        if setting
        else "*/game/*/analysis_data.json.gz"
    )
    return sorted(path.parent for path in root.glob(pattern))


def condition_of(record: Mapping[str, Any]) -> dict[str, str]:
    """setting / collusion_strategy / transform / monitor_arm of a saved record."""
    return dict(record["key"]["params"]["condition"])


def resolve_saved_condition(
    folder: Path, variant: Optional[str] = None
) -> OptimizationResult:
    """Re-solve a saved condition exactly as recorded (inputs, settings, seed)."""
    record = load_saved_condition(folder, variant)["pipeline"]
    frames = load_condition({k: Path(v) for k, v in record["inputs"].items()})
    disable_tm, disable_um = VARIANT_FLAGS[record["variant"]]
    return solve_condition(
        frames["attacks"],
        frames["honeypots"],
        frames["honest"],
        settings=SolveSettings(**record["solve"]),
        seed=record["solver_seed"],
        disable_trusted_monitor=disable_tm,
        disable_untrusted_monitor=disable_um,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="plot_data",
        description="Compare saved conditions on shared axes (from their saved games).",
    )
    parser.add_argument("folders", nargs="+", type=Path, help="game output folders")
    parser.add_argument("--out", type=Path, required=True, help="figure folder")
    parser.add_argument(
        "--variant",
        default=None,
        choices=sorted(VARIANT_FLAGS),
        help="analysis variant to compare (default: each folder's primary one)",
    )
    args = parser.parse_args()
    games = [load_game(folder, args.variant) for folder in args.folders]
    plot_data(
        {game.name: game.result for game in games},
        destination_folder=args.out,
        bootstrap={game.name: game.replicates for game in games},
    )
    logger.info("Figures saved to %s", args.out)


if __name__ == "__main__":
    main()
