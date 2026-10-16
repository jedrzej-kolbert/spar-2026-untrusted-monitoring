import argparse
import logging
from pathlib import Path
from typing import Any

from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    RunLabel,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_optimization_config,
    plot_parameter_sensitivity,
    run_parameter_sensitivity_analysis,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot_parameter_sensitivity",
        description=(
            "Run blue-team optimization, perform one-at-a-time parameter sensitivity "
            "sweeps, and render sensitivity plots for the specified config."
        ),
    )
    parser.add_argument(
        "--config",
        "-cfg",
        required=True,
        help="Path to monitor YAML config used by monitor_solutions_inspect",
    )
    parser.add_argument(
        "--disable-trusted-monitor",
        default=False,
        help="Disable the trusted monitor in optimization",
    )
    parser.add_argument(
        "--variation-points",
        type=int,
        default=21,
        help="Number of sweep points per parameter (>=2)",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional destination directory. Defaults to the config's "
            "plots_destination_folder/parameter_sensitivity/<experiment_name>."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    normalized = load_and_normalize_config(cfg_path)
    if args.disable_trusted_monitor:
        normalized.defaults["disable_trusted_monitor"] = True
    defaults: dict[str, Any] = normalized.defaults or {}
    experiment_name = normalized.experiment_name

    dest_folder = (
        normalized.results_folder or "src/lasr_labs_2025_control_project/results"
    )
    destination = dest_folder / "parameter_sensitivity" / experiment_name

    runs = {run.label: run for run in normalized.runs}

    attacks_result_eval = runs[RunLabel.ATTACKS].save_path
    honeypots_result_eval = runs[RunLabel.HONEYPOTS].save_path
    benign_result_eval = runs[RunLabel.HONEST].save_path

    required_labels = {RunLabel.ATTACKS, RunLabel.HONEYPOTS, RunLabel.HONEST}
    missing = required_labels - set(runs.keys())
    if missing:
        raise ValueError(
            "Config must contain runs labelled 'attacks', 'honeypots', and 'honest' for sensitivity analysis. "
            f"Missing: {sorted(str(label) for label in missing)}"
        )

    attacks_df = build_jsonl_dataframe_from_eval_log(attacks_result_eval)
    honeypots_df = build_jsonl_dataframe_from_eval_log(honeypots_result_eval)
    benign_df = build_jsonl_dataframe_from_eval_log(benign_result_eval)

    num_tasks = int(defaults.get("num_tasks", 5000))
    blue_team_budget_fraction = float(defaults.get("blue_team_budget_fraction", 0.02))
    slack_honeypots = float(defaults.get("slack_honeypots", 5.0))

    logger.info(
        "Building optimization config (tasks=%s, budget_fraction=%.4f)",
        num_tasks,
        blue_team_budget_fraction,
    )
    base_config = build_optimization_config(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        slack_honeypots=slack_honeypots,
        blue_team_replace_fnr=0.0,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
        disable_trusted_monitor=defaults.get("disable_trusted_monitor", False),
        disable_untrusted_monitor=defaults.get("disable_untrusted_monitor", False),
    )

    logger.info(
        "Running parameter sensitivity analysis with %d variation points per parameter",
        args.variation_points,
    )
    sensitivity_results = run_parameter_sensitivity_analysis(
        base_config=base_config,
        variation_points=args.variation_points,
        max_iterations=100,
    )

    logger.info("Rendering sensitivity plots to %s", destination)
    plot_parameter_sensitivity(
        sensitivity_results=sensitivity_results,
        destination_folder=destination,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
