"""One-at-a-time sensitivity of a saved condition to the fixed game parameters.

Re-solves the condition from the monitor outputs, game parameters and variant
its analyse step recorded, then sweeps each game parameter:

    python -m lasr_labs_2025_control_project.scripts.plot_parameter_sensitivity \
        experiments/apps/game/<condition>-<hash> [--variant um_only]
"""

import argparse
import logging
from pathlib import Path

from lasr_labs_2025_control_project.scripts.plot_data import (
    VARIANT_FLAGS,
    load_condition,
    load_saved_condition,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    PAPER_GAME_PARAMETERS,
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
    parser.add_argument("folder", type=Path, help="an analyse output folder")
    parser.add_argument(
        "--variant",
        default=None,
        choices=sorted(VARIANT_FLAGS),
        help="analysis variant (default: the folder's primary one)",
    )
    parser.add_argument(
        "--variation-points",
        type=int,
        default=21,
        help="Number of sweep points per parameter (>=2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <folder>/parameter_sensitivity/<variant>",
    )
    args = parser.parse_args()

    record = load_saved_condition(args.folder, args.variant)["pipeline"]
    variant = record["variant"]
    destination = args.output_dir or args.folder / "parameter_sensitivity" / variant
    frames = load_condition({k: Path(v) for k, v in record["inputs"].items()})
    attacks_df, honeypots_df, benign_df = (
        frames["attacks"],
        frames["honeypots"],
        frames["honest"],
    )
    disable_trusted_monitor, disable_untrusted_monitor = VARIANT_FLAGS[variant]
    game = {**PAPER_GAME_PARAMETERS, **record["solve"]["game"]}
    num_tasks = int(game["num_tasks"])
    blue_team_budget_fraction = float(game["blue_team_budget_fraction"])
    slack_honeypots = float(game["slack_honeypots"])

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
        human_audit_success_rate=float(game.get("human_audit_success_rate", 1.0)),
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )

    logger.info(
        "Running parameter sensitivity analysis with %d variation points per parameter",
        args.variation_points,
    )
    sensitivity_results = run_parameter_sensitivity_analysis(
        base_config=base_config,
        variation_points=args.variation_points,
        max_iterations=record["solve"]["max_iterations"],
    )

    logger.info("Rendering sensitivity plots to %s", destination)
    plot_parameter_sensitivity(
        sensitivity_results=sensitivity_results,
        destination_folder=destination,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
