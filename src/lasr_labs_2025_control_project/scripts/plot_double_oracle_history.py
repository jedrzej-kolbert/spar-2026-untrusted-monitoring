"""Compare double-oracle guesses, candidates, and bounds at one usefulness point.

To capture Powell inputs and outputs, rerun only the requested usefulness point:
    uv run python src/lasr_labs_2025_control_project/scripts/plot_double_oracle_history.py \
        --config experiments/self_rec_attack_500/configs/monitor_sft.yaml \
        --usefulness 0.95 \
        --output-prefix experiments/self_rec_attack_500/results/optimizer_diagnostics/sft_u0p95_guess_comparison

To replot the captured run without rerunning optimization:
    uv run python src/lasr_labs_2025_control_project/scripts/plot_double_oracle_history.py \
        --analysis-data experiments/self_rec_attack_500/results/optimizer_diagnostics/sft_u0p95_guess_comparison.json \
        --usefulness 0.95 \
        --output-prefix /tmp/sft_u0p95_replot

Neither mode makes model calls or changes the source .eval files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_analysis(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    if "sweep" not in data and "point" in data:
        return {"sweep": [data["point"]], "source": data.get("source", str(path))}
    return data


def run_from_evals(
    config_path: Path,
    usefulness: float,
    max_iterations: int,
    best_response_method: str = "Powell",
) -> dict:
    """Rebuild the same game as plot_data and optimize only one usefulness point."""
    from lasr_labs_2025_control_project.scripts.plot_data import (
        build_jsonl_dataframe_from_eval_log,
    )
    from lasr_labs_2025_control_project.utils.config_loader import (
        RunLabel,
        load_and_normalize_config,
    )
    from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
        build_optimization_config,
        optimize_parameters,
    )

    normalized = load_and_normalize_config(config_path, config_type="monitor")
    runs = {run.label: run for run in normalized.runs}
    attacks_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.ATTACKS].save_path)
    honeypots_df = build_jsonl_dataframe_from_eval_log(
        runs[RunLabel.HONEYPOTS].save_path
    )
    benign_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.HONEST].save_path)
    base_config = build_optimization_config(
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        slack_honeypots=5.0,
        blue_team_replace_fnr=1 - usefulness,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
        disable_trusted_monitor=normalized.defaults.get(
            "disable_trusted_monitor", False
        ),
        disable_untrusted_monitor=normalized.defaults.get(
            "disable_untrusted_monitor", False
        ),
    )
    result = optimize_parameters(
        base_config=base_config,
        max_iterations=max_iterations,
        replace_fnr_values=[1 - usefulness],
        best_response_method=best_response_method,
    )
    return {"sweep": result.results}


def select_point(analysis: dict, usefulness: float) -> dict:
    matches = [
        point
        for point in analysis["sweep"]
        if math.isclose(float(point["usefulness"]), usefulness, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        available = ", ".join(str(point["usefulness"]) for point in analysis["sweep"])
        raise ValueError(
            f"Expected one point at usefulness {usefulness}, found {len(matches)}. "
            f"Available: {available}"
        )
    if not matches[0].get("history"):
        raise ValueError(f"No optimizer history at usefulness {usefulness}")
    return matches[0]


def history_rows(point: dict) -> list[dict]:
    rows = []
    for entry in point["history"]:
        row = {
            "iteration": int(entry["iteration"]),
            "lower_bound": float(entry["lower_bound"]),
            "red_lower": float(entry.get("red_lower", entry["lower_bound"])),
            "upper_bound": float(entry["upper_bound"]),
        }
        for name in ("blue_guess", "candidate_blue", "red_guess", "candidate_red"):
            for parameter, value in entry.get(name, {}).items():
                row[f"{name}.{parameter}"] = float(value)
        for name in ("red_guess_score", "candidate_red_score"):
            if name in entry:
                row[name] = entry[name]
        rows.append(row)
    return rows


def save_plot(rows: list[dict], point: dict, path: Path) -> None:
    """Plot only the guesses, candidates, their red scores, and the bounds."""
    required = (
        "red_guess_score",
        "candidate_red_score",
        "red_guess.red_team_roc_threshold",
        "candidate_red.red_team_roc_threshold",
        "red_guess.red_team_p_attack",
        "candidate_red.red_team_p_attack",
        "blue_guess.blue_team_honeypot_fraction",
        "candidate_blue.blue_team_honeypot_fraction",
    )
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(
            f"History lacks {missing}; rerun from .eval files using --config "
            "to record per-iteration guesses and candidates."
        )

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")

    def draw(axis, selected, curves, ylabel, *, log=False):
        for key, label, color in curves:
            axis.plot(
                [row["iteration"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label,
                color=color,
            )
        if log:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("iteration")
        axis.set_xticks([row["iteration"] for row in selected])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    red_scores = (
        ("red_guess_score", "red_guess_score", "#a77a16"),
        ("red_lower", "red_lower", "#c43c39"),
    )
    draw(axes[0, 0], rows, red_scores, "red_objective")
    axes[0, 0].set_title("red_guess_score vs red_lower")
    draw(axes[0, 1], rows[-6:], red_scores, "red_objective")
    axes[0, 1].set_title("red_guess_score vs red_lower (last six iterations)")

    draw(
        axes[1, 0],
        rows,
        (
            ("upper_bound", "upper_bound", "#2767af"),
            ("lower_bound", "lower_bound", "#c43c39"),
        ),
        "upper_bound / lower_bound",
    )
    draw(
        axes[1, 1],
        rows,
        (
            (
                "red_guess.red_team_roc_threshold",
                "red_guess",
                "#a77a16",
            ),
            (
                "candidate_red.red_team_roc_threshold",
                "candidate_red",
                "#c43c39",
            ),
        ),
        "red_team_roc_threshold",
    )
    draw(
        axes[2, 0],
        rows,
        (
            (
                "red_guess.red_team_p_attack",
                "red_guess",
                "#a77a16",
            ),
            (
                "candidate_red.red_team_p_attack",
                "candidate_red",
                "#c43c39",
            ),
        ),
        "red_team_p_attack",
        log=True,
    )
    draw(
        axes[2, 1],
        rows,
        (
            (
                "blue_guess.blue_team_honeypot_fraction",
                "blue_guess",
                "#a77a16",
            ),
            (
                "candidate_blue.blue_team_honeypot_fraction",
                "candidate_blue",
                "#2767af",
            ),
        ),
        "blue_team_honeypot_fraction",
    )
    fig.suptitle(f"usefulness = {point['usefulness']:.2f}")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis-data", type=Path)
    source.add_argument(
        "--config", type=Path, help="Rerun this point from completed monitor .eval files"
    )
    parser.add_argument("--usefulness", type=float, default=0.95)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--best-response-method", choices=("Powell", "COBYQA"), default="Powell"
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.usefulness <= 1:
        parser.error("--usefulness must be between 0 and 1")

    if args.config:
        analysis = run_from_evals(
            args.config,
            args.usefulness,
            args.max_iterations,
            args.best_response_method,
        )
        source_path = args.config
    else:
        analysis = load_analysis(args.analysis_data)
        source_path = analysis.get("source", str(args.analysis_data))
    point = select_point(analysis, args.usefulness)
    rows = history_rows(point)
    if "red_guess_score" not in rows[0]:
        parser.error(
            "The saved history predates guess logging; use --config to rerun "
            "this usefulness point from its completed .eval files."
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(f"{args.output_prefix}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(f"{args.output_prefix}.json")
    json_path.write_text(
        json.dumps(
            {"source": str(source_path), "point": point, "diagnostic_rows": rows},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    svg_path = Path(f"{args.output_prefix}.svg")
    save_plot(rows, point, svg_path)
    final = rows[-1]
    print(f"Saved {len(rows)} iterations to {csv_path}, {json_path}, and {svg_path}")
    print(
        f"Final: lower={final['lower_bound']:.9f}, "
        f"upper={final['upper_bound']:.9f}"
    )
    if "candidate_red_score" in final:
        print(
            f"Final red search: red_guess_score={final['red_guess_score']:.9f}, "
            f"red_lower={final['red_lower']:.9f} "
            f"(candidate_red_score={final['candidate_red_score']:.9f})"
        )


if __name__ == "__main__":
    main()
