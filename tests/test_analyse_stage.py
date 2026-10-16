"""Analyse-stage helpers: which monitor variants run, how the run is named,
and how equilibrium strategies are logged and plotted."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

from lasr_labs_2025_control_project.scripts.plot_data import (  # noqa: E402
    VARIANT_FLAGS,
    log_strategies,
)
from lasr_labs_2025_control_project.tracking import names  # noqa: E402
from lasr_labs_2025_control_project.utils.numerical_pipeline import (  # noqa: E402
    OptimizationResult,
    bootstrap_band,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (  # noqa: E402
    construct_parameter_spaces,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (  # noqa: E402
    plot_strategies,
    strategy_parameters,
)


def test_variants_disable_the_other_monitor():
    # (disable_trusted_monitor, disable_untrusted_monitor)
    assert VARIANT_FLAGS == {
        "all_monitors": (False, False),
        "um_only": (True, False),
        "tm_only": (False, True),
    }


def test_condition_name():
    assert (
        names.condition_name("apps", "causal", "minified", "base")
        == "apps.causal.minified.base"
    )


def test_bootstrap_band_skips_failed_replicates():
    def payload(safeties):
        return {
            "sweep": [
                {"usefulness": u, "safety": s} for u, s in zip((0.5, 1.0), safeties)
            ]
        }

    replicates = [payload((0.2, 0.4)), payload((0.4, 0.8)), {"error": "boom"}]
    band = bootstrap_band(replicates, ci=0.0)
    assert band["usefulness"] == [0.5, 1.0]
    assert band["safety_median"] == [0.30000000000000004, 0.6000000000000001]
    assert bootstrap_band([{"error": "boom"}], ci=0.8) == {}


# A future strategy variable, to check it needs no code changes to show up.
NEW_PARAM = "blue_team_inverse_honeypot_fraction"


def _sweep_entry(
    usefulness: float, honeypot_fraction: float, converged: bool = True
) -> dict[str, Any]:
    blue = {"blue_team_honeypot_fraction": honeypot_fraction, NEW_PARAM: 0.1}
    return {
        "usefulness": usefulness,
        "safety": 0.5,
        "converged": converged,
        "best_parameters": {
            **blue,
            "red_team_roc_threshold": -1.0,
            "red_team_p_attack": 1e-3,
        },
        "blue_mixed_strategy": [
            (blue, 0.7),
            ({**blue, "blue_team_honeypot_fraction": 0.9}, 0.3),
        ],
        "red_mixed_strategy": [({}, 0.5), ({}, 0.5)],
    }


class _RecordingTracker:
    def __init__(self) -> None:
        self.curves: dict[str, str] = {}
        self.rows: list[dict[str, Any]] = []

    def define_curve(self, y: str, x: str) -> None:
        self.curves[y] = x

    def log(self, data: dict[str, Any]) -> None:
        self.rows.append(dict(data))


def test_log_strategies_plots_every_variable_against_usefulness() -> None:
    tracker = _RecordingTracker()
    payload = {"sweep": [_sweep_entry(1.0, 0.2), _sweep_entry(0.5, 0.4)]}
    log_strategies(tracker, "um_only", payload)

    x = "strategy/um_only/usefulness"
    assert set(tracker.curves.values()) == {x}
    assert {
        "strategy/um_only/blue_team_honeypot_fraction",
        f"strategy/um_only/{NEW_PARAM}",
        "strategy/um_only/red_team_roc_threshold",
        "strategy/um_only/red_team_p_attack",
        "strategy/um_only/blue_top_probability",
        "strategy/um_only/blue_support_size",
        "strategy/um_only/red_nash_support_size",
    } == set(tracker.curves)
    assert [row[x] for row in tracker.rows] == [0.5, 1.0]
    assert tracker.rows[0]["strategy/um_only/blue_team_honeypot_fraction"] == 0.4
    assert tracker.rows[0]["strategy/um_only/blue_top_probability"] == 0.7
    assert tracker.rows[0]["strategy/um_only/red_nash_support_size"] == 2


def test_strategy_figure_includes_new_variables(tmp_path: Path) -> None:
    spaces = construct_parameter_spaces(-3.0, 1.0)
    spaces[NEW_PARAM] = {"min": 0.0, "max": 1.0, "scale": "linear"}
    result = OptimizationResult(
        config=SimpleNamespace(parameter_spaces=spaces),
        results=[_sweep_entry(1.0, 0.2, converged=False), _sweep_entry(0.5, 0.4)],
    )
    assert strategy_parameters(result) == [
        "blue_team_honeypot_fraction",
        NEW_PARAM,
        "red_team_roc_threshold",
        "red_team_p_attack",
    ]

    path = tmp_path / "strategies.png"
    replicates = [{"sweep": [_sweep_entry(1.0, 0.3)]}, {"error": "boom"}]
    plot_strategies({"cond": result}, str(path), bootstrap={"cond": replicates})
    assert path.stat().st_size > 0
