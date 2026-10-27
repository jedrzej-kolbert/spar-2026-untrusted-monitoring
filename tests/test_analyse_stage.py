"""Analyse-stage helpers: which monitor variants run, how the game and analysis
runs are named, how ROC curves and equilibrium strategies are logged, and the
figures (strategies, model caption)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from lasr_labs_2025_control_project.scripts.plot_data import (  # noqa: E402
    Game,
    analysis_name,
    analysis_run_name,
    analysis_variants,
    log_payload_overview,
    log_strategies,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (  # noqa: E402
    OptimizationResult,
    bootstrap_band,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (  # noqa: E402
    visualization as viz,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (  # noqa: E402
    construct_parameter_spaces,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (  # noqa: E402
    plot_strategies,
    strategy_parameters,
)


def test_variants_follow_config_and_flags():
    assert [v[0] for v in analysis_variants({})] == ["all_monitors", "um_only"]
    assert analysis_variants({"disable_trusted_monitor": True}) == [
        ("um_only", True, False)
    ]
    assert analysis_variants({"disable_untrusted_monitor": True}) == [
        ("tm_only", False, True)
    ]


def test_analysis_name_prefers_condition():
    normalized = SimpleNamespace(experiment_name="Causal Minified")
    condition = {
        "setting": "apps",
        "collusion_strategy": "causal",
        "transform": "minified",
        "monitor_arm": "base",
    }
    assert (
        analysis_name(normalized, {"wandb": {"condition": condition}})
        == "apps.causal.minified.base"
    )
    assert analysis_name(normalized, {}) == "causal_minified"


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


def _game(name: str) -> Game:
    return Game(name, SimpleNamespace(), {})  # type: ignore[arg-type]


def test_analysis_run_is_named_after_its_conditions():
    games = {
        n: _game(n) for n in ("apps.self_rec.none.base", "apps.self_rec.minified.sft")
    }
    assert analysis_run_name(games, None) == "analysis.apps.self_rec"
    assert analysis_run_name(games, "no_mitigation") == "analysis.no_mitigation"
    assert analysis_run_name({"a": _game("apps.causal.none.base")}, None) == (
        "analysis.apps.causal.none.base"
    )


class _RocTracker:
    def __init__(self) -> None:
        self.rocs: dict[str, Any] = {}
        self.summaries: list[Any] = []

    def log_roc(self, key: str, curves: Any, title: str) -> None:
        self.rocs[key] = curves

    def summary(self, data: Any) -> None:
        self.summaries.append(data)


def test_roc_curves_are_logged_with_every_point():
    # Two points share an fpr: a vertical step that metric panels would drop.
    payload = {
        "roc_curves": {
            "untrusted_collusion": {
                "fpr": [0.0, 0.5, 0.5, 1.0],
                "tpr": [0.0, 0.25, 0.75, 1.0],
            }
        }
    }
    tracker = _RocTracker()
    log_payload_overview(tracker, payload)  # type: ignore[arg-type]
    points = tracker.rocs["roc/untrusted_collusion"]["untrusted_collusion"]
    assert points == [(0.0, 0.0), (0.5, 0.25), (0.5, 0.75), (1.0, 1.0)]


def _result(models: dict[str, str]) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(monitor_roc_fn_metadata={"models": models})
    )


def test_models_caption_names_u_h_t_once_or_per_condition():
    same = {
        "U": "deepseek-ai/DeepSeek-V3.1",
        "H": "Qwen/Qwen3.8-27B",
        "T": "tinker/openai/gpt-oss-20b",
    }
    caption = viz.models_caption({"a": _result(same), "b": _result(same)})
    assert (
        caption
        == "U: deepseek-ai/DeepSeek-V3.1 · H: Qwen/Qwen3.8-27B · T: openai/gpt-oss-20b"
    )
    other = {**same, "T": "openai/gpt-4o-mini"}
    lines = viz.models_caption({"a": _result(same), "b": _result(other)})
    assert lines is not None and lines.count("\n") == 1 and lines.startswith("a: U:")
    assert viz.models_caption({"a": SimpleNamespace(config=None)}) is None


def test_caption_sits_under_the_legend(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="curve")
    legend = ax.legend(loc="upper left")
    token = viz._CAPTION.set("U: a · H: b · T: c")
    try:
        viz._save_figure(fig, tmp_path / "f.png")
    finally:
        viz._CAPTION.reset(token)
    (text,) = [t for t in fig.texts if t.get_text().startswith("U:")]
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    box = fig.transFigure.inverted().transform_bbox(legend.get_window_extent(renderer))
    x, y = text.get_position()
    assert y < box.y0 and abs(x - box.x0) < 1e-6 and text.get_fontsize() < 10
    assert (tmp_path / "f.png").exists()
    plt.close(fig)
