"""Analyse-stage helpers: which monitor variants run, how the game and analysis
runs are named, how ROC curves are logged, and the figures' model caption."""

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
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (  # noqa: E402
    bootstrap_band,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (  # noqa: E402
    visualization as viz,
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
