"""Analyse-stage helpers: which monitor variants run, and how the run is named."""

from types import SimpleNamespace

from lasr_labs_2025_control_project.scripts.plot_data import (
    analysis_name,
    analysis_variants,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import bootstrap_band


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
