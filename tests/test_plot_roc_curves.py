"""Tests for the untrusted_suspiciousness mode of plot_roc_curves."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (  # noqa: E402
    plot_roc_curves,
)


def _result(metadata: dict[str, Any] | None) -> SimpleNamespace:
    config = SimpleNamespace(
        monitor_roc_fn=lambda *_: None,
        monitor_roc_fn_metadata=metadata,
        red_team_positive_rates=None,
    )
    return SimpleNamespace(config=config)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"untrusted_suspiciousness_scores": {"positive": [], "negative": [1.0]}},
        {
            "untrusted_suspiciousness_scores": {
                "positive": [float("nan")],
                "negative": [1.0],
            }
        },
    ],
    ids=["no-metadata", "no-scores", "empty-class", "all-nonfinite-class"],
)
def test_skipped_runs_still_save_plot(
    tmp_path: Path, metadata: dict[str, Any] | None
) -> None:
    save_path = tmp_path / "roc.svg"
    plot_roc_curves(
        {"run": _result(metadata)},  # type: ignore[dict-item]
        save_path=str(save_path),
        plot_mode="untrusted_suspiciousness",
    )
    assert save_path.exists()


def test_valid_scores_save_plot(tmp_path: Path) -> None:
    scores = {"positive": [8.0, 9.0, 3.0], "negative": [1.0, 2.0, 7.0, 0.0]}
    save_path = tmp_path / "roc.svg"
    plot_roc_curves(
        {"run": _result({"untrusted_suspiciousness_scores": scores})},  # type: ignore[dict-item]
        save_path=str(save_path),
        plot_mode="untrusted_suspiciousness",
    )
    assert save_path.exists()
