"""Minification-baseline helpers are tested without live model calls."""

import pytest

from experiments.grouped_apps_smoke.minification_baseline import (
    metrics,
    minify,
    opposite_marker,
)


def test_comment_only_bit_is_erased_for_both_payloads() -> None:
    code = "# Secret bit: 0\nprint('hello')"
    opposite = opposite_marker(code, 0)
    assert "Secret bit: 1" in opposite
    assert minify(code) == minify(opposite)


def test_unused_assignment_value_survives_minification() -> None:
    code = "secret_bit = 1  # Secret bit: 1\nprint('hello')"
    opposite = opposite_marker(code, 1)
    assert "secret_bit = 0" in opposite
    assert minify(code) != minify(opposite)
    assert "secret_bit" not in minify(code)


def test_counterfactual_does_not_modify_a_used_variable() -> None:
    with pytest.raises(ValueError, match="used by the program"):
        opposite_marker("secret_bit = 0\nprint(secret_bit)", 0)


def test_missing_markers_stop_the_control() -> None:
    with pytest.raises(ValueError, match="no known marker"):
        opposite_marker("print('hello')", 0)


def test_missing_scores_remain_in_denominators() -> None:
    rows = [
        {"payload": 0, "probabilities": (0.9, 0.1), "execution": {"passed": True}},
        {"payload": 1, "probabilities": None, "execution": {"passed": True}},
    ]
    result = metrics(rows)
    assert result["records"] == 2
    assert result["decoding_accuracy"] == 0.5
    assert result["correct_bit_probability_mean"] is None
    assert result["missing_scores"] == 1
