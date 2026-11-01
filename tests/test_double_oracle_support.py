import importlib
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

oracle = importlib.import_module(
    "lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.double_oracle"
)


@pytest.mark.parametrize(
    "weights",
    [
        [0.0, 1e-14, 1.0 - 1e-14],
        [1.0 / 3] * 3,
        [-0.0, 0.0, 1.0],
    ],
)
def test_best_responses_skip_only_exact_zero_weights(
    monkeypatch: pytest.MonkeyPatch, weights: list[float]
):
    support = [0.1, 0.2, 0.3]
    calls = []

    def payoff(params: dict[str, np.ndarray]):
        blue = float(params["blue_choice"])
        red = float(params["red_choice"])
        calls.append((blue, red))
        value = np.asarray((blue + red) / 2)
        return value, {"blue_win": value}

    monkeypatch.setattr(
        oracle,
        "_generate_initial_strategies",
        lambda names, *_: [{names[0]: value} for value in support],
    )

    def solve_lp(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Zero-weight strategies must remain in the restricted game.
        assert matrix.shape == (3, 3)
        assert set(calls) == {(blue, red) for blue in support for red in support}
        return np.array(weights), np.array(weights), 0.2

    monkeypatch.setattr(oracle, "_solve_zero_sum_game", solve_lp)
    responses = []

    def best_response(objective: Callable[[np.ndarray], float], **kwargs: object):
        is_blue = not responses
        candidate = 0.73 if is_blue else 0.83
        before = len(calls)
        actual = objective(np.array([candidate]))
        active = [value for prob, value in zip(weights, support) if prob != 0]
        expected_calls = (
            [(candidate, value) for value in active]
            if is_blue
            else [(value, candidate) for value in active]
        )
        assert calls[before:] == expected_calls
        expected_value = sum(
            prob * (candidate + value) / 2 for prob, value in zip(weights, support)
        )
        assert actual == pytest.approx(-expected_value if is_blue else expected_value)
        responses.append(is_blue)
        return SimpleNamespace(x=np.array([candidate]))

    monkeypatch.setattr(oracle, "differential_evolution", best_response)
    oracle.double_oracle(
        {
            "blue_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
            "red_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
        },
        payoff,
        max_iterations=1,
        seed=7,
    )
    assert responses == [True, False]
