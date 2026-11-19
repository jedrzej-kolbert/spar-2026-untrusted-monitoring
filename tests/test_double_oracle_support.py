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


def test_initial_guess_on_a_bound_is_accepted_by_differential_evolution():
    # The i2-t05-lt-s2 minified+paraphrased game: red's first guess sits on this lower
    # bound, which scipy rescales to -1.1e-16 and rejects as outside the bounds.
    from scipy.optimize import differential_evolution

    bounds = [(-14.866125101756724, 3.9911250343067226), (1e-06, 0.1)]
    guess = np.array([b[0] for b in bounds])
    with pytest.raises(ValueError, match="outside the specified bounds"):
        differential_evolution(lambda x: 0.0, bounds=bounds, x0=guess, maxiter=1)
    inside = oracle._strictly_inside(guess, bounds)
    differential_evolution(lambda x: 0.0, bounds=bounds, x0=inside, maxiter=1)
    assert np.allclose(inside, guess, rtol=0, atol=1e-10)
