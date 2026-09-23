"""Regression tests for double-oracle local best-response failures."""

from importlib import import_module

import numpy as np
from scipy.optimize import OptimizeResult


def test_known_strategies_keep_bounds_ordered_when_local_search_misses(monkeypatch):
    module = import_module(
        "lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.double_oracle"
    )

    def initial_strategies(names, _param_data, _rng):
        assert len(names) == 1
        return [{names[0]: 0.0}, {names[0]: 1.0}]

    def stuck_local_search(_objective, x0, **_kwargs):
        return OptimizeResult(x=x0.copy(), success=True, nfev=1, message="stuck")

    monkeypatch.setattr(module, "_generate_initial_strategies", initial_strategies)
    monkeypatch.setattr(module, "minimize", stuck_local_search)

    def payoff(params):
        value = params["blue_choice"] - params["red_choice"]
        return np.asarray(value), {"blue_win": np.asarray(value)}

    result = module.double_oracle(
        {
            "blue_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
            "red_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
        },
        payoff,
        max_iterations=3,
    )

    iteration = result["history"][-1]
    assert result["red_best_response"]["red_choice"] == 1.0
    assert iteration["lower_bound"] <= iteration["upper_bound"]
    assert result["converged"] is False
    assert set(iteration) == {
        "iteration",
        "num_blue_strategies",
        "num_red_strategies",
        "equilibrium_value",
        "upper_bound",
        "lower_bound",
        "gap",
        "blue_probs",
        "red_probs",
    }


def test_cobyqa_searches_log_parameter_in_log_coordinates(monkeypatch):
    module = import_module(
        "lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.double_oracle"
    )

    def initial_strategies(names, _param_data, _rng):
        if names == ["blue_choice"]:
            return [{"blue_choice": 0.0}, {"blue_choice": 1.0}]
        return [{"red_attack": 1e-6}, {"red_attack": 1e-1}]

    monkeypatch.setattr(module, "_generate_initial_strategies", initial_strategies)

    def payoff(params):
        value = params["blue_choice"] + (
            np.log10(params["red_attack"]) + 3.0
        ) ** 2
        return np.asarray(value), {"blue_win": np.asarray(value)}

    result = module.double_oracle(
        {
            "blue_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
            "red_attack": {"min": 1e-6, "max": 1e-1, "scale": "log"},
        },
        payoff,
        max_iterations=1,
        best_response_method="COBYQA",
    )

    assert np.isclose(result["red_best_response"]["red_attack"], 1e-3, rtol=0.05)
