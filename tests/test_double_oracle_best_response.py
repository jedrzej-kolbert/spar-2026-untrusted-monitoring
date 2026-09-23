"""Regression tests for double-oracle local best-response failures."""

from importlib import import_module

import numpy as np
from scipy.optimize import OptimizeResult

from lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.prima_bobyqa import (
    minimize_prima_bobyqa,
)


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
    assert iteration["raw_blue_score"] < iteration["best_existing_blue_score"]
    assert iteration["raw_red_score"] > iteration["best_existing_red_score"]
    assert iteration["used_existing_blue"] is True
    assert iteration["used_existing_red"] is True
    assert iteration["lower_bound"] <= iteration["upper_bound"]
    assert result["converged"] is False
    assert iteration["termination_reason"] == "unverified_best_response"


def test_prima_bridge_evaluates_log_parameters_in_original_coordinates(tmp_path):
    runner = tmp_path / "runner"
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('EVAL 0.5 0.5', flush=True)\n"
        "score = float(sys.stdin.readline())\n"
        "print(f'RESULT 0 0.5 0.5 {score:.17g} 1', flush=True)\n"
    )
    runner.chmod(0o755)
    seen = []

    def objective(values):
        seen.append(values.copy())
        return float(values[0] + values[1])

    result = minimize_prima_bobyqa(
        objective,
        np.array([0.2, 1e-4]),
        [(0.0, 1.0), (1e-6, 1e-2)],
        ["linear", "log"],
        runner=runner,
    )

    assert result.success
    assert result.nfev == 1
    assert np.allclose(seen[0], [0.5, 1e-4])
    assert np.allclose(result.x, seen[0])
    assert np.isclose(result.fun, objective(result.x))
