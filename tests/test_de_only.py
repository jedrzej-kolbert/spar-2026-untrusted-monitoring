import numpy as np

from lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers.double_oracle import (
    double_oracle,
)


def test_direct_de_finds_quadratic_response():
    def payoff(params):
        value = params["blue_choice"] + (params["red_choice"] - 0.2) ** 2
        return np.asarray(value), {"blue_win": np.asarray(value)}

    result = double_oracle(
        {
            "blue_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
            "red_choice": {"min": 0.0, "max": 1.0, "scale": "linear"},
        },
        payoff,
        max_iterations=1,
    )
    assert abs(result["red_best_response"]["red_choice"] - 0.2) < 0.05
