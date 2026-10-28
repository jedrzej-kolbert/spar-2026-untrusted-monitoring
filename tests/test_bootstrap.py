from types import SimpleNamespace

import numpy as np
import pandas as pd

from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    bootstrap as bootstrap_module,
)


def test_replicate_returns_only_band_data(monkeypatch):
    monkeypatch.setattr(
        bootstrap_module,
        "build_optimization_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "optimize_parameters",
        lambda **kwargs: SimpleNamespace(
            results=[
                {"safety": np.float64(0.4), "usefulness": 0.8},
                {"safety": np.nan, "usefulness": 0.5},
            ]
        ),
    )

    frame = pd.DataFrame({"score": [1]})
    payload = bootstrap_module.run_single_replicate(
        3,
        condition_seed=7,
        attacks_df=frame,
        honeypots_df=frame,
        benign_df=frame,
        disable_trusted_monitor=False,
        disable_untrusted_monitor=False,
    )

    assert payload == {
        "bootstrap_replicate": 3,
        "sweep": [
            {"safety": 0.4, "usefulness": 0.8},
            {"safety": None, "usefulness": 0.5},
        ],
    }
    band = bootstrap_module.bootstrap_band(
        [
            payload,
            {
                "sweep": [
                    {"safety": 0.6, "usefulness": 0.8},
                    {"safety": 0.2, "usefulness": 0.5},
                ]
            },
        ],
        ci=0.8,
    )
    assert band["safety_median"] == [0.5, 0.2]
