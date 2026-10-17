"""Nonparametric bootstrap of the full analysis for one condition.

Each replicate resamples the attack / honeypot / benign dataframes with
replacement (rows, i.e. whole samples -- preserving the joint dependence
structure the stats model exists to capture), rebuilds the joint ROC model,
re-solves the game, and returns the analysis payload. Used by the analyse
stage (``scripts/plot_data.py``) and the standalone resumable runner
(``scripts/bootstrap.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, cast

import numpy as np
import pandas as pd

from .optimization import (
    PAPER_GAME_PARAMETERS,
    build_optimization_config,
    optimize_parameters,
)
from .results_io import build_analysis_payload

# Per-replicate payload size knobs: no optimizer history, coarser surface grid
# than the point estimate (31x21 instead of 61x41).
SURFACE_T_POINTS = 31
SURFACE_F_POINTS = 21


def run_single_replicate(
    replicate: int,
    *,
    condition_seed: int,
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    disable_trusted_monitor: bool,
    disable_untrusted_monitor: bool,
) -> Dict[str, Any]:
    """Resample, re-fit, re-solve; return the analysis payload for one replicate.

    A failed replicate (e.g. a degenerate resample making the LP solver choke,
    observed as "HiGHS Status 4: Solve error") is recorded as an error row
    rather than killing the whole run; downstream analysis filters on the
    presence of "sweep" and can report the failure rate from "error".
    """
    try:
        seeds = np.random.SeedSequence([condition_seed, replicate]).generate_state(3)
        config = build_optimization_config(
            num_tasks=PAPER_GAME_PARAMETERS["num_tasks"],
            blue_team_budget_fraction=PAPER_GAME_PARAMETERS[
                "blue_team_budget_fraction"
            ],
            slack_honeypots=PAPER_GAME_PARAMETERS["slack_honeypots"],
            blue_team_replace_fnr=0.0,
            attacks_df=attacks_df.sample(
                len(attacks_df), replace=True, random_state=int(seeds[0])
            ),
            honeypots_df=honeypots_df.sample(
                len(honeypots_df), replace=True, random_state=int(seeds[1])
            ),
            benign_df=benign_df.sample(
                len(benign_df), replace=True, random_state=int(seeds[2])
            ),
            disable_trusted_monitor=disable_trusted_monitor,
            disable_untrusted_monitor=disable_untrusted_monitor,
        )
        result = optimize_parameters(base_config=config, max_iterations=100)
        payload = build_analysis_payload(
            result,
            include_history=False,
            joint_t_points=SURFACE_T_POINTS,
            joint_f_points=SURFACE_F_POINTS,
        )
    except Exception as exc:
        return {
            "bootstrap_replicate": replicate,
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload["bootstrap_replicate"] = replicate
    return payload


def run_replicates(
    replicates: Iterable[int],
    *,
    n_jobs: int = -1,
    **kwargs: Any,
) -> list[Dict[str, Any]]:
    """Run the given replicate indices in parallel (kwargs as run_single_replicate)."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_replicate)(r, **kwargs) for r in replicates
    )
    return cast(list[Dict[str, Any]], list(results))


def bootstrap_band(
    payloads: Sequence[Dict[str, Any]], ci: float
) -> Dict[str, list[float]]:
    """Percentile band of safety per sweep point across successful replicates.

    The usefulness axis is taken from the first successful replicate, matching
    the published band figures.
    """
    sweeps = [p["sweep"] for p in payloads if "sweep" in p]
    if not sweeps:
        return {}
    usefulness = [float(entry["usefulness"]) for entry in sweeps[0]]
    safety = np.array(
        [[np.nan if e["safety"] is None else e["safety"] for e in s] for s in sweeps],
        dtype=float,
    )
    tail = 100.0 * (1.0 - ci) / 2.0
    lo, med, hi = np.nanpercentile(safety, [tail, 50.0, 100.0 - tail], axis=0)
    return {
        "usefulness": usefulness,
        "safety_median": med.tolist(),
        "safety_lo": lo.tolist(),
        "safety_hi": hi.tolist(),
    }
