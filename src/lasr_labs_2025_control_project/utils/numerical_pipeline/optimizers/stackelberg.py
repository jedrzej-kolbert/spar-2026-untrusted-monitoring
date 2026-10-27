"""Stackelberg equilibrium optimizer for blue/red parameter search.

This module mirrors the ``double_oracle`` public interface but solves the
leader-follower (blue-leads, red-follows) variant. The blue team commits to a
pure strategy and anticipates the red team's best response. The implementation
combines a coarse grid search with warm-started local optimization so it can be
used as a drop-in replacement without a large runtime penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

Strategy = Dict[str, float]
ParamConfig = Dict[str, Dict[str, Union[str, float]]]
ObjectiveFn = Callable[
    [Dict[str, np.ndarray]], Tuple[np.ndarray, Dict[str, np.ndarray]]
]


def _midpoint(min_val: float, max_val: float, scale: str) -> float:
    scale = scale.lower()
    if scale == "log" and min_val > 0 and max_val > 0:
        return float(np.exp((np.log(min_val) + np.log(max_val)) / 2.0))
    return float((min_val + max_val) / 2.0)


def _clip_value(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return float(value)


def _strategy_exists(
    strategies: List[Strategy], candidate: Mapping[str, float], tol: float = 1e-8
) -> bool:
    candidate_keys = tuple(sorted(candidate.keys()))
    for strategy in strategies:
        if tuple(sorted(strategy.keys())) != candidate_keys:
            continue
        if all(
            abs(float(strategy[name]) - float(candidate[name])) <= tol
            for name in candidate_keys
        ):
            return True
    return False


def _generate_initial_strategies(
    param_names: List[str],
    param_data: Mapping[str, Mapping[str, Union[str, float]]],
    rng: np.random.Generator,
    max_strategies: int = 20,
) -> List[Strategy]:
    if not param_names:
        return [dict()]

    strategies: List[Strategy] = []
    candidate_values: List[List[float]] = []

    for name in param_names:
        config = param_data[name]
        min_val = float(config["min"])
        max_val = float(config["max"])
        scale = str(config.get("scale", "linear"))

        values: List[float] = [min_val, max_val, _midpoint(min_val, max_val, scale)]

        for _ in range(2):
            if scale.lower() == "log" and min_val > 0 and max_val > 0:
                sampled = float(np.exp(rng.uniform(np.log(min_val), np.log(max_val))))
            else:
                sampled = float(rng.uniform(min_val, max_val))
            values.append(sampled)

        deduped: List[float] = []
        for value in values:
            clipped = _clip_value(value, min_val, max_val)
            if not any(abs(clipped - existing) <= 1e-8 for existing in deduped):
                deduped.append(clipped)

        candidate_values.append(deduped)

    for combo in product(*candidate_values):
        strategy = {name: float(val) for name, val in zip(param_names, combo)}
        if not _strategy_exists(strategies, strategy):
            strategies.append(strategy)
        if len(strategies) >= max_strategies:
            break

    attempts = 0
    while len(strategies) < max_strategies and attempts < 50:
        candidate: Strategy = {}
        for name in param_names:
            config = param_data[name]
            min_val = float(config["min"])
            max_val = float(config["max"])
            scale = str(config.get("scale", "linear"))

            if scale.lower() == "log" and min_val > 0 and max_val > 0:
                value = float(np.exp(rng.uniform(np.log(min_val), np.log(max_val))))
            else:
                value = float(rng.uniform(min_val, max_val))

            candidate[name] = _clip_value(value, min_val, max_val)

        if not _strategy_exists(strategies, candidate):
            strategies.append(candidate)
        attempts += 1

    return strategies


def _strategies_to_key(
    blue_strategy: Mapping[str, float],
    red_strategy: Mapping[str, float],
    blue_names: List[str],
    red_names: List[str],
    precision: int = 12,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    blue_key = tuple(
        round(float(blue_strategy[name]), precision) for name in blue_names
    )
    red_key = tuple(round(float(red_strategy[name]), precision) for name in red_names)
    return blue_key, red_key


def _compute_bounds(
    param_data: Mapping[str, Mapping[str, Union[str, float]]],
    names: List[str],
) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for name in names:
        config = param_data[name]
        bounds.append((float(config["min"]), float(config["max"])))
    return bounds


def _clip_array_to_bounds(
    values: np.ndarray, bounds: List[Tuple[float, float]]
) -> np.ndarray:
    clipped = np.array(values, dtype=float)
    for idx, (lower, upper) in enumerate(bounds):
        if lower is not None:
            clipped[idx] = max(clipped[idx], float(lower))
        if upper is not None:
            clipped[idx] = min(clipped[idx], float(upper))
    return clipped


def _dict_from_vector(names: List[str], values: np.ndarray) -> Strategy:
    return {name: float(val) for name, val in zip(names, values)}


@dataclass
class _LeaderEvaluation:
    value: float
    red_vector: np.ndarray
    outcomes: Dict[str, float]
    blue_strategy: Strategy
    red_success: bool


def stackelberg_equilibrium(
    param_data: ParamConfig,
    function: ObjectiveFn,
    max_iterations: int,
    seed: Optional[int] = None,
) -> Dict[
    str,
    Union[
        float,
        Dict[str, float],
        List[Dict[str, float]],
        bool,
        Dict[str, Union[float, List[float], List[Strategy]]],
        List[Tuple[Strategy, float]],
    ],
]:
    tolerance = 1e-3
    rng = np.random.default_rng(seed)

    blue_param_names = [name for name in param_data.keys() if name.startswith("blue_")]
    if not blue_param_names and "blue_team_honeypot_fraction" in param_data:
        blue_param_names = ["blue_team_honeypot_fraction"]

    red_param_names = [name for name in param_data.keys() if name.startswith("red_")]
    if not red_param_names:
        red_param_names = [
            name for name in param_data.keys() if name not in blue_param_names
        ]

    if not blue_param_names or not red_param_names:
        raise ValueError(
            "blue_param_names and red_param_names must contain at least one parameter each"
        )

    unused = set(param_data.keys()) - set(blue_param_names) - set(red_param_names)
    if unused:
        raise ValueError(f"Unassigned parameters found: {sorted(unused)}")

    bounds_blue = _compute_bounds(param_data, blue_param_names)
    bounds_red = _compute_bounds(param_data, red_param_names)

    pair_cache: Dict[
        Tuple[Tuple[float, ...], Tuple[float, ...]], Tuple[float, Dict[str, float]]
    ] = {}

    def evaluate_pair(
        blue_vec: np.ndarray, red_vec: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        blue_strategy = _dict_from_vector(blue_param_names, blue_vec)
        red_strategy = _dict_from_vector(red_param_names, red_vec)
        key = _strategies_to_key(
            blue_strategy, red_strategy, blue_param_names, red_param_names
        )
        if key in pair_cache:
            return pair_cache[key]

        params: Dict[str, np.ndarray] = {}
        for name in blue_param_names:
            params[name] = np.array(float(blue_strategy[name]))
        for name in red_param_names:
            params[name] = np.array(float(red_strategy[name]))

        payoff_np, outcomes = function(params)
        payoff = float(np.asarray(payoff_np).item())
        outcomes_float = {k: float(np.asarray(v).item()) for k, v in outcomes.items()}
        pair_cache[key] = (payoff, outcomes_float)
        return payoff, outcomes_float

    leader_cache: Dict[Tuple[float, ...], _LeaderEvaluation] = {}
    history: List[Dict[str, Union[int, float, List[float], bool]]] = []
    last_red_best = None

    blue_seed_strategies = _generate_initial_strategies(
        blue_param_names, param_data, rng
    )
    red_seed_strategies = _generate_initial_strategies(red_param_names, param_data, rng)

    if not blue_seed_strategies or not red_seed_strategies:
        raise ValueError("Failed to generate initial strategies for Stackelberg search")

    blue_seed_vectors = [
        np.array([strategy[name] for name in blue_param_names], dtype=float)
        for strategy in blue_seed_strategies
    ]
    red_seed_vectors = [
        np.array([strategy[name] for name in red_param_names], dtype=float)
        for strategy in red_seed_strategies
    ]

    follower_max_iter = max(50, min(400, max(2 * max_iterations, 100)))

    def solve_follower(
        blue_vec: np.ndarray, warm_start: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float, Dict[str, float], bool]:
        candidate_vectors: List[np.ndarray] = []
        if warm_start is not None:
            candidate_vectors.append(warm_start.copy())
        candidate_vectors.extend(red_seed_vectors)

        best_value = float("inf")
        best_vec = None
        best_outcomes: Dict[str, float] = {}

        for candidate in candidate_vectors:
            clipped = _clip_array_to_bounds(candidate, bounds_red)
            value, outcomes = evaluate_pair(blue_vec, clipped)
            if value < best_value:
                best_value = value
                best_vec = clipped
                best_outcomes = outcomes

        if best_vec is None:
            best_vec = _clip_array_to_bounds(red_seed_vectors[0], bounds_red)
            best_value, best_outcomes = evaluate_pair(blue_vec, best_vec)

        red_vec = best_vec
        red_value = best_value
        red_outcomes = best_outcomes

        def maybe_update(candidate_vec: np.ndarray) -> float:
            nonlocal red_vec, red_value, red_outcomes
            candidate = _clip_array_to_bounds(candidate_vec, bounds_red)
            candidate_value, candidate_outcomes = evaluate_pair(blue_vec, candidate)
            improvement = red_value - candidate_value
            if improvement > 1e-9:
                red_vec = candidate
                red_value = candidate_value
                red_outcomes = candidate_outcomes
            return improvement

        def follower_objective(values: np.ndarray) -> float:
            clipped = _clip_array_to_bounds(values, bounds_red)
            value, _ = evaluate_pair(blue_vec, clipped)
            return value

        res_lbfgs = minimize(
            follower_objective,
            best_vec,
            method="L-BFGS-B",
            bounds=bounds_red,
            options={
                "maxiter": follower_max_iter,
                "ftol": tolerance * 0.1,
                "gtol": 1e-6,
            },
        )

        overall_success = bool(res_lbfgs.success)
        lbfgs_improvement = 0.0
        if isinstance(res_lbfgs.x, np.ndarray):
            lbfgs_improvement = maybe_update(res_lbfgs.x)

        movement = float(np.linalg.norm(red_vec - best_vec, ord=1))
        iterations = int(getattr(res_lbfgs, "nit", 0) or 0)
        need_refine = (
            (not res_lbfgs.success)
            or lbfgs_improvement <= tolerance * 0.1
            or movement <= 1e-9
            or iterations <= 2
        )

        if need_refine:
            res_powell = minimize(
                follower_objective,
                red_vec,
                method="Powell",
                bounds=bounds_red,
                options={
                    "maxiter": max(follower_max_iter, 200),
                    "ftol": tolerance * 0.1,
                    "xtol": tolerance * 0.1,
                },
            )
            if isinstance(res_powell.x, np.ndarray):
                maybe_update(res_powell.x)
            overall_success = overall_success or bool(res_powell.success)

        return red_vec, red_value, red_outcomes, overall_success

    def evaluate_blue(
        values: np.ndarray,
    ) -> Tuple[float, np.ndarray, Dict[str, float], Strategy, bool]:
        clipped = _clip_array_to_bounds(values, bounds_blue)
        blue_strategy = _dict_from_vector(blue_param_names, clipped)
        key = tuple(round(float(blue_strategy[name]), 12) for name in blue_param_names)
        cached = leader_cache.get(key)
        if cached is not None:
            return (
                float(cached.value),
                cached.red_vector.copy(),
                dict(cached.outcomes),
                dict(cached.blue_strategy),
                cached.red_success,
            )

        nonlocal last_red_best
        red_vec, red_value, red_outcomes, red_success = solve_follower(
            clipped, last_red_best
        )
        last_red_best = red_vec.copy()

        leader_cache[key] = _LeaderEvaluation(
            value=red_value,
            red_vector=red_vec.copy(),
            outcomes=dict(red_outcomes),
            blue_strategy=dict(blue_strategy),
            red_success=red_success,
        )

        red_strategy_values = _dict_from_vector(red_param_names, red_vec)

        history.append(
            {
                "iteration": len(history),
                "objective": red_value,
                "blue_strategy": [blue_strategy[name] for name in blue_param_names],
                "red_best_response": [
                    red_strategy_values[name] for name in red_param_names
                ],
                "solver_success": red_success,
            }
        )

        return red_value, red_vec, red_outcomes, blue_strategy, red_success

    best_value = float("-inf")
    best_blue_vec: Optional[np.ndarray] = None
    best_red_vec: Optional[np.ndarray] = None
    best_outcomes: Dict[str, float] = {}
    best_delta = float("inf")
    num_updates = 0

    def consider_blue_candidate(values: np.ndarray) -> Tuple[float, bool]:
        nonlocal \
            best_value, \
            best_blue_vec, \
            best_red_vec, \
            best_outcomes, \
            best_delta, \
            num_updates
        value, red_vec, outcomes, blue_strategy, _ = evaluate_blue(values)
        improved = False
        if value > best_value + 1e-9:
            previous = best_value
            delta = value - previous if np.isfinite(previous) else 0.0
            best_value = value
            best_blue_vec = np.array(
                [blue_strategy[name] for name in blue_param_names], dtype=float
            )
            best_red_vec = red_vec.copy()
            best_outcomes = dict(outcomes)
            best_delta = abs(delta)
            num_updates += 1
            improved = True
        return value, improved

    for seed in blue_seed_vectors:
        consider_blue_candidate(seed)

    if max_iterations > len(blue_seed_vectors):
        leader_guess = (
            best_blue_vec.copy()
            if best_blue_vec is not None
            else _clip_array_to_bounds(blue_seed_vectors[0], bounds_blue)
        )

        def leader_objective(values: np.ndarray) -> float:
            value, _ = consider_blue_candidate(values)
            return -value

        res_leader = minimize(
            leader_objective,
            leader_guess,
            method="L-BFGS-B",
            bounds=bounds_blue,
            options={"maxiter": max_iterations, "ftol": tolerance * 0.25, "gtol": 1e-6},
        )

        if res_leader.success and isinstance(res_leader.x, np.ndarray):
            consider_blue_candidate(res_leader.x)

        if best_blue_vec is not None:
            for idx, (lower, upper) in enumerate(bounds_blue):
                span = float(upper - lower)
                if not np.isfinite(span) or span <= 0.0:
                    continue

                base_step = max(span * 0.05, tolerance)
                baseline_val = best_blue_vec[idx]

                for direction in (-1.0, 1.0):
                    step = base_step
                    stepped_once = False
                    while step >= tolerance * 0.1:
                        candidate = best_blue_vec.copy()
                        candidate[idx] = _clip_value(
                            baseline_val + direction * step, lower, upper
                        )
                        if abs(candidate[idx] - baseline_val) <= 1e-9:
                            break

                        previous_best = best_value
                        consider_blue_candidate(candidate)
                        stepped_once = True

                        if best_value <= previous_best + tolerance * 0.25:
                            step *= 0.5
                            continue
                        break

                    if not stepped_once:
                        continue

        restart_limit = min(5, max(0, max_iterations - len(blue_seed_vectors)))
        if restart_limit > 0:
            for _ in range(restart_limit):
                random_vec = np.empty(len(blue_param_names), dtype=float)
                for idx, name in enumerate(blue_param_names):
                    cfg = param_data[name]
                    lower = float(cfg["min"])
                    upper = float(cfg["max"])
                    scale = str(cfg.get("scale", "linear")).lower()
                    if scale == "log" and lower > 0 and upper > 0:
                        random_vec[idx] = float(
                            np.exp(rng.uniform(np.log(lower), np.log(upper)))
                        )
                    else:
                        random_vec[idx] = float(rng.uniform(lower, upper))
                    random_vec[idx] = _clip_value(random_vec[idx], lower, upper)

                consider_blue_candidate(random_vec)

                res_restart = minimize(
                    leader_objective,
                    random_vec,
                    method="L-BFGS-B",
                    bounds=bounds_blue,
                    options={
                        "maxiter": max(10, max_iterations // 2),
                        "ftol": tolerance * 0.5,
                        "gtol": 1e-6,
                    },
                )

                if res_restart.success and isinstance(res_restart.x, np.ndarray):
                    consider_blue_candidate(res_restart.x)

    if best_blue_vec is None or best_red_vec is None:
        raise RuntimeError("Stackelberg optimizer failed to obtain a feasible solution")

    best_blue_strategy = _dict_from_vector(blue_param_names, best_blue_vec)
    best_red_strategy = _dict_from_vector(red_param_names, best_red_vec)

    best_score = float(best_value)

    if not np.isfinite(best_delta):
        best_delta = 0.0
    if num_updates <= 1:
        best_delta = 0.0
    converged = best_delta <= tolerance

    blue_mixed_strategy: List[Tuple[Strategy, float]] = [(best_blue_strategy, 1.0)]
    red_mixed_strategy: List[Tuple[Strategy, float]] = [(best_red_strategy, 1.0)]

    return {
        "best_score": best_score,
        "red_best_response": best_red_strategy,
        "blue_mixed_strategy": blue_mixed_strategy,
        "red_mixed_strategy": red_mixed_strategy,
        "history": history,  # type: ignore[return-value]
        "outcome_distribution": best_outcomes,
        "converged": converged,
    }
