"""Double Oracle optimizer for blue/red parameter search.

This module implements a Double Oracle algorithm for computing an approximate
Nash equilibrium in a two-player zero-sum game. The game models an adversarial
optimization problem where:
  - Blue player (defender): selects defensive parameters (honeypot allocation)
  - Red player (attacker): selects adversarial parameters (attack prob, threshold)

The algorithm iteratively expands each player's mixed strategy support by adding
best responses computed against the opponent's current mixed strategy. At each
iteration, the restricted game (using only strategies discovered so far) is solved
exactly via linear programming to obtain equilibrium mixed strategies. The algorithm
terminates when the gap between upper and lower bounds converges.

Key differences from grid_search:
  - Supports mixed strategies (probability distributions over pure strategies)
  - Estimates an equilibrium using locally optimized best responses; the
    stopping gap is not a global equilibrium certificate
  - More computationally expensive but theoretically sound for adversarial settings
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linprog, minimize

Strategy = Dict[str, float]
ParamConfig = Dict[str, Dict[str, Union[str, float]]]
ObjectiveFn = Callable[
    [Dict[str, np.ndarray]], Tuple[np.ndarray, Dict[str, np.ndarray]]
]


def _midpoint(min_val: float, max_val: float, scale: str) -> float:
    """Return a scale-aware midpoint between ``min_val`` and ``max_val``.

    For log-scale parameters, computes geometric mean. For linear-scale,
    computes arithmetic mean.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        scale: "log" or "linear"

    Returns:
        Midpoint value respecting the scale
    """

    scale = scale.lower()
    if scale == "log" and min_val > 0 and max_val > 0:
        return float(np.exp((np.log(min_val) + np.log(max_val)) / 2.0))
    return float((min_val + max_val) / 2.0)


def _clip_value(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    """Clip ``value`` to the inclusive interval ``[lower, upper]``.

    Args:
        value: Value to clip
        lower: Lower bound (None means no lower bound)
        upper: Upper bound (None means no upper bound)

    Returns:
        Clipped value
    """

    if lower is not None:
        value = max(value, float(lower))
    if upper is not None:
        value = min(value, float(upper))
    return float(value)


def _strategy_exists(
    strategies: List[Strategy], candidate: Mapping[str, float], tol: float = 1e-8
) -> bool:
    """Return ``True`` if ``candidate`` already exists in ``strategies``.

    Checks for approximate equality within tolerance to handle floating point issues.

    Args:
        strategies: List of existing strategy dicts
        candidate: Candidate strategy to check
        tol: Tolerance for floating point comparison

    Returns:
        True if candidate is already in strategies (within tolerance)
    """

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
    """Generate a diverse seed set of strategies for one player.

    Creates an initial support for the double oracle algorithm by:
    1. Enumerating a coarse grid: bounds, midpoint, and random samples per parameter
    2. Taking cartesian product (truncated to max_strategies)
    3. Random sampling to fill remaining capacity with unique points

    This initialization helps avoid poor local equilibria by starting with
    diverse strategies.

    Args:
        param_names: List of parameter names for this player
        param_data: Parameter metadata (min, max, scale per parameter)
        rng: Random number generator
        max_strategies: Maximum number of initial strategies

    Returns:
        List of diverse initial strategies (each a dict of param name -> value)
    """

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
    """Build a hashable cache key for a blue/red strategy pair."""

    blue_key = tuple(
        round(float(blue_strategy[name]), precision) for name in blue_names
    )
    red_key = tuple(round(float(red_strategy[name]), precision) for name in red_names)
    return blue_key, red_key


def _solve_zero_sum_game(
    payoff_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve the restricted game exactly via linear programming.

    Solves for the Nash equilibrium of a two-player zero-sum game using
    the standard LP formulation. The payoff matrix represents blue's utility
    (rows=blue strategies, cols=red strategies).

    The LP formulation:
    - Blue maximizes min expected payoff over red's actions
    - Red minimizes max expected payoff over blue's actions

    Returns the equilibrium mixed strategies for blue and red, together with the
    game value. The payoff matrix is shifted upward to ensure feasibility for the
    LP formulation, and the original value is recovered afterwards.

    Args:
        payoff_matrix: 2D array where entry [i,j] is blue's payoff when blue
            plays strategy i and red plays strategy j

    Returns:
        Tuple of (blue_probs, red_probs, game_value) where:
            - blue_probs: probability distribution over blue strategies
            - red_probs: probability distribution over red strategies
            - game_value: expected payoff at equilibrium

    Raises:
        ValueError: If matrix is not 2D or has invalid dimensions
        RuntimeError: If LP fails to solve
    """

    matrix = np.asarray(payoff_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("payoff_matrix must be 2D")

    num_blue, num_red = matrix.shape
    if num_blue == 0 or num_red == 0:
        raise ValueError("payoff_matrix must contain at least one strategy per player")

    # Shift matrix to ensure all entries are positive for LP feasibility
    min_val = matrix.min()
    shift = -min_val + 1e-9 if min_val < 0 else 0.0
    shifted = matrix + shift

    # Solve blue team's LP: maximize value v such that v <= expected payoff for each red action
    c = np.zeros(num_blue + 1)
    c[-1] = -1.0  # Maximize v (minimize -v)

    A_ub = np.zeros((num_red, num_blue + 1))
    b_ub = np.zeros(num_red)
    for j in range(num_red):
        A_ub[j, :num_blue] = -shifted[:, j]  # -sum(p_i * payoff[i,j])
        A_ub[j, -1] = 1.0  # + v
        # Constraint: v <= sum(p_i * payoff[i,j]) for all j

    A_eq = np.zeros((1, num_blue + 1))
    A_eq[0, :num_blue] = 1.0  # sum(p_i) = 1
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * num_blue + [(None, None)]

    res_blue = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res_blue.success:
        raise RuntimeError(f"Failed to solve blue LP: {res_blue.message}")

    # Extract and normalize blue probabilities
    blue_probs = np.clip(res_blue.x[:num_blue], 0.0, None)
    if blue_probs.sum() <= 0:
        blue_probs = np.full(num_blue, 1.0 / num_blue)
    else:
        blue_probs /= blue_probs.sum()

    game_value = float(res_blue.x[-1] - shift)  # Undo the shift

    # Solve red team's LP: minimize value v such that v >= expected payoff for each blue action
    c_red = np.zeros(num_red + 1)
    c_red[-1] = 1.0  # Minimize v

    A_ub_red = np.zeros((num_blue, num_red + 1))
    b_ub_red = np.zeros(num_blue)
    for i in range(num_blue):
        A_ub_red[i, :num_red] = shifted[i, :]  # sum(q_j * payoff[i,j])
        A_ub_red[i, -1] = -1.0  # - v
        # Constraint: sum(q_j * payoff[i,j]) <= v for all i

    A_eq_red = np.zeros((1, num_red + 1))
    A_eq_red[0, :num_red] = 1.0  # sum(q_j) = 1
    b_eq_red = np.array([1.0])

    bounds_red = [(0.0, None)] * num_red + [(None, None)]

    res_red = linprog(
        c_red,
        A_ub=A_ub_red,
        b_ub=b_ub_red,
        A_eq=A_eq_red,
        b_eq=b_eq_red,
        bounds=bounds_red,
        method="highs",
    )

    # Extract and normalize red probabilities
    if res_red.success:
        red_probs = np.clip(res_red.x[:num_red], 0.0, None)
        if red_probs.sum() <= 0:
            red_probs = np.full(num_red, 1.0 / num_red)
        else:
            red_probs /= red_probs.sum()
    else:
        # Fallback to uniform if red LP fails
        red_probs = np.full(num_red, 1.0 / num_red)

    return blue_probs, red_probs, game_value


def _compute_bounds(
    param_data: Mapping[str, Mapping[str, Union[str, float]]], names: List[str]
) -> List[Tuple[float, float]]:
    """Extract (min, max) bounds for the given parameter names."""

    bounds: List[Tuple[float, float]] = []
    for name in names:
        if name not in param_data:
            raise KeyError(f"Missing parameter configuration for '{name}'")
        config = param_data[name]
        bounds.append((float(config["min"]), float(config["max"])))
    return bounds


def _clip_array_to_bounds(
    values: np.ndarray, bounds: List[Tuple[float, float]]
) -> np.ndarray:
    """Clip a NumPy vector to match parameter bounds."""

    clipped = np.array(values, dtype=float)
    for idx, (lower, upper) in enumerate(bounds):
        clipped[idx] = _clip_value(clipped[idx], lower, upper)
    return clipped


def double_oracle(
    param_data: ParamConfig,
    function: ObjectiveFn,
    max_iterations: int,
    best_response_method: str = "Powell",
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
    """Compute a Nash equilibrium using the Double Oracle algorithm.

    Iteratively builds up mixed strategies for both players:
    1. Solve the restricted game (using current strategy supports) via LP
    2. Compute best responses to opponent's mixed strategy via local optimization
    3. Add new best responses to strategy supports
    4. Repeat until convergence (gap between bounds < epsilon)

    With locally optimized best responses, a small reported gap does not establish
    that neither player can improve by deviating elsewhere in the strategy space.

    Args:
        param_data: Parameter metadata describing bounds and scales, with keys like
            'blue_team_honeypot_fraction', 'red_team_roc_threshold', 'red_team_p_attack'.
            Each value is a dict with 'min', 'max', and 'scale' ('linear' or 'log').
        function: Callable that accepts a dictionary of NumPy arrays (parameter values)
            and returns a tuple ``(blue_win_rate, outcome_dict)``. Blue wants to maximize,
            red wants to minimize.
        max_iterations: Maximum number of outer Double Oracle iterations.
        best_response_method: Local method used for best-response searches.

    Returns:
        Dictionary containing:
            * ``best_score`` – estimated blue win rate against the discovered red response.
            * ``red_best_response`` – the best response red strategy (unmixed, for exploitation).
            * ``blue_mixed_strategy`` – list of (strategy, probability) tuples for blue team Nash equilibrium.
            * ``red_mixed_strategy`` – list of (strategy, probability) tuples for red team Nash equilibrium.
            * ``history`` – per-iteration trace (objective, bounds, probabilities).
            * ``outcome_distribution`` – final outcome breakdown at equilibrium (weighted sum over both mixed strategies).
            * ``converged`` – whether the local-response stopping tolerance was satisfied.
    """

    epsilon = 1e-3  # Convergence tolerance for gap between upper/lower bounds
    if best_response_method not in ("Powell", "COBYQA"):
        raise ValueError(f"Unsupported best-response method: {best_response_method}")
    rng = np.random.default_rng()

    # Separate parameters by player (blue vs red)
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

    # Cache to avoid re-evaluating the same strategy pairs
    cache: Dict[
        Tuple[Tuple[float, ...], Tuple[float, ...]], Tuple[float, Dict[str, float]]
    ] = {}

    def evaluate(blue: Strategy, red: Strategy) -> Tuple[float, Dict[str, float]]:
        """Evaluate objective function for a blue/red strategy pair with caching."""
        key = _strategies_to_key(blue, red, blue_param_names, red_param_names)
        if key in cache:
            return cache[key]

        # Convert strategies to numpy arrays for the objective function
        params: Dict[str, np.ndarray] = {}
        for name in blue_param_names:
            params[name] = np.array(float(blue[name]))
        for name in red_param_names:
            params[name] = np.array(float(red[name]))

        payoff_np, outcomes = function(params)
        payoff = float(np.asarray(payoff_np).item())
        outcomes_float = {k: float(np.asarray(v).item()) for k, v in outcomes.items()}
        cache[key] = (payoff, outcomes_float)
        return payoff, outcomes_float

    # Generate diverse initial strategy sets for both players
    initial_blue_strategies = _generate_initial_strategies(
        blue_param_names, param_data, rng
    )
    initial_red_strategies = _generate_initial_strategies(
        red_param_names, param_data, rng
    )

    blue_strategies = list(initial_blue_strategies)
    red_strategies = list(initial_red_strategies)

    # Build the initial payoff matrix using the current strategy supports.
    payoff_matrix = np.zeros((len(blue_strategies), len(red_strategies)))
    for i, blue in enumerate(blue_strategies):
        for j, red in enumerate(red_strategies):
            payoff_matrix[i, j] = evaluate(blue, red)[0]

    # Extract parameter bounds for continuous optimization
    bounds_blue = _compute_bounds(param_data, blue_param_names)
    bounds_red = _compute_bounds(param_data, red_param_names)

    def minimize_best_response(
        objective: Callable[[np.ndarray], float],
        guess: np.ndarray,
        bounds: List[Tuple[float, float]],
        names: List[str],
    ) -> Any:
        if best_response_method == "Powell":
            return minimize(
                objective,
                guess,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 200},
            )

        # COBYQA searches in log coordinates for parameters declared log-scale.
        log_indices = [
            idx
            for idx, name in enumerate(names)
            if str(param_data[name].get("scale", "linear")).lower() == "log"
        ]
        solver_guess = guess.copy()
        solver_bounds = list(bounds)
        for idx in log_indices:
            if bounds[idx][0] <= 0:
                raise ValueError(
                    f"Log-scale parameter {names[idx]} requires positive bounds"
                )
            solver_guess[idx] = np.log(solver_guess[idx])
            solver_bounds[idx] = tuple(np.log(bounds[idx]))

        def solver_objective(values: np.ndarray) -> float:
            actual = values.copy()
            for idx in log_indices:
                actual[idx] = np.exp(actual[idx])
            return objective(actual)

        result = minimize(
            solver_objective,
            solver_guess,
            method="COBYQA",
            bounds=solver_bounds,
            options={"maxfev": 3000, "scale": True},
        )
        for idx in log_indices:
            result.x[idx] = np.exp(result.x[idx])
        return result

    # Initial guesses for best response optimization
    blue_guess = np.array(
        [blue_strategies[0][name] for name in blue_param_names], dtype=float
    )
    red_guess = np.array(
        [red_strategies[0][name] for name in red_param_names], dtype=float
    )

    # Initialize mixed strategies (uniform over initial supports)
    blue_probs = np.full(len(blue_strategies), 1.0 / len(blue_strategies))
    red_probs = np.full(len(red_strategies), 1.0 / len(red_strategies))
    game_value = float(np.mean(payoff_matrix))

    # Initialize bounds tracking
    upper_bound = float("inf")
    lower_bound = float("-inf")
    gap = float("inf")

    best_blue_strategy = blue_strategies[0]
    best_red_strategy = red_strategies[0]

    history: List[Dict[str, Any]] = []
    converged = False

    for iteration in range(max_iterations):
        # Step 1: solve the restricted game exactly to obtain mixed strategies.
        blue_probs, red_probs, game_value = _solve_zero_sum_game(payoff_matrix)

        def blue_expected_value(values: np.ndarray) -> float:
            """Expected payoff for a candidate blue strategy."""

            candidate = {
                name: float(val) for name, val in zip(blue_param_names, values)
            }
            expected = 0.0
            for prob, red in zip(red_probs, red_strategies):
                payoff, _ = evaluate(candidate, red)
                expected += prob * payoff
            return expected

        def red_expected_value(values: np.ndarray) -> float:
            """Expected payoff (from blue's perspective) for a red strategy."""

            candidate = {name: float(val) for name, val in zip(red_param_names, values)}
            expected = 0.0
            for prob, blue in zip(blue_probs, blue_strategies):
                payoff, _ = evaluate(blue, candidate)
                expected += prob * payoff
            return expected

        def blue_objective(x: np.ndarray) -> float:
            return -blue_expected_value(x)

        def red_objective(x: np.ndarray) -> float:
            return red_expected_value(x)

        # Step 2: compute best responses with local continuous optimization.
        blue_start = blue_guess.copy()
        red_start = red_guess.copy()
        blue_guess_score = blue_expected_value(blue_start)
        red_guess_score = red_expected_value(red_start)
        best_existing_blue_score = float(np.max(payoff_matrix @ red_probs))
        best_existing_red_score = float(np.min(blue_probs @ payoff_matrix))

        res_blue = minimize_best_response(
            blue_objective, blue_guess, bounds_blue, blue_param_names
        )
        candidate_blue = _clip_array_to_bounds(
            res_blue.x if np.all(np.isfinite(res_blue.x)) else blue_guess, bounds_blue
        )
        if blue_expected_value(candidate_blue) < blue_guess_score:
            candidate_blue = blue_start
        best_blue_strategy = {
            name: float(val) for name, val in zip(blue_param_names, candidate_blue)
        }
        blue_upper = blue_expected_value(candidate_blue)
        blue_guess = candidate_blue

        res_red = minimize_best_response(
            red_objective, red_guess, bounds_red, red_param_names
        )
        candidate_red = _clip_array_to_bounds(
            res_red.x if np.all(np.isfinite(res_red.x)) else red_guess, bounds_red
        )
        if red_expected_value(candidate_red) > red_guess_score:
            candidate_red = red_start
        best_red_strategy = {
            name: float(val) for name, val in zip(red_param_names, candidate_red)
        }
        red_lower = red_expected_value(candidate_red)
        red_guess = candidate_red

        upper_bound = blue_upper
        lower_bound = red_lower
        raw_gap = upper_bound - lower_bound
        if raw_gap < -1e-8:
            raise RuntimeError(
                "Invalid best-response bounds: "
                f"lower={lower_bound}, upper={upper_bound}"
            )
        gap = max(0.0, raw_gap)

        history.append(
            {
                "iteration": iteration,
                "best_response_method": best_response_method,
                "num_blue_strategies": float(len(blue_strategies)),
                "num_red_strategies": float(len(red_strategies)),
                "equilibrium_value": game_value,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "red_lower": red_lower,
                "raw_gap": raw_gap,
                "gap": gap,
                "blue_probs": blue_probs.tolist(),
                "red_probs": red_probs.tolist(),
                "blue_guess": dict(zip(blue_param_names, map(float, blue_start))),
                "candidate_blue": best_blue_strategy.copy(),
                "red_guess": dict(zip(red_param_names, map(float, red_start))),
                "candidate_red": best_red_strategy.copy(),
                "blue_guess_score": blue_guess_score,
                "candidate_blue_score": blue_upper,
                "best_existing_blue_score": best_existing_blue_score,
                "red_guess_score": red_guess_score,
                "candidate_red_score": red_lower,
                "best_existing_red_score": best_existing_red_score,
                "res_blue_success": bool(res_blue.success),
                "res_blue_nit": (
                    int(res_blue.nit) if res_blue.get("nit") is not None else None
                ),
                "res_blue_nfev": (
                    int(res_blue.nfev) if res_blue.get("nfev") is not None else None
                ),
                "res_blue_message": str(res_blue.message),
                "res_red_success": bool(res_red.success),
                "res_red_nit": (
                    int(res_red.nit) if res_red.get("nit") is not None else None
                ),
                "res_red_nfev": (
                    int(res_red.nfev) if res_red.get("nfev") is not None else None
                ),
                "res_red_message": str(res_red.message),
            }
        )

        added_new = False

        added_blue = not _strategy_exists(blue_strategies, best_blue_strategy)
        if added_blue:
            blue_strategies.append(best_blue_strategy)
            new_row = np.array(
                [evaluate(best_blue_strategy, red)[0] for red in red_strategies]
            )
            payoff_matrix = np.vstack([payoff_matrix, new_row[np.newaxis, :]])
            added_new = True

        added_red = not _strategy_exists(red_strategies, best_red_strategy)
        if added_red:
            red_strategies.append(best_red_strategy)
            new_col = np.array(
                [evaluate(blue, best_red_strategy)[0] for blue in blue_strategies]
            )
            payoff_matrix = np.hstack([payoff_matrix, new_col[:, np.newaxis]])
            added_new = True

        history[-1]["added_blue_strategy"] = added_blue
        history[-1]["added_red_strategy"] = added_red

        if gap <= epsilon and not added_new:
            converged = True
            break

    # Compute the final outcome distribution using red's best response against blue's Nash equilibrium
    # This gives us the worst-case outcome distribution when red exploits blue's equilibrium strategy
    final_outcomes: Dict[str, float] = {}

    for blue_idx, blue_prob in enumerate(blue_probs):
        if blue_prob < 1e-12:
            continue
        blue_strategy = blue_strategies[blue_idx]
        # Use red's best response strategy (not the mixed Nash equilibrium)
        _, outcomes = evaluate(blue_strategy, best_red_strategy)
        for key, value in outcomes.items():
            final_outcomes[key] = final_outcomes.get(key, 0.0) + blue_prob * value

    # Use the discovered red response against blue's restricted-game mixture.
    # This aligns with the outcome_distribution computation above, but remains
    # an estimate unless the red best-response search was globally validated.
    best_score = lower_bound if np.isfinite(lower_bound) else game_value

    # Build the blue mixed strategy as a list of (strategy, probability) tuples
    blue_mixed_strategy: List[Tuple[Strategy, float]] = []
    for blue_idx, blue_prob in enumerate(blue_probs):
        if blue_prob > 1e-12:
            blue_mixed_strategy.append((blue_strategies[blue_idx], float(blue_prob)))

    # Build the red mixed strategy as a list of (strategy, probability) tuples
    red_mixed_strategy: List[Tuple[Strategy, float]] = []
    for red_idx, red_prob in enumerate(red_probs):
        if red_prob > 1e-12:
            red_mixed_strategy.append((red_strategies[red_idx], float(red_prob)))

    best_red_response = {k: float(v) for k, v in best_red_strategy.items()}

    return {
        "best_score": float(best_score),
        "red_best_response": best_red_response,  # Red team best response (unmixed, for exploitation)
        "blue_mixed_strategy": blue_mixed_strategy,  # Blue Nash equilibrium
        "red_mixed_strategy": red_mixed_strategy,  # Red Nash equilibrium
        "history": history,  # type: ignore[return]
        "outcome_distribution": final_outcomes,
        "converged": converged,
    }
