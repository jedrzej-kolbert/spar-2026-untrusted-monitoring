"""Optional PRIMA BOBYQA best-response search through the C runner.

PRIMA is a local optimizer. Its exit status does not certify a global best
response, so the double-oracle caller must verify returned payoffs against
strategies it already knows.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult


def minimize_prima_bobyqa(
    objective: Callable[[np.ndarray], float],
    guess: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    scales: Sequence[str],
    *,
    maxfun: int = 3000,
    runner: str | Path | None = None,
) -> OptimizeResult:
    """Minimize a bounded objective with the PRIMA C runner.

    Linear and log-scale parameters are normalized to [0, 1]. The runner path
    may be supplied here or in ``PRIMA_BOBYQA_RUNNER``. No external model calls
    are made by this adapter; it evaluates the supplied objective locally.
    """

    executable = runner or os.environ.get("PRIMA_BOBYQA_RUNNER")
    if not executable:
        raise RuntimeError(
            "PRIMA-BOBYQA requires PRIMA_BOBYQA_RUNNER; "
            "see docs/prima-bobyqa.md for build instructions"
        )
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"PRIMA BOBYQA runner is not executable: {executable}")

    guess = np.asarray(guess, dtype=float)
    bounds_array = np.asarray(bounds, dtype=float)
    if not np.all(np.isfinite(guess)):
        raise ValueError("PRIMA BOBYQA requires a finite starting guess")
    if bounds_array.shape != (len(guess), 2) or len(scales) != len(guess):
        raise ValueError("PRIMA bounds and scales must match the guess dimensions")
    if not np.all(np.isfinite(bounds_array)) or not np.all(
        bounds_array[:, 0] < bounds_array[:, 1]
    ):
        raise ValueError("PRIMA BOBYQA requires finite, nonempty bounds")
    if maxfun < 1:
        raise ValueError("maxfun must be positive")

    lower = bounds_array[:, 0].copy()
    upper = bounds_array[:, 1].copy()
    log_indices: list[int] = []
    for idx, scale in enumerate(scales):
        if scale == "log":
            if lower[idx] <= 0:
                raise ValueError("Log-scale PRIMA bounds must be positive")
            lower[idx], upper[idx] = np.log(lower[idx]), np.log(upper[idx])
            log_indices.append(idx)
        elif scale != "linear":
            raise ValueError(f"Unknown PRIMA parameter scale: {scale}")
    span = upper - lower
    solver_guess = guess.copy()
    for idx in log_indices:
        if solver_guess[idx] <= 0:
            raise ValueError("Log-scale PRIMA guess must be positive")
        solver_guess[idx] = np.log(solver_guess[idx])
    normalized_guess = np.clip((solver_guess - lower) / span, 0.0, 1.0)

    def denormalize(normalized: np.ndarray) -> np.ndarray:
        if normalized.shape != guess.shape or not np.all(np.isfinite(normalized)):
            raise RuntimeError("PRIMA runner returned invalid coordinates")
        if np.any(normalized < -1e-8) or np.any(normalized > 1 + 1e-8):
            raise RuntimeError("PRIMA runner returned coordinates outside bounds")
        actual = lower + np.clip(normalized, 0.0, 1.0) * span
        for idx in log_indices:
            actual[idx] = np.exp(actual[idx])
        return actual

    command = [str(executable), str(len(guess)), str(maxfun)] + [
        format(x, ".17g") for x in normalized_guess
    ]
    result: OptimizeResult | None = None
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for line in process.stdout:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "EVAL" and len(fields) == len(guess) + 1:
                    point = denormalize(np.array([float(x) for x in fields[1:]]))
                    value = float(objective(point))
                    if not np.isfinite(value):
                        raise ValueError("PRIMA objective returned a non-finite value")
                    process.stdin.write(f"{value:.17g}\n")
                    process.stdin.flush()
                elif fields[0] == "RESULT" and len(fields) == len(guess) + 4:
                    status = int(fields[1])
                    point = denormalize(
                        np.array([float(x) for x in fields[2 : 2 + len(guess)]])
                    )
                    result = OptimizeResult(
                        x=point,
                        fun=float(fields[2 + len(guess)]),
                        nfev=int(fields[3 + len(guess)]),
                        success=status in (0, 1),
                        status=status,
                        message=f"PRIMA BOBYQA status {status}",
                    )
                else:
                    raise RuntimeError(f"Unexpected PRIMA runner output: {line.strip()}")
        except Exception:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdin.close()
            process.stdout.close()
        exit_code = process.wait()
        if exit_code or result is None:
            errors.seek(0)
            raise RuntimeError(
                f"PRIMA BOBYQA runner failed (exit {exit_code}): {errors.read()}"
            )
    return result
