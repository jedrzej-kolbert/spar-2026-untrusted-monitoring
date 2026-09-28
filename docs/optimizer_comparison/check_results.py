"""Check the archived experiment; no model calls required."""

import json
import math
from pathlib import Path

folder = Path(__file__).parent
data = json.loads((folder / "results.json").read_text())
methods = (
    "main_powell",
    "powell_guess_only",
    "de_raw_only",
    "de_raw_known_policy",
    "pr_de",
)
for arm in ("passive", "sft"):
    for method in methods:
        rows = data["curves"][arm][method]
        assert len(rows) == 10
        assert all(math.isfinite(r["safety"]) and 0 <= r["safety"] <= 1 for r in rows)
        assert all(a["usefulness"] > b["usefulness"] for a, b in zip(rows, rows[1:]))
        reversals = sum(
            b["safety"] < a["safety"] - 1e-8 for a, b in zip(rows, rows[1:])
        )
        print(f"{arm}: {method}: {reversals} monotonicity reversals")
        if method.startswith("de_") or method == "pr_de":
            assert reversals == 0
        if method == "de_raw_only":
            assert all(
                r["converged"] and r["last_lower"] <= r["last_upper"] + 1e-8
                for r in rows
            )
        if arm == "sft" and method in ("main_powell", "powell_guess_only"):
            assert reversals > 0

pairs = data["sft_seed_repeats"]["offsets"]
assert len(pairs) == 4
assert all(p["0.99"]["safety"] <= p["0.98"]["safety"] + 1e-8 for p in pairs.values())
print("Raw DE is monotone in both arms and all four repeat pairs.")
