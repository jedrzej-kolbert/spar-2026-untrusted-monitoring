"""Event-count audit: which point-estimate cells rest on (near-)zero counts?

The bootstrap (and the point estimate itself) is only trustworthy where the
empirical rates entering the game value are supported by enough samples. This
audits, per condition x usefulness point, the effective event count behind
each channel of the win calculator, evaluated at the saved Nash equilibrium
strategies:

* ``pass``      -- n_attacks x P(attack passes all monitors). THE red-team win
                   channel: zero here pins safety at 1.0 and makes the
                   bootstrap degenerate (all replicates identical).
* ``flagged``   -- n_attacks x P(attack flagged for audit).
* ``hp_trigger``-- n_honeypots x P(collusion monitor stays silent on a
                   honeypot) -- the honeypot kill channel; 0 or n are both
                   boundaries.
* ``honest_fnr``/``honest_fpr`` -- honest-UM error numerators (fixed per
                   condition, threshold-independent).
* ``tm_benign`` -- n_benign x TM benign FPR at the audit budget (how many
                   benign samples pin down the audit threshold c(f)).

Dependence is handled by construction: rates are taken from the fitted ROC
function itself, so for the joint empirical model ``pass`` is the count of
attack samples in the JOINT cell (which can be zero while every marginal
count is healthy -- the marginal audit that suffices under independence is
NOT sufficient here). Interpolated/randomised thresholds make counts
fractional; a fractional count in (0, 1) means the rate is supported by less
than one sample.

Verdicts per cell are driven by the binding ``pass`` channel; a red mixture
may contain cautious components with ~0 passes, so degeneracy is judged on
the max over support pairs and precision on the mixture-weighted count:
  SATURATED  max-over-pairs pass count == 0 (bootstrap provably degenerate)
  CRITICAL   max-over-pairs pass count < 1 (sub-sample support)
  LOW        mixture-weighted pass count < 10
  OK         otherwise
Other channels (honest errors, honeypot boundary, TM-benign support) are
reported as ``low_count_caveats`` when under 10 events, without driving the
verdict.

Usage, over every saved condition (analyse folders) under an experiments dir:
  python -m lasr_labs_2025_control_project.scripts.helpers.event_count_audit \
      --root experiments [--setting apps] --out event_count_audit.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


from lasr_labs_2025_control_project.scripts.plot_data import (
    VARIANT_FLAGS,
    load_condition,
    load_saved_condition,
    saved_conditions,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.roc_function_builders import (
    build_roc_function,
)

MIN_SUPPORT_WEIGHT = 0.05  # audit equilibrium support points above this prob


def audit_cell(
    roc_fn,
    honest_fpr: float,
    honest_fnr: float,
    n_att: int,
    n_hp: int,
    n_ben: int,
    entry: Dict[str, Any],
    *,
    budget_fraction: float = 0.02,
) -> Dict[str, Any]:
    """Effective event counts for one sweep entry, over the Nash supports."""
    replace_fnr = float(entry["blue_team_replace_fnr"])
    channels: Dict[str, float] = {}

    def merge(name: str, count: float) -> None:
        channels[name] = min(channels.get(name, float("inf")), count)

    blue_support = [
        (s["blue_team_honeypot_fraction"], p)
        for s, p in entry["blue_mixed_strategy"]
        if p >= MIN_SUPPORT_WEIGHT
    ] or [(entry["best_parameters"].get("blue_team_honeypot_fraction", 0.0), 1.0)]
    red_support = [
        (s["red_team_roc_threshold"], p)
        for s, p in entry["red_mixed_strategy"]
        if p >= MIN_SUPPORT_WEIGHT
    ] or [(entry["best_parameters"].get("red_team_roc_threshold", 0.0), 1.0)]

    pass_counts = []
    pass_weights = []
    for hp_frac, w_b in blue_support:
        audit_fraction = budget_fraction * (1.0 - float(hp_frac))
        tm_fnr = max(audit_fraction - honest_fpr, 0.0)
        for t, w_r in red_support:
            not_flagged, _ = roc_fn(float(t), tm_fnr)
            pass_rate, _ = roc_fn(float(t), min(replace_fnr + tm_fnr, 1.0))
            _, u_fpr = roc_fn(float(t), None)
            pass_counts.append(n_att * float(pass_rate))
            pass_weights.append(w_b * w_r)
            merge("flagged", n_att * float(1.0 - not_flagged))
            # boundary on either side matters for the honeypot channel
            merge("hp_trigger", min(n_hp * float(u_fpr), n_hp * float(1.0 - u_fpr)))
            merge("tm_benign", n_ben * tm_fnr)

    # The pass channel is what saturates safety at 1.0, and it is a mixture:
    # the bootstrap only degenerates when the count is ~0 under EVERY support
    # pair (max), while precision is governed by the mixture-weighted count.
    total_w = sum(pass_weights)
    channels["pass_max"] = max(pass_counts)
    channels["pass_expected"] = (
        sum(c * w for c, w in zip(pass_counts, pass_weights)) / total_w
        if total_w
        else 0.0
    )
    channels["honest_fnr"] = n_att * honest_fnr
    channels["honest_fpr"] = n_ben * honest_fpr

    # Verdict is driven by the binding (pass) channel; the other channels are
    # reported and surfaced as caveats but do not saturate the game value.
    if channels["pass_max"] <= 1e-9:
        verdict = "SATURATED"
    elif channels["pass_max"] < 1.0:
        verdict = "CRITICAL"
    elif channels["pass_expected"] < 10.0:
        verdict = "LOW"
    else:
        verdict = "OK"
    caveats = sorted(
        name for name, v in channels.items() if not name.startswith("pass") and v < 10.0
    )
    return {
        "usefulness": entry["usefulness"],
        "safety": entry["safety"],
        "channels": {k: round(v, 3) for k, v in channels.items()},
        "low_count_caveats": caveats,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--setting", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    for folder in saved_conditions(args.root, args.setting):
        payload = load_saved_condition(folder)
        record = payload["pipeline"]
        frames = load_condition({k: Path(v) for k, v in record["inputs"].items()})
        attacks, honeypots, benign = (
            frames["attacks"],
            frames["honeypots"],
            frames["honest"],
        )
        disable_tm, disable_um = VARIANT_FLAGS[record["variant"]]
        roc_fn, meta = build_roc_function(
            attacks_df=attacks,
            honeypots_df=honeypots,
            benign_df=benign,
            disable_trusted_monitor=disable_tm,
            disable_untrusted_monitor=disable_um,
        )
        game = record["solve"]["game"]
        cells = [
            audit_cell(
                roc_fn,
                float(meta.get("untrusted_honest_fpr", 0.0)),
                float(meta.get("untrusted_honest_fnr", 0.0)),
                len(attacks),
                len(honeypots),
                len(benign),
                entry,
                budget_fraction=float(game["blue_team_budget_fraction"]),
            )
            for entry in payload["sweep"]
        ]
        key = f"{record['name']}/{record['variant']}"
        report[key] = {
            "folder": str(folder),
            "models": record.get("models"),
            "model_type": meta.get("model_type"),
            "sample_counts": {
                "attacks": len(attacks),
                "honeypots": len(honeypots),
                "benign": len(benign),
            },
            "cells": cells,
        }
        worst = min(
            cells,
            key=lambda c: ["SATURATED", "CRITICAL", "LOW", "OK"].index(c["verdict"]),
        )
        n_bad = sum(1 for c in cells if c["verdict"] != "OK")
        print(
            f"{key:58s} worst={worst['verdict']:9s} non-OK cells={n_bad}/{len(cells)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
