"""Render house-style safety/usefulness band plots from saved bootstrap replicates.

One figure per (setting, U, panel), with the transforms (none / minified /
paraphrased) as labelled curves: bootstrap median plus a percentile band (80%
by default), plus one assembled grid per (setting, U). A panel is a collusion
strategy and monitor arm; label-control arms are left out. Reads every
game folder under --root (the pipeline's experiments dir):

    python -m lasr_labs_2025_control_project.scripts.plot_bootstrap_curves \
        --root experiments --out figures/bands [--variant all_monitors] [--ci 0.80]
"""

import argparse
import json
from pathlib import Path

from lasr_labs_2025_control_project.scripts.plot_data import (
    bootstrap_path,
    condition_of,
    load_saved_condition,
    read_replicates,
    saved_conditions,
)
from lasr_labs_2025_control_project.tracking import names
from lasr_labs_2025_control_project.utils.numerical_pipeline.bootstrap import (
    bootstrap_band,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (
    PLOT_FILE_FORMAT,
    plot_safety_usefulness_bands,
    plot_safety_usefulness_bands_grid,
)


#: (collusion strategy, monitor arm) -> panel title, in plotting order
PANELS = {
    ("acausal", "base"): "Acausal",
    ("causal", "base"): "Causal",
    ("causal", "sft"): "Causal + SFT",
    ("additive", "sft"): "Additive",
    ("self_rec", "base"): "Passive Self-Rec",
    ("self_rec", "sft"): "Passive Self-Rec + SFT",
}
TRANSFORMS = {"none": "None", "minified": "Minified", "paraphrased": "Paraphrased"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--setting", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="all_monitors")
    parser.add_argument("--ci", type=float, default=0.80)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help=(
            "event_count_audit.json (helpers/event_count_audit.py). When "
            "given, ALSO emit *_diagnostic figures with X markers on cells "
            "the audit marked SATURATED or CRITICAL."
        ),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text()) if args.audit else {}

    # (setting, U) -> panel title -> transform label -> (band, saturated cells)
    groups: dict[tuple[str, str], dict[str, dict[str, tuple[dict, dict]]]] = {}
    for folder in saved_conditions(args.root, args.setting):
        record = load_saved_condition(folder)["pipeline"]
        condition = condition_of(record)
        title = PANELS.get((condition["collusion_strategy"], condition["monitor_arm"]))
        replicates = bootstrap_path(folder, args.variant)
        if title is None or not replicates.exists():
            continue
        band = bootstrap_band(read_replicates(replicates), args.ci)
        report = audit.get(f"{record['name']}/{args.variant}") or {}
        saturated = {
            c["usefulness"]: c["verdict"]
            for c in report.get("cells", [])
            if c["verdict"] in ("SATURATED", "CRITICAL")
        }
        group = (
            condition["setting"],
            names.short_model((record.get("models") or {}).get("U", "")),
        )
        label = TRANSFORMS.get(condition["transform"], condition["transform"])
        groups.setdefault(group, {}).setdefault(title, {})[label] = (band, saturated)

    ci_label = f"{args.ci:.0%} CI, bootstrap"
    for (setting, model), panels in sorted(groups.items()):
        out_dir = args.out / f"{setting}-{model}"
        ordered = {t: panels[t] for t in PANELS.values() if t in panels}
        for title, curves in ordered.items():
            bands = {label: band for label, (band, _) in curves.items()}
            slug = names.clean(title)
            plot_safety_usefulness_bands(
                bands,
                save_path=str(
                    out_dir / f"{slug}_safety_usefulness_bands.{PLOT_FILE_FORMAT}"
                ),
                ci_label=f"{title}, {ci_label}",
            )
            if audit:
                plot_safety_usefulness_bands(
                    bands,
                    save_path=str(
                        out_dir
                        / f"{slug}_safety_usefulness_bands_diagnostic.{PLOT_FILE_FORMAT}"
                    ),
                    ci_label=f"{title}, {ci_label}",
                    saturated={label: sat for label, (_, sat) in curves.items()},
                )
        grid = {
            title: {label: band for label, (band, _) in curves.items()}
            for title, curves in ordered.items()
        }
        plot_safety_usefulness_bands_grid(
            grid,
            save_path=str(
                out_dir / f"assembled_safety_usefulness_bands.{PLOT_FILE_FORMAT}"
            ),
            ci_label=f"{args.ci:.0%} CI",
            suptitle=f"{setting} / {model}",
        )
        if audit:
            plot_safety_usefulness_bands_grid(
                grid,
                save_path=str(
                    out_dir
                    / f"assembled_safety_usefulness_bands_diagnostic.{PLOT_FILE_FORMAT}"
                ),
                ci_label=f"{args.ci:.0%} CI",
                suptitle=f"{setting} / {model}",
                saturated={
                    title: {label: sat for label, (_, sat) in curves.items()}
                    for title, curves in ordered.items()
                },
            )


if __name__ == "__main__":
    main()
