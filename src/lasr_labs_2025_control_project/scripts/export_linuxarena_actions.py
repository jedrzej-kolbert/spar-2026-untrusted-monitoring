"""Export native LinuxArena views with the separate Control Tower environment.

Manifest format: [{"path": "run.eval", "actor_id": "deepseek-ai/DeepSeek-V3.1",
                   "population": "attacks"}]. Paths may be manifest-relative.
Populations are attacks, honest, and honeypots. No model calls are made here.
"""

import argparse
import json
from pathlib import Path

from lasr_labs_2025_control_project.utils.linuxarena_actions import (
    ActionSource,
    export_actions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-actor", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cot-level", choices=("none", "text", "cot"), default="none")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sources = [
        ActionSource(
            path=(args.manifest.resolve().parent / item["path"]).resolve(),
            actor_id=item["actor_id"],
            population=item["population"],
        )
        for item in manifest
    ]
    summary = export_actions(
        sources,
        args.target_actor,
        args.output_dir,
        cot_level=args.cot_level,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
