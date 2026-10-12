"""Offline JSONL preparation/comparison and explicitly opt-in monitor scoring."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lasr_labs_2025_control_project.utils.action_records import (  # noqa: E402
    compare_scores,
    prepare_exports,
)
from lasr_labs_2025_control_project.utils.action_scoring import score_prepared  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--exports", nargs="+", type=Path, required=True)
    prepare.add_argument("--target-actor", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--include-honest", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--prepared", type=Path, required=True)
    score.add_argument(
        "--models",
        type=Path,
        required=True,
        help="JSON mapping recognition/trusted_suspiciousness/untrusted_suspiciousness to model IDs",
    )
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--split", choices=("train", "val", "test"), default="test")
    score.add_argument("--concurrency", type=int, default=1)
    score.add_argument("--allow-paid", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--prepared", type=Path, required=True)
    compare.add_argument("--scores", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--split", choices=("train", "val", "test"), default="test")
    compare.add_argument("--cohort", choices=("all", "baseline"), default="all")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_exports(
            args.exports, args.target_actor, args.output, args.include_honest
        )
    elif args.command == "score":
        result = asyncio.run(
            score_prepared(
                args.prepared,
                args.output,
                json.loads(args.models.read_text()),
                args.split,
                args.allow_paid,
                args.concurrency,
            )
        )
    else:
        result = compare_scores(
            args.prepared, args.scores, args.output, args.split, args.cohort
        )
    print(json.dumps(result, sort_keys=True))
    if args.command == "compare" and not result["complete"]:
        raise SystemExit(1)
    if args.command == "score" and result.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
