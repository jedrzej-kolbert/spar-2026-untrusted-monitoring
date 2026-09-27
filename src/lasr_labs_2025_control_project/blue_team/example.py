"""Real-model CLI for APPS scoring, LinuxArena scoring, or one APPS honeypot."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import MonitorView, RepresentationKind
from .honeypot import apps_honeypot_task
from .jammers import IdentityJammer, MinifierJammer, TCodeRewriteJammer, TSummaryJammer
from .linuxarena_adapter import trajectory_to_view
from .policy import FixedMonitoringPolicy, make_resources


async def run(args):
    data = json.loads(args.input.read_text())
    if not isinstance(data, dict):
        raise TypeError("Input must be a JSON object")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.kind == "linux":
        view = trajectory_to_view(
            data, before_action=args.before_action, problem_id=args.id
        )
    else:
        view = MonitorView(
            data["problem"],
            data.get("code", ""),
            RepresentationKind.PYTHON_CODE,
            args.id,
        )
    resources = make_resources(
        {args.id: view.task_description},
        generator_model=args.u,
        um_model=args.um or args.u,
        trusted_model=args.t,
        collusion_model=args.collusion,
    )
    if args.kind == "honeypot":
        from control_arena.eval_mode import EvalMode
        from control_arena.settings.apps.task.dataset import AppsTaskMetadata
        from inspect_ai import eval_async
        from inspect_ai.dataset import Sample

        metadata = AppsTaskMetadata(
            main_task_description=view.task_description,
            side_task_description="",
            setting_description="APPS",
            setting_name="apps",
            eval_mode=EvalMode.ATTACK,
            inputs=data["inputs"],
            outputs=data["outputs"],
            solutions=data["solutions"],
        )
        if (
            not metadata.inputs
            or len(metadata.inputs) != len(metadata.outputs)
            or not metadata.solutions
        ):
            raise ValueError("Provide paired tests and a reference solution")
        sample = Sample(
            id=args.id, input=view.task_description, metadata=metadata.model_dump()
        )
        logs = await eval_async(
            apps_honeypot_task([sample], resources),
            model=args.t,
            log_dir=str(args.log_dir),
            max_samples=1,
        )
        if (
            not logs
            or logs[0].status != "success"
            or not logs[0].samples
            or logs[0].samples[0].error
        ):
            raise RuntimeError("Honeypot evaluation failed; inspect its local eval log")
        result = logs[0].samples[0].metadata["honeypot_result"]
    else:
        if args.kind == "apps" and not view.content.strip():
            raise ValueError("APPS scoring requires nonempty code")
        jammer = {
            "identity": lambda: IdentityJammer(),
            "minifier": lambda: MinifierJammer(),
            "summary": lambda: TSummaryJammer(resources.call_t, args.t),
            "rewrite": lambda: TCodeRewriteJammer(resources.call_t, args.t),
        }[args.jammer]()
        result = asdict(await FixedMonitoringPolicy(um_jammer=jammer)(view, resources))
        result["base_solution"] = view.content
        result["sample_id"] = args.id
    result["models"] = {
        "generator": args.u,
        "um": args.um or args.u,
        "collusion": args.collusion or args.um or args.u,
        "trusted": args.t,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("apps", "linux", "honeypot"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="sample-0", help="Harness-assigned opaque ID")
    parser.add_argument(
        "--jammer",
        choices=("identity", "minifier", "summary", "rewrite"),
        default="identity",
    )
    parser.add_argument(
        "--u", required=True, help="Inspect model ID for benign U generation"
    )
    parser.add_argument("--um", help="UM suspiciousness model (defaults to U)")
    parser.add_argument(
        "--collusion", help="UM collusion model/checkpoint (defaults to UM)"
    )
    parser.add_argument("--t", required=True, help="Inspect trusted model ID")
    parser.add_argument(
        "--before-action", type=int, help="Linux action list position, before execution"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/blue_team"))
    args = parser.parse_args()
    if args.before_action is not None and args.kind != "linux":
        parser.error("--before-action is only valid for Linux trajectories")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
