"""Planning and running a graph: what's done, what to run, and the paid gate."""

from __future__ import annotations

from dataclasses import dataclass

from lasr_labs_2025_control_project.pipeline.execute import EXECUTORS, RunContext
from lasr_labs_2025_control_project.pipeline.graph import Graph, Node, build_graph
from lasr_labs_2025_control_project.pipeline.schema import Config
from lasr_labs_2025_control_project.pipeline.store import Store
from lasr_labs_2025_control_project.tracking import PipelineContext


#: plan statuses
INPUT, DONE, WANDB, RUN, MISSING = "input", "done", "from W&B", "run", "missing"


class PlanError(RuntimeError):
    """The selected stages need outputs that don't exist."""


@dataclass
class Step:
    node: Node
    status: str


def plan(graph: Graph, cfg: Config, store: Store, ctx: RunContext) -> list[Step]:
    """Each needed node's status, recording the outputs already available.

    Walks back from the analyses: an available output (done locally, given in
    ``inputs``, or fetched from W&B) ends the walk, so nothing upstream of it
    is rerun or even looked up.
    """
    statuses: dict[str, str] = {}

    def visit(node: Node) -> None:
        if node.hash in statuses:
            return
        output = store.done(node)
        status = INPUT if node.external is not None else DONE
        if output is None and cfg.reuse == "wandb":
            output, status = store.fetch(node), WANDB
        if output is not None:
            ctx.outputs[node.hash] = output
            statuses[node.hash] = status
            return
        statuses[node.hash] = RUN if node.stage in cfg.stages else MISSING
        for dep in node.deps.values():
            visit(dep)

    for target in graph.targets:
        visit(target)
    steps = [
        Step(node, statuses[node.hash]) for node in graph.nodes if node.hash in statuses
    ]

    blocked = sorted(
        {
            f"{dep.id} (needed by {step.node.id})"
            for step in steps
            if step.status == RUN
            for dep in step.node.deps.values()
            if statuses[dep.hash] == MISSING
        }
    )
    if blocked:
        raise PlanError(
            "these outputs don't exist and their stage isn't selected:\n  "
            + "\n  ".join(blocked)
            + "\nAdd the stage to `stages`, or point `inputs.<id>` at an existing output."
        )
    return steps


def describe(steps: list[Step], store: Store) -> str:
    width = max(len(step.node.id) for step in steps)
    lines = []
    for step in steps:
        node = step.node
        where = node.external if node.external is not None else store.folder(node)
        cost = " $" if step.status == RUN and node.paid else ""
        lines.append(f"  {node.id:<{width}}  {step.status:<8}{cost:2}  {where}")
    return "\n".join(lines)


def run_pipeline(cfg: Config, pipeline: PipelineContext) -> list[Step]:
    """Plan the experiment; run what's missing unless that needs allow_paid."""
    graph = build_graph(cfg)
    store = Store(cfg.experiments_dir, cfg.setting.name)
    ctx = RunContext(cfg, store, pipeline)
    steps = plan(graph, cfg, store, ctx)
    todo = [step.node for step in steps if step.status == RUN]
    print(f"Plan for {cfg.name} ({len(todo)} to run; $ = calls paid models):")
    print(describe(steps, store))
    if any(node.paid for node in todo) and not cfg.allow_paid:
        print(
            "\nNothing was run: this plan calls paid models. Re-run with allow_paid=true."
        )
        return steps
    for index, node in enumerate(todo, start=1):
        # print, not logging: Inspect reconfigures logging once its first eval runs
        print(f"\n[{index}/{len(todo)}] {node.id} -> {store.folder(node)}", flush=True)
        try:
            ctx.outputs[node.hash] = EXECUTORS[node.stage](node, ctx)
        except Exception:
            print(
                f"\n{node.id} failed; finished outputs are kept, so re-running "
                "resumes here",
                flush=True,
            )
            raise
    if todo:
        print(f"\nDone: {len(todo)} step(s) run.")
    return steps
