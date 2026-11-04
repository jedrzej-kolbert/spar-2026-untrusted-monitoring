"""Save a W&B view of a project's RL runs, sections in reading order.

train, validation, test and the ICRL tables open at the top; then progress and the
_detail sections, collapsed. The Workspaces API cannot edit a personal workspace,
so this saves a named view (in the project's Views list); in it, "Copy to my
workspace" makes it your default. Rerun after new metrics appear; with --url it
updates that view instead of saving a new one.

    uv run --with wandb-workspaces python experiments/collusion-rl-v2/workspace.py \\
        <project> [--url <saved view url>]

--curves saves a second view instead: the test safety-usefulness curves of the
scoring's game runs, grouped by run.arm and channel (condition.transform).
"""

from __future__ import annotations


import click

ENTITY = "spar-untrusted-monitoring-collusion"
ORDER = (
    "train",
    "validation",
    "test",
    "icrl",
    "progress",
    "train_detail",
    "validation_detail",
    "test_detail",
)
OPEN = {"train", "validation", "test", "icrl"}
# A plot_table chart is stored under "<key>_table".
ROC_FIELDS = {"x": "fpr", "y": "tpr", "class": "class"}


def panel(key: str, kind: str) -> object:
    import wandb_workspaces.reports.v2 as wr

    if kind == "table-file" and key.endswith("_table") and "roc_" in key:
        return wr.CustomChart(
            query={"summaryTable": {"tableKey": key}},
            chart_name="wandb/area-under-curve/v0",
            chart_fields=ROC_FIELDS,
            chart_strings={
                "title": key.removesuffix("_table"),
                "x-axis-title": "False positive rate",
                "y-axis-title": "True positive rate",
            },
        )
    if kind == "table-file" and key.endswith("safety_usefulness_table"):
        return wr.CustomChart(
            query={"summaryTable": {"tableKey": key}},
            chart_name="wandb/line/v0",
            chart_fields={"x": "usefulness", "y": "safety", "stroke": "class"},
            chart_strings={"title": "Test safety vs usefulness"},
        )
    if kind == "table-file":
        return wr.WeavePanelSummaryTable(table_name=key)
    # Test numbers are logged once: a value reads better than a one-point line.
    if key.startswith(("test/", "test_detail/")):
        return wr.ScalarChart(title=key, metric=key)
    return wr.LinePlot(title=key, y=[key])


def curves_view(project: str, name: str) -> object:
    """Test safety-usefulness curves of the game runs, one line per arm and channel.

    Each game run logs its sweep as rows of (<variant>/usefulness, <variant>/safety);
    the lines are their mean over seeds, the band min to max. report_test puts
    run.arm on the game runs.
    """
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    panels = [
        wr.LinePlot(
            title=f"Test safety vs usefulness, {variant}",
            x=f"{variant}/usefulness",
            y=[f"{variant}/safety"],
            title_x="usefulness",
            title_y="safety",
            range_x=(0, 1),
            range_y=(0, 1),
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            line_marks={f"{variant}/safety": "points"},
        )
        for variant in ("all_monitors", "um_only")
    ]
    return ws.Workspace(
        entity=ENTITY,
        project=project,
        name=name,
        sections=[ws.Section(name="safety_usefulness", panels=panels, is_open=True)],
        runset_settings=ws.RunsetSettings(
            filters=[
                ws.Metric("JobType") == "game",
                ws.Metric("State") == "finished",
                ws.Config("run.arm").isin(
                    ["base", "sft", "i1-base", "i2", "g1-base", "while-loop"]
                ),
            ],
            groupby=[ws.Config("run.arm"), ws.Config("condition.transform")],
        ),
    )


@click.command()
@click.argument("project")
@click.option("--url", default=None, help="A saved view to update.")
@click.option("--name", default=None, help="Name of a new view.")
@click.option(
    "--curves",
    is_flag=True,
    help="Save the test safety-usefulness view instead (game runs by arm, channel).",
)
def main(project: str, url: str | None, name: str | None, curves: bool) -> None:
    import wandb
    import wandb_workspaces.workspaces as ws

    if curves:
        view = curves_view(project, name or "Test safety-usefulness")
        if url:
            saved = ws.Workspace.from_url(url)
            saved.sections, saved.runset_settings = view.sections, view.runset_settings
            saved.save()
            view = saved
        else:
            view = view.save_as_new_view()
        click.echo(view.url)
        return
    name = name or "RL runs"
    keys: dict[str, str] = {}
    # lazy=False: lazy runs (wandb >= 0.22) carry no history_keys.
    for run in wandb.Api().runs(f"{ENTITY}/{project}", lazy=False):
        for key, info in run.history_keys["keys"].items():
            keys.setdefault(key, info["typeCounts"][0]["type"])
    sections = []
    for section in ORDER:
        names = sorted(k for k in keys if k.split("/")[0] == section)
        if names:
            sections.append(
                ws.Section(
                    name=section,
                    panels=[panel(k, keys[k]) for k in names],
                    is_open=section in OPEN,
                )
            )
    if url:
        view = ws.Workspace.from_url(url)
        view.sections = sections
        view.save()
    else:
        view = ws.Workspace(
            entity=ENTITY, project=project, name=name, sections=sections
        )
        view = view.save_as_new_view()
    click.echo(f"{view.url}: {[s.name for s in sections]}")


if __name__ == "__main__":
    main()
