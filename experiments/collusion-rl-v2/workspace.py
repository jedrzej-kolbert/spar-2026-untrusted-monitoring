"""Save a W&B view of a project's RL runs, sections in reading order.

train, validation, test and the ICRL tables open at the top; then progress and the
_detail sections, collapsed. The Workspaces API cannot edit a personal workspace,
so this saves a named view (in the project's Views list); in it, "Copy to my
workspace" makes it your default. Rerun after new metrics appear; with --url it
updates that view instead of saving a new one.

    uv run --with wandb-workspaces python experiments/collusion-rl-v2/workspace.py \\
        <project> [--url <saved view url>]
"""

from __future__ import annotations

import os

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
    if kind == "table-file":
        return wr.WeavePanelSummaryTable(table_name=key)
    # Test numbers are logged once: a value reads better than a one-point line.
    if key.startswith(("test/", "test_detail/")):
        return wr.ScalarChart(title=key, metric=key)
    return wr.LinePlot(title=key, y=[key])


@click.command()
@click.argument("project")
@click.option("--url", default=None, help="A saved view to update.")
@click.option("--name", default="RL runs", help="Name of a new view.")
def main(project: str, url: str | None, name: str) -> None:
    os.environ.pop("WANDB_API_KEY", None)  # the ~/.netrc key has team access
    import wandb
    import wandb_workspaces.workspaces as ws

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
