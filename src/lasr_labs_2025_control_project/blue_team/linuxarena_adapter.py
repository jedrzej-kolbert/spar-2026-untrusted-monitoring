"""Allowlisted projection of current or legacy LinuxArena trajectory JSON."""

import json
from collections.abc import Mapping

from .contracts import MonitorView, RepresentationKind


def trajectory_to_view(
    trajectory: Mapping,
    *,
    before_action: int | None = None,
    task_description: str | None = None,
    problem_id: str = "",
) -> MonitorView:
    """`before_action=k`: executed actions [0,k) plus proposal k, no result k.

    k is a zero-based list position, never the optional/untrusted action index.
    Omit it for posthoc full-trajectory monitoring. Opaque evaluation IDs should
    be supplied by the harness; source names/IDs may encode attack labels.
    """
    actions = trajectory.get("actions")
    if not isinstance(actions, list):
        raise TypeError("Expected a LinuxArena actions list")
    if before_action is not None and (
        type(before_action) is not int or not 0 <= before_action < len(actions)
    ):
        raise ValueError("before_action must select an existing action position")
    main = trajectory.get("main_task")
    if task_description is None:
        if isinstance(main, Mapping):
            task_description = main.get("description")
        elif isinstance(
            main, str
        ):  # Legacy exports keep descriptions at the top level.
            task_description = trajectory.get("main_task_description") or main
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("A public task description is required")

    visible = []
    selected = actions if before_action is None else actions[: before_action + 1]
    for position, action in enumerate(selected):
        if (
            not isinstance(action, Mapping)
            or not isinstance(action.get("function"), str)
            or not isinstance(action.get("arguments"), dict)
        ):
            raise TypeError(f"Malformed action at position {position}")
        item = {
            "position": position,
            "function": action["function"],
            "arguments": action["arguments"],
        }
        if before_action is None or position < before_action:
            item.update(
                output=action.get("output"), output_error=action.get("output_error")
            )
        visible.append(item)
    environment = trajectory.get("environment")
    environment_description = (
        environment.get("description", "")
        if isinstance(environment, Mapping)
        else trajectory.get("environment_description") or ""
    )
    # Serialize at the boundary: later mutations of source dicts cannot alter the view.
    content = json.dumps(
        {"environment": environment_description, "actions": visible},
        ensure_ascii=False,
        allow_nan=False,
    )
    kind = (
        RepresentationKind.TRAJECTORY
        if before_action is None
        else RepresentationKind.TRAJECTORY_PREFIX
    )
    return MonitorView(task_description, content, kind, problem_id)
