from pathlib import Path
from typing import Optional

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.utils.utils import (
    ACTION_VIEW,
    TRAJECTORY_VIEW,
    ActionViewOptions,
    action_views,
    check_sample_is_valid,
    select_action_views,
)


def eval_to_dataset(
    eval_log_path: Path | str,
    user_prompt: Optional[str] = None,
    view: str = TRAJECTORY_VIEW,
    action_options: Optional[ActionViewOptions] = None,
) -> MemoryDataset:
    """
    Converts an eval log into a MemoryDataset.
    Saves all the contents from metadata, store and scores to sample metadata.
    If transformed_solution is present in the store, it is used as the solution.
    Otherwise, the base_solution is used.
    Keeps the original sample_id from the eval log.

    Args:
        eval_log_path: Path to the eval log.
        user_prompt: Optional. Prompt to wrap the solution in. If not provided, the solution is passed as is.
        view: "trajectory" (one sample per trajectory, base_solution as stored) or
            "action" (one sample per rated action, base_solution = that action's
            per-action view rendered from the sample's messages). Action samples
            get id "<trajectory id>#a<k>" and metadata trajectory_id, action_index,
            n_actions, n_actions_kept and solution_view; downstream plots take
            the max over a trajectory's actions.
        action_options: Options for the per-action views (view == "action").

    Returns:
        MemoryDataset: The dataset.
    """
    log = read_eval_log(eval_log_path)
    assert log.samples is not None, "No samples found in the eval log"
    print(f"Found {len(log.samples)} samples in the eval log")

    if user_prompt is None:
        user_prompt = "{}"

    dataset = []
    for sample in log.samples:
        metadata = {
            **sample.metadata,
            **sample.store,
        }  # order matters - store overrides metadata
        scores = sample.scores
        is_valid_sample = check_sample_is_valid(
            scores,
            sample.metadata.get("eval_mode"),
            base_solution=metadata.get("base_solution"),
            errored=getattr(sample, "error", None) is not None,
        )
        if not is_valid_sample:
            continue

        base_solution = metadata.get("base_solution", None)
        assert (
            base_solution is not None
        ), "No base_solution found in the sample store or metadata"

        if view == ACTION_VIEW:
            if "transformed_solution" in metadata:
                raise ValueError(
                    "Per-action views are rendered from the agent's messages; "
                    "transformed (paraphrased/minified) trajectories are not supported"
                )
            views = action_views(sample.messages, action_options)
            kept = select_action_views(views, sample.id, action_options)
            for index, action_view in kept:
                dataset.append(
                    Sample(
                        input=user_prompt.format(action_view),
                        metadata={
                            **metadata,
                            "scores": scores,
                            "base_solution": action_view,
                            "solution_view": ACTION_VIEW,
                            "trajectory_id": sample.id,
                            "action_index": index,
                            "n_actions": len(views),
                            "n_actions_kept": len(kept),
                        },
                        id=f"{sample.id}#a{index}",
                    )
                )
            continue
        if view != TRAJECTORY_VIEW:
            raise ValueError(f"Unknown view {view!r}")

        solution = metadata.get("transformed_solution", base_solution)

        dataset.append(
            Sample(
                input=user_prompt.format(solution),
                metadata={**metadata, "scores": scores},
                id=sample.id,
            )
        )
    print(f"Converted to dataset with {len(dataset)} valid samples (view={view})")

    return MemoryDataset(dataset)
