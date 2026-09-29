from pathlib import Path
from typing import Optional

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid


def eval_to_dataset(
    eval_log_path: Path | str, user_prompt: Optional[str] = None
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
            scores, sample.metadata.get("eval_mode")
        )
        if not is_valid_sample:
            continue

        base_solution = metadata.get("base_solution", None)
        assert (
            base_solution is not None
        ), "No base_solution found in the sample store or metadata"

        solution = metadata.get("transformed_solution", base_solution)

        dataset.append(
            Sample(
                input=user_prompt.format(solution),
                metadata={**metadata, "scores": scores},
                id=sample.id,
            )
        )
    print(f"Converted to dataset with {len(dataset)} valid samples")

    return MemoryDataset(dataset)
