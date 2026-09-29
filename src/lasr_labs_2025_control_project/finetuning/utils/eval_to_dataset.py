"""
Prepares the data for binary tuning.
Reads eval logs from generate_solutions_inspect.py or transform_solutions_inspect.py,
and saves successful solutions to a JSON file in our default Dataset format.

Usage:
    uv run eval-to-dataset --data-path <EVAL_LOG_PATH> --model-name <MODEL_NAME> --output-json <OUTPUT_JSON_PATH>
"""

import logging
from pathlib import Path

import click
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.self_rec_baselines.utils.scripts.data_models import (
    Dataset,
    ProblemSolutions,
    Solution,
)
from lasr_labs_2025_control_project.utils.utils import (
    check_sample_is_valid,
    remove_comments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def apply_python_code_format(x: str) -> str:
    """Format code with python markdown blocks if not already formatted."""
    x = x.strip()
    if not x.startswith("```python"):
        return f"```python\n{x}\n```"
    return x


def process_eval_log(
    data_path: str | Path,
    model_name: str,
    output_json_path: str | Path,
    transformed: bool = False,
    keep_comments: bool = False,
) -> Dataset:
    """Process eval log and extract successful solutions to Dataset format.

    Reads .eval files produced by generate_solutions_inspect.py or transform_solutions_inspect.py.
    Extracts solutions from metadata/store (base_solution or transformed_solution),
    problem statements from metadata (main_task_description), and filters based on scores.

    Args:
        data_path: Path to .eval log file
        model_name: Model name to assign to all solutions
        output_json_path: Path to output JSON file
        transformed: If True, use transformed_solution; if False, use base_solution
        keep_comments: If True, keep comments in base_solution (only when transformed=False)

    Returns:
        Dataset object containing successful solutions
    """
    logger.info(
        f"Starting process_eval_log with data_path={data_path}, model_name={model_name}, output_json_path={output_json_path}, "
        f"transformed={transformed}, keep_comments={keep_comments}"
    )

    data_path = Path(data_path)
    output_json_path = Path(output_json_path)

    if not data_path.exists():
        logger.error(f"Eval log file does not exist: {data_path}")
        raise FileNotFoundError(f"Eval log file does not exist: {data_path}")
    logger.info(f"Eval log file found: {data_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_json_path.parent}")

    # Read eval log using inspect_ai
    logger.info("Reading eval log...")
    log = read_eval_log(str(data_path))

    if log.samples is None:
        logger.error("No samples found in eval log")
        raise ValueError("No samples found in eval log")

    logger.info(f"Found {len(log.samples)} total samples in eval log")

    # Filter for successful solutions
    logger.info("Filtering for successful solutions...")
    successful_samples = []

    for sample in log.samples:
        # Check if sample has successful scores (both tasks for attacks, main task for honest)
        if not check_sample_is_valid(sample.scores, sample.metadata.get("eval_mode")):
            continue

        # Merge metadata and store (store takes precedence)
        sample_data = {**sample.metadata, **sample.store}
        transform_info = sample_data.get("transform_info")

        # Get solution based on transformed flag (similar to get_problem_and_solution_from_state)
        if transformed:
            # Require transformed solution - fail hard if not available
            if transform_info is None:
                raise ValueError(
                    f"Sample {sample.id} has no transform_info. "
                    f"Cannot use --transformed flag with eval logs that don't have transformed solutions. "
                    f"Remove --transformed flag or use a transformed eval log."
                )
            if (
                transform_info.get("type") == "none"
                or "transformed_solution" not in sample_data
            ):
                raise ValueError(
                    f"Sample {sample.id} does not contain transformed solution (transform_info.type={transform_info.get('type')}). "
                    f"Cannot use --transformed flag with eval logs that don't have transformed solutions. "
                    f"Remove --transformed flag or use a transformed eval log."
                )
            solution = sample_data["transformed_solution"]
        else:
            # Use base solution - fail hard if not available
            base_solution = sample_data.get("base_solution")
            if base_solution is None:
                raise ValueError(
                    f"Sample {sample.id} has no base_solution. "
                    f"This eval log may not be from generate_solutions_inspect.py or transform_solutions_inspect.py. "
                    f"Make sure the eval log contains base_solution in metadata/store."
                )
            # Optionally remove comments from base solution
            solution = (
                base_solution if keep_comments else remove_comments(base_solution)
            )

        # Get problem statement - fail hard if not available
        problem_statement = sample_data.get("main_task_description")
        if problem_statement is None:
            raise ValueError(
                f"Sample {sample.id} has no main_task_description. "
                f"This eval log may not be from generate_solutions_inspect.py or transform_solutions_inspect.py. "
                f"Make sure the eval log contains main_task_description in metadata."
            )

        # Get problem ID - try different field names
        problem_id = sample_data.get("id") or sample.id
        if isinstance(problem_id, str):
            # Try to extract numeric ID from string ID
            try:
                problem_id = int(problem_id.split("_")[-1])
            except (ValueError, IndexError):
                # If can't extract, use hash of sample id
                problem_id = hash(problem_id) % (10**8)

        successful_samples.append(
            {
                "sample_id": sample.id,
                "problem_id": problem_id,
                "problem_statement": problem_statement,
                "solution": solution,
                "transform_info": transform_info,
            }
        )

    logger.info(f"Found {len(successful_samples)} successful solutions")

    if len(successful_samples) == 0:
        logger.warning("No successful samples found!")
        return Dataset(problems=[])

    # Group solutions by problem_id
    logger.info("Grouping solutions by problem...")
    problems_dict: dict[int, dict] = {}

    for sample_data in successful_samples:
        problem_id = sample_data["problem_id"]
        transform_info = sample_data["transform_info"]

        if problem_id not in problems_dict:
            problems_dict[problem_id] = {
                "apps_problem_id": problem_id,
                "problem_statement": sample_data["problem_statement"],
                "solutions": [],
            }

        # Format solution with markdown
        if transform_info is not None and transform_info["type"] == "llm":
            formatted_solution = sample_data["solution"]
        else:
            formatted_solution = apply_python_code_format(sample_data["solution"])

        problems_dict[problem_id]["solutions"].append(
            Solution(model=model_name, solution=formatted_solution)
        )

    # Build Dataset
    logger.info("Building Dataset object...")
    dataset = Dataset(
        problems=[
            ProblemSolutions(
                apps_problem_id=problem_data["apps_problem_id"],
                problem_statement=problem_data["problem_statement"],
                solutions=problem_data["solutions"],
            )
            for problem_data in problems_dict.values()
        ]
    )

    num_solutions = sum(len(p.solutions) for p in dataset)
    logger.info(
        f"Dataset contains {len(dataset)} problems with {num_solutions} total solutions"
    )

    # Save to JSON
    logger.info(f"Writing dataset to JSON: {output_json_path}")
    dataset.to_json(str(output_json_path))
    logger.info("Dataset successfully written to JSON.")

    return dataset


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to eval log file (.eval) from generate_solutions_inspect.py or transform_solutions_inspect.py",
    required=True,
)
@click.option(
    "--model-name",
    type=str,
    help="Model name to assign to all solutions (default: auto-detect from eval log)",
    required=False,
    default=None,
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to output JSON file",
    required=True,
)
@click.option(
    "--transformed",
    is_flag=True,
    default=False,
    help="Use transformed_solution from eval log instead of base_solution",
)
@click.option(
    "--keep-comments",
    is_flag=True,
    default=False,
    help="Keep comments in base_solution (only applies when --transformed is not set)",
)
def main(
    data_path: str | Path,
    model_name: str | None,
    output_json: str | Path,
    transformed: bool,
    keep_comments: bool,
):
    """
    Convert eval logs to Dataset JSON for binary fine-tuning.

    Works with .eval files produced by generate_solutions_inspect.py or transform_solutions_inspect.py.
    Automatically reads solutions, problem statements, and scores from metadata/store.
    Supports both APPS and BigCodeBench datasets.

    Model name is auto-detected from the eval log metadata. Override with --model-name if needed.
    Use --transformed to extract transformed solutions (from transform_solutions_inspect.py).
    Use --remove-comments to strip comments from base solutions (only when --transformed is not set).
    """
    try:
        # Auto-detect model name if not provided
        if model_name is None:
            logger.info("Model name not provided, auto-detecting from eval log...")
            log = read_eval_log(str(data_path))

            # Try to get generator model from sample store (works for both generate and transform)
            if log.samples and len(log.samples) > 0:
                sample = log.samples[0]
                # Merge metadata and store (store takes precedence)
                sample_data = {**sample.metadata, **sample.store}

                # Try to get generator model from generator_info
                generator_info = sample_data.get("generator_info")
                if generator_info and isinstance(generator_info, dict):
                    model_info = generator_info.get("model_info")
                    if model_info and isinstance(model_info, dict):
                        model_name = model_info.get("model")
                        if model_name:
                            logger.info(
                                f"Auto-detected generator model from sample metadata: {model_name}"
                            )

                # Fallback to eval.model if generator_info not found
                if not model_name:
                    model_name = log.eval.model
                    logger.info(f"Auto-detected model from eval log: {model_name}")
            else:
                # No samples, use eval.model as fallback
                model_name = log.eval.model
                logger.info(f"Auto-detected model from eval log: {model_name}")

        dataset = process_eval_log(
            data_path,
            model_name,
            output_json,
            transformed,
            keep_comments,
        )
        num_solutions = sum(len(p.solutions) for p in dataset)
        logger.info(f"Wrote {num_solutions} solutions to {output_json}")
    except Exception:
        logger.exception("An error occurred during processing.")
        raise


if __name__ == "__main__":
    main()
