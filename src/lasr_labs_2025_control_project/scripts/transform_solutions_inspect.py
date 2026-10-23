"""Transform stage: signal jamming (minifier or LLM paraphraser) over solutions.

Reads a generation's .eval log and keeps each sample's original scores, so the
transformed log can be monitored like the original. The pipeline
(``lasr_labs_2025_control_project.pipeline``) calls ``run_transform``.
"""

import shutil
from typing import Any, Optional

import python_minifier
from inspect_ai import Epochs, Task, eval, task
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.util import store

from lasr_labs_2025_control_project.scripts.helpers.eval_to_dataset import (
    eval_to_dataset,
)
from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    TransformInfo,
    TransformType,
)
from lasr_labs_2025_control_project.tracking import StageRun, input_counts
from lasr_labs_2025_control_project.utils import tinker_provider  # noqa: F401  (registers tinker/ provider)
from lasr_labs_2025_control_project.utils.config_loader import TransformRunConfig
from lasr_labs_2025_control_project.utils.utils import (
    is_python_solution,
    remove_comments,
)


@solver
def minify_solution() -> Solver:
    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        assert isinstance(state.input, str), (
            "State input must be a string, check your .eval file"
        )
        code = state.input.replace("```python\n", "").replace("```", "")
        output = python_minifier.minify(
            code,
            remove_literal_statements=True,
            hoist_literals=True,
            combine_imports=True,
            rename_locals=True,
            rename_globals=True,
            remove_annotations=True,
            remove_object_base=True,
            convert_posargs_to_args=True,
            remove_asserts=True,
            remove_debug=True,
            remove_explicit_return_none=True,
            constant_folding=True,
        )

        response = ChatMessageAssistant(content=output)
        state.messages.append(response)
        state.output = ModelOutput.from_message(response)

        store().set("transformed_solution", output)
        store().set(
            "transform_info",
            TransformInfo(type=TransformType.MINIFIER, model_info=None),
        )

        return state

    return solve


@solver
def summarize_solution(
    model_info: ModelInfo,
    remove_comments_from_solution: bool = False,
    config: Optional[GenerateConfig] = None,
):
    """
    Summarizes the solution using the summarizer model.

    Args:
        model_info: The summarizer model and prompts; the base (user) prompt
            contains a `{solution}` placeholder.
        config: Sampling settings (seed, temperature) for the summarizer.
    """
    model = get_model(model_info.model)

    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Get solution from the dataset"""
        # Get the solution from messages

        assert isinstance(state.input, str), (
            "State input must be a string, not list[ChatMessage], check your .eval file"
        )
        code = state.input
        state.output = await model.generate(
            [
                ChatMessageSystem(content=model_info.system_prompt),
                ChatMessageUser(
                    content=model_info.base_prompt.format(
                        solution=remove_comments(code)
                        if remove_comments_from_solution
                        else code
                    )
                ),
            ],
            config=config or GenerateConfig(),
        )

        state.messages.append(state.output.message)

        summarizer_info = ModelInfo(
            system_prompt=model_info.system_prompt,
            base_prompt=model_info.base_prompt,
            model=model.name,
        )

        store().set("transformed_solution", state.output.completion)
        store().set(
            "transform_info",
            TransformInfo(
                type=TransformType.LLM,
                model_info=summarizer_info,
                remove_comments=remove_comments_from_solution,
            ),
        )

        return state

    return solve


def create_named_scorer(scorer_name: str) -> Scorer:
    @scorer(metrics=[], name=scorer_name)
    def named_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return state.metadata["scores"][scorer_name]

        score.__name__ = scorer_name
        score.__qualname__ = scorer_name

        return score

    return named_scorer()


@task
def transform_solutions(
    run: TransformRunConfig,
    config: Optional[GenerateConfig] = None,
) -> Task:
    """
    Applies paraphrasing (summarisation) or minification to solutions in the eval log.
    Keeps the same scorers and their values as in the eval log.

    Args:
        eval_log_path: Path to the eval log.
        transform_type: Type of transform to apply.
        summarizer_system_prompt: System prompt to use for paraphrasing.
        summarizer_base_prompt: Base (user) prompt to use for paraphrasing.

    """
    dataset = eval_to_dataset(run.eval_path)
    if any(
        not is_python_solution((sample.metadata or {}).get("setting_name"))
        for sample in dataset
    ):
        # The minifier parses Python and the summariser prompts describe code.
        raise NotImplementedError(
            "Paraphrasing and minification are not available for BashArena yet"
        )

    scorer_names = []
    if dataset[0].metadata and "scores" in dataset[0].metadata:
        scorer_names = list(dataset[0].metadata["scores"].keys())
    else:
        raise ValueError("No scores found in the eval log samples' metadata")

    if run.transform.type == TransformType.LLM:
        assert run.transform.model_info is not None, (
            "Model info is required for LLM transform"
        )
        solver = summarize_solution(
            run.transform.model_info,
            run.transform.remove_comments,
            config,
        )
    elif run.transform.type == TransformType.MINIFIER:
        solver = minify_solution()
    else:
        raise ValueError(f"Invalid transform type: {run.transform.type}")

    return Task(
        dataset=dataset,
        solver=solver,
        metadata={
            "transform_info": TransformInfo(
                type=run.transform.type,
                model_info=run.transform.model_info,
                remove_comments=run.transform.remove_comments,
            ),
        },
        scorer=[create_named_scorer(scorer_name) for scorer_name in scorer_names],
    )


def run_transform(
    run: TransformRunConfig,
    tracker: StageRun,
    *,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
    max_connections: int = 48,
) -> Any:
    """Transform ``run.eval_path`` under ``tracker``; save to ``run.save_path``."""
    config = GenerateConfig(seed=seed, temperature=temperature)
    with tracker:
        result = eval(
            tasks=transform_solutions(run, config),
            epochs=Epochs(1, ["pass_at_1"]),
            max_connections=max_connections,
            max_sandboxes=max_connections,
            reasoning_tokens=0,
            reasoning_effort="minimal",  # openai models
            limit=run.limit,
            metadata=tracker.eval_metadata(),
        )
        run.save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result[0].location, run.save_path)
        tracker.finish_eval(
            result[0],
            run.save_path,
            limit=run.limit,
            metadata={"label": run.label, "transform": run.transform.type},
            extra_counts=input_counts(
                run.eval_path, result[0].eval.dataset.samples or 0
            ),
        )
    return result[0]
