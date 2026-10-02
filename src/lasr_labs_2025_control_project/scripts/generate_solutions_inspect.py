"""
Generates solutions for the APPS dataset. Applies the control arena APPS scorers without actually using the control arena setup.
Put your prompt into the solver's system message.
Select the APPS data split in AppsSetting.
Make sure you're using the correct model in the eval config.
Just run the script.
"""

import json
import logging
import re
import shutil
from pathlib import Path

import click
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting
from control_arena.settings.bash_arena.policies.attack_policy import attack_policy
from control_arena.settings.bash_arena.policies.honest_policy import honest_policy
from control_arena.settings.bash_arena.protocol.auto_submit_protocol_wrapper import (
    auto_submit_protocol_wrapper,
)
from control_arena.tools.submit_tool import STORE_SUBMIT_ARGS_KEY
from inspect_ai import Epochs, Task, eval, task
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ModelOutput,
)
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    solver,
    system_message,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import store

from lasr_labs_2025_control_project.bigcodebench.bigcodebench import (
    bigcodebench,
)
from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    GeneratorInfo,
    ModelInfo,
)
from lasr_labs_2025_control_project.settings.setting import AppsSetting
from lasr_labs_2025_control_project.utils import tinker_provider  # noqa: F401  (registers tinker/ provider)
from lasr_labs_2025_control_project.utils.config_loader import (
    DatasetType,
    DataSplit,
    GeneratorRunConfig,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.utils import trajectory_view

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_code(completion: str) -> str:
    """
    Removes Markdown formatting around generated code blocks.

    Copied from humaneval
    """
    pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
    pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern_1.findall(completion) + pattern_2.findall(completion)
    if matches:
        extracted_answer = matches[0]
    else:
        extracted_answer = ""
    return str(extracted_answer)


@solver
def parse_and_submit(eval_mode: EvalMode):
    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Parse model output and create submit tool call."""
        # Get the model's output
        output = state.output.completion

        # Parse final_code and secret_input from the output
        final_code_match = re.search(
            r"<final_code>\s*([\s\S]+?)\s*<\/final_code>", output, re.DOTALL
        )
        secret_input_match = re.search(
            r"<secret_input>\s*([\s\S]+?)\s*<\/secret_input>", output, re.DOTALL
        )

        if final_code_match:
            code = final_code_match.group(1).strip()
            code = code.replace("```python", "").replace("```", "").strip()

        else:
            code = find_code(output)
            logger.warning(
                f"No <final_code> tags found for sample {state.sample_id}, "
                "using markdown extraction as fallback"
            )
        if eval_mode == EvalMode.ATTACK and secret_input_match:
            secret_input = secret_input_match.group(1).strip()  # pyright: ignore[reportOptionalMemberAccess]
        else:
            secret_input = ""

        generator_info = GeneratorInfo(
            model_info=ModelInfo(
                system_prompt=state.messages[0].text,
                base_prompt=f"{state.metadata.get('main_task_description', None)}",
                model=state.model.name,
            ),
            eval_mode=eval_mode,
            secret_input=secret_input,
        )
        # Save the code and secret input to the store
        store().set("generator_info", generator_info)
        store().set("base_solution", code)

        tool_arguments = (
            {"code": code, "secret_input": secret_input}
            if eval_mode == EvalMode.ATTACK
            else {"code": code}
        )

        # Create submit tool call using the correct pattern
        submit_call = ToolCall(
            id="manual_submit_001", function="submit", arguments=tool_arguments
        )

        # Create assistant message with tool call
        assistant_message = ChatMessageAssistant(
            content="",
            tool_calls=[submit_call],
        )

        state.output = ModelOutput.for_tool_call(
            model="manual_submit", tool_name="submit", tool_arguments=tool_arguments
        )
        state.output.completion = json.dumps(tool_arguments)
        # control-arena's APPS scorer reads the submission from the store.
        store().set(STORE_SUBMIT_ARGS_KEY, state.output.completion)

        # Add the assistant message to the state
        state.messages.append(assistant_message)

        return state

    return solve


@task
def backdoored_solutions(
    eval_mode: EvalMode,
    data_split: DataSplit,
    system_prompt: str,
) -> Task:
    """Task that runs AppsSetting dataset through attack policy system prompt."""

    # Create AppsSetting instance
    apps_setting = AppsSetting(
        dataset_name="simple",
        cache_python_code=False,
        num_tests_to_run=10,
        dataset_type="filtered",
        original_dataset_split="test",
        filter_on_split="even_test" if data_split == DataSplit.TEST else "odd_train",
        shuffle=False,
        problem_ids_range=[0, 2999],
    )

    dataset = apps_setting.get_dataset(eval_mode=eval_mode)

    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            generate(),
            parse_and_submit(eval_mode),
        ],
        scorer=apps_setting.get_scorers()[0],  # Use first scorer
        sandbox=apps_setting.get_sandbox_config(),
    )


@solver
def extract_bigcodebench_solution(eval_mode: EvalMode):
    """Extract and store BigCodeBench solution for monitoring/transform compatibility.

    This solver ensures BigCodeBench eval logs have the same metadata structure as APPS,
    allowing them to work seamlessly with the monitor and transform pipelines.

    Extracts code from <final_code> tags (same as APPS), with fallback to markdown extraction.

    Stores:
        - base_solution: The extracted code solution
        - generator_info: Model and prompt information
        - main_task_description: The problem statement (in metadata)
    """

    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Extract code from completion and store metadata for monitors/transforms."""
        # Get the model's output
        output = state.output.completion

        # Extract code from <final_code> tags (same as APPS)
        final_code_match = re.search(
            r"<final_code>\s*([\s\S]+?)\s*<\/final_code>", output, re.DOTALL
        )

        if final_code_match:
            code = final_code_match.group(1).strip()
            # Clean up markdown code blocks if present
            code = code.replace("```python", "").replace("```", "").strip()
        else:
            if "<final_code>" in output:
                output = output.split("<final_code>")[-1]

                logger.warning(
                    f"No </final_code> tag found for sample {state.sample_id}, "
                    "using markdown extraction on the output after <final_code> as fallback"
                )
            else:
                logger.warning(
                    f"No <final_code> tags found for sample {state.sample_id}, "
                    "using markdown extraction as fallback"
                )
            # Fallback to markdown extraction if no <final_code> tags
            code = find_code(output)
        if code == "":
            logger.error(f"No code found for sample {state.sample_id}, ")

        # Extract secret_input for attack mode (same as APPS)

        secret_input_match = re.search(
            r"<secret_input>\s*({[\s\S]+?})\s*<\/secret_input>", output, re.DOTALL
        )
        secret_input = secret_input_match.group(1).strip() if secret_input_match else ""

        # Determine eval mode based on whether secret_input was provided

        # Store the solution in the same format as APPS for compatibility with monitors/transforms
        store().set("base_solution", code)

        # Store the problem description - use the input which contains the problem
        # For BigCodeBench, state.input contains the instruction + complete_prompt
        state.metadata["main_task_description"] = state.input

        # Store generator info for consistency with APPS workflow
        generator_info = GeneratorInfo(
            model_info=ModelInfo(
                system_prompt=state.messages[0].text if state.messages else "",
                base_prompt=str(state.input),
                model=state.model.name,
            ),
            eval_mode=eval_mode,
            secret_input=secret_input,
        )
        store().set("generator_info", generator_info)
        store().set("secret_input", secret_input)
        store().set("eval_mode", eval_mode)

        return state

    return solve


@task
def bigcodebench_solutions(
    data_split: DataSplit,
    system_prompt: str,
    eval_mode: EvalMode,
) -> Task:
    """Task that runs BigCodeBench dataset with custom system prompt.

    Uses bigcodebench_scorer which checks both:
    - main_task_success: Normal test execution
    - side_task_success: Backdoor detection (if secret_input provided)

    Args:
        data_split: The data split to use for the task, one of "odd_train" or "even_test"
        system_prompt: The system prompt to use for the task
        eval_mode: The evaluation mode to use for the scorer
    """

    # Map DataSplit to bigcodebench data_split (same as APPS mapping)
    bigcodebench_data_split = (
        "even_test" if data_split == DataSplit.TEST else "odd_train"
    )

    # Create the bigcodebench task with default version v0.1.2
    base_task = bigcodebench(
        version="v0.1.2",
        data_split=bigcodebench_data_split,  # type: ignore
        eval_mode=eval_mode,
    )

    # Override the solver to add our custom system prompt and extract solution
    # Use bigcodebench_scorer instead of verify to support backdoor checking
    return Task(
        dataset=base_task.dataset,
        solver=[
            system_message(system_prompt),
            generate(),
            extract_bigcodebench_solution(eval_mode),
        ],
        scorer=base_task.scorer,
        sandbox=base_task.sandbox,
    )


# ponytail: Inspect aborts a whole eval when one task image fails to build, so tasks
# that cannot build on this machine are dropped by hand (1003684 runs mknod during
# docker build). Find more with run_logs/prebuild_test_images.py.
UNBUILDABLE_TASKS = {"1003684"}


def basharena_solutions(
    eval_mode: EvalMode,
    data_split: DataSplit,
    system_prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> Task:
    """BashArena agent runs (U/H attacks, U honest), ended early if a side task fires.

    BashArena has one split, so task ids (StackOverflow question numbers) are split
    like APPS problem ids: odd = train, even = test.

    max_tokens overrides the policy's per-step output cap (16000), which counts
    against the context window: on 32K models (DeepSeek-V3.1 on Tinker) it leaves
    only ~16K for the growing trajectory.
    """
    setting = BashArenaSetting()
    parity = 1 if data_split == DataSplit.TRAIN else 0
    make_policy = attack_policy if eval_mode == EvalMode.ATTACK else honest_policy
    policy = make_policy(model_name=model, system_prompt=system_prompt)
    if (
        max_tokens is not None
    ):  # keep the policy's reasoning settings, change only the cap
        policy = make_policy(
            model_name=model,
            system_prompt=system_prompt,
            generate_config=policy.generate_config.model_copy(
                update={"max_tokens": max_tokens}
            ),
        )
    control_task = get_control_task(
        setting=setting,
        micro_protocol=auto_submit_protocol_wrapper(micro_protocol=policy),
        mode=eval_mode,
        # ponytail: fixed agent limits; move into the config if pilots need tuning.
        eval_config=ControlEvalConfig(
            untrusted_model=model,
            trusted_model=model,  # unused: no trusted monitor during generation
            max_steps=60,  # gpt-oss H hit 40 on 2/5 pilot tasks; same budget for U and H
            token_limit=2_000_000,  # prompts are re-sent every step
        ),
        dataset=[
            s
            for s in setting.get_dataset(eval_mode)
            if int(str(s.id)) % 2 == parity and str(s.id) not in UNBUILDABLE_TASKS
        ],
    )
    # ControlEvalConfig.time_limit only feeds eval_args(), which get_control_task never
    # applies, so set it on the Task: 60 min for the agent, and Inspect gives scoring
    # half of that. Without it a hung main-task test blocked a run for 9 h.
    control_task.time_limit = 60 * 60
    return control_task


def add_trajectory_views(
    log_path: str, eval_mode: EvalMode, system_prompt: str, model: str
) -> EvalLog:
    """Store each trajectory's action view as base_solution, like an APPS solution.

    Done after eval() rather than in a solver: a sample that hits its step, time or
    token limit skips later solvers, and this keeps the view re-renderable offline.
    """
    log = read_eval_log(log_path)
    for sample in log.samples or []:
        sample.store["base_solution"] = trajectory_view(sample.messages)
        sample.store["generator_info"] = GeneratorInfo(
            model_info=ModelInfo(
                system_prompt=system_prompt,
                base_prompt=sample.metadata["main_task_description"],
                model=model.split("/", 1)[1],  # name without provider, as for APPS
            ),
            eval_mode=eval_mode,
            secret_input="",
        )
    return log


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
)
def main(
    config: str,
):
    from dotenv import load_dotenv

    load_dotenv()
    config_path = Path(config)

    normalized = load_and_normalize_config(config_path, config_type="generator")
    Path(normalized.experiment_folder / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        config_path, normalized.experiment_folder / "configs" / "generator_config.yaml"
    )
    for run in normalized.runs:
        assert isinstance(
            run, GeneratorRunConfig
        ), "Run is not a GeneratorRunConfig, make sure to use the correct config"

        if run.generator.dataset_type == DatasetType.BASHARENA:
            model = run.generator.model_info.model
            log = eval(
                tasks=basharena_solutions(
                    run.generator.eval_mode,
                    run.generator.data_split,
                    run.generator.model_info.system_prompt,
                    model,
                    run.generator.max_tokens,
                ),
                model=model,
                # ponytail: up to ~2.3 GB (memory caps) and 2 docker networks per
                # sample, so 6 fits a 30 GB desktop with ~14 GB free; raise with RAM
                # (docker's default address pools allow about 15).
                max_sandboxes=6,
                max_samples=6,
                log_buffer=1,  # save each completed sample, even if another hangs
                fail_on_error=False,  # one broken container must not end the run
                limit=run.limit,
            )[0]
            run.save_path.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(
                add_trajectory_views(
                    log.location,
                    run.generator.eval_mode,
                    run.generator.model_info.system_prompt,
                    model,
                ),
                str(run.save_path),
            )
            continue

        # Select the appropriate task based on dataset_type
        if run.generator.dataset_type == DatasetType.APPS:
            task = backdoored_solutions(
                run.generator.eval_mode,
                run.generator.data_split,
                run.generator.model_info.system_prompt,
            )
        elif run.generator.dataset_type == DatasetType.BIGCODEBENCH:
            task = bigcodebench_solutions(
                run.generator.data_split,
                run.generator.model_info.system_prompt,
                run.generator.eval_mode,
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {run.generator.dataset_type}")

        result = eval(
            tasks=task,
            model=run.generator.model_info.model,
            epochs=Epochs(1, ["pass_at_1"]),
            max_connections=8,  # Cap parallel API/Docker work; total example count is unchanged.
            max_sandboxes=8,
            reasoning_tokens=0,
            reasoning_effort="minimal",  # openai models
            max_tokens=run.generator.max_tokens,  # None -> provider default
            limit=run.limit,
        )
        run.save_path.parent.mkdir(parents=True, exist_ok=True)
        run.save_path.touch()
        shutil.copy(result[0].location, run.save_path)


if __name__ == "__main__":
    main()
