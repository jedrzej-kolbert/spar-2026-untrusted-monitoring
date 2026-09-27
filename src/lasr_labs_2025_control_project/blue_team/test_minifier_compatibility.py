"""Check existing YAML -> .eval -> SFT JSON contracts; no APIs or sandbox."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score
from inspect_ai.solver import solver
from inspect_ai.util import store

from ..finetuning.utils.eval_to_dataset import process_eval_log
from ..scripts.helpers.eval_to_dataset import eval_to_dataset
from ..scripts.transform_solutions_inspect import (
    create_named_scorer,
    transform_solutions,
)
from ..utils.config_loader import load_and_normalize_config
from ..utils.paraphrasers.paraphrasers import MinifierParaphraser
from .contracts import MonitorView, RepresentationKind
from .jammers import MinifierJammer


async def check():
    code = "```python\n# comment\nvalue: int = 2\nassert value > 0\nprint(value)\n```"
    legacy = MinifierParaphraser()([code])[0]
    view = MonitorView("Print two", code, RepresentationKind.PYTHON_CODE)
    assert (await MinifierJammer()(view))[0].content == legacy

    @solver
    def source():
        async def solve(state, generate):
            store().set("base_solution", state.input)
            return state

        return solve

    with TemporaryDirectory(prefix="minifier-compatibility-") as temporary:
        folder = Path(temporary)
        for name, value in (
            ("apps_scorer", {"main_task_success": "C", "side_task_success": "C"}),
            ("bigcodebench_scorer", {"main_task_success": "C"}),
            ("verify", "C"),
        ):
            sample = Sample(
                id=1,
                input=code,
                metadata={
                    "id": 1,
                    "main_task_description": "Print two",
                    "scores": {name: Score(value=value)},
                },
            )
            logs = await eval_async(
                Task(
                    dataset=[sample], solver=source(), scorer=create_named_scorer(name)
                ),
                model="mockllm/model",
                log_dir=str(folder / "source"),
                display="none",
            )
            assert logs[0].status == "success", logs[0].error
            config = folder / "minified.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "experiment_folder": str(folder),
                        "experiment_name": "compatibility",
                        "results_folder": "results",
                        "defaults": {
                            "transform": {"type": "minifier", "remove_comments": False}
                        },
                        "runs": {
                            "honest": {
                                "eval_path": logs[0].location,
                                "save_path": "transformed.eval",
                            }
                        },
                    }
                )
            )
            run = load_and_normalize_config(config, config_type="transform").runs[0]
            transformed = await eval_async(
                transform_solutions(run),
                model="mockllm/model",
                log_dir=str(folder / "transformed"),
                display="none",
            )
            log = transformed[0]
            assert log.status == "success", log.error
            result = log.samples[0]
            assert result.id == sample.id and result.output.completion == legacy
            assert result.store == {
                "transformed_solution": legacy,
                "transform_info": {
                    "type": "minifier",
                    "model_info": None,
                    "remove_comments": False,
                },
            }
            assert result.metadata["base_solution"] == code
            assert result.scores[name].value == value
            dataset = eval_to_dataset(log.location)
            assert len(dataset) == 1 and dataset[0].input == legacy
            raw = process_eval_log(
                log.location, "mock/U", folder / f"{name}.json", transformed=True
            )
            assert raw.problems[0].solutions[0].solution == f"```python\n{legacy}\n```"
            print(f"{name}: legacy code, .eval fields and SFT JSON preserved")
    print("Minifier workflow compatibility checks passed.")


if __name__ == "__main__":
    asyncio.run(check())
