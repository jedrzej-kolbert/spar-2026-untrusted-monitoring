"""All three consumers drop zero-action and errored BashArena samples (prior S3 wiring)."""

from pathlib import Path
from types import SimpleNamespace

from inspect_ai.scorer import Score

from lasr_labs_2025_control_project.finetuning.utils import eval_to_dataset as sft
from lasr_labs_2025_control_project.scripts import tinker_sft_dynamics as dynamics
from lasr_labs_2025_control_project.scripts.helpers import eval_to_dataset as monitor


def _sample(i, view, error=None):
    return SimpleNamespace(
        id=str(i),
        metadata={
            "eval_mode": "attack",
            "setting_name": "bash_arena",
            "main_task_description": f"task {i}",
        },
        store={"base_solution": view},
        scores={
            "main_task_success": Score(
                value={"main_task_success": "I", "main_task_score": 0.0}
            ),
            "side_task_success": Score(value="C"),
        },
        error=error,
    )


SAMPLES = [
    _sample(2, "<action_1>\n<tool>bash</tool>\n<cmd>ls</cmd>\n</action_1>"),
    _sample(4, ""),  # zero actions, side task already "C"
    _sample(6, "<action_1>x</action_1>", error="scoring timed out"),
]
LOG = SimpleNamespace(samples=SAMPLES, status="success")


def test_consumers_drop_zero_action_and_errored(monkeypatch, tmp_path):
    for module in (sft, dynamics, monitor):
        monkeypatch.setattr(module, "read_eval_log", lambda *a, **k: LOG)
    assert [s.id for s in monitor.eval_to_dataset("x.eval")] == ["2"]
    assert len(dynamics.load_eval_examples(Path("a"), Path("h"), "base_solution")) == 2
    source = tmp_path / "a.eval"
    source.touch()
    dataset = sft.process_eval_log(source, "U", tmp_path / "a.json")
    assert [p.apps_problem_id for p in dataset] == [2]
