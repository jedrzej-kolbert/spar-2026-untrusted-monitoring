"""The BashArena task must carry a wall-clock limit that Inspect actually applies.

ControlEvalConfig(time_limit=...) is silently dropped by get_control_task; with no
limit, task 43890461 sat in a hung main-task test for over nine hours.
"""

from types import SimpleNamespace

from control_arena import EvalMode
from inspect_ai import Task, eval

from lasr_labs_2025_control_project.scripts import generate_solutions_inspect as gen
from lasr_labs_2025_control_project.utils.config_loader import DataSplit


def test_inspect_applies_the_time_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gen, "BashArenaSetting", lambda: SimpleNamespace(get_dataset=lambda mode: [])
    )
    monkeypatch.setattr(gen, "attack_policy", lambda **_: None)
    monkeypatch.setattr(gen, "auto_submit_protocol_wrapper", lambda **_: None)
    monkeypatch.setattr(gen, "get_control_task", lambda **_: Task())
    task = gen.basharena_solutions(
        EvalMode.ATTACK, DataSplit.TRAIN, "prompt", "mockllm/model"
    )
    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    assert log.eval.config.time_limit == 60 * 60
