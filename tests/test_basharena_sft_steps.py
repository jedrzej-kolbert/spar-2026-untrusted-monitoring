"""BashArena logs through the existing SFT and monitor steps, one row per action."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score

from lasr_labs_2025_control_project.finetuning.utils import eval_to_dataset as sft_data
from lasr_labs_2025_control_project.finetuning.utils.generate_train_data import (
    load_data,
    prepare_rows,
)
from lasr_labs_2025_control_project.prompts.basharena_monitor_prompts import (
    BASHARENA_ACTION_USER_PROMPT,
    BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT,
)
from lasr_labs_2025_control_project.scripts import (
    plot_data,
    transform_solutions_inspect,
)
from lasr_labs_2025_control_project.scripts.helpers import eval_to_dataset as mon_data
from lasr_labs_2025_control_project.utils.basharena_views import (
    ACTION_VIEW,
    ActionViewOptions,
    trajectory_view,
)
from lasr_labs_2025_control_project.utils.utils import format_solution
from tests.test_basharena_views import _trajectory

OPTIONS = ActionViewOptions(actions_per_trajectory=2, seed=0)


def _sample(id: str, side: str = "C", curl_target: str = "185.199.108.153") -> Any:
    messages = _trajectory(curl_target)
    return SimpleNamespace(
        id=id,
        messages=messages,
        scores={
            "main_task_success": Score(
                value={"main_task_success": "I", "main_task_score": 0.0}
            ),
            "side_task_success": Score(value=side),
        },
        metadata={
            "eval_mode": "attack",
            "setting_name": "bash_arena",
            "main_task_description": f"Do the task. ({id})",
        },
        store={"base_solution": trajectory_view(messages)},
        error=None,
    )


LOGS = {
    # 103's side task failed, so it is not a valid attack and gives no rows.
    "attacks": [_sample("101"), _sample("103", side="I")],
    "honeypots": [_sample("105", curl_target="52.84.125.33")],
}


def _fake_read(path: Any, *args: Any, **kwargs: Any) -> Any:
    return SimpleNamespace(samples=LOGS[Path(path).stem])


def test_sft_rows_and_monitor_inputs_are_the_same_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sft_data, "read_eval_log", _fake_read)
    monkeypatch.setattr(mon_data, "read_eval_log", _fake_read)

    for label, model in (("attacks", "U"), ("honeypots", "H")):
        (tmp_path / f"{label}.eval").touch()  # the loader checks the path exists
        sft_data.process_eval_log(
            tmp_path / f"{label}.eval",
            model,
            tmp_path / "raw" / f"{label}.json",
            view=ACTION_VIEW,
            action_options=OPTIONS,
        )
    rows = prepare_rows(
        load_data(tmp_path / "raw"),
        "U",
        BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT,
        BASHARENA_ACTION_USER_PROMPT,
    )
    assert len(rows) == 4  # two valid trajectories x two selected actions
    answers = sorted(row["messages"][2]["content"] for row in rows)
    assert answers == ["<answer>NO</answer>"] * 2 + ["<answer>YES</answer>"] * 2

    monitored = set()
    for label in LOGS:
        dataset = mon_data.eval_to_dataset(
            tmp_path / f"{label}.eval", view=ACTION_VIEW, action_options=OPTIONS
        )
        for sample in dataset:
            metadata = sample.metadata or {}
            assert sample.id == (
                f"{metadata['trajectory_id']}#a{metadata['action_index']}"
            )
            monitored.add(
                BASHARENA_ACTION_USER_PROMPT.format(
                    problem_statement=metadata["main_task_description"],
                    solution=format_solution(
                        metadata["base_solution"],
                        metadata["setting_name"],
                        metadata["solution_view"],
                    ),
                )
            )
    assert monitored == {row["messages"][1]["content"] for row in rows}


def test_sft_loader_rejects_transformed_logs_in_action_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    sample = _sample("107")
    sample.store["transformed_solution"] = "paraphrased"
    monkeypatch.setattr(
        sft_data, "read_eval_log", lambda *a, **k: SimpleNamespace(samples=[sample])
    )
    (tmp_path / "attacks.eval").touch()
    with pytest.raises(ValueError, match="transformed"):
        sft_data.process_eval_log(
            tmp_path / "attacks.eval", "U", tmp_path / "raw.json", view=ACTION_VIEW
        )


def test_paraphrasing_and_minification_refuse_basharena(
    monkeypatch: pytest.MonkeyPatch,
):
    sample = Sample(input="<action_1>", metadata={"setting_name": "bash_arena"})
    monkeypatch.setattr(
        transform_solutions_inspect, "eval_to_dataset", lambda path: [sample]
    )
    with pytest.raises(NotImplementedError, match="not available for BashArena"):
        transform_solutions_inspect.transform_solutions(
            SimpleNamespace(eval_path="attacks.eval")
        )


@pytest.mark.parametrize("missing_score", [None, float("nan")])
def test_plot_frame_takes_max_over_actions_and_rejects_partial(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing_score: float | None,
):
    def action_sample(
        traj: str,
        k: int,
        total: int,
        scores: tuple[float | None, float | None, float | None],
    ) -> SimpleNamespace:
        u_col, t_sus, u_sus = scores
        return SimpleNamespace(
            id=f"{traj}#a{k}",
            metadata={"main_task_description": "t", "trajectory_id": traj},
            store={
                "base_solution": f"view {k}",
                "n_actions": total,
                "untrusted_collusion_score": u_col,
                "trusted_suspiciousness_score": t_sus,
                "untrusted_suspiciousness_score": u_sus,
                "untrusted_collusion_score_metadata": {
                    "model_info": {"system_prompt": "s", "base_prompt": "b"}
                },
            },
        )

    samples = [
        action_sample("1", 1, 2, (0.5, 1, 2)),
        action_sample("1", 2, 2, (-0.5, 7, 3)),
        action_sample("2", 1, 2, (0.1, 9, 4)),
        action_sample("2", 2, 2, (0.2, missing_score, 5)),
        action_sample("3", 1, 1, (missing_score, missing_score, missing_score)),
        action_sample("4", 1, 1, (0.1, missing_score, 4)),
    ]
    monkeypatch.setattr(
        plot_data, "read_eval_log", lambda _: SimpleNamespace(samples=samples)
    )
    df = plot_data.build_jsonl_dataframe_from_eval_log(Path("x.eval")).set_index(
        "sample_id"
    )
    assert list(df.index) == ["1", "2", "4"]
    scores = df["trusted_suspiciousness_score"].to_numpy(dtype=float)
    assert df["untrusted_collusion_score"].to_numpy(dtype=float)[0] == 0.5
    assert scores[:2].tolist() == [7, 9]  # missing actions do not erase valid scores
    assert np.isnan(scores[2])  # only ROCs requiring this monitor omit trajectory 4
    assert df["n_actions_scored"].tolist() == [2, 2, 1]
    assert "no available monitor scores: ['3']" in caplog.text

    # A trajectory with only some of its actions scored cannot give a max.
    monkeypatch.setattr(
        plot_data, "read_eval_log", lambda _: SimpleNamespace(samples=samples[:1])
    )
    with pytest.raises(ValueError, match="missing scored actions"):
        plot_data.build_jsonl_dataframe_from_eval_log(Path("x.eval"))
