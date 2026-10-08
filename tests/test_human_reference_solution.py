"""The human-reference baseline must read solutions from AppsSetting's samples.

control-arena >=4 moved the APPS fields under metadata["metadata"] and made each
reference solution a {"code": ...} dict; the baseline generator read the old
top-level list and silently dropped every sample.
"""

import json

import pytest
from control_arena import EvalMode
from inspect_ai.dataset import Sample

from lasr_labs_2025_control_project.self_rec_baselines.utils.scripts.data_generation import (
    human_reference_solution,
    process_single_sample,
)
from lasr_labs_2025_control_project.settings.setting import AppsSetting

REFERENCE = "print(int(input()) + 1)\n"


def _sample(solutions: list[str]) -> Sample:
    record = {
        "problem_id": 7,
        "question": "Print the input plus one.",
        "solutions": json.dumps(solutions),
        "input_output": json.dumps({"inputs": ["1\n"], "outputs": ["2\n"]}),
    }
    sample = AppsSetting().record_to_sample_factory(EvalMode.HONEST)(record)
    assert isinstance(sample, Sample)
    return sample


def test_human_reference_solution_reads_apps_setting_sample():
    assert human_reference_solution(_sample([REFERENCE, "pass"])) == REFERENCE


def test_human_provider_returns_the_reference():
    result = process_single_sample(
        _sample([REFERENCE]), "human", "human", data_split="train"
    )
    assert [s.solution for s in result.solutions] == [REFERENCE]


def test_human_reference_solution_rejects_empty_reference():
    with pytest.raises(ValueError):
        human_reference_solution(_sample([""]))
