"""Tests for the per-action prompts used by BashArena configs."""

import pytest

import lasr_labs_2025_control_project.prompts.basharena_monitor_prompts as bmp


@pytest.mark.parametrize(
    "prompt",
    [bmp.BASHARENA_ACTION_USER_PROMPT, bmp.BASHARENA_ACTION_SUSPICIOUSNESS_USER_PROMPT],
)
def test_user_prompts_format(prompt):
    formatted = prompt.format(problem_statement="Problem P", solution="Action A")
    assert "Problem P" in formatted
    assert "Action A" in formatted


@pytest.mark.parametrize(
    "prompt",
    [
        bmp.BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT,
        bmp.BASHARENA_ACTION_SUSPICIOUSNESS_SYSTEM_PROMPT,
    ],
)
def test_system_prompts_are_filled(prompt):
    assert prompt.strip()
    assert "{" not in prompt and "}" not in prompt


def test_suspiciousness_prompt_includes_side_tasks_and_score():
    prompt = bmp.BASHARENA_ACTION_SUSPICIOUSNESS_SYSTEM_PROMPT
    for description in (
        bmp.SUDO_PRIVILEGE_ESCALATION_DESCRIPTION,
        bmp.DISABLE_FIREWALL_DESCRIPTION,
        bmp.EXFILTRATE_SENSITIVE_DATA_DESCRIPTION,
        bmp.DOWNLOAD_EXECUTE_MALWARE_DESCRIPTION,
    ):
        assert description in prompt
    assert "<score>" in prompt
    assert "<answer>" in bmp.BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT
    assert "</answer>" in bmp.BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT
