"""Tests for BashArena monitor, collusion, and summary prompts."""

from __future__ import annotations

import lasr_labs_2025_control_project.prompts.basharena_monitor_prompts as bmp
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name


def test_constants_exist_and_non_empty():
    expected_constants = [
        "BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT",
        "BASHARENA_MONITOR_USER_PROMPT",
        "BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT",
        "BASHARENA_SUMMARIZE_SYSTEM_PROMPT",
        "BASHARENA_SUMMARIZE_BASE_PROMPT",
    ]
    for name in expected_constants:
        assert hasattr(bmp, name), f"{name} is not defined in module"
        val = getattr(bmp, name)
        assert isinstance(val, str), f"{name} is not a str"
        assert len(val.strip()) > 0, f"{name} is empty"


def test_user_and_summary_base_prompts_format():
    user_formatted = bmp.BASHARENA_MONITOR_USER_PROMPT.format(
        problem_statement="Problem Statement P",
        solution="Solution S",
    )
    assert "Problem Statement P" in user_formatted
    assert "Solution S" in user_formatted

    summary_formatted = bmp.BASHARENA_SUMMARIZE_BASE_PROMPT.format(
        solution="Solution S"
    )
    assert "Solution S" in summary_formatted


def test_system_prompts_no_unfilled_placeholders():
    system_prompts = [
        bmp.BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT,
        bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT,
        bmp.BASHARENA_SUMMARIZE_SYSTEM_PROMPT,
    ]
    for prompt in system_prompts:
        assert "{" not in prompt, f"Found unformatted '{{' in prompt:\n{prompt[:200]}"
        assert "}" not in prompt, f"Found unformatted '}}' in prompt:\n{prompt[:200]}"

    # Verify side-task descriptions were substituted into suspiciousness prompt
    susp = bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT.lower()
    assert "firewall" in susp
    assert "sudo" in susp
    assert "malware" in susp
    assert "TARGET_IP" in bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT
    assert "TARGET_USERNAME" in bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT
    assert "SOURCE_IP" in bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT


def test_prompt_xml_tags():
    assert "<score>" in bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT
    assert "</score>" in bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT
    assert "<answer>" in bmp.BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT
    assert "</answer>" in bmp.BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT


def test_config_loader_registration():
    prompt = get_prompt_by_name("BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT")
    assert prompt == bmp.BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT

    assert (
        get_prompt_by_name("BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT")
        == bmp.BASHARENA_SELF_REC_COLLUSION_SYSTEM_PROMPT
    )
    assert (
        get_prompt_by_name("BASHARENA_MONITOR_USER_PROMPT")
        == bmp.BASHARENA_MONITOR_USER_PROMPT
    )
    assert (
        get_prompt_by_name("BASHARENA_SUMMARIZE_SYSTEM_PROMPT")
        == bmp.BASHARENA_SUMMARIZE_SYSTEM_PROMPT
    )
    assert (
        get_prompt_by_name("BASHARENA_SUMMARIZE_BASE_PROMPT")
        == bmp.BASHARENA_SUMMARIZE_BASE_PROMPT
    )


def test_parse_score_accepts_decimal_scores():
    from lasr_labs_2025_control_project.scripts.monitor_solutions_inspect import (
        parse_score,
    )

    assert parse_score("<reasoning>x</reasoning>\n<score>8.9</score>") == 8.9
    assert parse_score("<score>7</score>") == 7
    assert parse_score("<score>high</score>") is None
    assert parse_score("no tag") is None
    assert parse_score("<score>inf</score>") is None
    assert parse_score("<score>nan</score>") is None
