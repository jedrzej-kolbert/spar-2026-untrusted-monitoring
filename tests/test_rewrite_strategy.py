import json
from pathlib import Path

import pytest

from lasr_labs_2025_control_project.scripts.rewrite_solutions_inspect import (
    CENTRAL_SENDER,
    RewriteSettings,
    with_strategy_result,
)


def settings(result: Path, system: str) -> RewriteSettings:
    return RewriteSettings(
        model="tinker/base",
        system_prompt=system,
        user_prompt="RL_SELF_REC_SENDER_USER_PROMPT",
        signal="RL_SIGNAL_ON",
        strategy_result=str(result),
    )


def result(tmp_path: Path, decoder: str) -> Path:
    path = tmp_path / "icrl_result.json"
    path.write_text(
        json.dumps({"winner": {"strategy": "rename vars", "decoder": decoder}})
    )
    return path


def test_a_sender_only_winner_gives_its_instructions_and_no_judge(
    tmp_path: Path,
) -> None:
    rewrite, inputs, judge = with_strategy_result(
        settings(result(tmp_path, ""), "RL_SELF_REC_SENDER_SYSTEM_PROMPT")
    )
    assert rewrite.strategy == "rename vars" and judge == "" and len(inputs) == 1


def test_a_centralized_winner_gives_its_judge_text_and_needs_its_sender_prompt(
    tmp_path: Path,
) -> None:
    path = result(tmp_path, "YES if it counts calls")
    _, _, judge = with_strategy_result(settings(path, CENTRAL_SENDER))
    assert judge == "YES if it counts calls"
    with pytest.raises(ValueError, match="centralized"):
        with_strategy_result(settings(path, "RL_SELF_REC_SENDER_SYSTEM_PROMPT"))
    with pytest.raises(ValueError, match="centralized"):
        with_strategy_result(settings(result(tmp_path, ""), CENTRAL_SENDER))


def test_learned_judge_prompts_fill_the_search_judge_template() -> None:
    from lasr_labs_2025_control_project.prompts.learned_judge_prompts import _judge
    from lasr_labs_2025_control_project.prompts.rl_prompts import (
        RL_CENTRAL_JUDGE_SYSTEM_PROMPT,
    )

    assert _judge("X") == RL_CENTRAL_JUDGE_SYSTEM_PROMPT.format(
        decoder_instructions="X"
    )
