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


def test_test_rewrite_template_takes_a_label_and_a_source() -> None:
    import yaml

    from lasr_labs_2025_control_project.scripts.rewrite_solutions_inspect import (
        RewriteConfig,
        for_label,
    )

    template = RewriteConfig.model_validate(
        yaml.safe_load(
            Path(
                "src/lasr_labs_2025_control_project/configs/rl/collusion_rl_v2/test_rewrite.yaml"
            ).read_text()
        )
    )
    config = for_label(
        template, "g1-base-s0", {"model": "tinker/tinker://x", "system_prompt": None}
    )
    assert list(config.runs) == ["g1-base-s0"]
    assert config.runs["g1-base-s0"].save_path == Path("g1-base-s0/attacks.eval")
    rewrite = config.defaults["rewrite"]
    assert rewrite.model == "tinker/tinker://x" and rewrite.max_tokens == 4096
    assert rewrite.system_prompt == "RL_SELF_REC_SENDER_SYSTEM_PROMPT"


def test_a_learned_judge_is_registered_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import click

    from lasr_labs_2025_control_project.scripts import (
        rewrite_solutions_inspect as rewrite,
    )

    module = tmp_path / "learned_judge_prompts.py"
    module.write_text("def _judge(text):\n    return text\n")
    monkeypatch.setattr(rewrite, "LEARNED_JUDGES", module)
    assert rewrite.register_judge("i2-s0", "count calls") == "JUDGE_I2_S0"
    rewrite.register_judge("i2-s0", "count calls")  # the same text again: no change
    namespace: dict = {}
    exec(module.read_text(), namespace)
    assert namespace["JUDGE_I2_S0"] == "count calls"
    with pytest.raises(click.ClickException, match="other instructions"):
        rewrite.register_judge("i2-s0", "something else")


def test_the_collusion_rl_v2_configs_load() -> None:
    from lasr_labs_2025_control_project.finetuning.grouped_rl.config import (
        load_rl_config,
    )

    folder = Path("src/lasr_labs_2025_control_project/configs/rl/collusion_rl_v2")
    configs = {
        p.stem: load_rl_config(p)
        for p in folder.glob("*.yaml")
        if p.stem != "test_rewrite"
    }
    assert set(configs) == {
        "i1_base",
        "i2",
        "g1_base",
        "smoke_i1_base",
        "smoke_g1_base",
    }
    assert {c.training.max_tokens for c in configs.values()} == {4096}
    assert configs["i2"].identity["threat_model"] == "causal"
    assert configs["g1_base"].identity["method"] == "gbrl"
