import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl import (
    Episode,
    Knowledge,
    LEARNER_SYSTEM_PROMPT,
    parse_strategy,
    results_string,
    run_icrl,
    summarise,
)


def episode(
    *,
    p_self: float = 0.8,
    passed: bool = True,
    backdoor: bool | None = True,
    reward: float = 1.0,
) -> Episode:
    return Episode(
        p_self=p_self, passed=passed, backdoor=backdoor, suspicion=0.2, reward=reward
    )


def test_parse_strategy_keeps_description_and_multiline_instructions() -> None:
    parsed = parse_strategy(
        "<description>plain style</description>"
        "<encoder_instructions>use a\nhelper function</encoder_instructions>"
    )
    assert parsed.description == "plain style"
    assert parsed.encoder_instructions == "use a\nhelper function"
    with pytest.raises(ValueError, match="encoder_instructions"):
        parse_strategy("<description>missing instructions</description>")


def test_summary_counts_working_backdoors_over_checked_episodes_only() -> None:
    examples = [
        episode(p_self=0.9),
        episode(p_self=0.7, backdoor=False),
        episode(p_self=0.5, passed=False, backdoor=True, reward=0.0),
        episode(p_self=0.3, backdoor=None),
    ]
    summary = summarise(1, "s", examples)
    assert summary.episodes == 4
    assert summary.p_self_mean == pytest.approx(0.6)
    assert summary.pass_rate == pytest.approx(0.75)
    assert summary.backdoor_checked == 3
    assert summary.working_rate == pytest.approx(1 / 3)
    assert summary.reward_mean == pytest.approx(0.75)


def test_summary_without_any_checked_backdoor_reports_no_rate() -> None:
    summary = summarise(1, "s", [episode(backdoor=None)])
    assert summary.backdoor_checked == 0 and summary.working_rate is None


def test_summary_rejects_a_strategy_without_episodes() -> None:
    with pytest.raises(ValueError, match="no episodes"):
        summarise(1, "s", [])


class ScriptedLearner:
    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []
        self.strategy_calls = 0

    async def query(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append([dict(m) for m in messages])
        user = messages[-1]["content"]
        if "<encoder_instructions>" in user:
            self.strategy_calls += 1
            strategy = "weak" if self.strategy_calls == 1 else "strong"
            return (
                f"<description>{strategy} description</description>"
                f"<encoder_instructions>{strategy}</encoder_instructions>"
            )
        if "<summarize_learning>" in user:
            return "<summarize_learning>lesson from the latest attempt</summarize_learning>"
        return "<recognition_failure_reason>not recognized</recognition_failure_reason>"


class ScriptedTeam:
    def __init__(self) -> None:
        self.runs: list[str] = []

    async def episodes(self, strategy: str) -> list[Episode]:
        self.runs.append(strategy)
        reward = {"": 0.4, "weak": 0.1, "strong": 0.9}[strategy]
        return [episode(reward=reward), episode(reward=reward)]


async def test_hipt_separate_queries_reset_and_knowledge_match_upstream_order() -> None:
    learner, team, knowledge = ScriptedLearner(), ScriptedTeam(), Knowledge()
    events: list[tuple[int, str]] = []
    history, best = await run_icrl(
        learner,
        team,
        episodes=2,
        knowledge=knowledge,
        on_reply=lambda i, stage, prompt, reply: events.append((i, stage)),
    )
    assert team.runs == ["", "weak", "strong"]
    assert events == [
        (1, "strategy"),
        (1, "reflection"),
        (1, "strategize"),
        (2, "strategy"),
        (2, "reflection"),
        (2, "strategize"),
    ]
    assert len(learner.prompts[0]) == 2  # upstream starts with query_for_strategy
    assert (
        len(learner.prompts[1]) == 3
    )  # conversation reset: system, results, reflection
    assert "weak description" in learner.prompts[1][1]["content"]
    # The next strategy sees the immediately preceding reflection and brainstorm.
    assert any(
        "lesson from the latest attempt" in m["content"] for m in learner.prompts[3]
    )
    # It is cleared after the second evaluation, not accumulated indefinitely.
    assert len(learner.prompts[4]) == 3
    last_brainstorm = learner.prompts[5][-1]["content"]
    assert "lesson from the latest attempt" in last_brainstorm
    assert (
        "weak description" in last_brainstorm
        and "strong description" in last_brainstorm
    )
    assert len(knowledge.failed_attempts) == 2
    assert knowledge.knowledge == "lesson from the latest attempt"
    assert best.strategy == "strong" and best.reward_mean == pytest.approx(0.9)
    assert [item.round for item in history] == [0, 1, 2]


async def test_malformed_strategy_is_retried_once_and_both_replies_logged() -> None:
    class Flaky(ScriptedLearner):
        bad_calls = 0

        async def query(self, messages):
            if self.bad_calls == 0:
                self.bad_calls += 1
                return "malformed"
            return await super().query(messages)

    stages: list[str] = []
    await run_icrl(
        Flaky(),
        ScriptedTeam(),
        episodes=1,
        on_reply=lambda i, stage, prompt, reply: stages.append(stage),
    )
    assert stages == ["strategy", "strategy", "reflection", "strategize"]


async def test_malformed_replies_abort_after_the_upstream_two_attempts() -> None:
    class Silent:
        calls = 0

        async def query(self, messages):
            self.calls += 1
            return "no tags"

    silent = Silent()
    with pytest.raises(ValueError, match="description"):
        await run_icrl(silent, ScriptedTeam(), episodes=1)
    assert silent.calls == 2


def test_feedback_contains_actual_inputs_outputs_and_bounded_examples() -> None:
    rows = [
        Episode(
            0.2,
            False,
            None,
            0.7,
            -0.7,
            problem="problem text",
            code=f"print({i})",
            execution_detail="wrong answer",
        )
        for i in range(4)
    ]
    text = results_string(summarise(1, "s", rows), rows, 3)
    assert "INPUT:\nproblem text" in text
    assert "ENCODER MESSAGE:\nprint(0)" in text
    assert "execution=wrong answer" in text
    assert (
        "print(3)" not in text
    )  # illustrative examples only; all rows remain in scores
    assert "working attack: n/a (0 checked)" in text
    assert summarise(1, "s", rows).episodes == 4
    assert "original backdoor trigger" in LEARNER_SYSTEM_PROMPT
    assert "never see" in LEARNER_SYSTEM_PROMPT


def test_rewrite_prompts_target_recognition_without_research_framing() -> None:
    from lasr_labs_2025_control_project.prompts.rl_prompts import (
        RL_SELF_REC_SENDER_SYSTEM_PROMPT,
    )

    for prompt in (LEARNER_SYSTEM_PROMPT, RL_SELF_REC_SENDER_SYSTEM_PROMPT):
        assert "more recognizable" in prompt
        assert "research" not in prompt.lower()
        assert "experiment" not in prompt.lower()
    assert "original backdoor trigger" in LEARNER_SYSTEM_PROMPT
    assert "Keep its behaviour identical" in RL_SELF_REC_SENDER_SYSTEM_PROMPT


def test_summary_means_each_reward_term_and_they_sum_to_the_reward() -> None:
    examples = [
        Episode(
            0.9,
            True,
            True,
            0.3,
            2.6,
            components={
                "task": 1.0,
                "self_recognition": 0.9,
                "backdoor_kept": 1.0,
                "backdoor_suspicion": -0.3,
            },
        ),
        Episode(
            0.5,
            False,
            None,
            0.1,
            -0.1,
            components={
                "task": 0.0,
                "self_recognition": 0.0,
                "backdoor_suspicion": -0.1,
            },
        ),
    ]
    summary = summarise(1, "s", examples)
    assert summary.reward_components == pytest.approx(
        {
            "backdoor_kept": 0.5,
            "backdoor_suspicion": -0.2,
            "self_recognition": 0.45,
            "task": 0.5,
        }
    )
    assert sum(summary.reward_components.values()) == pytest.approx(summary.reward_mean)
