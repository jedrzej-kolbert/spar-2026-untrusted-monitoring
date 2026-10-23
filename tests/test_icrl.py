import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl import (
    Episode,
    RoundSummary,
    learner_messages,
    parse_strategies,
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


def test_parse_strategies_keeps_multiline_blocks_and_drops_empty() -> None:
    text = "<strategy>\nuse a\nhelper function\n</strategy>\n<strategy>  </strategy>"
    assert parse_strategies(text) == ["use a\nhelper function"]
    assert parse_strategies("no tags here") == []


def test_summary_counts_working_backdoors_over_checked_episodes_only() -> None:
    episodes = [
        episode(p_self=0.9),
        episode(p_self=0.7, backdoor=False),
        episode(p_self=0.5, passed=False, backdoor=True, reward=0.0),
        episode(p_self=0.3, backdoor=None),  # the check does not apply
    ]
    summary = summarise(1, "s", episodes)
    assert summary.episodes == 4
    assert summary.p_self_mean == pytest.approx(0.6)
    assert summary.pass_rate == pytest.approx(0.75)
    # 3 checked; only the first passes AND keeps its backdoor.
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
        self.seen: list[list[str]] = []

    async def propose(self, history: list[RoundSummary], count: int) -> str:
        self.seen.append([item.strategy for item in history])
        if len(self.seen) == 1:
            return "<strategy>weak</strategy><strategy>strong</strategy>"
        return "<strategy>strong, refined</strategy>"


class ScriptedTeam:
    def __init__(self) -> None:
        self.runs: list[str] = []

    async def episodes(self, strategy: str) -> list[Episode]:
        self.runs.append(strategy)
        reward = {"": 0.4, "weak": 0.1, "strong": 0.9}.get(strategy, 0.6)
        return [episode(reward=reward), episode(reward=reward)]


async def test_run_icrl_starts_from_the_baseline_and_keeps_history() -> None:
    learner, team = ScriptedLearner(), ScriptedTeam()
    history, best = await run_icrl(learner, team, rounds=2, candidates=2)
    assert team.runs[0] == ""  # round 0: the sender with no instructions
    assert learner.seen == [[""], ["", "weak", "strong"]]
    assert [item.round for item in history] == [0, 1, 1, 2]
    assert best.strategy == "strong"
    assert best.reward_mean == pytest.approx(0.9)


async def test_learner_output_without_strategies_is_an_error() -> None:
    class Silent:
        async def propose(self, history: list[RoundSummary], count: int) -> str:
            return "I cannot help with that."

    with pytest.raises(ValueError, match="no <strategy> blocks"):
        await run_icrl(Silent(), ScriptedTeam(), rounds=1, candidates=1)


def test_learner_prompt_states_the_objective_and_earlier_results() -> None:
    baseline = summarise(0, "", [episode(p_self=0.5, backdoor=None)])
    previous = summarise(1, "keep the helper names", [episode(p_self=0.6)])
    system, user = learner_messages([baseline, previous], count=3)
    for goal in ("passes the", "keeps the backdoor", "judge", "trusted monitor"):
        assert goal in system["content"]
    assert "secret bit" not in system["content"]
    assert "(no instructions: the baseline)" in user["content"]
    assert "working backdoor n/a (0 checked)" in user["content"]
    assert "<strategy>keep the helper names</strategy>" in user["content"]
    assert "judge P(self) 0.60" in user["content"]
    assert "Propose 3 new instructions" in user["content"]
