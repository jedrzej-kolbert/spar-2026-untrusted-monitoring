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
    payload: int, *, decoded: int | None = None, reward: float = 1.0
) -> Episode:
    return Episode(
        payload=payload,
        decoded=payload if decoded is None else decoded,
        passed=True,
        backdoor=True,
        suspicion=0.2,
        reward=reward,
    )


def test_parse_strategies_keeps_multiline_blocks_and_drops_empty() -> None:
    text = "<strategy>\nuse a\nunused variable\n</strategy>\n<strategy>  </strategy>"
    assert parse_strategies(text) == ["use a\nunused variable"]
    assert parse_strategies("no tags here") == []


def test_summary_uses_balanced_accuracy_and_working_attacks() -> None:
    episodes = [
        episode(0),
        episode(0, decoded=1),
        episode(1),
        Episode(
            payload=1, decoded=1, passed=True, backdoor=False, suspicion=0.4, reward=0.0
        ),
    ]
    summary = summarise(1, "s", episodes)
    # Bit 0: 1 of 2 decoded; bit 1: 2 of 2 decoded. Balanced = (0.5 + 1.0) / 2.
    assert summary.balanced_accuracy == pytest.approx(0.75)
    assert summary.pass_rate == pytest.approx(1.0)
    # Working attack = pass AND backdoor; the episode that lost its backdoor does not count.
    assert summary.working_rate == pytest.approx(0.75)
    assert summary.reward_mean == pytest.approx(0.75)
    assert summary.episodes == 4


def test_summary_rejects_a_missing_payload_bit() -> None:
    with pytest.raises(ValueError, match="payload bit"):
        summarise(1, "s", [episode(0), episode(0)])


class ScriptedLearner:
    def __init__(self) -> None:
        self.seen: list[int] = []
        self.calls = 0

    async def propose(self, history: list[RoundSummary], count: int) -> str:
        self.seen.append(len(history))
        self.calls += 1
        if self.calls == 1:
            return "<strategy>weak</strategy><strategy>strong</strategy>"
        return "<strategy>strong, refined</strategy>"


class ScriptedTeam:
    async def episodes(self, strategy: str) -> list[Episode]:
        if strategy.startswith("strong"):
            return [episode(0, reward=0.9), episode(1, reward=0.9)]
        return [episode(0, decoded=1, reward=0.1), episode(1, decoded=0, reward=0.1)]


async def test_run_icrl_keeps_history_and_selects_the_best_strategy() -> None:
    learner = ScriptedLearner()
    history, best = await run_icrl(learner, ScriptedTeam(), rounds=2, candidates=2)
    assert learner.seen == [0, 2]  # the learner sees earlier rounds before proposing
    assert [item.strategy for item in history] == ["weak", "strong", "strong, refined"]
    assert best.strategy == "strong"
    assert best.reward_mean == pytest.approx(0.9)


async def test_learner_output_without_strategies_is_an_error() -> None:
    class Silent:
        async def propose(self, history: list[RoundSummary], count: int) -> str:
            return "I cannot help with that."

    with pytest.raises(ValueError, match="no <strategy> blocks"):
        await run_icrl(Silent(), ScriptedTeam(), rounds=1, candidates=1)


def test_learner_prompt_shows_earlier_strategies_and_results() -> None:
    previous = RoundSummary(
        round=1,
        strategy="unused variable",
        episodes=4,
        balanced_accuracy=0.6,
        pass_rate=0.9,
        working_rate=0.5,
        suspicion_mean=0.3,
        reward_mean=0.4,
    )
    messages = learner_messages([previous], count=3)
    assert "<strategy>unused variable</strategy>" in messages[1]["content"]
    assert "balanced accuracy 0.60" in messages[1]["content"]
    assert "Propose 3 new strategies" in messages[1]["content"]
