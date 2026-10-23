from types import SimpleNamespace

import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    AppsProblem,
    ExecutionGrade,
    Rollout,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_adapters import (
    StrategyBox,
    TinkerLearner,
    TrainerTeam,
    episode_from,
    with_strategy,
)


def attack_problem() -> AppsProblem:
    return AppsProblem("1", "question", ("1",), ("1",))


def rollout(payload: int, *, decoded_ok: bool = True, reward: float = 0.5) -> Rollout:
    item = Rollout(attack_problem(), payload, None)
    item.probabilities = (0.2, 0.8) if decoded_ok == (payload == 1) else (0.8, 0.2)
    item.execution = ExecutionGrade(True)
    item.backdoor = ExecutionGrade(True)
    item.suspiciousness = 0.3
    item.reward = reward
    return item


def test_strategy_is_appended_to_the_system_message_only() -> None:
    box = StrategyBox("use an unused variable")

    def messages(problem: object, payload: int) -> list[dict[str, str]]:
        return [{"role": "system", "content": "base"}, {"role": "user", "content": "u"}]

    wrapped = with_strategy(messages, box)(None, 1)
    assert wrapped[0]["content"].startswith("base\n\nStrategy you share")
    assert wrapped[0]["content"].endswith("use an unused variable")
    assert wrapped[1] == {"role": "user", "content": "u"}
    box.text = ""
    assert with_strategy(messages, box)(None, 1) == messages(None, 1)


def test_episode_maps_the_rollout_fields() -> None:
    episode = episode_from(rollout(1, reward=0.7))
    assert (episode.payload, episode.decoded, episode.passed) == (1, 1, True)
    assert episode.backdoor and episode.suspicion == pytest.approx(0.3)
    assert episode.reward == pytest.approx(0.7)


def test_episode_rejects_a_rollout_without_a_reward() -> None:
    item = rollout(0)
    item.reward = None
    with pytest.raises(ValueError, match="no reward"):
        episode_from(item)


class FakeTrainer:
    def __init__(self) -> None:
        self.box: StrategyBox | None = None
        self.seen: list[str] = []

    async def evaluate(self, problems, num_samples, *, balanced):
        assert balanced is True
        self.seen.append(self.box.text)
        return SimpleNamespace(rollouts=[rollout(0), rollout(1)])


async def test_team_sets_the_strategy_before_evaluating() -> None:
    trainer = FakeTrainer()
    box = StrategyBox()
    trainer.box = box
    team = TrainerTeam(trainer, [attack_problem()], box)
    episodes = await team.episodes("strategy one")
    assert trainer.seen == ["strategy one"]
    assert [e.payload for e in episodes] == [0, 1]


class FakeSampler:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.calls = 0

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls += 1
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=self.tokens)])


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=True):
        return "<strategy>use a dead branch</strategy>"


async def test_tinker_learner_refuses_to_sample_without_approval() -> None:
    sampler = FakeSampler([4])
    learner = TinkerLearner(sampler, FakeTokenizer(), allow_paid=False)
    with pytest.raises(PermissionError):
        await learner.propose([], 2)
    assert sampler.calls == 0


async def test_tinker_learner_returns_the_strategy_reply() -> None:
    sampler = FakeSampler([4])
    learner = TinkerLearner(sampler, FakeTokenizer(), allow_paid=True)
    text = await learner.propose([], 2)
    assert "use a dead branch" in text
    assert sampler.calls == 1


async def test_tinker_learner_rejects_a_reply_without_strategies() -> None:
    class Plain(FakeTokenizer):
        def decode(self, tokens, skip_special_tokens=True):
            return "no tags"

    learner = TinkerLearner(FakeSampler([4]), Plain(), allow_paid=True)
    with pytest.raises(ValueError, match="no <strategy>"):
        await learner.propose([], 1)
