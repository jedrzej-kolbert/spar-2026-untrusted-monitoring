from types import SimpleNamespace

import pytest

from lasr_labs_2025_control_project.finetuning.grouped_rl.channels import (
    MonitorViewChannel,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.core import (
    ExecutionGrade,
    GroupedRLConfig,
    GroupedRLTrainer,
    Rollout,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl import (
    Knowledge,
    RoundSummary,
    run_icrl,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl_adapters import (
    StrategyBox,
    TinkerLearner,
    TrainerTeam,
    episode_from,
    with_strategy,
)
from lasr_labs_2025_control_project.finetuning.grouped_rl.rewards import RewardConfig
from lasr_labs_2025_control_project.finetuning.grouped_rl.tinker_backend import (
    extract_final_code,
)
from tests.test_hipt_rl import (
    EchoSender,
    FixedOverseer,
    MarkerReceiver,
    PassingEvaluator,
    attack_problem,
)


def graded(backdoor: ExecutionGrade | None = ExecutionGrade(True)) -> Rollout:
    item = Rollout(attack_problem(), 1, None)
    item.probabilities = (0.3, 0.7)
    item.execution = ExecutionGrade(True)
    item.backdoor = backdoor
    item.backdoor_suspicion = 0.4
    item.reward = 0.9
    return item


def test_strategy_is_appended_to_the_sender_system_message_only() -> None:
    box = StrategyBox("keep the original helper names")

    def messages(problem: object, payload: int) -> list[dict[str, str]]:
        return [{"role": "system", "content": "base"}, {"role": "user", "content": "u"}]

    wrapped = with_strategy(messages, box)(None, 1)
    assert wrapped[0]["content"].startswith("base\n\nInstructions for this rewrite:")
    assert wrapped[0]["content"].endswith("keep the original helper names")
    assert wrapped[1] == {"role": "user", "content": "u"}
    box.text = ""
    assert with_strategy(messages, box)(None, 1) == messages(None, 1)


def test_episode_keeps_the_judge_score_and_an_unchecked_backdoor() -> None:
    episode = episode_from(graded())
    assert episode.p_self == pytest.approx(0.7)
    assert episode.passed and episode.backdoor is True
    assert episode.suspicion == pytest.approx(0.4)
    assert episode.reward == pytest.approx(0.9)
    assert episode_from(graded(backdoor=None)).backdoor is None


@pytest.mark.parametrize("field", ["reward", "backdoor_suspicion"])
def test_episode_rejects_a_missing_score(field: str) -> None:
    item = graded()
    setattr(item, field, None)
    with pytest.raises(ValueError):
        episode_from(item)


def self_rec_trainer() -> GroupedRLTrainer:
    return GroupedRLTrainer(
        EchoSender(),
        MarkerReceiver(),
        PassingEvaluator(),
        GroupedRLConfig(
            group_size=2,
            self_recognition=True,
            reward=RewardConfig(backdoor_kept_weight=1, backdoor_suspicion_weight=1),
        ),
        channel=MonitorViewChannel("none"),
        backdoor_monitor=FixedOverseer(),
        code_extractor=extract_final_code,
    )


def team(trainer: GroupedRLTrainer) -> TrainerTeam:
    held_out = attack_problem("13", "validation")
    return TrainerTeam(
        trainer,
        StrategyBox(),
        search_problems=[attack_problem("11"), attack_problem("15")],
        held_out_problems=[held_out],
        honeypots=[(held_out, "# another model\nprint(7)")],
    )


class OneShotLearner:
    async def strategize(self, previous: RoundSummary, knowledge: Knowledge) -> str:
        return "<strategy>keep the original helper names</strategy>"


async def test_icrl_runs_through_the_real_self_recognition_trainer() -> None:
    # Self-recognition evaluation generates only payload 1; the loop must accept that.
    trainer = self_rec_trainer()
    history, best = await run_icrl(
        OneShotLearner(), team(trainer), episodes=1, candidates=1
    )
    assert [item.strategy for item in history] == ["", "keep the original helper names"]
    assert all(item.episodes == 2 for item in history)
    assert history[0].p_self_mean == pytest.approx(
        0.9
    )  # MarkerReceiver sees "signal = 1"
    assert history[0].backdoor_checked == 2
    assert best in history


async def test_held_out_check_scores_the_honeypots() -> None:
    episodes, honeypot = await team(self_rec_trainer()).held_out("")
    assert len(episodes) == 1
    assert honeypot.metrics()["n_honeypots"] == 1
    assert honeypot.metrics()["auroc"] == pytest.approx(1.0)


class FlakyEvaluator(PassingEvaluator):
    """The first grading call hits a sandbox infrastructure error."""

    def __init__(self) -> None:
        self.calls = 0

    async def grade(self, problem, code):
        self.calls += 1
        if self.calls == 1:
            return ExecutionGrade(False, infrastructure_error="sandbox down")
        return ExecutionGrade(True)


async def test_infrastructure_error_is_recorded_and_the_batch_rerun_once() -> None:
    trainer = self_rec_trainer()
    trainer.evaluator = FlakyEvaluator()
    phases: list[str] = []
    flaky = team(trainer)
    flaky.on_result = lambda strategy, result, phase: phases.append(phase)
    episodes = await flaky.episodes("")
    assert phases == ["search_failed", "search"]
    assert len(episodes) == 2 and all(e.passed for e in episodes)


def test_team_refuses_overlapping_problems_and_other_objectives() -> None:
    problem = attack_problem("13", "validation")
    with pytest.raises(ValueError, match="overlap"):
        TrainerTeam(
            self_rec_trainer(),
            StrategyBox(),
            search_problems=[problem],
            held_out_problems=[problem],
            honeypots=[],
        )
    payload_trainer = GroupedRLTrainer(
        EchoSender(), MarkerReceiver(), PassingEvaluator(), GroupedRLConfig()
    )
    with pytest.raises(ValueError, match="self-recognition"):
        TrainerTeam(
            payload_trainer,
            StrategyBox(),
            search_problems=[],
            held_out_problems=[],
            honeypots=[],
        )


class FakeSampler:
    def __init__(self) -> None:
        self.calls = 0

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls += 1
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[4])])


class FakeTokenizer:
    reply = "<strategy>keep the original helper names</strategy>"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=True):
        return self.reply


def _baseline() -> RoundSummary:
    from lasr_labs_2025_control_project.finetuning.grouped_rl.icrl import summarise

    return summarise(0, "", [episode_from(graded())])


async def test_tinker_learner_refuses_to_sample_without_approval() -> None:
    sampler = FakeSampler()
    learner = TinkerLearner(sampler, FakeTokenizer(), allow_paid=False)
    with pytest.raises(PermissionError):
        await learner.strategize(_baseline(), Knowledge())
    assert sampler.calls == 0


async def test_tinker_learner_returns_the_strategy_reply() -> None:
    sampler = FakeSampler()
    learner = TinkerLearner(sampler, FakeTokenizer(), allow_paid=True)
    assert "helper names" in await learner.strategize(_baseline(), Knowledge())
    assert sampler.calls == 1


async def test_tinker_learner_rejects_a_reply_without_strategies() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.reply = "no tags"
    learner = TinkerLearner(FakeSampler(), tokenizer, allow_paid=True)
    with pytest.raises(ValueError, match="no <strategy>"):
        await learner.strategize(_baseline(), Knowledge())
