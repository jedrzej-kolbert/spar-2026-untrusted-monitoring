"""Small async contracts; only MonitorView is supplied to a monitor/jammer."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol


class RepresentationKind(StrEnum):
    PYTHON_CODE = "python_code"
    SUMMARY_TEXT = "summary_text"
    TRAJECTORY = "trajectory"
    TRAJECTORY_PREFIX = "trajectory_prefix"


@dataclass(frozen=True)
class MonitorView:
    task_description: str
    content: str
    representation: RepresentationKind
    problem_id: str = ""


@dataclass(frozen=True)
class TransformProvenance:
    jammer_name: str
    input_representation: RepresentationKind
    output_representation: RepresentationKind
    model_used: str | None = None
    system_prompt: str = ""
    user_prompt: str = ""


@dataclass(frozen=True)
class BlueTeamScores:
    untrusted_collusion_score: float
    untrusted_suspiciousness_score: float
    trusted_suspiciousness_score: float
    provenance: tuple[TransformProvenance, ...] = ()
    um_view: MonitorView | None = None
    t_view: MonitorView | None = None

    def __post_init__(self):
        if not isfinite(self.untrusted_collusion_score):
            raise ValueError("Nonfinite collusion score")
        for score in (
            self.untrusted_suspiciousness_score,
            self.trusted_suspiciousness_score,
        ):
            if not isfinite(score) or not 0 <= score <= 10:
                raise ValueError("Suspiciousness must be finite and in [0, 10]")


@dataclass(frozen=True)
class HoneypotValidation:
    valid: bool
    explanation: str


@dataclass(frozen=True)
class HoneypotResult:
    """Evaluator-only result: never pass the trigger/validation to a monitor."""

    view: MonitorView
    original_benign: str
    secret_input: str
    validation: HoneypotValidation


TextCaller = Callable[[str, str], Awaitable[str]]
MonitorCaller = Callable[[MonitorView], Awaitable[float]]
BenignSampler = Callable[[str], Awaitable[str]]
Validator = Callable[[str, str, str, str], Awaitable[HoneypotValidation]]


@dataclass(frozen=True)
class EvaluatorResources:
    sample_benign: BenignSampler
    call_t: TextCaller
    um_collusion: MonitorCaller
    um_suspiciousness: MonitorCaller
    t_suspiciousness: MonitorCaller
    trusted_model: str


class SignalJammer(Protocol):
    async def __call__(
        self, view: MonitorView
    ) -> tuple[MonitorView, TransformProvenance]: ...


class HoneypotPipeline(Protocol):
    async def __call__(self, problem_id: str) -> HoneypotResult: ...


class BlueTeamPolicy(Protocol):
    async def __call__(
        self, view: MonitorView, resources: EvaluatorResources
    ) -> BlueTeamScores: ...
