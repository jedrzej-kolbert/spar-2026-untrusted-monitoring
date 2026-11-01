"""Run settings for the Inspect stages, and the prompt registry.

The pipeline (``lasr_labs_2025_control_project.pipeline``) builds these
dataclasses from its config; prompts are referred to by name everywhere and
resolved here.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

from control_arena.eval_mode import EvalMode

from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    MonitorConfig,
    TransformInfo,
)
from lasr_labs_2025_control_project.utils.basharena_views import (
    TRAJECTORY_VIEW,
    ActionViewOptions,
)


@dataclass(frozen=True, slots=True)
class RunMonitors:
    """Configuration for a single run monitors.

    Attributes:
        trusted_backdoor_monitor: The trusted backdoor monitor config.
        trusted_collusion_monitor: The trusted collusion monitor config.
        untrusted_backdoor_monitor: The untrusted backdoor monitor config.
        untrusted_collusion_monitor: The untrusted collusion monitor config.
    """

    trusted_backdoor_monitor: Optional[MonitorConfig]
    trusted_collusion_monitor: Optional[MonitorConfig]
    untrusted_backdoor_monitor: Optional[MonitorConfig]
    untrusted_collusion_monitor: Optional[MonitorConfig]


class DataSplit(StrEnum):
    """The data split to run the generator on."""

    TRAIN = "train"
    TEST = "test"


class DatasetType(StrEnum):
    """The dataset type to use for generation."""

    APPS = "apps"
    BIGCODEBENCH = "bigcodebench"


@dataclass(frozen=True, slots=True)
class RunGenerator:
    """Configuration for a single generator run.

    Attributes:
        model_info: The model info.
        eval_mode: The evaluation mode.
        data_split: The data split.
        dataset_type: The dataset type (apps or bigcodebench).
        max_tokens: Optional per-run completion token cap passed to inspect's
            eval(); None keeps the provider default (open models that reason at
            length before emitting <final_code> need more than the tinker
            provider's 4096 default).
    """

    model_info: ModelInfo
    eval_mode: EvalMode
    data_split: DataSplit
    dataset_type: DatasetType = DatasetType.APPS
    max_tokens: Optional[int] = None


class RunLabel(StrEnum):
    """The label of the run."""

    ATTACKS = "attacks"
    HONEYPOTS = "honeypots"
    HONEST = "honest"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for a single monitoring run.

    Attributes:
        label: The label of the run.
        eval_path: The path to the eval file.
        save_path: The path to the save file.
        limit: The limit of the run.
    """

    label: RunLabel
    save_path: Path
    limit: Optional[int]


@dataclass(frozen=True, slots=True)
class GeneratorRunConfig(RunConfig):
    """Configuration for a single generator run."""

    generator: RunGenerator


@dataclass(frozen=True, slots=True)
class TransformRunConfig(RunConfig):
    """Configuration for a single transform run."""

    eval_path: Path
    transform: TransformInfo


@dataclass(frozen=True, slots=True)
class MonitorRunConfig(RunConfig):
    """Configuration for a single monitoring run.

    view: "trajectory" scores each finished trajectory once; "action" scores every
    action in BashArena's per-action format (see basharena_views.action_views),
    with the same action_options as the SFT data.
    """

    eval_path: Path
    monitors: RunMonitors
    view: str = TRAJECTORY_VIEW
    action_options: Optional[ActionViewOptions] = None


# Prompt registry - populated at module level
_PROMPT_REGISTRY: dict[str, str] = {}


def _init_prompt_registry() -> None:
    """Automatically initialize the prompt registry with all string prompts defined in the prompts package."""
    if _PROMPT_REGISTRY:
        return

    import importlib
    import pkgutil

    # Find all modules in the prompts package
    prompts_pkg = "lasr_labs_2025_control_project.prompts"
    try:
        pkg = importlib.import_module(prompts_pkg)
    except ImportError:
        raise RuntimeError(f"Could not import prompts package: {prompts_pkg}")

    # Iterate through all modules in the prompts package
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, prompts_pkg + "."):
        if ispkg:
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue  # skip modules that fail to import

        # Add all UPPERCASE string variables to the registry
        for attr in dir(mod):
            value = getattr(mod, attr)
            if isinstance(value, str):
                _PROMPT_REGISTRY[attr] = value


def get_prompt_by_name(prompt_name: str) -> str:
    """Retrieve a prompt by name from the registry."""
    _init_prompt_registry()
    try:
        return _PROMPT_REGISTRY[prompt_name]
    except KeyError:
        available = ", ".join(_PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown prompt '{prompt_name}'. Available: {available}"
        ) from None
