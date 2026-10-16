"""Typed schema for the pipeline config in ``conf/``.

Hydra validates the composed YAML against these dataclasses, so a misspelled
key or a wrong type fails before anything runs. Each section is one class; to
add an option, add a field here (with its default) and use it in
``graph.py``. Prompt fields hold prompt *names* from
``lasr_labs_2025_control_project.prompts``, as everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

STAGES = ("generate", "transform", "finetune", "monitor", "analyse")
MONITORS = ("untrusted_collusion", "trusted_suspiciousness", "untrusted_suspiciousness")
#: analysis variant -> monitors it needs
VARIANTS = {
    "all_monitors": ("untrusted_collusion", "trusted_suspiciousness"),
    "um_only": ("untrusted_collusion",),
    "tm_only": ("trusted_suspiciousness",),
}
SOLVERS = ("double_oracle", "stackelberg")
NO_TRANSFORM = "none"


@dataclass
class ModelsConfig:
    """Model roles: U untrusted (policy + untrusted monitors, the model SFT'd),
    T trusted monitor and paraphraser, H honeypot generator."""

    U: str = MISSING
    T: str = MISSING
    H: str = MISSING
    #: completion cap for generation; None keeps the provider default
    generation_max_tokens: Optional[int] = None


@dataclass
class ActionViewConfig:
    """Per-action views (``view: action``, BashArena); see utils/basharena_views."""

    max_history_chars: Optional[int] = None
    #: SFT only: at most this many actions per trajectory (monitoring keeps all)
    actions_per_trajectory: Optional[int] = None


@dataclass
class SettingPrompts:
    #: collusion strategy -> U's attack-policy prompt
    attack: dict[str, str] = field(default_factory=dict)
    honest: str = MISSING
    #: H's prompt; the same in every condition, so honeypots never depend on red's strategy
    honeypot: str = MISSING
    #: collusion strategy -> untrusted collusion monitor system prompt
    collusion: dict[str, str] = field(default_factory=dict)
    collusion_user: str = MISSING
    #: transform -> untrusted suspiciousness monitor system prompt
    suspiciousness: dict[str, str] = field(default_factory=dict)
    trusted_suspiciousness: str = MISSING
    suspiciousness_user: str = MISSING
    #: prompts the SFT'd collusion monitor is trained with
    sft_system: str = MISSING
    sft_user: str = MISSING


@dataclass
class SettingConfig:
    """A dataset and everything that depends on it (``conf/setting/``)."""

    name: str = MISSING
    #: inclusive [first, last] problem-id range (APPS); train uses odd ids, test even
    problem_ids: Optional[list[int]] = None
    #: "trajectory" (one solution per sample) or "action" (one per agent action)
    view: str = "trajectory"
    action_view: ActionViewConfig = field(default_factory=ActionViewConfig)
    #: transforms this setting supports
    transforms: list[str] = field(default_factory=lambda: [NO_TRANSFORM])
    #: generation limit used when a section sets none (settings that need one)
    default_generation_limit: Optional[int] = None
    #: parallel samples for settings with their own generation runtime
    generation_concurrency: int = 1
    prompts: SettingPrompts = field(default_factory=SettingPrompts)


@dataclass
class TransformConfig:
    """A signal-jamming transform; ``type`` is "minifier" or "llm"."""

    type: str = MISSING
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    base_prompt: Optional[str] = None
    remove_comments: bool = True


@dataclass
class GenerationConfig:
    """Sampling settings shared by every model call in the pipeline."""

    #: None keeps each provider's default
    temperature: Optional[float] = None
    max_connections: int = 8
    transform_max_connections: int = 48


@dataclass
class TrainingConfig:
    rank: int = 32
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 64
    #: held-out fraction for the post-training collapse check (Tinker)
    val_frac: float = 0.05
    #: save and score a checkpoint every N steps (Tinker; 0 = final only)
    checkpoint_every_steps: int = 0
    #: score checkpoints on this experiment's test attacks vs honeypots (Tinker)
    checkpoint_auc: bool = False
    upsample_pos_factor: int = 1
    upsample_neg_factor: int = 1
    keep_comments: bool = False
    #: OpenAI backend only; None = OpenAI's automatic choice
    openai_n_epochs: Optional[int] = None


@dataclass
class ElicitationArm:
    """One way of fine-tuning U into a collusion monitor, trained once per
    transform that a monitor arm uses it on."""

    #: mode of the training generations: "attack" (backdoored) or "honest"
    eval_mode: str = "attack"
    #: U's training prompt; default: the attack (or honest) prompt for the collusion strategy
    attack_prompt: Optional[str] = None
    #: H's training prompt; default: the setting's honeypot (or honest) prompt
    honeypot_prompt: Optional[str] = None
    #: "author" (YES = written by U) or "permuted" (labels shuffled; a control)
    labels: str = "author"


@dataclass
class ElicitationLimits:
    #: problems attempted per training generation
    generate: Optional[int] = None
    #: training examples used per fine-tune
    examples: Optional[int] = None


@dataclass
class ElicitationConfig:
    """Fine-tuning U (the SFT arms)."""

    arms: dict[str, Optional[ElicitationArm]] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    limits: ElicitationLimits = field(default_factory=ElicitationLimits)
    #: refuse to train with fewer usable examples than this per label
    min_examples_per_label: int = 10


@dataclass
class MonitorArm:
    """Which collusion monitor scores the solutions."""

    #: "U" (base), "sft:<elicitation arm>", or a literal model id
    model: str = "U"
    #: default: the setting's collusion prompt for the collusion strategy
    collusion_prompt: Optional[str] = None
    #: "honest" scores U's honest solutions in the attacks slot (no-backdoor control)
    attacks_label: str = "attacks"
    #: transforms to measure this arm on; default: all of measurement.transforms
    transforms: Optional[list[str]] = None


@dataclass
class MeasurementLimits:
    #: problems attempted per test generation
    generate: Optional[int] = None
    #: samples scored per monitor run
    monitor: Optional[int] = None


@dataclass
class MeasurementConfig:
    """Test-split generation, transforms and monitoring."""

    collusion_strategy: str = "self_rec"
    #: U's attack prompt; default: the setting's prompt for the collusion strategy
    attack_prompt: Optional[str] = None
    honeypot_prompt: Optional[str] = None
    transforms: list[str] = field(
        default_factory=lambda: [NO_TRANSFORM, "minified", "paraphrased"]
    )
    arms: dict[str, Optional[MonitorArm]] = field(default_factory=dict)
    monitors: list[str] = field(default_factory=lambda: list(MONITORS))
    #: reuse generations across collusion strategies when their config matches
    share_generations: bool = True
    limits: MeasurementLimits = field(default_factory=MeasurementLimits)
    #: refuse to analyse with fewer usable samples than this per label
    min_samples_per_label: int = 10


@dataclass
class GameConfig:
    """Fixed (non-empirical) game parameters."""

    num_tasks: int = 5000
    slack_honeypots: float = 5.0
    blue_team_budget_fraction: float = 0.02
    #: P(a human audit of a real attack incriminates red)
    human_audit_success_rate: float = 1.0


@dataclass
class BootstrapConfig:
    replicates: int = 2
    ci: float = 0.80
    n_jobs: int = -1
    #: replicates per saved chunk (an interrupted bootstrap resumes from the last chunk)
    chunk_size: int = 25


@dataclass
class OptimizationSection:
    """Solving the game for each condition."""

    game: GameConfig = field(default_factory=GameConfig)
    #: blue_team_replace_fnr values swept; usefulness = 1 - cutoff
    regeneration_cutoffs: list[float] = field(
        default_factory=lambda: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99]
    )
    solver: str = "double_oracle"
    max_iterations: int = 100
    #: analysis variants (see VARIANTS); the first is the primary one
    variants: list[str] = field(default_factory=lambda: ["all_monitors", "um_only"])
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass
class Config:
    #: experiment name: run folder, W&B tags and config artifact
    name: str = "adhoc"
    seed: int = 0
    #: stages to run; outputs of other stages must already exist
    stages: list[str] = field(default_factory=lambda: list(STAGES))
    #: without it, a run that would call paid models only prints its plan
    allow_paid: bool = False
    allow_unlinked: bool = False
    #: "local", or "wandb" to also reuse matching outputs logged to W&B
    reuse: str = "local"
    experiments_dir: str = "experiments"
    #: node id -> existing output (path, wandb:<entity>/<project>/<name>:vN, or model id)
    inputs: dict[str, Any] = field(default_factory=dict)
    wandb_tags: list[str] = field(default_factory=list)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    setting: SettingConfig = field(default_factory=SettingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transforms: dict[str, TransformConfig] = field(default_factory=dict)
    elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    optimization: OptimizationSection = field(default_factory=OptimizationSection)


def register() -> None:
    ConfigStore.instance().store(name="pipeline_schema", node=Config)
