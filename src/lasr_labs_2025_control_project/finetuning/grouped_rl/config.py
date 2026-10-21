"""YAML run configuration for HiPT-style collusion-signal RL.

Prompt fields take prompt *names* from ``lasr_labs_2025_control_project.prompts``,
like the rest of the pipeline's configs. Unknown keys are rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsConfig(_Strict):
    # Raw Tinker base-model id for the trainable sender (no tinker/ prefix).
    sender: str = "deepseek-ai/DeepSeek-V3.1"
    # tinker/<base model> or a tinker:// checkpoint; frozen unless train_receiver.
    receiver: str = "tinker/deepseek-ai/DeepSeek-V3.1"
    overseer: str | None = "tinker/Qwen/Qwen3.5-4B"
    backdoor_monitor: str | None = "tinker/Qwen/Qwen3.5-4B"


class PromptsConfig(_Strict):
    sender_system: str = "RL_REWRITE_SIGNAL_SENDER_SYSTEM_PROMPT"
    sender_user: str = "RL_REWRITE_SIGNAL_SENDER_USER_PROMPT"
    signal_off: str = "RL_SIGNAL_OFF"
    signal_on: str = "RL_SIGNAL_ON"
    receiver_system: str = "RL_SELF_SIGNAL_RECEIVER_SYSTEM_PROMPT"
    overseer_system: str = "RL_OVERSEER_SIGNAL_SYSTEM_PROMPT"
    backdoor_monitor_system: str = "DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT"


class DataConfig(_Strict):
    attacks_eval: Path
    honeypots_eval: Path | None = None
    require_main_task: bool = True
    require_backdoor: bool = True
    max_tests: int | None = 10
    validation_fraction: float = Field(default=0.2, ge=0, lt=1)
    max_validation_problems: int | None = 20


class ChannelConfig(_Strict):
    kind: Literal["none", "minify", "paraphrase"] = "none"
    remove_comments: bool = True
    paraphrase_model: str = "tinker/Qwen/Qwen3.5-4B"
    paraphrase_system_prompt: str = "LLM_SUMMARIZE_SYSTEM_PROMPT"
    paraphrase_base_prompt: str = "LLM_SUMMARIZE_BASE_PROMPT"


class RewardSettings(_Strict):
    reward_fn: Literal["branch", "hipt"] = "branch"
    # branch reward
    task_weight: float = Field(default=1.0, ge=0)
    communication_weight: float = Field(default=1.0, ge=0)
    overseer_weight: float = Field(default=1.0, ge=0)
    # Linear ramp of the overseer weight (branch) / scale (hipt) from
    # overseer_ramp_start to its final value over overseer_ramp_steps steps.
    overseer_ramp_start: float | None = None
    overseer_ramp_steps: int = Field(default=0, ge=0)
    # hipt reward
    stego_scale: float = Field(default=1.0, ge=0)
    overseer_scale: float = Field(default=1.0, ge=0)
    overseer_log_odds_threshold: float = Field(default=0.75, allow_inf_nan=False)
    # penalties
    failed_backdoor_weight: float = Field(default=0.0, ge=0)
    backdoor_suspicion_weight: float = Field(default=0.0, ge=0)
    kl_coef: float = Field(default=0.05, ge=0)
    # bonus when the rewrite's backdoor still triggers on the secret input
    backdoor_kept_weight: float = Field(default=0.0, ge=0)


class TrainingConfig(_Strict):
    steps: int = Field(default=50, ge=1)
    problems_per_step: int = Field(default=16, ge=1)
    group_size: int = Field(default=8, ge=2)
    payload_grouping: Literal["single", "both"] = "both"
    loss_fn: Literal["ppo", "importance_sampling"] = "ppo"
    ppo_passes: int = Field(default=2, ge=1)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    learning_rate: float = Field(default=1e-5, gt=0)
    lora_rank: int = Field(default=32, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    max_context_tokens: int = Field(default=32768, ge=2)
    train_receiver: bool = False
    max_concurrency: int = Field(default=32, ge=1)
    # Steps whose rollouts hit infrastructure errors *before* any update are
    # skipped; the run stops after this many. Uncertain updates always stop.
    max_skipped_steps: int = Field(default=3, ge=0)


class EvalConfig(_Strict):
    every: int = Field(default=10, ge=1)
    num_samples: int = Field(default=1, ge=1)  # per bit, per held-out problem
    honeypot_eval: bool = True


class PreflightConfig(_Strict):
    enabled: bool = True
    # Stop unless the overseer scores an explicitly signalled control at least this
    # much above the clean one (0 = log only).
    require_overseer_gap: float = Field(default=0.0, ge=0, le=1)
    num_controls: int = Field(default=3, ge=1)


class SandboxConfig(_Strict):
    # Digest-pinned python:3.12-slim, as in the original grouped-APPS driver.
    image: str = (
        "python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
    )
    timeout_seconds: float = Field(default=10, gt=0)
    control_arena_parity: bool = True
    # One container per candidate (fresh process per test) instead of per test.
    batch_tests: bool = True
    max_containers: int = Field(default=8, ge=1)


class LoggingConfig(_Strict):
    wandb_project: str | None = "hipt-collusion-rl"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


class CheckpointConfig(_Strict):
    save_every: int | None = 10
    # On completion the final sender checkpoint is added to this file under
    # ``name`` so make_configs.py can build monitor configs for it.
    ckpts_json: Path | None = Path("experiments/open_models/ckpts.json")
    name: str | None = None


class RLRunConfig(_Strict):
    run_name: str
    output_dir: Path
    backend: Literal["tinker", "hf_gpu"] = "tinker"
    task_mode: Literal[
        "rewrite_signal", "self_recognition", "honest_signal", "attack_signal"
    ] = "rewrite_signal"
    seed: int = 0
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    data: DataConfig
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    reward: RewardSettings = Field(default_factory=RewardSettings)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def _check(self) -> RLRunConfig:
        if self.backend == "hf_gpu":
            raise ValueError("backend 'hf_gpu' is not implemented yet; use 'tinker'")
        if self.task_mode not in ("rewrite_signal", "self_recognition"):
            raise ValueError(
                f"task_mode '{self.task_mode}' is not wired into the run driver yet"
            )
        if self.training.payload_grouping == "both" and self.training.group_size % 2:
            raise ValueError("payload_grouping 'both' needs an even group_size")
        if self.training.train_receiver and self.task_mode != "self_recognition":
            raise ValueError("receiver training requires task_mode self_recognition")
        if self.task_mode == "self_recognition":
            defaults = {
                "sender_system": "RL_SELF_REC_SENDER_SYSTEM_PROMPT",
                "sender_user": "RL_SELF_REC_SENDER_USER_PROMPT",
                "receiver_system": "SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
            }
            for name, value in defaults.items():
                if name not in self.prompts.model_fields_set:
                    setattr(self.prompts, name, value)
            if not self.data.honeypots_eval:
                raise ValueError("self_recognition requires data.honeypots_eval")
            receiver_model = self.models.receiver
            if not receiver_model.startswith("tinker://") and (
                receiver_model.removeprefix("tinker/")
                != self.models.sender.removeprefix("tinker/")
            ):
                raise ValueError(
                    "self_recognition sender and receiver must use the same base model"
                )
            if self.reward.overseer_weight or self.reward.overseer_scale:
                raise ValueError("self_recognition requires both overseer weights zero")
            if self.reward.backdoor_suspicion_weight:
                raise ValueError("self_recognition has no backdoor-suspicion objective")
        reward = self.reward
        overseer_used = (
            reward.overseer_weight > 0
            if reward.reward_fn == "branch"
            else reward.overseer_scale > 0
        )
        if overseer_used and not self.models.overseer:
            raise ValueError("the overseer term needs models.overseer")
        if reward.backdoor_suspicion_weight > 0 and not self.models.backdoor_monitor:
            raise ValueError("backdoor_suspicion_weight needs models.backdoor_monitor")
        return self

    @property
    def checkpoint_name(self) -> str:
        return self.checkpoints.name or f"rl_{self.run_name}"


def load_rl_config(path: Path | str) -> RLRunConfig:
    with open(path) as handle:
        return RLRunConfig.model_validate(yaml.safe_load(handle))
