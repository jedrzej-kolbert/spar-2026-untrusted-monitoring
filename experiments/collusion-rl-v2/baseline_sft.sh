#!/usr/bin/env bash
# Rescore the SFT-judge test baselines (V4.1 none and paraphrased SFT judges) on today's serving,
# in W&B project collusion-rl-v2. Args: extra pipeline overrides (allow_paid=true to run).
set -euo pipefail
cd /home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/icrl
source experiments/collusion-rl-v2/env.sh
V=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=rlv2_baseline_sft \
  models.generation_max_tokens=65536 '~measurement.arms.base' 'stages=[transform,monitor,game,analysis]' \
  'measurement.transforms=[none,paraphrased]' \
  optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
  "+inputs.generate.test.attacks=$V/generate/apps-262144_ft_a1611f-test-attacks-self_rec_subtle_attack-6f5c6a86b195/attacks.eval" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.generate.test.honest=$V/generate/apps-262144_ft_a1611f-test-honest-honest-267d38248808/honest.eval" \
  "+inputs.finetune.self_rec.none=tinker/tinker://2fcd7478-7f4f-5b17-abf4-6afc1d3b4b02:train:0/sampler_weights/self_rec_base" \
  "+inputs.finetune.self_rec.paraphrased=tinker/tinker://5d7065c1-0581-5cc2-a31b-e465de6721af:train:0/sampler_weights/self_rec_paraphrased" \
  'wandb_tags=[collusion-rl-v2,baseline-sft]' "$@"
