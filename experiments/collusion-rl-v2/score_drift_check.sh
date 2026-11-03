#!/usr/bin/env bash
# Score the first 20 original test attacks (no jamming, base judge) with today's monitor code
# (one-request YES/NO), to compare with the stored V4.1 scores. Extra args: allow_paid=true.
set -euo pipefail
cd "$(dirname "$0")/../.."
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=collusion-rl-v2-smoke
V=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
M=$V/monitor
base=262144_ft_a1611f-self_rec_collusion
exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=rlv2_prefill_check \
  models.generation_max_tokens=65536 '~measurement.arms.sft' 'stages=[monitor]' \
  'measurement.transforms=[none]' measurement.limits.monitor=20 \
  "+inputs.generate.test.attacks=$V/generate/apps-262144_ft_a1611f-test-attacks-self_rec_subtle_attack-6f5c6a86b195/attacks.eval" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.monitor.none.base.honest=$M/apps-262144_ft_a1611f-test-honest-honest.mon-$base-57a5bb885f16/honest.eval" \
  "+inputs.monitor.none.base.honeypots=$M/untracked-honeypots.mon-$base-0e73330bf758/honeypots.eval" \
  'wandb_tags=[collusion-rl-v2,wiring]' "$@"
