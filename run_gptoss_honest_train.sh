#!/usr/bin/env bash
# Usage: run_gptoss_honest_train.sh <log> [overrides...]  gpt-oss-120b (H) honest code on the train split.
set -uo pipefail
cd "$(dirname "$0")"
log=$1; shift
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
# Only generate.train.honest_honeypots is left to run: U's train honest code is the finished V4.1 log.
uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec_controls \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=self_rec_v41 \
  models.generation_max_tokens=65536 "stages=[generate]" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.generate.train.honest=$PWD/experiments/apps/generate/apps-262144_ft_a1611f-train-honest-honest-c7c7f56a99bd/honest.eval" \
  'wandb_tags=[gpt-oss,honest-train]' "$@" > "$log" 2>&1
echo "exit=$?" >> "$log"
