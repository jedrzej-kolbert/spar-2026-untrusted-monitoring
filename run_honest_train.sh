#!/usr/bin/env bash
# Usage: run_honest_train.sh <log> [overrides...]  V4.1-Flash honest code on the train split.
set -uo pipefail
cd "$(dirname "$0")"
log=$1; shift
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec_controls \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=self_rec_v41 \
  models.generation_max_tokens=65536 "stages=[generate]" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.generate.train.honest_honeypots=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps/generate/apps-gpt_oss_120b-train-honeypots-self_rec_subtle_attack-bc819434c195/honeypots.eval" \
  'wandb_tags=[v41,honest-train]' "$@" > "$log" 2>&1
echo "exit=$?" >> "$log"
