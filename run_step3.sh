#!/usr/bin/env bash
# Usage: run_step3.sh <stage,stage,...> <log> [extra overrides...]   (add allow_paid=true to run paid steps)
# Plan step 3 (V4.1 SFT judges, full limits) at the 65,536-token cap, with the V3.1 run's test and train honeypots.
set -uo pipefail
cd "$(dirname "$0")"
stages="[$1]" log=$2
shift 2
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=self_rec_v41 \
  models.generation_max_tokens=65536 "stages=$stages" \
  optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
  elicitation.training.checkpoint_every_steps=9 elicitation.training.checkpoint_auc=true \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.generate.train.honeypots=$X/SFT_self_rec_train/generator/honeypots.eval" \
  'wandb_tags=[v41,sft]' "$@" > "$log" 2>&1
echo "exit=$?" >> "$log"
