#!/usr/bin/env bash
# Usage: run_step2.sh <stage,stage,...> <log> [extra overrides...]   e.g. run_step2.sh generate logs/v41/step2_generate.log
# Plan step 2 (V4.1 base arm, full limits) at the 65,536-token cap, with the V3.1 run's test honeypots.
set -uo pipefail
cd "$(dirname "$0")"
stages="[$1]" log=$2
shift 2
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name=self_rec_v41 \
  models.generation_max_tokens=65536 '~measurement.arms.sft' "stages=$stages" \
  optimization.bootstrap.replicates=200 \
  '+inputs.generate.test.honeypots=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64/self_rec_attack_test/generator/honeypots.eval' \
  allow_paid=true 'wandb_tags=[v41,base]' "$@" > "$log" 2>&1
echo "exit=$?" >> "$log"
