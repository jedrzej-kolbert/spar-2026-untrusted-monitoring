#!/usr/bin/env bash
# Usage: run_v41.sh <name> <log> [extra hydra overrides...]  (V4.1 self-rec pipeline, generate only)
set -uo pipefail
cd "$(dirname "$0")"
name=$1 log=$2
shift 2
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" name="$name" \
  limits=dry stages=[generate] allow_paid=true "$@" > "$log" 2>&1
echo "exit=$?" >> "$log"
