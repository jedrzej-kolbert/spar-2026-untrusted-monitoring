#!/usr/bin/env bash
# Usage: run_central_merged_game.sh <transform> <arm> <t> [overrides...]
# Free: one game from the merged monitor files in experiments/central_merged/.
set -uo pipefail
cd "$(dirname "$0")"
transform=$1 arm=$2 t=$3
shift 3
F=$PWD/experiments/central_merged/${transform}_${arm}_$t
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" "name=central_t_${transform}_${arm}_$t" \
  models.generation_max_tokens=65536 '~measurement.arms.sft' 'stages=[game]' \
  "measurement.transforms=[$transform]" \
  optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
  "+inputs.monitor.$transform.base.attacks=$F/attacks.eval" \
  "+inputs.monitor.$transform.base.honeypots=$F/honeypots.eval" \
  "+inputs.monitor.$transform.base.honest=$F/honest.eval" \
  allow_unlinked=true 'wandb_tags=[v41,central-icrl,t-comparison]' "$@"
