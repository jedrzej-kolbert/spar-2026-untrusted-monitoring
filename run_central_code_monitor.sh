#!/usr/bin/env bash
# Usage: run_central_code_monitor.sh <transform> <label> <collusion prompt or -> [overrides...]
# U monitors on one arm's rewrites, on a code channel. T is merged in later and honeypot or
# honest monitors come from existing runs, so only the rewrites are paid for.
set -uo pipefail
cd "$(dirname "$0")"
transform=$1 label=$2 prompt=$3
shift 3
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
G=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/icrl/experiments/icrl-central-r2-20261101/game
M=$PWD/experiments/apps/monitor
case $transform in
  none) honest=apps-262144_ft_a1611f-test-honest-honest.mon-262144_ft_a1611f-self_rec_collusion-57a5bb885f16
        honeypots=untracked-honeypots.mon-262144_ft_a1611f-self_rec_collusion-0e73330bf758 ;;
  minified) honest=apps-262144_ft_a1611f-test-honest-honest.min.mon-262144_ft_a1611f-self_rec_collusion-c60920deaeaa
        honeypots=untracked-honeypots.min.mon-262144_ft_a1611f-self_rec_collusion-aee175c0c5c5 ;;
esac
args=()
[ "$prompt" != - ] && args+=("+measurement.arms.base.collusion_prompt=$prompt")
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" "name=central_${label}_$transform" \
  models.generation_max_tokens=65536 '~measurement.arms.sft' "stages=[transform,monitor]" \
  "measurement.transforms=[$transform]" 'measurement.monitors=[untrusted_collusion,untrusted_suspiciousness]' \
  'optimization.variants=[um_only]' \
  "+inputs.generate.test.attacks=$G/$label/attacks.eval" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.monitor.$transform.base.honest=$M/$honest/honest.eval" \
  "+inputs.monitor.$transform.base.honeypots=$M/$honeypots/honeypots.eval" \
  'wandb_tags=[v41,central-icrl]' ${args[@]+"${args[@]}"} "$@"
