#!/usr/bin/env bash
# Usage: run_central_game.sh <label> <stages> <collusion prompt or -> [extra overrides...]
# One centralized-ICRL safety game on rewritten attacks, paraphrased only. Honest scores are
# reused: the game reads no honest collusion score. Without allow_paid=true this prints the plan.
set -uo pipefail
cd "$(dirname "$0")"
label=$1 stages=$2 prompt=$3
shift 3
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
G=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/icrl/experiments/icrl-central-r2-20261101/game
M=$PWD/experiments/apps/monitor
args=()
[ "$prompt" != - ] && args+=("+measurement.arms.base.collusion_prompt=$prompt")
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=untrusted-runs
exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
  "models.U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'" "name=central_$label" \
  models.generation_max_tokens=65536 '~measurement.arms.sft' "stages=[$stages]" \
  'measurement.transforms=[paraphrased]' \
  optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
  "+inputs.generate.test.attacks=$G/$label/attacks.eval" \
  "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
  "+inputs.monitor.paraphrased.base.honest=$M/apps-262144_ft_a1611f-test-honest-honest.para-qwen3_5_4b-llm_summarize.mon-262144_ft_a1611f-self_rec_collusion-8062ab61d0cd/honest.eval" \
  'wandb_tags=[v41,central-icrl]' ${args[@]+"${args[@]}"} "$@"
