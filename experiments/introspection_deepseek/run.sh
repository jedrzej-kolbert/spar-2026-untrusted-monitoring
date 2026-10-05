#!/usr/bin/env bash
# Does introspection (self-prediction) training make DeepSeek-V3.1 better at
# recognising its own APPS solutions? See README.md for the design.
#
# REQUIREMENTS: .env with TINKER_API_KEY; Docker running (APPS scorer, stage
# `generate` only); uv.
#
# USAGE: bash experiments/introspection_deepseek/run.sh [stage]
#   stage: fetch | generate | build | train | accuracy | compare | all (default)
#          | monitor | plot   (standard pipeline; not part of `all`, need MONITORED)
# Optional cross-prediction control (train DeepSeek on gpt-oss-120b's behaviour):
#   CROSS=1 bash experiments/introspection_deepseek/run.sh build   (then train, ...)
#
# Knobs (env vars): MODEL, CROSS_MODEL, PER_FILE, EPOCHS, LR, RANK, BATCH, EVERY,
# ATTACKS, HONEYPOTS (existing .eval logs to evaluate on instead of generating),
# MONITORED (dir of existing monitor logs attacks/honest/honeypots.eval that
# already carry the trusted/untrusted suspiciousness scores).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-all}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3.1}"
CROSS_MODEL="${CROSS_MODEL:-openai/gpt-oss-120b}"
PER_FILE="${PER_FILE:-400}"   # 26 train files -> ~10k examples
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
RANK="${RANK:-32}"
BATCH="${BATCH:-64}"
EVERY="${EVERY:-40}"          # AUC checkpoint cadence (gradient updates)
CROSS="${CROSS:-0}"

EXP=experiments/introspection_deepseek
RUNS=$EXP/runs
REL=$RUNS/introspection_self_prediction/dataset_release
GEN=$RUNS/apps_test/generator
# Point these at existing generator logs to skip `generate` (and Docker).
ATTACKS="${ATTACKS:-$GEN/attacks.eval}"
HONEYPOTS="${HONEYPOTS:-$GEN/honeypots.eval}"

if [ "$CROSS" = 1 ]; then ARM=cross; LABEL_MODEL="$CROSS_MODEL"; else ARM=self; LABEL_MODEL="$MODEL"; fi
ARM_DIR=$RUNS/$ARM
CKPT_FILE=$ARM_DIR/checkpoint_path.txt

py() { PYTHONPATH=src uv run python -m "$@"; }

stage_fetch() {
  echo "=== fetch Binder et al. prompts (labels are made per model in 'build')"
  [ -d "$REL" ] || git clone --depth 1 \
    https://github.com/felixbinder/introspection_self_prediction.git \
    "$RUNS/introspection_self_prediction"
}

stage_generate() {
  echo "=== APPS self-recognition eval set: $MODEL attacks vs $CROSS_MODEL honeypots"
  [ -f "$ATTACKS" ] && [ -f "$HONEYPOTS" ] && { echo "exists, skipping"; return; }
  docker info >/dev/null 2>&1 || { echo "Docker is not running (APPS scorer)." >&2; exit 1; }
  py lasr_labs_2025_control_project.scripts.generate_solutions_inspect \
    --config "$EXP/generator_config.yaml"
}

stage_build() {
  echo "=== [$ARM] training data labelled by $LABEL_MODEL's own answers"
  [ -f "$ARM_DIR/train.jsonl" ] && { echo "exists, skipping"; return; }
  py lasr_labs_2025_control_project.finetuning.introspection_data build \
    --data-dir "$REL/train" --model "$LABEL_MODEL" \
    --out "$ARM_DIR/train.jsonl" --per-file "$PER_FILE"
}

stage_train() {
  echo "=== [$ARM] LoRA on $MODEL; self-rec AUC every $EVERY updates"
  [ -f "$CKPT_FILE" ] && { echo "exists: $(cat "$CKPT_FILE")"; return; }
  # --val-frac 0: the trainer's held-out check is for YES/NO labels.
  py lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
    --train-jsonl "$ARM_DIR/train.jsonl" --base-model "$MODEL" \
    --rank "$RANK" --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH" \
    --checkpoint-name "introspection-$ARM" --val-frac 0 \
    --checkpoint-every-steps "$EVERY" \
    --attacks-eval "$ATTACKS" --honeypots-eval "$HONEYPOTS" \
    --dynamics-results "$ARM_DIR/dynamics" \
    --out-path-file "$CKPT_FILE"
}

stage_accuracy() {
  echo "=== self-prediction accuracy on held-out tasks (did training work?)"
  local ckpt; ckpt=$(cat "$CKPT_FILE")
  [ -f "$RUNS/accuracy_base.json" ] || py lasr_labs_2025_control_project.finetuning.introspection_data \
    accuracy --data-dir "$REL/test" --model "$MODEL" --out "$RUNS/accuracy_base.json"
  py lasr_labs_2025_control_project.finetuning.introspection_data accuracy \
    --data-dir "$REL/test" --model "$ckpt" --out "$ARM_DIR/accuracy.json"
}

stage_compare() {
  echo "=== paired self-rec AUC: base vs trained arm(s)"
  local args=(--model "$MODEL")
  for arm in self cross; do
    [ -f "$RUNS/$arm/checkpoint_path.txt" ] && args+=(--model "$(cat "$RUNS/$arm/checkpoint_path.txt")")
  done
  py lasr_labs_2025_control_project.scripts.compare_self_rec_auc \
    --attacks-eval "$ATTACKS" --honeypots-eval "$HONEYPOTS" \
    "${args[@]}" --out "$RUNS/compare"
}

PIPE=$RUNS/pipeline
read -ra PIPE_ARMS <<< "${PIPE_ARMS:-base self cross}"   # e.g. PIPE_ARMS=self to run one arm

arm_model() { if [ "$1" = base ]; then echo "$MODEL"; else cat "$RUNS/$1/checkpoint_path.txt"; fi; }
arm_name() {
  case "$1" in
    base) echo "Base $MODEL" ;;
    self) echo "Introspection SFT (own answers)" ;;
    cross) echo "Control SFT ($CROSS_MODEL answers)" ;;
  esac
}

stage_monitor() {
  echo "=== standard pipeline: score each arm as the collusion monitor on $MONITORED"
  [ -d "${MONITORED:-}" ] || { echo "set MONITORED=<dir of existing monitor logs>" >&2; exit 2; }
  for arm in "${PIPE_ARMS[@]}"; do
    [ "$arm" = base ] || [ -f "$RUNS/$arm/checkpoint_path.txt" ] || continue
    [ -f "$PIPE/$arm/honeypots.eval" ] && { echo "[$arm] exists, skipping"; continue; }
    mkdir -p "$PIPE/$arm"
    sed -e "s|__ARM__|$arm|g" -e "s|__NAME__|$(arm_name "$arm")|" \
      -e "s|__MODEL__|$(arm_model "$arm")|" -e "s|__MONITORED__|$(realpath "$MONITORED")|" \
      "$EXP/monitor_config.template.yaml" > "$PIPE/$arm/monitor_config.yaml"
    py lasr_labs_2025_control_project.scripts.monitor_solutions_inspect \
      --config "$PIPE/$arm/monitor_config.yaml"
  done
}

stage_plot() {
  echo "=== standard pipeline plots (histograms, ROC, safety-usefulness) across arms"
  local cfgs=()
  for arm in "${PIPE_ARMS[@]}"; do
    [ -f "$PIPE/$arm/honeypots.eval" ] && cfgs+=("$PIPE/$arm/monitor_config.yaml")
  done
  py lasr_labs_2025_control_project.scripts.plot_data --config "${cfgs[@]}" --grouped-path arms
}

case "$STAGE" in
  fetch|generate|build|train|accuracy|compare|monitor|plot) "stage_$STAGE" ;;
  all) stage_fetch; stage_generate; stage_build; stage_train; stage_accuracy; stage_compare ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
