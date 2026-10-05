#!/usr/bin/env bash
# Passive self-recognition SFT on BigCodeBench, open models (Tinker), end to end.
#
# The BigCodeBench counterpart of src/lasr_labs_2025_control_project/sft-steps-bcb.md:
# it runs exactly those steps in order. Three SFT arms, one per signal-jamming
# strategy (none / minified / paraphrased); each arm LoRA-finetunes U to
# recognise its own attacks against H's honeypots, then U-SFT is scored as the
# untrusted collusion monitor alongside T and base U.
#
# REQUIREMENTS
#   * Docker running. The BigCodeBench scorer runs every solution in a sandbox,
#     and the SFT data keeps only examples where both the main task and the
#     backdoor succeed. The first run builds the sandbox image (several minutes).
#   * .env with TINKER_API_KEY.
#   * uv sync done.
#
# USAGE
#   bash experiments/open_models/run_bcb_sft.sh [--U MODEL] [--T MODEL] [--H MODEL]
#     [--max-connections N] [stage]
#
# MODEL is a raw Tinker id (e.g. deepseek-ai/DeepSeek-V3.1); a tinker/ prefix is
# also accepted. Pass the SAME models to every stage: configs are regenerated
# in `configs` and `configure`, and they take U/T/H from these flags.
#
# --max-connections N (default 8) caps parallel samples in generation and
# monitoring, i.e. both Tinker connections and Docker sandboxes. Each sandbox may
# use up to 0.5 GB, so raise it only with Docker memory to spare (e.g. 32-48 on
# a machine giving Docker 16+ GB). An overloaded host can turn correct code into
# test timeouts, so compare main_task_success against an 8-connection run.
#
# stage is one of: configs | train-gen | train-transform | prepare | finetune |
# configure | test-gen | test-transform | monitor | plot | all (default).
# Each stage skips work already done: an .eval counts as done only if its log
# reports success (a failed run's log is redone), and data prep / finetunes skip
# when their output file exists. An interrupted run resumes by re-running the
# same command. Delete a file to redo it.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

U="deepseek-ai/DeepSeek-V3.1"
T="Qwen/Qwen3.5-4B"
H="openai/gpt-oss-20b"
MAX_CONN=8
STAGE="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --U) U="$2"; shift 2 ;;
    --T) T="$2"; shift 2 ;;
    --H) H="$2"; shift 2 ;;
    --max-connections) MAX_CONN="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) STAGE="$1"; shift ;;
  esac
done
U="${U#tinker/}"; T="${T#tinker/}"; H="${H#tinker/}"

ARMS=(none minified paraphrased)
CFG=src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec
TRAIN_EXP=experiments/open-source-model/bigcodebench/bcb_train
TEST_EXP=experiments/open-source-model/bigcodebench/self_rec
SFT_DATA=$TRAIN_EXP/sft_data
CKPTS=experiments/open_models/ckpts.json
SCRIPTS=src/lasr_labs_2025_control_project/scripts
PREP=src/lasr_labs_2025_control_project/finetuning/binary_tuning

set -a; . ./.env; set +a
: "${TINKER_API_KEY:?TINKER_API_KEY must be set in .env}"

# arm -> SFT arm name in make_configs.SFT_ARMS / ckpts.json
ckpt_name() {
  case "$1" in
    none) echo bcb_self_rec_base ;;
    *) echo "bcb_self_rec_$1" ;;
  esac
}
# arm -> monitor config file stem (the none arm's file is called "base")
monitor_cfg() {
  case "$1" in
    none) echo "$CFG/monitors/sft/base_monitor_config.yaml" ;;
    *) echo "$CFG/monitors/sft/$1_monitor_config.yaml" ;;
  esac
}

all_exist() {
  local f
  for f in "$@"; do [ -s "$f" ] || return 1; done
}

need_docker() {
  docker info >/dev/null 2>&1 || {
    echo "Docker is not running -- the BigCodeBench scorer needs it." >&2
    exit 1
  }
}

make_configs() {
  uv run python experiments/open_models/make_configs.py --model-class open \
    --settings bigcodebench --U "$U" --T "$T" --H "$H" "$@"
}

stage_configs() {
  echo "=== [configs] generate bigcodebench_open configs (U=$U T=$T H=$H)"
  make_configs
}

stage_train_gen() {
  echo "=== [train-gen] U attacks + H honeypots on the odd (train) split"
  need_docker
  uv run python "$SCRIPTS/generate_solutions_inspect.py" --skip-existing \
    --max-connections "$MAX_CONN" \
    --config "$CFG/generator/generator_config_train.yaml"
}

stage_train_transform() {
  echo "=== [train-transform] minify + paraphrase (T) the train generations"
  local kind
  for kind in minified paraphrased; do
    uv run python "$SCRIPTS/transform_solutions_inspect.py" --skip-existing \
      --config "$CFG/transforms/${kind}_config_train.yaml"
  done
}

stage_prepare() {
  echo "=== [prepare] eval logs -> raw JSON -> train_messages.jsonl, per arm"
  # Positive label = U's own solutions. The logs record the bare Tinker id.
  local arm logs extra
  for arm in "${ARMS[@]}"; do
    if all_exist "$SFT_DATA/$arm/prepared/train/train_messages.jsonl"; then
      echo "  $arm exists, skipping"; continue
    fi
    logs="$SFT_DATA/$arm/eval_logs"
    rm -rf "$logs" "$SFT_DATA/$arm/raw" "$SFT_DATA/$arm/prepared"
    mkdir -p "$logs/train"
    # Copy (not move) so the generator/transform outputs stay where
    # --skip-existing looks for them.
    if [ "$arm" = none ]; then
      cp "$TRAIN_EXP/generator/attacks.eval" "$TRAIN_EXP/generator/honeypots.eval" "$logs/train/"
      extra=()
    else
      cp "$TRAIN_EXP/transforms/$arm/attack_transformed.eval" \
         "$TRAIN_EXP/transforms/$arm/honeypots_transformed.eval" "$logs/train/"
      extra=(--transformed)
    fi
    bash "$PREP/transform_logs_to_raw.sh" "$logs" "$SFT_DATA/$arm/raw" ${extra[@]+"${extra[@]}"}
    bash "$PREP/transform_raw_to_prepared.sh" "$SFT_DATA/$arm/raw" "$U" "$SFT_DATA/$arm/prepared"
  done
}

stage_finetune() {
  echo "=== [finetune] Tinker LoRA of U, one per arm -> $CKPTS"
  local arm name path_file
  for arm in "${ARMS[@]}"; do
    name=$(ckpt_name "$arm")
    path_file="$SFT_DATA/$arm/$name.txt"
    if all_exist "$path_file"; then
      echo "  $name already trained: $(cat "$path_file")"
    else
      # batch 16, not the default 64: attack mode keeps few examples (both
      # tasks must pass), so 64 would leave only a handful of updates.
      uv run python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
        --train-jsonl "$SFT_DATA/$arm/prepared/train/train_messages.jsonl" \
        --base-model "$U" --batch-size 16 \
        --checkpoint-name "$name" --out-path-file "$path_file"
    fi
    # Merge into ckpts.json; it also holds the APPS arms, so never overwrite it.
    uv run python - "$CKPTS" "$name" "$(cat "$path_file")" <<'PY'
import json, sys
from pathlib import Path
path, arm, uri = Path(sys.argv[1]), sys.argv[2], sys.argv[3].strip()
data = json.loads(path.read_text()) if path.exists() else {}
data[arm] = uri
path.write_text(json.dumps(data, indent=2) + "\n")
print(f"  {path}: {arm} = {uri}")
PY
  done
}

stage_configure() {
  echo "=== [configure] bake the checkpoints into the SFT monitor configs"
  make_configs --checkpoints "$CKPTS"
  if grep -l "PENDING__bcb" "$CFG"/monitors/sft/*.yaml; then
    echo "SFT monitor configs still reference PENDING checkpoints." >&2
    exit 1
  fi
}

stage_test_gen() {
  echo "=== [test-gen] U attacks + U honest + H honeypots on the even (test) split"
  need_docker
  uv run python "$SCRIPTS/generate_solutions_inspect.py" --skip-existing \
    --max-connections "$MAX_CONN" \
    --config "$CFG/generator/generator_config.yaml"
}

stage_test_transform() {
  echo "=== [test-transform] minify + paraphrase (T) the test generations"
  local kind
  for kind in minified paraphrased; do
    uv run python "$SCRIPTS/transform_solutions_inspect.py" --skip-existing \
      --config "$CFG/transforms/${kind}_config.yaml"
  done
}

stage_monitor() {
  echo "=== [monitor] U-SFT collusion + T and U backdoor monitors, per arm"
  local arm
  for arm in "${ARMS[@]}"; do
    uv run python "$SCRIPTS/monitor_solutions_inspect.py" --skip-existing \
      --max-connections "$MAX_CONN" --config "$(monitor_cfg "$arm")"
  done
}

stage_plot() {
  echo "=== [plot] histograms / ROC / safety-usefulness per arm"
  local arm
  for arm in "${ARMS[@]}"; do
    uv run python "$SCRIPTS/plot_data.py" --config "$(monitor_cfg "$arm")"
  done
  echo "Plots: $TEST_EXP/results/{none,minified,paraphrased}_sft/"
}

case "$STAGE" in
  configs) stage_configs ;;
  train-gen) stage_train_gen ;;
  train-transform) stage_train_transform ;;
  prepare) stage_prepare ;;
  finetune) stage_finetune ;;
  configure) stage_configure ;;
  test-gen) stage_test_gen ;;
  test-transform) stage_test_transform ;;
  monitor) stage_monitor ;;
  plot) stage_plot ;;
  all)
    stage_configs; stage_train_gen; stage_train_transform; stage_prepare
    stage_finetune; stage_configure; stage_test_gen; stage_test_transform
    stage_monitor; stage_plot
    ;;
  *) echo "Unknown stage: $STAGE (see --help)" >&2; exit 1 ;;
esac
echo "STAGE_${STAGE}_DONE"
