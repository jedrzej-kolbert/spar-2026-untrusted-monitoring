# This script transforms all eval logs in eval_logs/test and eval_logs/train to the raw dataset format.
# For each .eval file in eval_logs/{test,train}, it writes a .json file with the same base name to raw/{test,train}.
# Model names are automatically detected from the eval log metadata (no specific filename required).
#
# DIRECTORY STRUCTURE:
# The eval_logs_dir must contain train/ and/or test/ subdirectories with .eval files.
# If either subdirectory is missing or empty, it will be skipped with a warning.
#
# Usage: bash transform_logs_to_raw.sh <eval_logs_dir> [raw_dir] [--keep-comments] [--transformed] [--action-views]
# eval_logs_dir is a required argument.
# If raw_dir is not provided, defaults to <parent of eval_logs_dir>/raw.
# By default, comments are removed from base solutions. If --keep-comments is given, comments are kept. Ignored if --transformed is given.
# If --transformed is given as an argument, transformed solutions from .eval files are used.
# If --action-views is given (BashArena), each trajectory becomes one row per rated action, in
# the same per-action format the monitor uses (view: action). Optional env vars, which must
# match the monitor config's defaults.action_view: ACTION_MAX_HISTORY_CHARS,
# ACTIONS_PER_TRAJECTORY, ACTION_SEED. Then run transform_raw_to_prepared.sh with
# --system-prompt BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT --user-prompt BASHARENA_ACTION_USER_PROMPT.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default directories
DEFAULT_EVAL_LOGS_DIR="$ROOT_DIR/../../eval_logs"

# Parse arguments
KEEP_COMMENTS=""
TRANSFORMED=""

# Helper to check for flag in arguments
has_flag() {
  local flag="$1"
  shift
  for arg in "$@"; do
    if [ "$arg" = "$flag" ]; then
      return 0
    fi
  done
  return 1
}

# Remove flag from arguments
remove_flag() {
  local flag="$1"
  shift
  local new_args=()
  for arg in "$@"; do
    if [ "$arg" != "$flag" ]; then
      new_args+=("$arg")
    fi
  done
  echo "${new_args[@]}"
}

# Parse --keep-comments and --transformed flags
ARGS=("$@")
if has_flag "--keep-comments" "${ARGS[@]}"; then
  KEEP_COMMENTS="--keep-comments"
  ARGS=($(remove_flag "--keep-comments" "${ARGS[@]}"))
fi
if has_flag "--transformed" "${ARGS[@]}"; then
  TRANSFORMED="--transformed"
  ARGS=($(remove_flag "--transformed" "${ARGS[@]}"))
fi
ACTION_VIEWS=""
if has_flag "--action-views" "${ARGS[@]}"; then
  ACTION_VIEWS="--action-views"
  ARGS=($(remove_flag "--action-views" "${ARGS[@]}"))
fi

# Now parse positional arguments
RAW_DIR_PROVIDED=false
if [ "${#ARGS[@]}" -eq 2 ]; then
  EVAL_LOGS_DIR="${ARGS[0]}"
  RAW_DIR="${ARGS[1]}"
  RAW_DIR_PROVIDED=true
elif [ "${#ARGS[@]}" -eq 1 ]; then
  EVAL_LOGS_DIR="${ARGS[0]}"
  # Default: put raw/ in parent directory of eval_logs
  RAW_DIR="$(dirname "$EVAL_LOGS_DIR")/raw"
else
  echo "Error: eval_logs_dir is required."
  echo "Usage: bash transform_logs_to_raw.sh <eval_logs_dir> [raw_dir] [--keep-comments] [--transformed] [--action-views]"
  echo "  eval_logs_dir: Directory containing train/ and/or test/ subdirectories with .eval files (required)"
  echo "  raw_dir: Output directory (defaults to <parent of eval_logs_dir>/raw if not specified)"
  echo "  Optionally add --keep-comments and/or --transformed as flags to pass to eval_to_dataset."
  exit 1
fi

# Check that eval_logs_dir exists
if [ ! -d "$EVAL_LOGS_DIR" ]; then
  echo "Error: eval_logs_dir '$EVAL_LOGS_DIR' does not exist."
  exit 1
fi

# Safety check: if using default raw_dir, check if train/test subdirectories exist and are not empty
if [ "$RAW_DIR_PROVIDED" = false ]; then
  for SPLIT in test train; do
    SPLIT_DIR="$RAW_DIR/$SPLIT"
    if [ -d "$SPLIT_DIR" ] && [ -n "$(ls -A "$SPLIT_DIR" 2>/dev/null)" ]; then
      echo "Error: Default output directory '$SPLIT_DIR' already exists and is not empty."
      echo "Please either:"
      echo "  1. Explicitly specify a raw_dir: bash transform_logs_to_raw.sh <eval_logs_dir> <raw_dir>"
      echo "  2. Remove the existing directory: rm -rf '$SPLIT_DIR'"
      echo "  3. Remove both train and test: rm -rf '$RAW_DIR/train' '$RAW_DIR/test'"
      exit 1
    fi
  done
fi

for SPLIT in test train; do
  IN_DIR="$EVAL_LOGS_DIR/$SPLIT"
  OUT_DIR="$RAW_DIR/$SPLIT"

  # Check if input directory exists
  if [ ! -d "$IN_DIR" ]; then
    echo "Warning: Input directory $IN_DIR does not exist, skipping."
    continue
  fi

  # Check if there are .eval files
  shopt -s nullglob
  EVAL_FILES=("$IN_DIR"/*.eval)
  if [ "${#EVAL_FILES[@]}" -eq 0 ]; then
    echo "No .eval files found in $IN_DIR, skipping."
    shopt -u nullglob
    continue
  fi

  # Create output directory and clear if it exists
  if [ -d "$OUT_DIR" ]; then
    echo "Clearing output directory $OUT_DIR"
    rm -rf "$OUT_DIR"/*
  else
    mkdir -p "$OUT_DIR"
  fi
  for EVAL_PATH in "${EVAL_FILES[@]}"; do
    BASE_NAME="$(basename "$EVAL_PATH" .eval)"
    OUT_PATH="$OUT_DIR/$BASE_NAME.json"

    echo "Processing $EVAL_PATH -> $OUT_PATH"
    # Build command with optional flags (model name auto-detected from eval log)
    cmd=(uv run src/lasr_labs_2025_control_project/finetuning/utils/eval_to_dataset.py --data-path "$EVAL_PATH" --output-json "$OUT_PATH")
    if [ -n "$KEEP_COMMENTS" ]; then
      cmd+=(--keep-comments)
    fi
    if [ -n "$TRANSFORMED" ]; then
      cmd+=(--transformed)
    fi
    if [ -n "$ACTION_VIEWS" ]; then
      cmd+=(--action-views --action-seed "${ACTION_SEED:-0}")
      if [ -n "${ACTION_MAX_HISTORY_CHARS:-}" ]; then
        cmd+=(--max-history-chars "$ACTION_MAX_HISTORY_CHARS")
      fi
      if [ -n "${ACTIONS_PER_TRAJECTORY:-}" ]; then
        cmd+=(--actions-per-trajectory "$ACTIONS_PER_TRAJECTORY")
      fi
    fi
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
  done
  shopt -u nullglob
done
