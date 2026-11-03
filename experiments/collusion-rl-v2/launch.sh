#!/usr/bin/env bash
# Usage: launch.sh <arm> <seed> [run.py options...]
#   arm: i1-base | i2 | g1-base (a YAML here). Without --allow-paid this only loads the
#   data, writes the split and checks the sandbox (free). With it, the run is detached
#   and logs to runs/<arm>-s<seed>.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
arm=$1 seed=$2
shift 2
source experiments/collusion-rl-v2/env.sh
cmd=(uv run python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run
  --config "experiments/collusion-rl-v2/$arm.yaml" --seed "$seed" "$@")
if [[ " $* " == *" --allow-paid "* ]]; then
  # The run refuses uncommitted code too, but only in its log once detached.
  git diff --quiet HEAD -- || { echo "commit first: tracked files differ from HEAD" >&2; exit 1; }
  log="experiments/collusion-rl-v2/runs/$arm-s$seed.log"
  mkdir -p "$(dirname "$log")"
  setsid nohup "${cmd[@]}" > "$log" 2>&1 < /dev/null &
  echo "started $arm seed $seed (pid $!), log $log"
else
  "${cmd[@]}"
fi
