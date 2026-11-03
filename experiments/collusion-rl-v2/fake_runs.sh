#!/usr/bin/env bash
# Example W&B runs of every arm with fake models (free); see fake_runs.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/collusion-rl-v2/env.sh
export WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-collusion-rl-v2-smoke}
uv run python experiments/collusion-rl-v2/fake_runs.py
