#!/usr/bin/env bash
# Retry only the errored/missing samples of the train-honest log.
cd "$(dirname "$0")"
set -a; . ./.env; set +a
export WANDB_MODE=disabled
D=experiments/apps/generate/apps-262144_ft_a1611f-train-honest-honest-c7c7f56a99bd
uv run python finish_honest.py > logs/retry.log 2>&1
echo "exit=$?" >> logs/retry.log
