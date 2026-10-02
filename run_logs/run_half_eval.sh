#!/bin/bash
# Usage: run_half_eval.sh ARM   (monitors + standard plots on the half-split test set)
set -euo pipefail
cd /home/jkk/spar-2026-untrusted-monitoring.feat-basharena-generation
set -a; . /home/jkk/spar-2026-untrusted-monitoring/.env; set +a
export PYTHONPATH=src
echo "=== eval $1 start $(date -Is)"
exec /home/jkk/spar-2026-untrusted-monitoring.deps-control-arena-17/.venv/bin/python -u run_logs/run_half_eval.py "$1"
