#!/bin/bash
# Larger self-recognition SFT on the finished train U/H data, with checkpoint ROC on
# the finished test U/H data. Does not need the honest batch; no monitors or plots.
set -euo pipefail
cd /home/jkk/spar-2026-untrusted-monitoring.feat-basharena-generation
set -a; . /home/jkk/spar-2026-untrusted-monitoring/.env; set +a
export PYTHONPATH=src
echo $$ > run_logs/half_sft.pid
echo "=== SFT stage start $(date -Is)"
exec /home/jkk/spar-2026-untrusted-monitoring.deps-control-arena-17/.venv/bin/python -u run_logs/run_half_experiment.py --train-only
