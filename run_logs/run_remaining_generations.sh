#!/bin/bash
# Remaining half-split generations only (train H, then test U, H, honest); no SFT.
set -euo pipefail
cd /home/jkk/spar-2026-untrusted-monitoring.feat-basharena-generation
set -a; . /home/jkk/spar-2026-untrusted-monitoring/.env; set +a
export PYTHONPATH=src
PY=/home/jkk/spar-2026-untrusted-monitoring.deps-control-arena-17/.venv/bin/python
echo $$ > run_logs/remaining_generations.pid
"$PY" run_logs/run_half_experiment.py --check
echo "=== Remaining generations start $(date -Is)"
exec "$PY" -u run_logs/run_half_experiment.py --generate-only
