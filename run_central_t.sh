#!/usr/bin/env bash
# Stronger T on all honest and attack sets. User approved on 2 Nov, inside the $45 total.
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python experiments/central_t_score.py --frac 1 --cap 5.5 > experiments/central_t/run.log 2>&1
echo "$?" > experiments/central_t/exit_code
