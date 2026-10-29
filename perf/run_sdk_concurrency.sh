#!/usr/bin/env bash
# Warm-up pass, then each (SDK, concurrency) cell twice: forward order, then
# reversed, so drift in server load or caching doesn't favour one cell.
set -u
cd "$(dirname "$0")/.."
O=perf/results/2026-10-29
A=../monitor-connections-16/.venv/bin/python  # tinker 0.30.0 (main's lock)
B=../tinker-sdk-036/.venv/bin/python          # tinker 0.36.0 (PR #108's lock)
export PYTHONPATH=$PWD/src                    # same provider code in both

cell() {
  echo "$(date -u +%FT%TZ) start $*" >> $O/progress.log
  "$1" perf/tinker_sdk_concurrency.py run --inputs $O/inputs.json --results $O/results.jsonl \
    --concurrency "$2" --tag "$3" --phase "${4:-measure}" >> $O/run.log 2>&1
  echo "$(date -u +%FT%TZ) exit=$? $*" >> $O/progress.log
}

cell $A 8 warmup warmup
cell $A 8 r1; cell $A 16 r1; cell $B 8 r1; cell $B 16 r1
cell $B 16 r2; cell $B 8 r2; cell $A 16 r2; cell $A 8 r2
$B perf/tinker_sdk_concurrency.py summary --results $O/results.jsonl > $O/summary.tsv 2>> $O/run.log
echo "$(date -u +%FT%TZ) DONE" >> $O/progress.log
