#!/usr/bin/env bash
# Detached, so the benchmark survives the Claude session ending.
cd "$(dirname "$0")/.."
setsid nohup bash perf/run_sdk_concurrency.sh < /dev/null > /dev/null 2>&1 &
echo "launched pid $!"
