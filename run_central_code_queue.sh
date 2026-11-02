#!/usr/bin/env bash
# Usage: run_central_code_queue.sh <transform> <label:prompt>... One monitor run after another.
cd "$(dirname "$0")"
transform=$1
shift
for item in "$@"; do
  label=${item%%:*} prompt=${item##*:}
  [ "$prompt" = none ] && prompt=-
  ./run_central_code_monitor.sh "$transform" "$label" "$prompt" allow_paid=true > "logs/v41/central_${label}_$transform.log" 2>&1
  echo "$label $transform: exit $?" >> logs/v41/central_code.status
done
