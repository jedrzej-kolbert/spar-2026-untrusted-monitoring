#!/usr/bin/env bash
# Prints each finished W&B step, any traceback, and the exit line of logs/v41/pilot.log; exits when the run ends.
cd "$(dirname "$0")"
log=${1:-logs/v41/pilot.log}
seen=0 errs=0
while true; do
  n=$(grep -cE 'View run [^ ]+ · [a-z0-9]+' "$log" 2>/dev/null)
  if [ "$n" -gt "$seen" ]; then grep -oE 'View run [^ ]+ · [a-z0-9]+' "$log" | tail -n +$((seen + 1)); seen=$n; fi
  e=$(grep -cE 'Error executing|^Traceback' "$log" 2>/dev/null)
  if [ "$e" -gt "$errs" ]; then grep -E 'Error executing|^Traceback' "$log" | tail -n +$((errs + 1)); errs=$e; fi
  if grep -qE '^exit=' "$log"; then grep -E '^exit=|^Done' "$log"; break; fi
  pgrep -f 'name=self_rec_v41 ' > /dev/null || { echo "pipeline process gone without exit line"; break; }
  sleep 30
done
