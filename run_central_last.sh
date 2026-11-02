#!/usr/bin/env bash
# The last T comparison game, cut off by a reboot at 100 of 200 replicates.
cd "$(dirname "$0")"
./run_central_merged_game.sh none empty old > logs/v41/central_t_none_empty_old.log 2>&1
echo "none:empty:old: exit $?" >> logs/v41/central_merged.status
