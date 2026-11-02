#!/usr/bin/env bash
# Usage: run_central_queue.sh <label:prompt>... Runs games one after another: a pipeline command must
# not run while a game writes its bootstrap file. "none" as prompt keeps the base judge.
cd "$(dirname "$0")"
for item in "$@"; do
  label=${item%%:*} prompt=${item##*:}
  [ "$prompt" = none ] && prompt=-
  ./run_central_game.sh "$label" transform,monitor,game,analysis "$prompt" allow_paid=true > "logs/v41/central_$label.log" 2>&1
  echo "$label: exit $?" >> logs/v41/central_games.status
done
