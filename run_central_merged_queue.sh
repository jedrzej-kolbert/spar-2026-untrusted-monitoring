#!/usr/bin/env bash
# All T comparison games, one after another: free CPU, no paid models in these plans.
cd "$(dirname "$0")"
for item in paraphrased:seed1:new paraphrased:loop:new paraphrased:empty:new paraphrased:sft:new paraphrased:base:new \
            none:seed1:new none:loop:new none:empty:new none:sft:new none:base:new \
            none:seed1:old none:loop:old none:empty:old; do
  IFS=: read -r transform arm t <<< "$item"
  ./run_central_merged_game.sh "$transform" "$arm" "$t" > "logs/v41/central_t_${transform}_${arm}_$t.log" 2>&1
  echo "$item: exit $?" >> logs/v41/central_merged.status
done
