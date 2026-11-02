#!/usr/bin/env bash
# Step 2 phase B1: monitors, game and analysis on the untransformed solutions only.
cd "$(dirname "$0")"
exec ./run_step2.sh generate,transform,monitor,game,analysis logs/v41/step2_b1.log 'measurement.transforms=[none]'
