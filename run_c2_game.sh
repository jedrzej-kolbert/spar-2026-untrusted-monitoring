#!/usr/bin/env bash
# Step 3: the none SFT game + analysis (free; no allow_paid), restarted with PR #106 applied.
cd "$(dirname "$0")"
exec ./run_step3.sh game,analysis logs/v41/step3_c2_game.log 'measurement.transforms=[none]'
