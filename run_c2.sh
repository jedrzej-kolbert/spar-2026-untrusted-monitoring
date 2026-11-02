#!/usr/bin/env bash
# Step 3 phase C2: the none SFT judge (finetune, 3 SFT monitors), then its game and analysis.
cd "$(dirname "$0")"
exec ./run_step3.sh generate,transform,finetune,monitor,game,analysis logs/v41/step3_c2.log 'measurement.transforms=[none]' allow_paid=true
