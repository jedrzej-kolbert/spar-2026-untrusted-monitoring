#!/usr/bin/env bash
# Step 3 phase C3: the paraphrased (llm_summarize) SFT judge: train paraphrase, finetune, 3 SFT monitors, game, analysis.
cd "$(dirname "$0")"
exec ./run_step3.sh generate,transform,finetune,monitor,game,analysis logs/v41/step3_c3.log 'measurement.transforms=[none,paraphrased]' allow_paid=true
