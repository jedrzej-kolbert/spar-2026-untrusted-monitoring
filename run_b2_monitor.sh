#!/usr/bin/env bash
# Step 2 phase B2, paid part: paraphrase plus monitors for all transforms (untransformed ones are reused).
# No game stage, so it can run while B1's bootstrap is still using the CPU.
cd "$(dirname "$0")"
exec ./run_step2.sh generate,transform,monitor logs/v41/step2_b2_monitor.log
