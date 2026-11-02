#!/usr/bin/env bash
# Step 3 phase C1: V4.1 train attacks only (the SFT YES data).
cd "$(dirname "$0")"
exec ./run_step3.sh generate logs/v41/step3_c1_generate.log allow_paid=true
