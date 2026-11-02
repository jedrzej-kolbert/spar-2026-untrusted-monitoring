#!/usr/bin/env bash
# Retry of the B2 paid part after a W&B race stopped it before monitor.paraphrased.base.honest; finished steps are reused.
cd "$(dirname "$0")"
exec ./run_step2.sh generate,transform,monitor logs/v41/step2_b2_monitor_retry.log
