#!/usr/bin/env bash
# Step 2 phase B2: the last paid monitor (retry) and the free minified/paraphrased games + final analysis, chunk_size 100.
# A pipeline start plans the whole graph and reads every game's bootstrap file; on main a read during a chunk
# write deletes that file (PR #95). So each pipeline starts only while no game is writing its bootstrap file.
cd "$(dirname "$0")"
b1=logs/v41/step2_b1.log b2=logs/v41/step2_b2_monitor.log b2_retry=logs/v41/step2_b2_monitor_retry.log
say() { echo "$(date -u +%H:%M) $*"; }

until grep -q '^exit=' "$b1" && [ "$(grep -cE 'View run [^ ]+\.min\.mon-[^ ]+ · ' "$b2")" -ge 3 ]; do sleep 60; done
grep -q '^exit=0' "$b1" || { say "B1 failed: $(tail -1 "$b1"); stopping"; exit 1; }

# The first B2 monitor run stopped on a W&B race before monitor.paraphrased.base.honest; the retry runs that one.
say "B1 done: starting the B2 monitor retry"
./run_b2_monitor_retry.sh &
sleep 90  # the retry's planning reads the game files; let it finish before the minified game starts writing

say "starting the minified game"
./run_step2.sh game,analysis logs/v41/step2_game_min.log 'measurement.transforms=[none,minified]' optimization.bootstrap.chunk_size=100
say "minified game run: $(tail -1 logs/v41/step2_game_min.log)"

wait
grep -q '^exit=0' "$b2_retry" || { say "B2 monitor retry failed: $(tail -1 "$b2_retry"); stopping"; exit 1; }
say "B2 monitors done; starting the paraphrased game and final analysis"
./run_step2.sh game,analysis logs/v41/step2_game_final.log optimization.bootstrap.chunk_size=100
say "final run: $(tail -1 logs/v41/step2_game_final.log)"
