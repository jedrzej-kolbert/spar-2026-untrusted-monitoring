#!/usr/bin/env bash
# Rewrites with seed 1's strategy, waits for the empty game, then runs the while-loop and seed 1 games.
cd "$(dirname "$0")"
G=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/icrl/experiments/icrl-central-r2-20261101/game
"$G/run_rewrites.sh" seed1:1
grep -q 'seed1: exit 0' "$G/rewrites.status" || { echo "rewrite failed" >> logs/v41/central_games.status; exit 1; }
until grep -q '^empty: exit' logs/v41/central_games.status 2>/dev/null; do sleep 30; done
exec ./run_central_queue.sh loop:CENTRAL_WHILE_LOOP_JUDGE_PROMPT seed1:CENTRAL_SEED1_JUDGE_PROMPT
