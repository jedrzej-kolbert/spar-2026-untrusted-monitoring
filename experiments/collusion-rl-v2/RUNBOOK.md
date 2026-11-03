# collusion-rl-v2 runbook

Branch `exp/collusion-rl-v2`. W&B: `spar-untrusted-monitoring-collusion/collusion-rl-v2`
(smoke and wiring runs: `collusion-rl-v2-smoke`). Every command runs from the repo root.
Budget cap $200.

## Files

| File | What |
|---|---|
| `i1-base.yaml` | I1-base: ICRL sender-only, frozen base judge, 8 rounds + no-strategy round, 32 search problems, top 3 on validation, $9 cap per seed |
| `i2.yaml` | I2: centralized ICRL (learner writes sender and judge text), paired reward, $10 cap per seed |
| `g1-base.yaml` | G1-base: GBRL, frozen base judge, 20 steps, validation every 10, $25 cap per seed |
| `smoke/*.yaml` | Wiring-test versions (done 3 Nov) |
| `env.sh` | W&B entity/project and the ~/.netrc key |
| `launch.sh` | Search/training run: free offline check, or detached paid run with `--allow-paid` |
| `test.sh` | Test scoring: `rewrite` (paid), `score` (plan, or paid with `allow_paid=true`), `report` (free) |
| `baseline_sft.sh` | Rescore the two V4.1 SFT judges on the test set |
| `workspace.py` | Saved W&B view with sections in reading order |
| `fake_runs.sh/.py` | Free example runs with fake models |
| `score_drift_check.sh`, `score_drift_repeat.py` | The 3 Nov score-drift checks |

Code: `src/lasr_labs_2025_control_project/finetuning/grouped_rl/` (`run.py`, `config.py`,
`icrl_adapters.py`, `icrl.py`, `report_test.py`), `scripts/rewrite_solutions_inspect.py`,
`tracking/core.py` (`roc_chart`, `log_roc(step=)`).

## Order and commands

1. Baselines on today's serving (running since 3 Nov):
   `experiments/collusion-rl-v2/test.sh score baseline-base allow_paid=true` (attacks = the original
   309 test attacks) and `experiments/collusion-rl-v2/baseline_sft.sh allow_paid=true`. ~$45.
2. Each arm, seeds 0, 1, 2 (free check first, then paid):
   `experiments/collusion-rl-v2/launch.sh i1-base <seed>` then `... --allow-paid`;
   the same for `g1-base`, then `i2`.
3. Test for each finished seed:
   - ICRL: `test.sh rewrite <arm>-s<seed> strategy=wandb:spar-untrusted-monitoring-collusion/collusion-rl-v2/strategy-<arm>-s<seed>:v0`
   - GBRL: `test.sh rewrite g1-base-s<seed> model=<checkpoint of best_validation_step in status.json>`
   - I2: `test.sh rewrite i2-s<seed> central=wandb:spar-untrusted-monitoring-collusion/collusion-rl-v2/strategy-i2-s<seed>:v0`
     (also saves the learned judge text; `score` turns it into prompt `JUDGE_I2_S<SEED>`)
   - then `test.sh score <label> allow_paid=true` and `test.sh report <label> <entity/project/run_id>`.
4. `uv run --with wandb-workspaces python experiments/collusion-rl-v2/workspace.py collusion-rl-v2`.
