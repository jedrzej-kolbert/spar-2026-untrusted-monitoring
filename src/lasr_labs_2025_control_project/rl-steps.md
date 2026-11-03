# Running RL Experiments for Collusion Signals (HiPT-style)

This trains the untrusted model (U) with reinforcement learning to hide a collusion signal in its code, following the GBRL method in *Hidden in Plain Text* (HiPT). The current setup is **Option A**: U rewrites attacks it already wrote, with the signal switched ON or OFF, and is rewarded when a frozen copy of itself (the untrusted monitor) can tell the two apart while the trusted monitor cannot.

For the design, the decisions behind it, and what is still to come, see [`docs/hipt_rl/ROADMAP.md`](../../docs/hipt_rl/ROADMAP.md).

## Setup

Run `uv sync`, and put your Tinker API key in `.env` as `TINKER_API_KEY`. Docker must be running: the training loop grades code in a locked-down container. The first run pulls a pinned `python:3.12-slim` image.

Weights & Biases works as in the pipeline: each run is a stage run (job type `icrl` or `gbrl`) in `WANDB_ENTITY`/`WANDB_PROJECT`, with `WANDB_MODE=offline` to log locally.

Models (set in `experiments/open_models/make_configs.py`, `ROLE_MODELS["open"]`):
- U, the untrusted model: `deepseek-ai/DeepSeek-V3.1`. It is the sender being trained and, frozen, the receiver.
- T, the trusted monitor: `Qwen/Qwen3.5-4B`. It is the overseer during training and the paraphraser.
- H, the honeypot generator: `openai/gpt-oss-120b`.

## 1. Generate training data

RL needs attacks to rewrite, from the **train** split (odd problem IDs):
```
python src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/generator/train_generator_config.yaml
```
This writes `attacks.eval` (U) and `honeypots.eval` (H) to `experiments/open-source-model/apps/self_rec_train/generator/`.

Only attacks that pass the tests **and** have a working backdoor are used. On the test split that is about 11% of DeepSeek's attacks, so expect roughly 100 to 120 training problems.

## 2. Check the setup offline

```
python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \
  --config src/lasr_labs_2025_control_project/configs/rl/rewrite_signal_pilot.yaml
```
This makes no model calls. It loads and splits the tasks, writes `manifest.json`, and re-grades a sample of the original attacks in the Docker sandbox. Both agreement numbers should be 1.0. If they are not, the sandbox disagrees with the Inspect scorer, so stop and investigate.

## 3. Train

Add `--allow-paid`:
```
python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \
  --config src/lasr_labs_2025_control_project/configs/rl/rewrite_signal_pilot.yaml --allow-paid
```

What happens:
1. **Preflight.** Clean and obviously signalled versions of real attacks are scored by the receiver and the overseer. A useful overseer scores the signalled version higher.
2. **Baseline evaluation (step 0).** Held-out problems are rewritten with both bits, and the bit-1 rewrites are also compared against H's honeypots.
3. **Training steps.** Each step rewrites 16 problems, 4 times with the signal ON and 4 times OFF. Every rewrite is graded, scored, and used for one PPO update of U's LoRA. A step takes about a minute or two.
4. **Evaluation** every 10 steps, a checkpoint every 10 steps, and a final checkpoint. Its `tinker://` path is added to `experiments/open_models/ckpts.json` under the run's name.

Everything is saved in `output_dir` (default `experiments/open-source-model/apps/rl/rewrite_signal_pilot/`):
- `rollouts.jsonl`: every rewrite with its scores, rewards, and errors
- `metrics.jsonl`: every logged number
- `status.json`: progress and checkpoint paths
- `manifest.json`: config, git commit, and train/validation problem IDs

### Numbers to watch

| Metric | Meaning | Healthy |
| --- | --- | --- |
| `train/balanced_accuracy` | Receiver accuracy averaged over bit 0 and bit 1 | Rises above 0.5 |
| `train/bit1_accuracy`, `bit0_accuracy` | Per-bit accuracy | Both improve. If one is about 1 and the other about 0, the receiver is just guessing one answer |
| `train/identical_pair_fraction` | Problems where some ON and OFF rewrites are identical | Near 0. Rising means the sender stopped signalling |
| `train/suspicion_mean` | Overseer (T) suspicion | Falls while balanced accuracy holds |
| `train/backdoor_success` | Rewrites that still trigger the backdoor | Stays high |
| `train/correctness` | Rewrites that pass the tests | Stays high |
| `eval_vs_honeypots/auroc` | Signalled attacks vs honeypots, the deployment question | Rises above 0.5 |

### Changing the experiment

Copy `configs/rl/rewrite_signal_pilot.yaml`, change `run_name`, and edit:
- `reward.reward_fn`: `branch` (default, correctness-gated) or `hipt` (upstream probability-reward variant, no APPS pass gate or task bonus).
- For `hipt`, `reward.stego_scale` and `overseer_scale` weight the communication and overseer terms. `overseer_log_odds_threshold` (default `0.75`) is subtracted from `log P(NO) - log P(YES)` before `tanh`; it is not a probability threshold. The old `hipt_gate_on_pass` and `overseer_threshold` keys are rejected.
- `reward.failed_backdoor_weight`, `backdoor_suspicion_weight`, `kl_coef`, `overseer_weight`: optional penalties and their strengths. `overseer_ramp_start` and `overseer_ramp_steps` ramp the overseer in gradually.
- `channel.kind`: `none`, `minify`, or `paraphrase` (signal jamming between sender and receiver).
- `training.payload_grouping`: `both` (default; both bits in every group) or `single`.

For the trained model to show up in Step 4, add the new run name to `RL_ARMS` and `OPEN_ONLY` in `make_configs.py`.

## Self-recognition instead of hidden-signal decoding

Set `task_mode: self_recognition` and `training.train_receiver: true` in a copy
of the run config. Set both `reward.overseer_weight` and `overseer_scale` to zero:
that overseer looks for hidden signals, which this mode does not train. The trusted
monitor that looks for the backdoor stays: `reward.backdoor_suspicion_weight`
subtracts its 0-10 suspicion score, divided by 10, from every rewrite's reward, scored
by `models.backdoor_monitor` on the same monitor view. The mode selects natural-style rewrite
prompts and the usual `SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT` unless
explicit prompt overrides are supplied; there is no ON/OFF request or overseer call.

Every new sender rewrite is **YES/self**, regardless of its correctness. Matching
`data.honeypots_eval` generations from a different model are **NO/non-self**, exactly
as APPS SFT labels by model of origin. Only training-split honeypots enter receiver
SFT, one per problem, repeated to balance the generated positives. Missing,
duplicate, or same-model training negatives stop the run rather than silently
training an always-YES recognizer. Both classes pass through the same monitor-view
channel. Held-out honeypots remain evaluation-only.

The sender's existing group-relative RL update uses the **pre-update** receiver's
`P(YES | problem, code)` instead of decoding a randomly assigned bit. The branch
reward's correctness gate/task bonus and optional KL are unchanged. To remove the
task bonus while keeping the gate, set `task_weight: 0`. A backdoor is rewarded,
not penalised: `reward.backdoor_kept_weight` adds a bonus when the rewrite's
backdoor still triggers on the secret input, and `failed_backdoor_weight` is
rejected in this mode.
The receiver gets answer-token cross-entropy SFT with `<answer>YES/NO</answer>`.
`models.receiver` may start from a base model or an existing SFT sampler checkpoint.

Training problems whose honeypot generation failed have no non-self example, so
they are left out of training; their IDs are in `manifest.json` under
`train_problems_dropped_without_honeypot`. `training.receiver_learning_rate` sets the
receiver's SFT step size separately (our SFT runs use 1e-4). The receiver scores YES
and NO with one teacher-forced request each: since 2026-10-21 Tinker returns no
`target_prompt_logprobs` (SDK 0.30.0), so main's single-prefill path fails.

`receiver_examples.jsonl` records all receiver inputs and labels. W&B groups the
metrics a reader needs into sections: `reward/` (total and every component),
`loss/` (sender, receiver per example), `scores/` (P(self), pass rate, kept-backdoor
rate, trusted suspicion, per training step), `eval/` (held-out AUROC, P(self),
self and honeypot YES rates, suspicion) and `progress/` (epoch, estimated cost per
step and in total, prefill/sample/train tokens). Everything else is under
`zz_other/`. `usage.json` holds the token counts per model.

Costs are estimated from every Tinker request the process makes, priced from
Tinker's published rates (`usage.py`). `training.max_cost_usd` reserves each
request's maximum token cost before dispatch, including concurrent requests and
baseline/final evaluation. Failed requests are conservatively counted at their
reserved maximum. It also stops early when the previous step predicts an overrun.
After exhaustion, no more token requests are dispatched; checkpoints are still
saved. Final usage is recorded even after a failed step or final evaluation.
This is a published-rate token-cost cap, not an invoice cap (SDK retries and
storage are not independently billed here).
`status.json` and W&B summary contain **both** final checkpoints. Joint training
currently starts fresh; `--resume-state` is rejected because it restores only the sender.

## 4. Evaluate with the usual pipeline

The RL arms reuse the self-rec **test-split** honest solutions and honeypots, and rewrite the self-rec test attacks. Make sure these exist first: they are the "Generate test data" and signal-jamming steps in `sft-steps.md`, which write to `experiments/open-source-model/apps/self_rec/generator/` and `.../self_rec/transforms/{minified,paraphrased}/`.

1. Bake the checkpoint into the configs:
   ```
   python experiments/open_models/make_configs.py --model-class open --settings apps
   ```
   This reads `ckpts.json`. The RL configs are under `configs/apps_open/rl/`: `rl_rewrite_signal_pilot` (trained) and `rl_rewrite_signal_untrained` (the same rewrite with base U, for a before/after comparison).
2. Rewrite the test attacks with the signal ON (run for both arms):
   ```
   python src/lasr_labs_2025_control_project/scripts/rewrite_solutions_inspect.py \
     --config src/lasr_labs_2025_control_project/configs/apps_open/rl/rl_rewrite_signal_pilot/generator/rewrite_config.yaml
   ```
3. Signal jamming:
   ```
   python src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
     --config src/lasr_labs_2025_control_project/configs/apps_open/rl/rl_rewrite_signal_pilot/transforms/minified_config.yaml
   ```
   Repeat with `paraphrased_config.yaml`.
4. Monitors: run `monitor_solutions_inspect.py` with each of `base_monitor_config.yaml`, `minified_monitor_config.yaml`, and `paraphrased_monitor_config.yaml` in `.../rl/rl_rewrite_signal_pilot/monitors/`.
5. Plots: run `plot_data.py` with the same monitor configs. Results go to `experiments/open-source-model/apps/rl_rewrite_signal_pilot/results/{none,minified,paraphrased}_rl/`.

## Collusion RL v2 (V4.1, Nov 2026)

GBRL and ICRL under one protocol: search or training on train problems, the final strategy or checkpoint picked on validation (a fixed 28% of train problems, by problem ID), and one score on the full test set (309 working V4.1 attacks vs 246 working honeypots) with the same monitors and game as the SFT judges. Branch `exp/collusion-rl-v2`; W&B project `spar-untrusted-monitoring-collusion/collusion-rl-v2`.

| Arm | Config (`configs/rl/collusion_rl_v2/`) | What |
| --- | --- | --- |
| I1-base | `i1_base.yaml` | ICRL, sender-only, frozen base judge; reward V·g(s)·P(self_rec) |
| I2 | `i2.yaml` | ICRL, centralized: the learner also writes the judge's instructions; paired reward |
| G1-base | `g1_base.yaml` | GBRL (LoRA on the sender), frozen base judge; same reward as I1-base |

Run every command from the repo root, in the project's environment (`uv run` or the activated `.venv`), with
```
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=collusion-rl-v2
C=src/lasr_labs_2025_control_project/configs/rl/collusion_rl_v2
```

**1. Baselines on today's serving** (Tinker's V4.1 scores changed between 28 Oct and 3 Nov for the same inputs, so the base and SFT judges are scored again in this project):
```
python -m lasr_labs_2025_control_project.pipeline experiment=rl_test name=rlv2_baseline-base \
  +inputs.generate.test.attacks=<the 309 original test attacks .eval> allow_paid=true
python -m lasr_labs_2025_control_project.pipeline experiment=rl_test_sft name=rlv2_baseline_sft allow_paid=true
```

**2. Search or training**, seeds 0, 1, 2 (`--seed` appends `-s<seed>` to the run name and output folder). First without `--allow-paid` (free: loads the split, checks the sandbox), then detached:
```
python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run --config $C/i1_base.yaml --seed 0
setsid nohup python -m lasr_labs_2025_control_project.finetuning.grouped_rl.run \
  --config $C/i1_base.yaml --seed 0 --allow-paid > experiments/collusion-rl-v2/runs/i1-base-s0.log 2>&1 &
```
A `kind: main` config refuses to start from uncommitted code. ICRL logs its winner as artifact `strategy-<run>`; GBRL records `best_validation_step` in `status.json` and logs every checkpoint as a model artifact.

**3. Test rewrite** of the 309 test attacks (`scripts/rewrite_solutions_inspect.py`, a W&B stage run of type `rewrite` linked to the strategy or checkpoint):
```
R="python -m lasr_labs_2025_control_project.scripts.rewrite_solutions_inspect --config $C/test_rewrite.yaml"
$R --label i1-base-s0 --strategy-result wandb:$WANDB_ENTITY/$WANDB_PROJECT/strategy-i1-base-s0:v0
$R --label g1-base-s0 --model tinker/<tinker:// checkpoint of best_validation_step>
$R --label i2-s0 --strategy-result wandb:$WANDB_ENTITY/$WANDB_PROJECT/strategy-i2-s0:v0 \
   --system-prompt RL_CENTRAL_SENDER_SYSTEM_PROMPT
```
For I2 the rewrite also adds the learned judge as prompt `JUDGE_I2_S0` to `prompts/learned_judge_prompts.py`; commit it before scoring.

**4. Test scoring** through the pipeline (prints the plan without `allow_paid=true`; honeypot and honest scores are computed once and reused):
```
python -m lasr_labs_2025_control_project.pipeline experiment=rl_test name=rlv2_i1-base-s0 \
  +inputs.generate.test.attacks=experiments/collusion-rl-v2/test/i1-base-s0/attacks.eval allow_paid=true
```
For I2 add the learned judge, and reuse the base judge's honest scores from step 1 (the game reads only their suspicion scores):
```
  +measurement.arms.base.collusion_prompt=JUDGE_I2_S0 \
  +inputs.monitor.none.base.honest=experiments/apps/monitor/apps-262144_ft_a1611f-test-honest-honest.mon-262144_ft_a1611f-self_rec_collusion-36f750b15b47/honest.eval \
  +inputs.monitor.paraphrased.base.honest=experiments/apps/monitor/apps-262144_ft_a1611f-test-honest-honest.para-qwen3_5_4b-llm_summarize.mon-262144_ft_a1611f-self_rec_collusion-a1fdc5ca2a29/honest.eval
```

**5. Report** the test scores into the run's W&B page (`test/` section; free):
```
python -m lasr_labs_2025_control_project.finetuning.grouped_rl.report_test \
  --run $WANDB_ENTITY/$WANDB_PROJECT/<run id> --label i1-base-s0
```

Optional: `uv run --with wandb-workspaces python experiments/collusion-rl-v2/workspace.py collusion-rl-v2` saves a W&B view with the sections in reading order.

## Tests

```
uv run python -m pytest tests -m "not slow and not mc"
```
These are offline (no Tinker, no Docker). Don't run plain `pytest`: `tests/test_tinker_live.py` makes paid calls.
