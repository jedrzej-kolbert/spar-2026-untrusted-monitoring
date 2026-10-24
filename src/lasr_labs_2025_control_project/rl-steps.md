# Running RL Experiments for Collusion Signals (HiPT-style)

This trains the untrusted model (U) with reinforcement learning to hide a collusion signal in its code, following the GBRL method in *Hidden in Plain Text* (HiPT). The current setup is **Option A**: U rewrites attacks it already wrote, with the signal switched ON or OFF, and is rewarded when a frozen copy of itself (the untrusted monitor) can tell the two apart while the trusted monitor cannot.

For the design, the decisions behind it, and what is still to come, see [`docs/hipt_rl/ROADMAP.md`](../../docs/hipt_rl/ROADMAP.md).

## Setup

Run `uv sync`, and put your Tinker API key in `.env` as `TINKER_API_KEY`. Docker must be running: the training loop grades code in a locked-down container. The first run pulls a pinned `python:3.12-slim` image.

Log in to Weights & Biases with `wandb login`, or set `logging.wandb_mode: offline` in the RL config and upload later with `wandb sync`.

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

## Tests

```
PYTHONPATH=src:. TINKER_API_KEY= pytest tests/test_hipt_rl.py tests/test_apps_grouped_rl.py tests/test_make_configs_rl.py
```
These are offline (no Tinker, no Docker). Don't run plain `pytest`: `tests/test_tinker_live.py` makes paid calls.
