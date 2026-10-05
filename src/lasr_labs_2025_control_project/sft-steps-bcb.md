# Running SFT Experiments with Passive Self-Recognition on BigCodeBench

These are the BigCodeBench (BCB) versions of the steps in `sft-steps.md`. The pipeline is the same; what changes is the config tree, a few file names, the output folders, and the checkpoint names. Differences from the APPS instructions are marked **BCB:**.

To run everything at once instead, use the script:
```
bash experiments/open_models/run_bcb_sft.sh --U deepseek-ai/DeepSeek-V3.1 --T Qwen/Qwen3.5-4B --H openai/gpt-oss-20b
```
It runs every step below in order, and you can run a single step by naming it (for example `... finetune`). See `--help` for the step names.

Generation and monitoring run 8 samples in parallel by default, and each one holds a Docker sandbox. On a machine with more memory, add `--max-connections 32` (or 48) to the script, or to `generate_solutions_inspect.py` / `monitor_solutions_inspect.py` when running steps by hand. That roughly quarters generation time. Each sandbox may use up to 0.5 GB. An overloaded machine turns correct code into test timeouts without any error, so check that `main_task_success` looks similar to an 8-connection run.

## Setup

Create a virtual environment for this project. Run the following command to install all dependencies:
```
uv sync
```

Select three Tinker models to use as:
-  untrusted generator (this will double as untrusted monitor) U
-  honeypot generator H
-  trusted monitor T

Save their model strings. Make sure you select models that are sampleable.

Configure the Tinker API key in the .env file.

**BCB: Docker must be running.** Every generated solution is run in a Docker sandbox to check whether it passes the tests and whether its backdoor works. The first generation run builds the sandbox image from `bigcodebench/Dockerfile`. The image is about 8.5 GB and takes several minutes to build. It needs `bigcodebench/docker-requirements.txt` (that file was added on the `bcb-fixes` branch; older checkouts of main lack it and the build fails).

Generation starts 8 sandboxes at once. If it fails with `no space left on device`, free up disk space, build the image once, and re-run:
```
docker compose -f src/lasr_labs_2025_control_project/bigcodebench/compose.yaml -p bcb-prebuild build
```

## Configs

**BCB:** You don't need to edit `make_configs.py`. Pass the models as flags, along with `--settings bigcodebench`:
```
python experiments/open_models/make_configs.py --model-class open --settings bigcodebench \
  --U deepseek-ai/DeepSeek-V3.1 --T Qwen/Qwen3.5-4B --H openai/gpt-oss-20b
```
The `tinker/` prefix is optional. Flags you leave out fall back to `ROLE_MODELS["open"]` in `make_configs.py`. The script prints `open roles: ...` so you can check what went into the configs.

**⚠️ Important:** pass the same three flags every time you re-run `make_configs.py` (including the `--checkpoints` step later). Without them, the configs are regenerated with the default models.

The configs are written to `src/lasr_labs_2025_control_project/configs/bigcodebench_open/`. Everything below uses `configs/bigcodebench_open/self_rec/`.

## A note about the number of code samples

**BCB:** There is no `problem_ids_range` to change. BCB has 1140 problems (v0.1.2). Training uses the 570 odd-indexed problems and testing uses the 570 even-indexed ones (see `bigcodebench/bigcodebench.py`).

### Dry run

To check the whole pipeline before a full run, add `--limit N` to the `make_configs.py` command. Every generator then stops after the first N problems; the later steps process whatever was generated. Re-running `make_configs.py` without `--limit` restores the full-size configs.

**⚠️ Delete the dry run's outputs before the full run** (`experiments/open-source-model/bigcodebench/`). The generator, transform and monitor scripts accept `--skip-existing`, which `run_bcb_sft.sh` always passes. With it, they skip any run whose `.eval` already holds a successful log, including a 10-problem one.

## Training data generation

To generate the training data, run this command:

```
python src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/generator/generator_config_train.yaml
```
**BCB:** the train config is `generator_config_train.yaml` (the APPS name is `train_generator_config.yaml`), and the output experiment is called `bcb_train`, not `self_rec_train`.

Check that `attacks.eval` (U) and `honeypots.eval` (H) were saved in `experiments/open-source-model/bigcodebench/bcb_train/generator`.

## Applying blue-team mitigation strategies to training data

We can apply the mitigation strategies of minification and paraphrasing to the training data, to test our red team's robustness to signal jamming.

Run from the repo root:

**Minification:**
```
uv run src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/transforms/minified_config_train.yaml
```

**Paraphrasing:**
```
uv run src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/transforms/paraphrased_config_train.yaml
```

Your data will appear in `experiments/open-source-model/bigcodebench/bcb_train/transforms/minified` and `experiments/open-source-model/bigcodebench/bcb_train/transforms/paraphrased`.

You now have 3 pairs of honeypot and attack training samples, one for each of the three arms (no signal jamming, minification, and paraphrasing):

| Arm | Attack file | Honeypot file |
|---|---|---|
| none | `bcb_train/generator/attacks.eval` | `bcb_train/generator/honeypots.eval` |
| minified | `bcb_train/transforms/minified/attack_transformed.eval` | `bcb_train/transforms/minified/honeypots_transformed.eval` |
| paraphrased | `bcb_train/transforms/paraphrased/attack_transformed.eval` | `bcb_train/transforms/paraphrased/honeypots_transformed.eval` |

**Important:** Each pair of eval files must sit in a `train` subdirectory. Give each arm its own folder and copy its pair in (from the repository root). For example, for the none arm:

```sh
arm_dir=experiments/open-source-model/bigcodebench/bcb_train/sft_data/none/eval_logs
mkdir -p "$arm_dir/train"
cp experiments/open-source-model/bigcodebench/bcb_train/generator/{attacks,honeypots}.eval "$arm_dir/train/"
ls "$arm_dir/train"
```

Repeat for `minified` and `paraphrased` using the files in the table. The next commands expect the eval files there.

## Prepare data for fine-tuning

Next, we need to convert the .eval files into .json files.

Repeat these commands for each arm, with `arm` set to `none`, `minified` or `paraphrased`:
```
arm_root=experiments/open-source-model/bigcodebench/bcb_train/sft_data/$arm
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh $arm_root/eval_logs $arm_root/raw [--transformed]
```
**Important:** the first argument is the *parent* directory of the `train` subdirectory.

Include the `--transformed` flag for the minified and paraphrased arms.

You will find the .json files in `$arm_root/raw/train`.

Then run:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh $arm_root/raw <untrusted_generator_model_name> $arm_root/prepared
```
Repeat for each arm. The model name is U's bare Tinker id, with no `tinker/` prefix (e.g. `deepseek-ai/DeepSeek-V3.1`). `transform_logs_to_raw.sh` logs the name it found (`Auto-detected generator model ...`). If the name you pass matches none of the solutions, `transform_raw_to_prepared.sh` stops with an error that lists the models it found.

The script prints the number of positives (U's attacks) and negatives (H's honeypots). Only attacks and honeypots that pass the tests *and* have a working backdoor are kept, so expect far fewer examples than the 570 problems.

Check out `src/lasr_labs_2025_control_project/finetuning/binary_tuning/README.md` for more info on how these two commands work.
Check that `train_messages.jsonl` shows up under `$arm_root/prepared/train` for each arm.

## Fine-tune the untrusted model

Run this command to fine-tune the model through the Tinker API:
```
checkpoint_name=<name>
python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl $arm_root/prepared/train/train_messages.jsonl \
      --base-model deepseek-ai/DeepSeek-V3.1 --batch-size 16 \
      --checkpoint-name $checkpoint_name --out-path-file experiments/open_models/$checkpoint_name.txt \
      [--max-examples N] [--seed 0] [--val-frac 0.05]
```
**⚠️ Important (BCB):** the checkpoint names have a `bcb_` prefix, so they don't clash with the APPS arms in `ckpts.json`:
- No signal jamming: `checkpoint_name=bcb_self_rec_base`
- Paraphrased: `checkpoint_name=bcb_self_rec_paraphrased`
- Minified: `checkpoint_name=bcb_self_rec_minified`

`--batch-size 16` replaces the default 64 because the training set is small (see above). With 64 there would be only a handful of updates.

Pass the raw Tinker model ID to `--base-model`, without the `tinker/` provider prefix used in generated monitor configs.

The trainer prints the `tinker://` sampler-weights path and writes it to the `--out-path-file`. Add that path to `experiments/open_models/ckpts.json` under the matching arm name, **next to** any APPS entries already there:

```json
{
  "self_rec_base": "tinker://... (APPS, leave as is)",
  "bcb_self_rec_base": "tinker://<run-id>/sampler_weights/bcb_self_rec_base",
  "bcb_self_rec_minified": "tinker://<run-id>/sampler_weights/bcb_self_rec_minified",
  "bcb_self_rec_paraphrased": "tinker://<run-id>/sampler_weights/bcb_self_rec_paraphrased"
}
```
Create the file if it does not exist.

The optional checkpoint learning-dynamics tracking described in `sft-steps.md` works the same way here. Use the BCB test `attacks.eval` / `honeypots.eval` from the next step.

## Generate test data

```
python3 src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/generator/generator_config.yaml
```
Your testing data should appear in `experiments/open-source-model/bigcodebench/self_rec/generator` as `honest.eval`, `honeypots.eval`, and `attacks.eval`.

Apply minification and paraphrasing. Run these two commands:

```
python3 src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/transforms/minified_config.yaml
```
```
python3 src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/transforms/paraphrased_config.yaml
```
Check that your transformed testing data has saved under `experiments/open-source-model/bigcodebench/self_rec/transforms/minified` and `experiments/open-source-model/bigcodebench/self_rec/transforms/paraphrased`.

## Let's test the untrusted and trusted monitors!

We need to pass the `tinker://` paths of the fine-tuned checkpoints to our monitor config files. To do so, re-run `make_configs.py` with the **same model flags as before**:
```
python3 experiments/open_models/make_configs.py --model-class open --settings bigcodebench \
  --U deepseek-ai/DeepSeek-V3.1 --T Qwen/Qwen3.5-4B --H openai/gpt-oss-20b \
  --checkpoints experiments/open_models/ckpts.json
```
Check that none of `configs/bigcodebench_open/self_rec/monitors/sft/*.yaml` still contains `PENDING__bcb`.

Run this command to test the untrusted and trusted monitors when no signal jamming is applied:
```
python3 src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/monitors/sft/base_monitor_config.yaml
```
To test the monitors against the signal-jammed arms, re-run the previous command, replacing the config file name with `minified_monitor_config.yaml` and/or `paraphrased_monitor_config.yaml`.

## Plot your results
```
python3 src/lasr_labs_2025_control_project/scripts/plot_data.py --config src/lasr_labs_2025_control_project/configs/bigcodebench_open/self_rec/monitors/sft/base_monitor_config.yaml
```

To generate plots for the minified and paraphrased arms, replace `base_monitor_config.yaml` with `minified_monitor_config.yaml` and/or `paraphrased_monitor_config.yaml`.

Plots for the no-signal-jamming arm appear in `experiments/open-source-model/bigcodebench/self_rec/results/none_sft`. Plots for the signal-jamming arms appear in `results/minified_sft` and `results/paraphrased_sft` in the same folder.
Upload those graphs to the team Google Drive and share any key notes/observations you make!
