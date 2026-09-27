# Running SFT Experiments with Passive Self-Recognition

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

## Configs

Before making your config files, open `experiments/open_models/make_configs.py` to line 142:
```
#: role -> model string, per model class. U untrusted, T trusted/weak, H honeypot.
ROLE_MODELS: dict[str, dict[str, str]] = {
    "closed": {
        "U": "openai/gpt-4.1-2025-04-14",
        "T": "openai/gpt-4o-mini",
        "H": "google/gemini-2.5-flash",
    },
    "open": {
        "U": "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "T": "tinker/Qwen/Qwen3.5-4B",
        "H": "tinker/openai/gpt-oss-120b",
    },
}
```
This is where you'll set U, T, and H. Paste the model strings you found earlier into the corresponding locations under `"open"`.

Generate all config files:
```
python experiments/open_models/make_configs.py --model-class open
```

## A note about changing the number of code samples

Line 173 of `scripts/generate_solutions_inspect.py` is where the number of problems are defined:
```
...
shuffle=False,
problem_ids_range=[0, 2999],
```
You don't have to change anything here for now: just know changing this setting will change how many problems the model will solve.

For training, the models will get all the odd problems in the range you specify, and for testing, the models will get all the even problems.

**Note:** If trying to conserve tokens, I'd recommend lowering the upper limit of `problem_ids_range` to maybe 2000, but **only when you are generating training data** because usually fewer problems are enough to still achieve high self-rec accuracy.

## Training data generation

To generate the training data, run this command:

```
python src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/generator/train_generator_config.yaml
```
Check that your eval files were saved in `experiments/open-source-model/apps/self_rec_train/generator`.

## Applying blue-team mitigation strategies to training data 

We can apply the mitigation strategies of minification and paraphrasing to the training data, to test our red team's robustness to signal jamming. 

Run from the repo root: 

**Minification:** 
```
uv run src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/transforms/minified_config_train.yaml 
```

**Paraphrasing:**
```
uv run src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/transforms/paraphrased_config_train.yaml 
```

Your data will appear in `experiments/open-source-model/apps/self_rec_train/transforms/minified` and `experiments/open-source-model/apps/self_rec_train/transforms/paraphrased`.

You now have 3 pairs of honeypot and attack training samples, for each of three arms (no signal jamming, minification, and paraphrasing).

**Important:** Move each pair of generated eval files into a `train` subdirectory. Repeat these commands three times, for each pair (from the repository root):

```sh
eval_files_dir=<path/to/eval/files>
mkdir -p "$eval_files_dir/train"
mv "$eval_files_dir"/*.eval "$eval_files_dir/train/"
ls "$eval_files_dir/train"
```

The next commands expect the eval files there.

## Prepare data for fine-tuning

Next, we need to convert the .eval files into .json files.

Repeat these commands for each pair of .eval files:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh <eval_logs_dir> <eval_logs_dir>/raw [--transformed]
```
**Important:** `<eval_logs_dir>` should point to the *parent* directory of the `train` subdirectory where all pairs of the .eval files are.

Include the `--transformed` flag for the minified and paraphrased arms.

You will find the .json files in `<eval_logs_dir>/raw`.

Then run:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh <eval_logs_dir>/raw <untrusted_generator_model_name>
```
Repeat thrice for each arm.

Check out `src/lasr_labs_2025_control_project/finetuning/binary_tuning/README.md` for more info on how these two commands work.
Check that `train_messages.jsonl` shows up under `<eval_logs_dir>/prepared/train` for each arm.

## Fine-tune the untrusted model

Run this command to fine-tune the model through the Tinker API:
```
checkpoint_name=<name> 
python -m src.lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl <eval_logs_dir>/prepared/train/train_messages.jsonl \
      --base-model <supported_tinker_model_id> \
      --checkpoint-name $checkpoint_name --out-path-file experiments/open_models/$checkpoint_name.txt \
      [--max-examples N] [--seed 0] [--val-frac 0.05]
```
**⚠️ Important:** Change `checkpoint_name` to the corresponding names for each signal jamming regime: 
- No signal jamming: `checkpoint_name=self_rec_base`
- Paraphrased: `checkpoint_name=self_rec_paraphrased`
- Minified: `checkpoint_name=self_rec_minified`

Pass the raw Tinker model ID to `--base-model` (for example, `Qwen/Qwen3-8B`), without the `tinker/` provider prefix used in generated monitor configs. Check Tinker's supported model list if the ID is rejected.

The trainer prints the `tinker://` sampler-weights path and writes it to `experiments/open_models/<checkpoint_name>.txt`. Add that path to `experiments/open_models/ckpts.json` under the matching arm name:

```json
{
  "<checkpoint-name>": "tinker://<run-id>/sampler_weights/<checkpoint-name>"
}
```
Create the file if it does not exist. `--checkpoint-name` is the remote sampler-weights name; it does not write to `ckpts.json`.

## Optional: track checkpoint learning dynamics

To track ROC AUC during training, use prepared training JSONL and held-out attack/honeypot `.eval` files:

```bash
export PYTHONPATH="$PWD/src"
python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
  --train-jsonl /path/to/train_messages.jsonl \
  --base-model deepseek-ai/DeepSeek-V3.1 \
  --rank 32 --lr 1e-4 --epochs 3 --batch-size 64 --val-frac 0 \
  --checkpoint-every-steps 12 \
  --attacks-eval /path/to/test/attacks.eval \
  --honeypots-eval /path/to/test/honeypots.eval \
  --dynamics-results results/my-run
```

For transformed data, add `--transformed` here and during training-data preparation.
The extra flags save/evaluate checkpoints every 12 updates and at the end.
`results/my-run/` contains `dynamics.csv` (checkpoint URIs, loss, AUC, sample count),
`dynamics.svg` (AUC over updates), and `dynamics.csv.manifest.json` (evaluation provenance).

## Generate test data

```
python3 src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/generator/generator_config.yaml
```
Your testing data should appear in `experiments/open-source-model/apps/self_rec/generator` as `honest.eval`, `honeypots.eval`, and `attacks.eval`.

Apply minification and paraphrasing. Run these two commands: 

```
python3 src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/transforms/minified_config.yaml
```
```
python3 src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/transforms/paraphrased_config.yaml
```
Check that your transformed testing data has saved under `experiments/open-source-model/apps/self_rec/transforms/minified` and `experiments/open-source-model/apps/self_rec/transforms/paraphrased`. 

## Let's test the untrusted and trusted monitors!

We need to pass `tinker://` paths to the fine-tuning checkpoints to our monitor config files. To do so, re-run this command:
```
python3 experiments/open_models/make_configs.py --checkpoints experiments/open_models/ckpts.json
```

Run this command to test the untrusted and trusted monitors when no signal jamming is applied:
```
python3 src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/monitors/sft/base_monitor_config.yaml
```
To test the monitors against the signal-jammed arms, re-run the previous command, replacing the config file name with `minified_monitor_config.yaml` and/or `paraphrased_monitor_config.yaml`.

## Plot your results
```
python3 src/lasr_labs_2025_control_project/scripts/plot_data.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/monitors/sft/base_monitor_config.yaml
```

To generate plots for the minifed and paraphrased arms, replace `base_monitor_config.yaml` with `minified_monitor_config.yaml` and/or `paraphrased_monitor_config.yaml`. 

Plots for the no-signal-jamming arm appear in `experiments/open-source-model/apps/self_rec/results/none_sft`. Plots for the signal-jamming arms appear in `results/minified_sft` and `results/paraphrased_sft`. 
Upload those graphs to the team Google Drive and share any key notes/observations you make!
