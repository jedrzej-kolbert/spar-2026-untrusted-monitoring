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

**Important:** Move the generated eval files into a `train` subdirectory. From the repository root:

```sh
generator_dir=experiments/open-source-model/apps/self_rec_train/generator
mkdir -p "$generator_dir/train"
mv "$generator_dir"/*.eval "$generator_dir/train/"
ls "$generator_dir/train"
```

The next commands expect the eval files there.

Next, we need to convert the .eval files into .json files and prepare the data for fine-tuning. 

## Prepare data for fine-tuning 

Run these commands:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh experiments/open-source-model/apps/self_rec_train/generator

bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh experiments/open-source-model/apps/self_rec_train/raw <untrusted_generator_model_name>
```
Check out `src/lasr_labs_2025_control_project/finetuning/binary_tuning/README.md` for more info on how these two commands work. 
Check that `train_messages.jsonl` shows up under `experiments/open-source-model/apps/self_rec_train/prepared`. 

## Fine-tune the untrusted model 

Run this command to fine-tune the model through the Tinker API: 

```
python -m src.lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl experiments/open-source-model/apps/self_rec_train/prepared/train/train_messages.jsonl \
      --base-model <supported_tinker_model_id> \
      [--rank 32] [--lr 1e-4] [--epochs 3] [--batch-size 64] \
      --checkpoint-name self_rec_base --out-path-file experiments/open_models/self_rec_base.txt \
      [--max-examples N] [--seed 0] [--val-frac 0.05]
```
Pass the raw Tinker model ID to `--base-model` (for example, `Qwen/Qwen3-8B`), without the `tinker/` provider prefix used in generated monitor configs. Check Tinker's supported model list if the ID is rejected.

The trainer prints the `tinker://` sampler-weights path and writes it to `experiments/open_models/self_rec_base.txt`. Add that path to `experiments/open_models/ckpts.json` under the matching arm name:

```json
{
  "self_rec_base": "tinker://<run-id>/sampler_weights/self_rec_base"
}
```

Create the file if it does not exist. `--checkpoint-name` is the remote sampler-weights name; it does not write to `ckpts.json`.

## Generate test data 

```
python3 src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py \
  --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/generator/generator_config.yaml
```
Your testing data should appear in `experiments/open-source-model/apps/self_rec/generator` as `honest.eval`, `honeypots.eval`, and `attacks.eval`. 

## Let's test the untrusted and trusted monitors!

We need to pass `tinker://` paths to the fine-tuning checkpoints to our monitor config files. To do so, re-run this command: 
```
python3 experiments/open_models/make_configs.py --checkpoints experiments/open_models/ckpts.json
```

Run this command to test the untrusted and trusted monitors: 
```
python3 src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/monitors/sft/base_monitor_config.yaml
```
## Plot your results 
```
python3 src/lasr_labs_2025_control_project/scripts/plot_data.py --config src/lasr_labs_2025_control_project/configs/apps_open/self_rec/monitors/sft/base_monitor_config.yaml
```
You'll find all plots in `experiments/open-source-model/apps/self_rec/results/none_sft`. Upload those graphs to the team Google Drive and share any key notes/observations you make!
