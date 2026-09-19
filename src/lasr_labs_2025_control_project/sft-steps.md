# End-to-end Pipeline for Running SFT Experiments with Passive Self-Recognition 

## Setup 

Create a virtual environment for this project. Run the following command to install all dependencies: 
```
uv sync
```

Select two models from the Tinker model directory to use as untrusted generator and honeypot generator. Save their model strings. 

Configure the Tinker API key in the .env file. 

Open `configs/apps_open/honest_generator_config.yaml`. You should see something like: 
```
...
runs: 
    honest: 
        ...
        model: tinker/<untrusted_generator_model_name>
    honeypots: 
        ...
        model: tinker/<honeypot_generator_model_name>

```
Create a copy of the config file, and paste the model strings into their corresponding locations. 

**Important:** In your config file, change the experiment_name variable to something different from whatever is already there. This is important to make sure you don't overwrite someone else's data!  
Your generated data will show up in `spar-2026-untrusted-monitoring/experiments/<experiment_name>`. 

## A note about changing the number of code samples 

Open `scripts/generate_solutions_inspect.py` and scroll down to line 173: 
```
...
shuffle=False, 
problem_ids_range=[0, 1000], # old: 2999
```
You don't have to change anything here for now: just know that setting the range from 0-1000 will give each model (untrusted and honeypot generators) 500 coding problems to solve. 

For training, the models will get all the even problems in the range you specify, and for testing, the models will get all the odd problems. 

Note that if you ever want to increase or decrease the number of samples you generate per model, this is where you go to do it (max 2999). I will try to make this easier to do by adding a "problem ids range" flag later. 

## Training data generation 

To generate the .eval files, run this command: 

```
python src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py --config <path/to/config/file>
```
Check that your eval files were saved in `experiments/<experiment_name>/generator`. 

**Important:** Create a subdirectory under the `generator` folder called `train`. This is necessary for the next commands to work.

Next, we need to convert the .eval files into .json files and prepare the data for fine-tuning. 

## Prepare data for fine-tuning 

Run these commands:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh experiments/<experiment_name>/generator

bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh experiments/<experiment_name>/raw <untrusted_generator_model_name>
```
Check out src/lasr_labs_2025_control_project/finetuning/binary_tuning/README.md for more info on how these two commands work. 
Check that `train_messages.jsonl` shows up under `experiments/<experiment_name>/prepared`. 

## Fine-tune the untrusted model 

Run this command to fine-tune the model through the Tinker API: 

```
python -m src.lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl <path/to/train_messages.jsonl> \
      --base-model <untrusted_generator_model_name> \
      [--rank 32] [--lr 1e-4] [--epochs 3] [--batch-size 64] \
      --checkpoint-name <checkpoint_name> [--out-path-file <file_name>.txt] \
      [--max-examples N] [--seed 0] [--val-frac 0.05]
```
This is copied directly from the docstring in src/lasr_labs_2025_control_project/finetuning/tinker_lora_finetune.py. 
Once the fine-tuning process is done, check the Tinker console to see that your checkpoint appeared under 'Checkpoints'. 