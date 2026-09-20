# End-to-end Pipeline for Running SFT Experiments with Passive Self-Recognition 

## Setup 

Create a virtual environment for this project. Run the following command to install all dependencies: 
```
uv sync
```

Select three models from the Tinker model directory to use as: 
-  untrusted generator (this will double as untrusted monitor)
-  honeypot generator
-  trusted monitor 
Save their model strings. 

Configure the Tinker API key in the .env file. 

## Configs 

The entire experiment is predicated on setting up your config files right. There are three configs we'll set up: 
- Config for fine-tuning data generation 
- Config for testing data generation 
- Config for the untrusted monitor 

### Config for Fine-tuning Data Generation:

Open `configs/apps_open/generator/honest_generator_config.yaml`. You should see something like: 
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
Paste the model strings into their corresponding locations. 

**Note:** Check out the folder path in `experiment_folder`. Your training data will show up in `spar-2026-untrusted-monitoring/<experiment_folder>`. Someone else's data might already be at that filepath; you can change that variable to save your data somewhere else.
For example: `experiment_folder: experiments/self_rec_train_#` 

### Config for Testing Data Generation and the Untrusted Monitor: 

 Open `configs/apps_open/generator/generator_config.yaml`, and repeat the process of pasting model strings. 
 
 Under the `honest` and `attacks` headers, you should paste the model string of the untrusted generator (U_G). Under the `honeypots` header, you should paste the model string of the honeypot generator (H).  

 The config for the untrusted monitor is in `configs/apps_open/monitors/base_monitor_config.yaml`.  
 - Place the model string for trusted monitor under `trusted_backdoor_monitor`. 
 - We won't replace the other two model strings yet; we'll do so after we fine-tune the untrusted generator. 

**⚠️ Important:** Check out the `experiment_folder` value in both of these configs; note that **they are the same**. You need to change this folder path so that you don't overwrite someone else's training run. **Please change both to be the same**. 
For example: 
```experiment_folder: experiments/apps/your_name/self_rec_1```

## A note about changing the number of code samples 

Line 173 of `scripts/generate_solutions_inspect.py` is where the number of problems are defined: 
```
...
shuffle=False, 
problem_ids_range=[0, 2999], 
```
You don't have to change anything here for now: just know changing this setting will change how many problems the model will solve.  

For training, the models will get all the even problems in the range you specify, and for testing, the models will get all the odd problems. 

## Training data generation 

To generate the .eval files, run this command: 

```
python src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py --config <path/to/config/file>
```
Check that your eval files were saved in `<experiment_folder>/generator`. 

**Important:** Create a subdirectory under the `generator` folder called `train`. Place the eval files you just generated into the `train` subdirectory. This is necessary for the next commands to work.

Next, we need to convert the .eval files into .json files and prepare the data for fine-tuning. 

## Prepare data for fine-tuning 

Run these commands:
```
bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh experiments/<experiment_name>/generator

bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh experiments/<experiment_name>/raw <untrusted_generator_model_name>
```
Check out src/lasr_labs_2025_control_project/finetuning/binary_tuning/README.md for more info on how these two commands work. 
Check that `train_messages.jsonl` shows up under `experiments/<experiment_folder>/prepared`. 

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

## Generate testing data 

```
python3 src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/generator/generator_config.yaml
```
Your testing data should appear in `<experiment_folder>/generator` as `honest.eval`, `honeypots.eval`, and `attacks.eval`. 

## Let's test the untrusted and trusted monitors!

Home stretch!
Open the Tinker console, go to Checkpoints, and grab the sampler checkpoint path from the fine-tuning run you just did. This is what you will pass to the monitor config file as the untrusted monitor. 

In `configs/apps_open/monitors/base_monitor_config.yaml`, pass the sampler checkpoint paths to the `model_name` values under `untrusted_collusion_monitor` and `untrusted_backdoor_monitor`. 

This should look like: 

```
...
monitors:
    - untrusted_collusion_monitor:
        model_info:
          model: tinker/tinker://<run_id>:train:0/sampler_weights/<name>
...
    - untrusted_backdoor_monitor:
        model_info:
          model: tinker/tinker://<run_id>:train:0/sampler_weights/<name>
...
```

Run this command to test the untrusted and trusted monitors: 

```
python3 src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py --config src/lasr_labs_2025_control_project/configs/apps_open/monitors/base_monitor_config.yaml
```
## Plot your results 
```
python3 src/lasr_labs_2025_control_project/scripts/plot_data.py --config src/lasr_labs_2025_control_project/configs/apps_open/monitors/base_monitor_config.yaml
```
You'll find all plots in `<experiment_folder>/results/none_sft`. Upload those graphs to the team Google Drive and share any key notes/observations you make! 