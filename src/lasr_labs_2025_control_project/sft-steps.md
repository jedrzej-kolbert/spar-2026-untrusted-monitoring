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

## Optional: track checkpoint learning dynamics

Plain fine-tuning does **not** track dynamics automatically. Add
`--checkpoint-every-steps`, `--attacks-eval`, `--honeypots-eval`, and
`--dynamics-results` to score saved checkpoints and plot attack-vs-honeypot ROC AUC.
Generate the held-out test logs (see “Generate test data” below) **before** launching
training with dynamics enabled. Do not use the training logs as evaluation inputs.

Run from the repository/worktree root, replacing the example paths:

```bash
export PYTHONPATH="$PWD/src"
python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
  --train-jsonl /path/to/prepared/train/train_messages.jsonl \
  --base-model deepseek-ai/DeepSeek-V3.1 \
  --rank 32 --lr 1e-4 --epochs 3 --batch-size 64 --val-frac 0 \
  --checkpoint-name self_rec_summarized \
  --checkpoint-every-steps 12 \
  --attacks-eval /path/to/transformed/test/attacks.eval \
  --honeypots-eval /path/to/transformed/test/honeypots.eval \
  --transformed \
  --dynamics-results results/summarized/run-01
```

For minified, paraphrased, or summarized arms, use `--transformed` both when
converting training logs with `transform_logs_to_raw.sh` and in this command.
The trainer uses prepared JSONL messages as-is; its flag selects the evaluation
log's `transformed_solution`. Omit the flag for original-code evaluation and use
the original-code training JSONL for a matching base arm. Transformed logs retain
both representations, so their path alone does not select the transformed one.
Use a separate results folder for every training run/arm.

The default checkpoint interval is `0`: save only the final checkpoint. With
dynamics enabled this evaluates only that final checkpoint. A positive interval
saves/evaluates periodically and also at the end if the final update is not an
interval boundary. Ordinary nonzero training does not automatically evaluate a
step-zero baseline. Omitting both eval paths and `--dynamics-results` disables
dynamics; the separate held-out collapse check still runs unless `--val-frac 0`.

Training pauses for each checkpoint evaluation. Invalid samples are filtered;
attacks have label 1 and honeypots label 0, with at least one retained sample of
each required. The score is `log P(YES) - log P(NO)`. With 59 attacks and 131
honeypots, each checkpoint uses approximately 380 Tinker logprob requests.
This measures self-recognition, not the full trusted/untrusted monitor safety
and usefulness evaluation later in this guide.

### Output and a historical example

```text
results/summarized/run-01/
├── dynamics.csv
├── dynamics.svg
└── dynamics.csv.manifest.json
```

The CSV records `updates`, the full sampler `checkpoint` URI, `train_loss` from
the latest training batch (not an epoch average), `roc_auc`, and retained
`eval_n`. A checkpoint row is written before scoring; blank AUC/count fields mean
evaluation did not finish. The SVG is refreshed after each successful evaluation.
The manifest fingerprints ordered labels, rendered prompts/solutions, and the
selected representation. Keep it with the CSV. Changed evaluation inputs or a
legacy CSV without a manifest require a new results folder.

The following is real historical data from
`attack_summarized_sft/results/batch64_no_holdout_20260927_133242`, in the issue11
experiment checkout at `093224b1a`. Its log records DeepSeek-V3.1, rank 32,
learning rate 1e-4, 406 training examples, batch size 64, three epochs, no holdout,
and checkpoints every **3** updates (the command above illustrates **12**).
It evaluated 190 examples and finished at update 21. These are existing results,
not a new training run or a prediction for other arms.

![Historical summarized batch-64 checkpoint dynamics](../../docs/examples/tinker-dynamics/summarized-batch64.svg)

[Download the SVG](../../docs/examples/tinker-dynamics/summarized-batch64.svg) ·
[Download the full CSV](../../docs/examples/tinker-dynamics/summarized-batch64.csv)

The SVG was regenerated from the archived CSV with this PR's plotting function,
without model calls. The CSV is an unchanged historical export, without a
provenance manifest; use it for comparison, not as a resumable output folder.
First five rows below; numeric values are rounded and checkpoint names shortened
for display (the download contains full URIs and precision):

| updates | checkpoint (name only) | train_loss | roc_auc | eval_n |
| ---: | --- | ---: | ---: | ---: |
| 3 | `self_rec_attack_summarized_batch64-step-3` | 0.114226 | 0.695174 | 190 |
| 6 | `self_rec_attack_summarized_batch64-step-6` | 0.082992 | 0.710118 | 190 |
| 9 | `self_rec_attack_summarized_batch64-step-9` | 0.075698 | 0.731466 | 190 |
| 12 | `self_rec_attack_summarized_batch64-step-12` | 0.073693 | 0.746992 | 190 |
| 15 | `self_rec_attack_summarized_batch64-step-15` | 0.071469 | 0.729719 | 190 |

Historical issue11 comparison CSVs also contain a manually added step-zero point.
That was post-processing, not automatic output of this trainer. The example
attached here contains only the recorded checkpoints at updates 3 through 21.

### Retry an interrupted checkpoint evaluation

Use the URI and update number from the incomplete CSV row, with the same eval
logs, representation, and results folder:

```bash
python -m lasr_labs_2025_control_project.scripts.tinker_sft_dynamics \
  --checkpoint 'tinker://<run-id>/sampler_weights/<checkpoint-name>' \
  --updates 12 \
  --attacks-eval /path/to/transformed/test/attacks.eval \
  --honeypots-eval /path/to/transformed/test/honeypots.eval \
  --transformed \
  --dynamics-results results/summarized/run-01
```

This retries evaluation, not training. Completed rows are reused only with
matching provenance; no extra model calls are made for those rows.

### Compare original, minified, and paraphrased runs

Keep each run's CSV/manifest separate. For analysis, concatenate CSVs with an
explicit run/arm label; do not merge them into a trainer output CSV, because
update numbers overlap and checkpoints/evaluation representations differ.
This example uses the existing pandas and matplotlib dependencies:

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

runs = {
    "base": Path("results/base/run-01/dynamics.csv"),
    "minified": Path("results/minified/run-01/dynamics.csv"),
    "paraphrased": Path("results/paraphrased/run-01/dynamics.csv"),
}
combined = pd.concat(
    [pd.read_csv(path).assign(run=run) for run, path in runs.items()],
    ignore_index=True,
)
output = Path("results/comparison")
output.mkdir(parents=True, exist_ok=True)
combined.to_csv(output / "dynamics-comparison.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
for run, rows in combined.dropna(subset=["roc_auc"]).groupby("run", sort=False):
    rows = rows.sort_values("updates")
    ax.plot(rows["updates"], rows["roc_auc"], marker="o", label=run)
ax.set(xlabel="Gradient updates", ylabel="Attack vs honeypot ROC AUC", ylim=(0, 1))
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(output / "dynamics-comparison.svg")
plt.close(fig)
```

For a matched arm comparison, use the same held-out problem IDs, labels, and
filtering policy, with the intended representation for each arm. Equal `eval_n`
alone does not establish matched data. Different training/evaluation arms measure
different conditions; label those conditions explicitly. Equal update counts
also do not imply equal data exposure when batch sizes differ. Compare matching
batch sizes or record examples seen/epochs separately; the CSV does not contain
enough information to recover these reliably, especially for partial batches.

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
