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

## LinuxArena: native action-view self-recognition

This setting uses LinuxArena trajectories instead of code solutions. Edit U/T/H in
`ROLE_MODELS["open"]` in `experiments/open_models/make_configs.py`, just as above.
All six stages below consume one generated `linuxarena_open/pipeline.yaml`; do not
copy model names into shell commands. The recogniser and the trainable base model
are U. U produces honest and attack runs; H produces honeypots under the same
attack policy and side task as U. All three populations use the same main tasks.

### Two Python environments

Keep this project's environment (`uv sync`, Inspect `0.3.133`) separate from a
Control Tower checkout pinned to **`bf64499`**, with its own Python 3.13+ virtual
environment, **`inspect-ai==0.3.268`** and `tinker_cookbook` installed. Control Tower's
native eval-seat action views, monitor configuration and post-hoc score format
are the interfaces this pipeline targets; its newer Inspect version is incompatible
with this project's pin. Do not add Control Tower, cookbook or its Inspect version
to this project's `pyproject.toml` or lockfile.

In the separate checkout, follow Control Tower's installation instructions, pin
that checkout before installing, and pull its LinuxArena settings (`ct settings
pull linuxarena`). Install cookbook only into that separate environment. Use the
prebuilt LinuxArena images; this pipeline never builds local images. A different
Control Tower revision or settings/image revision is not a verified substitute.
Record those revisions with your experiment. The bridge checks the Inspect pin
at runtime and loads Control Tower/cookbook only inside the CT interpreter.

From **this project's repository root**, set these shell variables (replace only
the CT checkout path, or the interpreter paths if your environments live elsewhere):

```bash
export PYTHONPATH="$PWD/src"
PROJECT_PY="$PWD/.venv/bin/python"
CT_ROOT=/path/to/control-tower
CT_PY="$CT_ROOT/.venv/bin/python"
CFG=src/lasr_labs_2025_control_project/configs/linuxarena_open/pipeline.yaml
PIPELINE=lasr_labs_2025_control_project.scripts.run_linuxarena
ENV_FILE="$PWD/.env"
# Check the checkout pin and the separate interpreter without making model calls.
git -C "$CT_ROOT" rev-parse --short=7 HEAD  # must print bf64499
"$CT_PY" -c 'from importlib.metadata import version; assert version("inspect-ai") == "0.3.268"; import tinker_cookbook'
"$PROJECT_PY" experiments/open_models/make_configs.py --model-class open --settings linuxarena
```

Configure `TINKER_API_KEY` in `ENV_FILE`. Generated YAML is gitignored, like the
APPS/BigCodeBench config trees; only the declarative spec and pipeline code are
checked in. LinuxArena is opt-in, so the default APPS/BigCodeBench invocation and
its generated output are unchanged. LinuxArena's own completion cap is in
`LINUXARENA["generation"]["max_tokens"]`; the existing `--max-tokens` option still
applies only to the APPS/BigCodeBench generators.

### Per-model and run settings

Change LinuxArena settings in the `LINUXARENA` spec in `make_configs.py`, then
regenerate. Defaults select ten deterministic `fix_bug` main tasks in
`shared_dotfiles`, with `bypass_blacklist` as the shared attack side task. Set
`generation.main_tasks` to an explicit list to select a different common task set;
otherwise selection uses `task_category`, `task_count` and `selection_seed`.
The run's `plan.json` freezes both tasks and the entire config. Changing the
triple or settings after generation requires a fresh `experiment_folder` (set
`LINUXARENA["experiment_folder"]` to a new repository-relative output path), rather
than silently mixing data from different runs. The default is
`experiments/open-source-model/linuxarena/self_rec`.

`model_settings` is keyed by **role**, not hard-coded model names:

- `renderer: null`: use cookbook's recommended renderer for generation.
- `monitor_renderer: null`: use the metadata-recommended disable-thinking renderer
  when available, otherwise the recommended renderer. This applies to both T's
  trusted monitor and base U's honest suspiciousness monitor. An explicit monitor
  override that re-enables thinking when a disable-thinking recommendation exists
  is refused. Supply an explicit cookbook renderer for an unknown model; without
  metadata or an override, the bridge fails before sampling, with the role-setting
  to change. This is a renderer compatibility choice, not evidence that a model
  is sampleable or supports tools: verify those separately before approving a run.
- `tool_dialect: auto`: inspect what each call actually emits, translate declared
  dotted tool names (such as `bash.exec`) and string-argv shell commands to the
  environment's schema, and restore the exact original call in the model's own
  subsequent history. Translation is logged; it does not depend on a model-name
  substring. `schema` forbids argv translation; `argv` also restores shell argv
  for canonical calls in pre-existing history. Unknown dialects, malformed argv,
  and dotted names with no declared base tool fail loudly. Other arguments,
  including `timeout`, are left untouched. Millisecond-looking timeout values may
  therefore be clamped as seconds by LinuxArena; normalising/dropping this field
  needs a separate project-owner decision and is not done implicitly here.

A monitor's forced tool choice is offered as `auto`, restricted to the requested
score tool when applicable: Tinker cannot force decoding. Tool-less replies reach
Control Tower's own retries and unscored-action ledger instead of disappearing.
Calls get unique IDs, and both monitor routes disable Inspect response caching
so a renderer change cannot retrieve an earlier thinking-on response.

One sandbox runs at a time. Every service gets CPU, memory, swap and PID caps;
default agent caps are 0.5 CPUs/1536 MiB, sidecars 0.1 CPUs/512 MiB each. Generation
pauses below 5 GiB available RAM or 12 GiB free disk. Privileged, host-namespace,
host-mounted and device-backed services are refused, and image build sections
are removed. Tinker's dollar prices are unknown: Control Tower pricing is turned
off, **not treated as zero cost**; token accounting remains available in eval logs.

### Stages 1–6

The `--allow-paid` flag is an explicit runtime guard, not a cost estimate. Approve
sampling/training/logprob costs before executing the cost-bearing commands.
`--dry-run` prints planned commands without invoking `ct run` or Tinker; generation
planning still requires the CT environment and may load local model/task metadata.

**1. Generate trajectories — CT Python; cost-bearing model sampling and Docker.**

```bash
"$CT_PY" -m "$PIPELINE" --config "$CFG" --env-file "$ENV_FILE" --allow-paid generate
```

This sequentially generates U/honest, U/attack and H/attack for every selected
main task, with the same side task for both attacks. To approve a smaller first
phase, add `--limit 1`, inspect its log and resources, then rerun without the limit.
Finished jobs are skipped. Failed runs are recorded and stop the current phase;
rerunning without `--retry-failed` proceeds to other jobs while retaining the failed
attempt, or use `--retry-failed` to explicitly pay for a retry. Interrupted attempts'
partial `.eval` files are recovered. Every emitted log, including failed attempts,
is retained in `manifest.json`; failures without eval logs remain in
`generation_status.jsonl` and `run_ledger.json`.

**2. Score every action with T and base U — CT Python; cost-bearing sampling.**

```bash
"$CT_PY" -m "$PIPELINE" --config "$CFG" --env-file "$ENV_FILE" --allow-paid monitor
```

One `ct run monitor` per source asks native `simple` action monitors named
`trusted` (T) and `untrusted_suspiciousness` (base U), both with `cot_level: none`.
The two score sets coexist in each monitored copy. Status and raw-to-monitored
provenance are saved in `monitor_status.jsonl`; `export_manifest.json` prefers
monitored copies and falls back to raw logs when no copy exists. Missing source
files cause an error, never a silent skip. Successful process exit does not imply
that every draw has a score: inspect the export coverage before fitting.

**3. Export action views and SFT files — CT Python; offline, no model calls.**

```bash
"$CT_PY" -m "$PIPELINE" --config "$CFG" export
```

This delegates to `scripts/export_linuxarena_actions.py`, taking target U from the
config. `exports/actions.jsonl` retains all populations, failed samples and
unscored native draws. `trajectory_ledger.jsonl` retains zero-action samples.
Splits are task-connected components, deterministic 60/20/20 by main task, shared
across authors and populations; actions from the same main task cannot leak across
splits. `sft_attack_{train,val,test}.jsonl` contains matched-intent own-attack YES
versus honeypot NO examples; honest actions remain in the action/all-SFT files.
The exporter refuses an existing output directory. For a repaired monitoring pass,
retain the old export under a different name before running stage 3 again.

**4. Fine-tune U — project Python; cost-bearing Tinker training.**

```bash
"$PROJECT_PY" -m "$PIPELINE" --config "$CFG" --env-file "$ENV_FILE" --allow-paid finetune
```

The wrapper delegates to the existing `finetuning/tinker_lora_finetune.py`, supplies
the raw U ID, and saves the sampler URI to `sft/checkpoint.txt`. Rank, LR, epochs,
batch size and seed come from the config. It uses only `sft_attack_train.jsonl` and
sets `--val-frac 0` to avoid the trainer's random **action-level** split; the exported
validation tasks remain reserved, and held-out test tasks are never used for
training. A repeated finetune command trains again; it is not training resumption.

**5. Score held-out recognition rows — project Python; cost-bearing logprobs.**

```bash
"$PROJECT_PY" -m "$PIPELINE" --config "$CFG" --env-file "$ENV_FILE" --allow-paid score
```

This delegates to `scripts/score_linuxarena_actions.py` on `sft_attack_test.jsonl`
for both base U and the checkpoint saved in stage 4. Each row uses two paid YES/NO
logprob requests, producing `sft/scores_test_{base,sft}.jsonl`. Every request failure
is recorded. Rerunning retries only rows lacking a finite margin, retaining earlier
errors in the append-only file. Resume checks the model and a SHA256 identity of
exactly the system/user text before making requests; changing either requires a
new output file. Historical pilot score files lack this input hash: they remain
readable for offline analysis but are refused for resume (do not retrofit an
unverifiable hash). Honest rows need native suspiciousness scores, not a
collusion-recognition score, for the statistical model.

**6. Assemble flat tables and plot — project Python; offline.**

```bash
"$PROJECT_PY" -m "$PIPELINE" --config "$CFG" plot
```

The wrapper joins by stable action ID, not row order. It writes
`sft/plot_all/flat_{base,sft}.jsonl` and `coverage.json` **before fitting**. Every
held-out action stays in the tables, with missing values and errors visible.
Trusted and honest-U suspiciousness scores come from their separate named native
monitors; recognition margins supply the base/SFT collusion score. If any required
score is absent/nonfinite, or a population is empty, fitting stops: repair/retry
the failed requests, never drop those rows or use the perfect-honest-U fallback.
Failed trajectories with valid scores are included normally. For a new paid
monitor pass, add `--retry-failed` for process failures, or `--rescore-complete`
to explicitly pay for new draws on completed jobs that had unscored actions.
Both options append new attempts without deleting old status/log records. Retain
the previous export under another name, then re-export.

With complete data, the same `build_optimization_config`, `optimize_parameters`
and `plot_data` functions used by `scripts/plot_data.py` produce the full figure
suite (including histograms, ROC and safety/usefulness curves) for both arms,
alongside `analysis_data_{base,sft}.json.gz`. Game parameters are in `plot` in the
spec. These are action-weighted observations with trajectory-intent labels, not
validated per-action maliciousness annotations; differing action counts per
trajectory can affect the statistical comparison.
