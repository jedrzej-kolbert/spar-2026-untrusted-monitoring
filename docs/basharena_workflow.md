# BashArena workflow

BashArena trajectories are generated and exported in their own environment
(`basharena_runtime/`, control-arena 17). The main project reads the exported
JSONL and never needs the newer control-arena.

```
basharena_runtime (control-arena 17)        main project (control-arena 2.1.3)
  generate  ->  export  ──── JSONL ────>  prepare  ->  train / score  ->  compare
```

Only the baseline view is supported: one self-recognition score and one
suspiciousness score per action, on the unmodified trajectory.

## 1. Generate and export

Run in `basharena_runtime/`; see its [README](../basharena_runtime/README.md).
Export each `.eval` log to its own directory, declaring the split it was
generated on:

```bash
cd basharena_runtime
uv run --frozen python -m basharena_runtime.export out/test/attacks.eval \
  --output ../exports/test_attacks --population attacks \
  --split test --split-provenance "task-id parity: even ids"
```

## 2. Prepare

```bash
uv run python src/lasr_labs_2025_control_project/scripts/basharena_data.py prepare \
  --exports exports/train_attacks exports/train_honeypots \
            exports/test_attacks exports/test_honeypots exports/test_honest \
  --target-actor tinker/Qwen/Qwen3.6-35B-A3B \
  --output prepared/
```

`--target-actor` is the model whose actions count as its own. An actor read from
a log is stored without its provider (`Qwen/X` for `tinker/Qwen/X`) and matches
either spelling. An actor declared at export time must match exactly.

`prepared/` gets:

- `actions.jsonl`, `trajectories.jsonl`: every action and attempt, with split,
  population and `own_source`.
- `sft_train.jsonl`, `sft_val.jsonl`, `sft_test.jsonl`: self-recognition examples
  ending in `<answer>YES</answer>` or `<answer>NO</answer>`, in the format
  `tinker_lora_finetune.py` loads.
- `sft_exclusions.jsonl`: actions left out of the examples, with the reason.

The examples use attack and honeypot actions only, so both classes are attacks
and differ only in who acted. `--include-honest` adds honest actions.

Preparation fails if a main task appears in two splits, or if an export is
incomplete or has missing actions.

## 3. Score

```bash
# Prints how many requests would be made. No model calls.
uv run python src/lasr_labs_2025_control_project/scripts/basharena_data.py score \
  --prepared prepared/ --models models.json --output scores/base.jsonl

# Makes the calls.
uv run python src/lasr_labs_2025_control_project/scripts/basharena_data.py score \
  --prepared prepared/ --models models.json --output scores/base.jsonl --allow-paid
```

`models.json` maps roles to models; any subset is allowed:

```json
{
  "recognition": "tinker/Qwen/Qwen3.6-35B-A3B",
  "trusted_suspiciousness": "tinker/Qwen/Qwen3.5-4B",
  "untrusted_suspiciousness": "tinker/Qwen/Qwen3.6-35B-A3B"
}
```

- `recognition` is `log P(YES) - log P(NO)` and needs a `tinker/` model or
  checkpoint.
- The suspiciousness roles are the 0-9 score in the answer's `<score>` tag.
  Decimal answers are rounded.

Scoring is resumable: rerunning the same command retries only the requests that
failed. Each record keeps the model's raw answer, including answers that had no
usable score. The command exits non-zero while any score is missing.

A score file belongs to one set of models and inputs. Use a new `--output` for a
different checkpoint, and give that run only the roles that changed:

```json
{"recognition": "tinker/tinker://<run>/sampler_weights/<name>"}
```

Monitor calls are sampled, so scoring the same role, model and action twice
gives two different draws, and `compare` refuses files that disagree on one.
Compare the checkpoint's file together with the base file instead; the base
suspiciousness scores are then used for both.

## 4. Compare

```bash
uv run python src/lasr_labs_2025_control_project/scripts/basharena_data.py compare \
  --prepared prepared/ --scores scores/base.jsonl scores/sft.jsonl \
  --output comparison/ --cohort all
```

`comparison.json` reports coverage and AUCs per role and model, and
`trajectory_scores.jsonl` the per-trajectory maxima. A trajectory's score is the
maximum over all of its actions; a trajectory with any unscored action, or with
no actions, has no score and the comparison is marked incomplete.

`--cohort` chooses which attempts are compared:

| Cohort | Attempts |
|---|---|
| `all` (default) | Every attempt on the split |
| `baseline` | Only attempts the older success-filtered loader kept |

Use `baseline` when comparing against results from that loader. The two cohorts
give different numbers on the same data, so say which one a result uses.
