# Hand-prepared SFT data

The fine-tuning scripts prepare their own training data from `.eval` logs (`--train-eval`; see the
top-level README). Use these scripts only when you need to inspect or edit the prepared dataset
before training, for example for the label-ablation controls.

```bash
# .eval logs (in <dir>/train/) -> raw JSON (add --transformed for minified/paraphrased logs)
bash transform_logs_to_raw.sh <dir> [<dir>/raw] [--transformed] [--keep-comments]

# raw JSON -> chat JSONL; <target_model> is the generator model labelled YES (U)
bash transform_raw_to_prepared.sh <dir>/raw <target_model> [<dir>/prepared]
```

The result is `<dir>/prepared/train/train_messages.jsonl`. Edit it if needed; the label
ablations are `finetuning/scripts/randomise_labels.py` and `permute_labels.py`. Then train on it
with `--train-jsonl` in place of `--train-eval`.

For more options (custom prompts, upsampling) run
`uv run src/lasr_labs_2025_control_project/finetuning/utils/generate_train_data.py --help`.
