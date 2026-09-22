# Tinker training cost estimates

The LoRA trainer counts the `model_input` tokens submitted in each completed
training batch. It logs step and cumulative counts locally and, with `--wandb`,
to W&B. Supply a training rate to also estimate dollar cost:

```bash
uv run src/lasr_labs_2025_control_project/finetuning/tinker_lora_finetune.py \
  --train-jsonl experiments/self_rec_train/prepared/train/train_messages.jsonl \
  --base-model deepseek-ai/DeepSeek-V3.1 \
  --wandb --train-rate-usd-per-million 3.718
```

The example rate is Tinker's listed DeepSeek-V3.1 training price on
2026-09-22. Check the current rate for your base model in the
[Tinker Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
page before each run. Without a rate, the trainer reports tokens but no dollars.

The dollar figure is an estimate for training batches only. It excludes
validation forward passes, checkpoint evaluation or sampling, and checkpoint
storage. Tinker's billing records are the source for actual charges.
