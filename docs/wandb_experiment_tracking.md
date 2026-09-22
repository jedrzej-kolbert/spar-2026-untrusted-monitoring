# W&B tracking for Tinker LoRA fine-tuning

The LoRA trainer can record loss, validation scores, ROC curves, and Tinker
checkpoint paths in W&B. Tracking is opt-in; `wandb` is already a project
dependency. Put `WANDB_API_KEY` in your local `.env` (see `env.example`), or log
in with `wandb login`, before starting. The trainer loads `.env` at startup.

```bash
uv run src/lasr_labs_2025_control_project/finetuning/tinker_lora_finetune.py \
  --train-jsonl experiments/self_rec_train/prepared/train/train_messages.jsonl \
  --base-model deepseek-ai/DeepSeek-V3.1 \
  --checkpoint-name self_rec_u \
  --wandb --wandb-entity s184361 \
  --wandb-project untrusted-monitoring-lora \
  --eval-every-epoch
```

Each training step records `train/loss`, `train/step`, and `train/epoch`.
With W&B enabled, each epoch also records `val/loss`. Saved checkpoints are
scored on the held-out rows, recording `val/collusion_accuracy`,
`val/collusion_auroc`, `val/yes_rate`, `val/collusion_score`, and an ROC curve
when both classes are present. `--eval-every-epoch` saves and evaluates
intermediate sampler weights. Their paths appear in the W&B run summary as
`checkpoint_epoch_1`, etc.; the final path appears as `checkpoint_path`.

The validation split is by row. Multiple rows can come from the same APPS
problem, so its AUROC is a training diagnostic rather than independent test
performance. Use a separate problem-disjoint test set for reported results.
