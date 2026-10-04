# Introspection training → self-recognition on APPS (DeepSeek-V3.1, Tinker)

**Question.** Does training DeepSeek-V3.1 to predict its own behaviour
(Binder et al., *Looking Inward*, arXiv:2410.13787) make it better than the base
model at recognising its own APPS solutions? This is the passive
self-recognition arm of the paper, where U is used as its own monitor.

**Metric.** ROC AUC of the collusion monitor's exact `log P(YES) − log P(NO)`
margin. The positive class is DeepSeek's own APPS attack solutions and the
negative class is gpt-oss-120b honeypots written from the same prompt. Base and
trained models are scored on the same examples, and the result is the paired
bootstrap 95% CI of ΔAUC.

## Design

| Step | What | Code |
|---|---|---|
| fetch | Binder et al. released prompts (26 train files and 25 held-out test files, no labels) | `git clone felixbinder/introspection_self_prediction` |
| generate | APPS **test** split, 300 problems: DeepSeek attacks, gpt-oss-120b honeypots | `generator_config.yaml` (needs Docker) |
| build | Sample DeepSeek's object-level answer at T=0, extract the property (first word, second character, …), and pair it with the hypothetical prompt | `finetuning/introspection_data.py build` |
| train | LoRA (rank 32, lr 1e-4, 1 epoch, batch 64), with self-rec AUC logged every 40 updates | `finetuning/tinker_lora_finetune.py` (existing) |
| accuracy | **Manipulation check:** self-prediction accuracy on held-out tasks, base vs trained, against each model's *own* object-level answers, plus the mode baseline | `finetuning/introspection_data.py accuracy` |
| compare | Paired ΔAUC with bootstrap CI | `scripts/compare_self_rec_auc.py` |

Before reading the APPS result, check two things:

1. **Did the training instil introspection?** The trained model's accuracy on
   held-out tasks should exceed both the base model's accuracy and the mode
   baseline. If it doesn't, the training failed, and the APPS number says
   nothing about introspection either way.
2. **Is any gain specific to self-knowledge?** Run the cross-prediction control
   (`CROSS=1`). It trains DeepSeek on gpt-oss-120b's behaviour with the same
   prompts and format. A gain shared by both arms comes from finetuning on this
   format, not from introspection. That control is the paper's key one.

None of the training data is APPS or code, so any change in APPS AUC comes from
transfer.

## Run

```bash
# .env: TINKER_API_KEY=...   Docker running for `generate`.
bash experiments/introspection_deepseek/run.sh            # self arm, all stages
CROSS=1 bash experiments/introspection_deepseek/run.sh build
CROSS=1 bash experiments/introspection_deepseek/run.sh train
CROSS=1 bash experiments/introspection_deepseek/run.sh accuracy
bash experiments/introspection_deepseek/run.sh compare    # picks up both arms
```

Outputs go to `experiments/introspection_deepseek/runs/` (gitignored):
`compare/summary.json` (AUCs and ΔAUC CIs), `compare/margins.csv`,
`{self,cross}/dynamics/dynamics.svg` (AUC over training), and
`accuracy_base.json` / `{self,cross}/accuracy.json`.

Knobs are environment variables: `PER_FILE` (default 400 → ~10k examples),
`EPOCHS`, `LR`, `RANK`, `BATCH`, `EVERY`, `MODEL`, and `CROSS_MODEL`.

## Caveats

- **Cost.** DeepSeek-V3.1 is a 671B MoE, so LoRA training and sampling on
  Tinker cost far more than on the Qwen3-30B U. Try a pilot first with
  `PER_FILE=50` and `limit: 50` in `generator_config.yaml`.
- **The chat template is untested here.** The container had no Hugging Face
  access, so the data path was smoke-tested with a stand-in tokenizer. On a
  first real run, check that the `build` labels look sensible
  (`head runs/self/train.jsonl`).
- **Single seed and a single eval set.** The CI covers sampling of APPS
  problems, not training-seed variance. If ΔAUC is borderline, retrain with
  another `--seed`.
