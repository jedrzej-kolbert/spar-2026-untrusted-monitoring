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

## Results (2026-10-04)

Evaluated on the existing full APPS test-split generations
(`experiments/exports/attack_SFT_full_test_batch64/self_rec_attack_test/generator/`,
114 valid DeepSeek-V3.1 attacks vs 246 gpt-oss-120b honeypots), not a fresh
`generate`. Defaults otherwise: 10,400 training examples per arm, rank 32,
lr 1e-4, 1 epoch, batch 64.

| Model | Held-out self-prediction acc (n=2,500) | Mode baseline | APPS self-rec AUC | ΔAUC vs base [95% CI] |
|---|---|---|---|---|
| Base DeepSeek-V3.1 | 0.396 | 0.301 | 0.543 | — |
| Trained on its own answers | 0.606 | 0.321 | 0.564 | +0.021 [−0.004, +0.047] |
| Control: trained on gpt-oss-120b's answers | 0.524 | 0.322 | 0.617 | +0.075 [+0.044, +0.108] |

Base and self-trained AUCs are from margins averaged over three scoring passes;
the control was scored once. Control minus self-trained: +0.053 [+0.028, +0.080].

- **The training instilled self-prediction** (0.396 → 0.606 on held-out tasks),
  so the APPS number is a fair test of the hypothesis.
- **It did not measurably improve self-recognition.** ΔAUC is about +0.02 with an
  interval that includes zero, and every model stays near chance.
- **The control gained more than the introspection arm.** So the small gains are
  not specific to self-knowledge. Note the control's label model is also the
  honeypot author, so its gain may be "learned what gpt-oss-120b is like" rather
  than a pure format effect; a control labelled by a third model would separate
  the two.

![results](results.svg)

Regenerate with `uv run python experiments/introspection_deepseek/plot_results.py`.

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

To evaluate on existing generator logs instead of generating (no Docker), set
`ATTACKS=<attacks.eval> HONEYPOTS=<honeypots.eval>` on every stage.

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
- **Check the `build` labels of any new label model** (`head runs/<arm>/train.jsonl`).
  DeepSeek-V3.1 labels are clean. gpt-oss-120b reasons before answering, and its
  labels were reasoning fragments until `sample_text` kept only its final
  channel; another reasoning model would need the same treatment.
- **Tinker's scoring is not deterministic.** Re-scoring the same model on the
  same examples moved its AUC by up to 0.011 (per-example margins correlate
  0.97). `compare`'s bootstrap CI does not include this, so do not read a CI
  that barely excludes zero as a result; re-run `compare` and average.
- **Single seed and a single eval set.** The CI covers sampling of APPS
  problems, not training-seed variance. If ΔAUC is borderline, retrain with
  another `--seed`.
