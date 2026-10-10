# Paraphrased attack SFT — full APPS test, batch 64, half-epoch dynamics

COMPLETE EXPORT. Three epochs completed, final checkpoint evaluated, all three
full-test monitoring logs completed, and eight final plots generated.
Final authorship ROC AUC: 0.760287 on 114 attacks and 246 honeypots.
Held-out accuracy: 0.75 on 20 examples; predicted YES rate: 0.20.

Models: U = deepseek-ai/DeepSeek-V3.1; H = openai/gpt-oss-120b;
T/paraphraser = Qwen/Qwen3.5-4B. Paraphrasing uses the existing LLM summarization
prompts with comments removed. Rank 32, learning rate 1e-4, three epochs,
batch 64, seed 0, 5% holdout: 406 prepared, 386 trained, 20 held out.

## Contents

- `SFT_self_rec_train/`: two original generation logs, two paraphrased training
  logs, exact prepared JSONL, and original/portable transformation configs.
- `self_rec_attack_test/generator/`: original full-test generation logs.
- `self_rec_attack_test/evaluation_inputs/`: paraphrased full-test inputs.
- `self_rec_attack_test/monitor/sft/`: final monitoring logs, 114 attacks,
  246 honeypots and 751 honest solutions. All completed without sample errors.
- `self_rec_attack_test/configs/monitor_sft.original.yaml`: config used by the run.
- `self_rec_attack_test/configs/monitor_sft.yaml`: portable plotting config.
- Top-level: eight final SVG plots, safety_usefulness.jsonl and analysis_data.json.gz.
- `SFT_Run_paraphrased_batch64/`: final checkpoint URI, completed dynamics
  CSV/SVG/provenance and full run log.
- `run_manifest.json`: original source hashes and model/training settings.
- `artifact_manifest.json`: hashes and sizes for every other packaged file.

Training contains 138 attacks and 268 honeypots. Test generations attempted
1,030 problems per class; 1,111 valid solutions are monitored. Authorship ROC
uses all 360 attacks/honeypots. Updates 0, 3, 7, 10, 14, 17, 21 correspond to
nominal epochs 0, .5, 1, 1.5, 2, 2.5, 3. Midpoints use the nearest batch
boundary: 192/386 examples (0.497 epochs).

Run portable configs from this package root using the experiment branch's code
and dependencies. Plotting with monitor_sft.yaml reads packaged monitoring logs
and regenerates top-level plots without model calls. Rerunning monitoring makes
model calls. Transformation configs also make model calls if rerun.

The earlier monitor_at_export.original.yaml and monitor_saved_checkpoint.yaml
are retained as historical snapshot configs; use monitor_sft.yaml for replay.
All .eval files are actual copies, with no external symlinks. Checkpoint weights
remain hosted by Tinker. Transforms preserve original program-test scores rather
than rerunning tests. Trusted monitoring reads original code with comments
removed; untrusted monitoring reads paraphrases. Successful monitoring does not
imply every parsed monitor score is present; the raw logs retain missing scores.
