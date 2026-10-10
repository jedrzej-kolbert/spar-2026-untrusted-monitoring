# Attack SFT — minified code, full APPS test evaluation, batch 64

Upload package for issue #25, following the baseline/honest-SFT folder layout.
Top-level SVGs are the completed minified-arm evaluation plots.

## Contents

- Top level: eight SVG plots, safety_usefulness.jsonl, analysis_data.json.gz, validation.json, and validation_report.md.
- self_rec_attack_test/generator/: original baseline test generation logs, 1,030 attempted problems per class.
- self_rec_attack_test/evaluation_inputs/: minified valid test solutions (114 attacks, 246 honeypots, 751 honest).
- self_rec_attack_test/monitor/sft/: final monitor results with the same sample counts.
- self_rec_attack_test/configs/: portable plotting config plus unchanged configs used for monitoring and transformation, and historical generator config.
- SFT_self_rec_train/generator/: original attack-mode training logs from U and H.
- SFT_self_rec_train/minification/: minified training logs, transform config, and prepared messages (406 examples: 138 attacks/YES and 268 honeypots/NO).
- SFT_self_rec_train/no_minification/prepared/: the original baseline's 406-example training messages for reference.
- SFT_Run_minified_batch64/: checkpoint URI/mapping, settings, and captured execution logs.
- checkpoint_manifest.csv and artifact_manifest.json: checkpoint identity and source paths/file hashes.

Models: U = deepseek-ai/DeepSeek-V3.1, H = openai/gpt-oss-120b, T = Qwen/Qwen3.5-4B.
The collusion monitor is the minified SFT checkpoint. The untrusted suspiciousness monitor uses base DeepSeek-V3.1. Trusted monitoring uses original code with comments removed; untrusted monitoring uses minified code.

Authorship ROC AUC: 0.969940 (baseline export: 0.978997), on the same 360 attacks/honeypots.
There are no sample errors or missing untrusted scores. Trusted scores are missing for 11 honest samples and two honeypots (12 token-limit truncations, one existing first-tag parser failure). See validation.json for IDs and validation_report.md for comparison caveats.

Only the minified SFT arm is packaged. Paraphrased, passive/shared evaluations and intermediate training dynamics are not included because they were not produced by this run.

## Replot locally without model calls

From this package root, with the original project environment available:

```bash
REPO=/home/jkk/spar-2026-untrusted-monitoring.experiment-issue25-signal-jamming
PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" \
  "$REPO/src/lasr_labs_2025_control_project/scripts/plot_data.py" \
  --config self_rec_attack_test/configs/monitor_sft.yaml
```

The portable config resolves inputs/results relative to the package root and will overwrite its existing plots when replotting. The *.original.yaml files preserve historical source paths and are provenance records. Replotting needs project code and dependencies; those are not bundled. Checkpoint weights remain remote, while the .eval files contain the completed evaluation records.
