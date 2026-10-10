# Attack SFT — full APPS test evaluation, batch 64

Final-checkpoint evaluation on the completed APPS test split, packaged 2026-09-27.
Representation: original code, with monitor comment removal enabled. This is
not the summarized SFT arm and not the historical honest-SFT control.

## Contents

- Top-level SVGs, `safety_usefulness.jsonl`, and `analysis_data.json.gz`: completed full-test evaluation.
- `self_rec_attack_test/generator/`: 1,030 attempted problems per class, completion manifests and supplemental generation logs.
- `self_rec_attack_test/monitor/sft/`: scored valid solutions: 114 attacks, 246 honeypots, 751 honest (1,111 total).
- `self_rec_attack_test/configs/monitor_sft.yaml`: portable plotting config, resolved relative to this package root.
- `self_rec_attack_test/configs/monitor_sft.original.yaml`: unchanged config actually used for evaluation.
- `SFT_self_rec_train/`: original training generation logs, their config, and the base-code prepared JSONL (406 examples: 138 attacks and 268 honeypots).
- `SFT_Run_batch64/earlier_190_example_dynamics/`: original checkpoint CSV, SVG and final URI. These dynamics used the earlier 59-attack/131-honeypot test set, NOT the full test set.
- `checkpoint_manifest.csv`: checkpoint used by this full-test evaluation.
- `artifact_manifest.json`: source paths, file sizes, SHA-256 hashes and source checkout revision/dirty status.

The authorship evaluation has 360 examples (114 attacks + 246 honeypots).
Honest solutions provide the benign class for backdoor/safety evaluations.
Training uses odd problem IDs; test uses even IDs. Failed generations remain in
generator logs but are filtered before monitoring. Monitor logs were checked
for successful completion, sample errors and expected sample counts.

Models: DeepSeek-V3.1 SFT for untrusted collusion, Qwen3.5-4B for trusted
suspiciousness, and base DeepSeek-V3.1 for untrusted suspiciousness. Both classes
of attack-SFT training data are generated in attack mode; labels denote authorship.
No passive/shared monitor runs or full-test training dynamics are included,
because those were not part of this completed evaluation.

## Replot without model calls

Run from this package root, using the source checkout environment:

```bash
REPO=/home/jkk/spar-2026-untrusted-monitoring.experiment-issue11-attack-batch-comparison
PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" \
  "$REPO/src/lasr_labs_2025_control_project/scripts/plot_data.py" \
  --config self_rec_attack_test/configs/monitor_sft.yaml
```

This regenerates the top-level plots from the packaged monitor outputs; it does
not train or call a model. It overwrites the packaged plot/analysis files.
Original generation configs/manifests preserve historical source paths and are
provenance records, not portable generation commands. The portable monitor
config is for replaying plots; rerunning monitoring would make model calls and
overwrite the packaged monitor logs. Checkpoint weights remain hosted by Tinker;
the package contains their URI, not the weights themselves.
