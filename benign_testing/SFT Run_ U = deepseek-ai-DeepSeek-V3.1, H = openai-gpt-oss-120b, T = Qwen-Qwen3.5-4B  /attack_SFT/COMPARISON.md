# Base, minified and paraphrased attack SFT — full APPS test

The four root-level comparison figures overlay all three conditions, using
blue for Base, orange for Minified and green for Paraphrased. Each is available
as SVG and PNG with full-range and zoomed panels:

- `untrusted_monitor_roc_curve`: collusion/authorship ROC (attacks vs honeypots).
- `trusted_monitor_roc_curve`: trusted suspiciousness ROC (attacks vs honest).
- `untrusted_suspiciousness_roc_curve`: untrusted suspiciousness ROC (attacks vs honest).
- `safety_usefulness_curves`: saved safety/usefulness sweep, with high-usefulness zoom.

These plots read the three exports' `analysis_data.json.gz` files. No model
calls or optimizer runs were performed. ROC lines connect the saved raw
empirical coordinates, without convex hulling or smoothing. Safety/usefulness
lines connect the ten saved optimizer points per condition.

| Monitor AUC | Base | Minified | Paraphrased |
| --- | ---: | ---: | ---: |
| Collusion/authorship | 0.978997 | 0.969940 | 0.757649 |
| Trusted suspiciousness | 0.872587 | 0.871230 | 0.862945 |
| Untrusted suspiciousness | 0.920229 | 0.928651 | 0.783943 |

Each export reports 114 attacks, 246 honeypots and 751 honest solutions.
Game parameters match: 5,000 tasks, audit budget 0.02 and slack honeypots 5.
Missing parsed scores can cause the trusted ROC/safety calculations to use
different valid subsets across runs. These are single-run comparisons;
training/holdout settings are not guaranteed identical across the three arms.
The paraphrased monitoring AUC above comes from the final monitoring records;
the separate training-dynamics evaluation reported 0.760287. This comparison
uses the final monitoring data consistently for all arms.

`comparison_roc_points.csv` and `comparison_safety_usefulness_points.csv` contain
the plotted coordinates. `comparison_sources.json` records source paths and
SHA-256 hashes. The plotting script includes runnable checks for matching game
parameters/counts, finite and monotone ROC data, three conditions per plot,
and SVG validity.

To regenerate from this folder with the project's Python environment:

```bash
/path/to/project/.venv/bin/python plot_comparison.py
```

The script requires the already-installed matplotlib, NumPy and scikit-learn.
