# SFT run: DeepSeek-V3.1 / GPT-OSS-120B / Qwen3.5-4B

This folder packages the inputs and results for the self-recognition SFT dynamics plot.

- **U / fine-tuned monitor:** `deepseek-ai/DeepSeek-V3.1`
- **H / honeypot generator:** `openai/gpt-oss-120b`
- **T / summariser:** `Qwen/Qwen3.5-4B` via Tinker
- Training used 1,470 examples per condition (772 honest DeepSeek and 698 GPT-OSS honeypots).
- The test ROC uses the same matched set at every point: 59 attacks (positive) and 131 honeypots (negative), 190 total.
- Update 0 is the unfine-tuned DeepSeek base model, scored separately on original and transformed solution text.
- For the no-summarisation test, the evaluator reads `base_solution` from each paired test `.eval`; for the summarised test it reads `transformed_solution`. Thus the two test conditions contain the same underlying examples.
- The prepared training JSONL files are the exact SFT inputs; neighboring `.eval` files preserve their source eval logs.
- The HTML chart is interactive and loads Plotly from its CDN. The SVG is a static rendering of the same CSV results.
- `attack_honeypot_checkpoint_scores.csv` contains the per-example YES-minus-NO log-probability margins; `attack_honeypot_sft_dynamics.csv` contains the ROC AUC and training-loss points.
- No additional Tinker calls were needed to assemble this package. All checkpoint scores were already evaluated.

Training losses at no-summarisation updates 22 and 44 are W&B `train/loss` values from the included run log. They are the per-update training losses recorded by W&B, not a separate full-dataset evaluation pass. Summarisation losses are the per-update losses saved by the trainer at each checkpoint.
