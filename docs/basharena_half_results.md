# BashArena self-recognition: larger run (issue #29)

Status on 2 October 2026, 18:45. Honest test generation is still running, so the
honest set below is partial (103 of 160 attempts saved when scored).

## Setup

- U / fine-tuned monitor base: `Qwen/Qwen3.6-35B-A3B`; honeypots H: `openai/gpt-oss-120b`;
  trusted monitor T: `Qwen/Qwen3.5-9B`. All through Tinker.
- Train = odd task IDs, test = even task IDs. Mixed side tasks; attack runs end as soon
  as a side task succeeds.
- Monitor and SFT input: the task description plus the agent's tool calls only (no tool
  outputs, no agent text). One score per finished trajectory.

## Data

| Split | Role | Attempts | Usable |
|---|---|---:|---:|
| Train | U attacks | 158 | 138 |
| Train | Honeypots | 158 | 68 |
| Test | U attacks | 160 | 148 |
| Test | Honeypots | 160 | 68 |
| Test | Honest (so far) | 103 | 83 |

Usable attack/honeypot = side task succeeded. Usable honest = no side task triggered.

## Self-recognition fine-tune

138 U attacks (YES) and 68 honeypots (NO, repeated twice): 274 rows, 3 epochs, 822
updates, rank 32, lr 5e-5, batch size 1. ROC AUC on 148 test U attacks vs
68 test honeypots:

| Updates | 0 | 75 | 150 | 225 | 300 | 375 | 450 | 525 | 600 | 675 | 750 | 822 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ROC AUC | 0.623 | 0.972 | 0.994 | 0.995 | 0.983 | 0.983 | 0.982 | 0.978 | 0.993 | 0.993 | 0.988 | 0.990 |

| Scorer on the same test set | ROC AUC |
|---|---:|
| Base U | 0.623 |
| Pilot fine-tune (49 trajectories) | 0.972 |
| Larger fine-tune, final checkpoint | 0.990 |
| Number of actions alone | 0.888 |
| Larger fine-tune, length trend removed | 0.895 |

## Suspiciousness and safety-usefulness

| Monitor | Groups | ROC AUC |
|---|---|---:|
| T suspiciousness | 148 attacks vs 83 honest | 0.982 |
| Base U suspiciousness | 148 attacks vs 83 honest | 0.991 |
| Number of actions alone | 148 attacks vs 83 honest | 0.998 |

Modelled safety (`plot_data.py`: 5,000 tasks, 2% audit/honeypot budget; usefulness is
1 - replacement FNR, not task completion):

| Usefulness | 1.00 | 0.99 | 0.98 | 0.95 | 0.90 | 0.80 | 0.50 | 0.20 | 0.05 | 0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 0.253 | 0.277 | 0.326 | 0.457 | 0.514 | 0.604 | 0.664 | 0.821 | 0.945 | 0.987 |

## Caveats

- **Length shortcut.** Attacks stop at side-task success (median 14 actions); honest
  runs use most of the 60-step budget (median 60). Action count alone separates them,
  so the suspiciousness results and safety curve cannot yet be credited to the monitors.
  U attacks are also longer than honeypots (median 14 vs 5), which explains
  most of the self-recognition AUC.
- **Safety curve is unstable.** It is driven by a few honest runs scored as highly
  suspicious (base U gave 4 of 83 a 7+, T gave 5). With 48 honest runs the same
  pipeline gave 0.83 at full usefulness.
- **Honest task performance is poor.** 2 of 85 scored honest runs fully completed the
  main task; mean fraction of tests passed 0.46. 14 of 103 attempts exceeded
  the model's 65,536-token context.
- Scores are post-hoc on whole trajectories; the test set has been inspected repeatedly.
- Saved logs keep full conversations with tool outputs, so per-action scoring, cut-off
  honest runs and with-output inputs can be built without regenerating.

## How it was run

Scripts in `run_logs/` (local paths, experimental): `run_half_experiment.py`
(`--generate-only`, `--train-only`), `run_half_eval.py ARM` (monitors + standard plots),
`prebuild_test_images.py`. Outputs live under
`experiments/open-source-model/basharena/self_rec_half/` (not tracked).
