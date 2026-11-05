# Five-epoch SFT on summarised attacks

Status: completed and published (2026-11-05). Training and scoring finished at 00:58 local time.

## Result

- **Final five-epoch AUROC: 0.808864**, independently recomputed from all 555 saved test scores.
- **Conservative uncached API-compute estimate: $12.432382**, including training and 3,360 checkpoint/held-out scoring requests, below the $20 cap. This is not an invoice reconciliation.
- [Mirrored result in Collusion RL v2](https://forge.coreweave.com/wandb/spar-untrusted-monitoring-collusion/collusion-rl-v2/runs/sft5p1105).
- [Two explicitly labelled 5-epoch SFT plots](https://forge.coreweave.com/wandb/spar-untrusted-monitoring-collusion/collusion-rl-v2?nw=06avz3a96jf): training loss and diagnostic checkpoint AUROC.
- Result artifact: `spar-untrusted-monitoring-collusion/collusion-rl-v2/sft-v41-summary-5epoch-results:v0`, with per-example scores, source history, preflight, ledger, configuration and budget-guard source.
- Machine-readable result and all six sampler-weight references: [`results.json`](results.json).

| Optimiser update | Diagnostic test AUROC |
|---:|---:|
| 9 | 0.656886 |
| 18 | 0.729287 |
| 27 | 0.784974 |
| 36 | 0.813791 |
| 45 | 0.817363 |
| **50 (five-epoch endpoint)** | **0.808864** |

Step 45 is higher than the final checkpoint, but the five-epoch endpoint was fixed in advance and was retained. These are diagnostic measurements on the reused test set, not a validation-selected best-checkpoint result. No new safety/usefulness claim is made.

## Setup

- Source: [Untrusted Collusion run `l0dbt9nf`](https://forge.coreweave.com/wandb/spar-untrusted-monitoring-collusion/untrusted-runs/runs/l0dbt9nf).
- Destination: `spar-untrusted-monitoring-collusion/collusion-rl-v2`, following the latest comparison project in `hipt-collusion-icrl-ledger`.
- Code: `main` at `2e97d827fe6074e13d3287c19a59c4faecdf0c75`, isolated on `exp/sft-v41-summary-5epoch`. The user's main checkout and dependency edits were left untouched.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144`, LoRA rank 32, learning rate 0.0001, batch size 64, five epochs (50 updates).
- Data: the existing Qwen3.5-4B summaries used by the three-epoch `moap3ypn` run. No attack generation or paraphrasing is repeated. There are 580 training and 30 held-out examples; diagnostic checkpoint AUROC uses the same 555 cached test attacks/honeypots.
- Split: the native pipeline derives its seed from the new node key (`353097607`), so its 30-example held-out slice is not identical to the old three-epoch run. The independent 555-example diagnostic test set is unchanged.
- Checkpoints: updates 9, 18, 27, 36, 45, 50. The endpoint is fixed at epoch five, not selected by diagnostic test AUROC.

## Budget

The user authorised a hard **$20 cap on new API compute**. The launcher reserves the full deterministic training cost before starting and refuses scoring requests that could take the total above $19, leaving $1 unallocated. Uncertain failed scoring calls retain their worst-case reservation.

Free tokenisation preflight counted 2,188,435 training tokens ($10.8984) and bounded all checkpoint/held-out scoring at $3.0680, for an uncached upper bound of **$13.9664**. The live ledger is `experiments/sft-v41-summary-5epoch/budget-ledger.json`; all per-example diagnostic scores are saved beside it in `scores.jsonl`.

This is token-based accounting, not invoice reconciliation. Checkpoint storage is an ongoing charge outside the API-compute estimate. No additional safety-game scoring is authorised by this run plan.

## Transfer and plots

`publish_results.py` checked completeness, budget and consistency before publication. It independently recomputed all six AUROCs from the saved per-example scores, copied the source metrics/configuration to a clearly labelled baseline in Collusion RL v2, uploaded the result files, and created a named view with two panels:

1. **5-epoch SFT — training loss (summarised attacks)**, versus optimiser update.
2. **5-epoch SFT — collusion AUROC (summarised test attacks)**, versus optimiser update, with the full 0–1 AUROC range.

The mirror is attribution-preserving, not a second training run. Existing three-epoch SFT and RL results are not overwritten. No new safety/usefulness result is inferred from AUROC alone.

Checks already passed: native main-branch pipeline dry run (exactly one paid fine-tuning node, all four inputs cached), free exact-token budget preflight, script syntax, W&B panel schema construction without publication, and three isolated spending-guard cases (reject before API call, reconcile a successful request, retain reservation on uncertain failure).

The first commit's repository-wide Pyright hook failed: its newly created worktree environment lacks the project dependencies (`control_arena`, `inspect_ai`, etc.), and the scan also reports unrelated pre-existing errors in BashArena and documentation scripts. No dependency files or unrelated code were changed to silence it. The experiment itself uses the existing compatible `v41-pilot/.venv` with this worktree's main-derived source. The other pre-commit checks passed; the formatter's changes were verified to preserve the publication script's AST. Pyright is skipped for the experiment-record commit, not reported as passing.
