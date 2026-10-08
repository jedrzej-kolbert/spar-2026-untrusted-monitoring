# Frozen minification baseline

`minification_baseline.py` evaluates the **untrained Qwen3-8B receiver** on all
eight saved pre-training validation messages from the wiring run. No new sender
sampling, optimizer update, adapter training, or checkpoint is performed.

The minifier matches the repository's `MinifierParaphraser`: remove comments and
literal/docstring statements, rename local/global symbols, and apply its existing
other options. The installed library defaults alone would preserve docstrings.
Every transformed program is graded in the pinned restricted Docker evaluator.

## Two clearly separated measurements

1. **Primary:** the original eight saved messages, with unchanged payload labels.
   Compare their existing unminified receiver scores with new minified scores.
2. **Balanced marker diagnostic:** add one opposite-bit control for each message,
   flipping explicit bit comments and unused `secret_bit` assignments only.
   This makes 16 balanced records. Controls are manually constructed diagnostic
   variants, **not** new sender generations. If a variable is used by the program,
   unknown marker syntax occurs, or infrastructure fails, stop instead of silently
   constructing a questionable control.

The original eight messages contain six zero bits and two one bits, so a constant
zero decoder already achieves 75%. The balanced diagnostic prevents that bias
from being mistaken for communication. Four of the eight marker pairs become
byte-identical after minification: those bit distinctions have been erased, and
no receiver can recover both equally likely labels from an identical input.
Identical receiver prompts reuse the same recorded probability vector, with an
explicit reuse flag; this avoids redundant paid scoring, not any metric rows.

The trusted offline preflight graded all 16 transformed programs successfully.
This checks supplied tests only, not universal semantic equivalence.

## Observability and cost

A separate, verified-private W&B run logs per-case recovery/confidence and final
primary/diagnostic aggregates, with **case index**, not training step, on its axis.
No loss curve is fabricated for this frozen evaluation. Source remains in local
records rather than W&B uploads. All results and failures are retained.

Paid scoring uses the **same cumulative $10 authorization ledger** as the wiring
run. Before each request, reserve conservative token costs; disable HTTP and
sampling retries. No fresh $10 allowance is created. Running without
`--allow-paid` does local transformation/grading only and makes no model calls.

The evaluation's `config.json`, `events.jsonl`, `results.jsonl`, `status.json`, and
`summary.json` live outside the disposable worktree under
`~/.local/share/spar-2026-runs/`.
