# Monitored grouped APPS wiring run

This driver is separate from the grouped learner's feature commit. It is a small
wiring experiment, not a claim of learning, stealth, or general APPS performance.

## Approved scope

The user explicitly approved paid Tinker calls on 2026-10-08 with a **$10 total
ceiling**, and requested loss curves, accessible loss components, decoding scores,
and rewards. The default run uses two independent Qwen3-8B LoRA adapters, rank 8,
four training tasks, four disjoint validation tasks, groups of four, and at most
10 alternating update steps. Original APPS test is untouched.

The cached training data has 134 eligible short introductory stdin tasks. The
pilot takes the eight shortest by supplied-test count and question length. Each
selected task has **one supplied test**. This deliberately narrow choice limits
execution and token costs; correctness measures only those supplied tests.

## Cost controls

- Prices were checked against the official Tinker model/pricing page on
  2026-10-08: per million tokens, prefill $0.195, sample $0.60, train $0.44.
- Before each sampling, scoring, or forward/backward call, reserve twice its
  uncached maximum token cost. Scoring is charged conservatively at the larger
  training rate plus one possible output token.
- Keep $2 reserved for storage and uncertainty. The plotted reservation is an
  **upper-bound reservation, not the observed invoice**.
- A locked, atomically updated authorization ledger retains reservations across
  process failures and restarts. Never refund failed or uncertain calls. Never
  create a fresh ledger to evade the original ceiling.
- Disable sampling retries and HTTP transport retries. Never blindly retry an
  optimizer update. Stop on infrastructure failures or insufficient budget.
- At most three attempts share this authorization. At most 11 one-hour sampler
  snapshots per role and one 24-hour final checkpoint per role per attempt keep
  the storage allowance bounded. Checkpoints expire; indefinite storage is not
  authorized by this pilot.
- Another day's run requires fresh pricing and human authorization.

## Monitoring and records

A verified-private W&B project `apps-grouped-wiring` receives scalar metrics and
run configuration only. Code logging, Git collection, and console capture are
disabled. Generated code and per-rollout probabilities/rewards stay in local
JSONL files; they are not uploaded as W&B artifacts.

Logged values include raw sender/receiver loss metrics, loss per active token,
receiver label/EOS NLL when the server exposes aligned token probabilities,
optimizer diagnostics when available, APPS correctness, decoding accuracy, joint
success, true-bit confidence, reward mean/min/max/std, task/communication reward
components, missing/infrastructure counts, and update flags. Absent components
are omitted rather than invented. Incomplete reward aggregates stay null.

Sender loss is signed and group-relative; near-zero or nonmonotonic objective
values do not by themselves imply no learning or successful learning. Inspect
reward variation, update receipts, and paired held-out scores. There is no critic,
PPO-clipping, reference-KL, or overseer component in the learner.

The baseline uses the frozen initial team. Validation repeats every two steps,
with the same ordered problems and payloads but independently sampled code.
All generated candidates, including broken code and missing scores, are retained.
Raw sampled token sequences are saved before scoring, including interrupted work.

Results live under `~/.local/share/spar-2026-runs/` and survive worktree cleanup:
`manifest.json`, `events.jsonl`, `rollouts.jsonl`, `status.json`, `summary.json`,
and W&B's local run files. Failure logs and uncertain update states are retained.

## Local checks

Without `--allow-paid`, the driver selects data, writes a manifest, and runs a
trusted echo fixture in the digest-pinned local Python 3.12 Docker image. It makes
no Tinker or W&B calls. The generated-model programs always run in the restricted
Docker evaluator, never on the host.

```bash
PYTHONPATH="$PWD/src:$PWD" python experiments/grouped_apps_smoke/run.py \
  --temp-root "$CLAUDE_JOB_DIR/tmp"

PYTHONPATH="$PWD/src:$PWD" TINKER_API_KEY='' WANDB_MODE=disabled \
  python -m pytest tests/test_grouped_apps_smoke_driver.py \
  tests/test_apps_grouped_rl.py -m 'not slow and not mc'
```

Do not run plain pytest; the repository contains paid/live tests. Paid execution
requires the explicit `--allow-paid` opt-in in addition to actual user approval.
