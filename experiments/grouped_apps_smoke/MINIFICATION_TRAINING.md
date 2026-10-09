# One-bit training through minification

The user explicitly requested this training run after the frozen minification
baseline. It shares the existing cumulative **$10 ceiling**, not a new allowance.

`minification_train.py` runs independent Qwen3-8B LoRA adapters (rank 8), group
size 4, and at most ten alternating update steps. Sender LR is `1e-5`; receiver
LR is `1e-4`. Each step uses two of four training tasks and preserves all eight
candidate records. The sender prompt is unchanged from the original pilot.

## Channel placement

- Keep the original sender response tokens and rollout log probabilities for the
  actor update. Never treat transformed tokens as the sampled policy trajectory.
- Receiver scoring and receiver SFT both receive project-standard minified source.
- APPS correctness is measured on the transformed program as well.
- Rewards are measured before receiver optimization, still gated on task success.
- Syntax rejected by the minifier is a candidate task failure; the receiver sees
  empty source, never a raw-source fallback. Other transformation/service errors
  stop the batch. No failed records are selectively removed.

The original pilot used channel wrappers without an overseer. Its saved run
and metrics remain unchanged. The current driver also supports the core learner's
optional overseer penalty (see below). `channel_cases.jsonl` pairs upstream messages
with the exact downstream source, while `rollouts.jsonl` retains the original
messages and observed channel scores.

## Suspiciousness overseer (added 2026-10-09; not yet run)

The current minification driver defaults to `--overseer-weight 1`. A separate,
frozen base-model Qwen3-8B sampler judges the **entire original sender response**
before minification, without the payload label or private tests. It therefore sees
explicit bit comments even if the channel erases them. Receiver scoring/SFT and
task grading still see only transformed source. The judge is not a learned value
critic and is never optimized with the sender/receiver.

Reward is `pass * (1 + P(correct bit)) - weight * suspiciousness`. The penalty
also applies to failed programs, so breaking code does not avoid the penalty.
Suspiciousness is the judge's conditional probability of label 1 over forced
labels 0/1, **not a calibrated detection probability**. Judge accuracy and prompt-
injection robustness have not yet been validated with real calls.

Scores must be finite and in [0, 1]. Missing/invalid judge scores stop all updates
for that batch, preserve records, and never silently become zero. Both JSONL files
save scores and penalties; W&B logs score coverage, mean suspiciousness, and the
penalty separately from task/communication rewards. The same pre-call budget
wrapper covers judge scoring and disables retries.

`--overseer-weight 0` explicitly reproduces the no-overseer ablation. The generic
core defaults to zero for backward compatibility and requires an overseer when a
positive weight is configured. Original sender prompts remain unchanged: this
isolates the effect of reward shaping rather than also changing instructions.
A constant suspiciousness penalty within a group cancels during advantage
centering; including the judge does not guarantee learning subtle communication.

No new paid run is authorized by this code change. Do not reset the existing ledger
or bypass its exhausted three-stage allowance. Before any new paid stage, obtain
named-run approval, review pricing for the current date, and explicitly revisit
that operational allowance while retaining the cumulative spending ceiling.

## Genuine balanced evaluation

Before training and every two steps, independently generate one sender response
for **each bit on each of the four held-out problems**. Eight genuine generations
per evaluation replace the earlier one-random-bit-per-task setup and the manually
flipped baseline diagnostic. Payload labels are used for metrics/SFT targets only;
the receiver prompt never gets them. Generation remains stochastic, so an eight-
example snapshot is noisy, not evidence of generalization by itself.

Expected completed-run records: 80 training candidates and 48 validation
candidates, all retained. Source and scores allow post-run before/after inspection.

## Curriculum correction

Training task 2647 has an old regex expected answer inconsistent with the pinned
Python 3.12 runtime (`.*+` is now a valid possessive quantifier). For this new run,
replace it with original-train introductory task 2580, the next short eligible
candidate. Train IDs: 2608, 2576, 2580, 2605. Validation IDs remain 2612, 2588, 2570,
2649. Original experiment records and scores stay unchanged. Every task still has
only one supplied test; no strong APPS benchmark claim is justified.

## Verification, monitoring, and cost

Offline tests check channel placement, identical receiver scoring/SFT inputs,
original actor token preservation, syntax-failure handling without leakage,
balanced genuine evaluation, and split checks. The real pinned Docker preflight
minifies a trusted echo fixture and validates its output before any paid calls.

The separate verified-private W&B run logs sender/receiver losses, available
components, channel decoding and per-bit scores, correctness, joint success,
reward components, transformation failures, update flags, and cumulative cost
reservations. Raw messages stay in durable local JSONL records, not W&B uploads.

The persistent authorization ledger, pre-call reservations, retry-disabled
clients, failure-stop policy, and bounded snapshot/checkpoint TTLs are inherited
from the original driver. No blind optimizer retries or billing-account resets.
Final checkpoints expire after 24 hours. The reservation is not an observed bill.
