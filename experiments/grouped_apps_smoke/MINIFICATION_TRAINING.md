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

The existing grouped learner is unchanged. Driver wrappers implement the channel;
`channel_cases.jsonl` pairs upstream messages with the exact downstream source,
while `rollouts.jsonl` retains the original messages and observed channel scores.
This is channel-robustness training, not a stealth or overseer experiment.

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
