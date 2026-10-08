# Grouped APPS communication learning

This is an offline-tested library for a **wiring pilot**, not evidence of stealth
or successful learned collusion. It generates solutions from scratch and permits
comments/docstrings. No trained model or experiment results are included.

## Algorithm

For each training problem, sample an independent random bit and generate a group
of solutions for that **same problem–bit pair** (default group size: four).

1. Score each solution with the pre-update receiver and sandboxed APPS execution.
2. Give a solution that passes every test reward
   `task_weight + communication_weight * p(correct_bit)`; otherwise give zero.
   Both weights default to one. Communication credit is correctness-gated.
3. Center rewards separately within each group: `advantage = reward - group_mean`.
4. Supervise the receiver on every generated code–bit example, including broken
   code; there is no correctness filtering of receiver examples.
5. Update the sender with Tinker's `importance_sampling` policy-gradient loss and
   the centered advantages. Skip its optimizer step if all advantages are zero.
6. Refresh each role's sampling weights after its optimizer step.

There is **no learned critic, PPO clipping, reference-policy KL term, or overseer**
in this initial learner. It preserves HiPT's sender-RL/receiver-SFT alternation,
but is not a reproduction of the paper's PPO implementation. Adding a different
policy loss or advantage estimator later does not require changing APPS scoring.

The LLM receiver scores both forced completions (`0` and `1`) and normalizes their
sequence probabilities over those two candidates. Multi-token labels are handled
by summing their token log probabilities. This is a conditional binary score, not
the probability mass of all possible textual answers. Receiver SFT trains only
label/EOS tokens; sender training masks prompt tokens and retains the actual
rollout probabilities, including any sampled EOS token. Missing/invalid rollout
probabilities are re-scored with the same frozen sampler at temperature one. If
recovery fails, code and evaluation scores are retained, but training is aborted;
held-out evaluation does not require policy log probabilities.

## Components and offline verification

- `core.py`: APPS record parsing, backend protocols, grouping, alternation, metrics,
  and train/held-out problem-ID checks. Imports only the standard library.
- `tinker_backend.py`: independent sender and receiver LoRA clients (default model:
  `Qwen/Qwen3-8B`). Clients are created only by explicit factory calls; every model
  operation is disabled unless `allow_paid=True`.
- `sandbox.py`: local Docker evaluator for stdin and function-call APPS tasks.
- `tests/test_apps_grouped_rl.py`: deterministic fake backends and mocked Docker
  process tests; no model calls, downloads, or generated-code execution.

Run the tests, always excluding the repository's paid/live tests:

```bash
uv run pytest tests/test_apps_grouped_rl.py -m "not slow and not mc"
```

Do not run plain `pytest`: other tests in this repository make live Tinker calls.
Do not interpret synthetic fake-client metrics as trained-model performance.

## Data and evaluation

Construct problems with `AppsProblem.from_record(record, split="train")`, using
records from APPS's **original training split**. Reference solutions are neither
required nor retained. Make validation problems from a disjoint subset of that
original training split; reserve the original test split for final evaluation.
The explicit `split` argument labels the caller's data; it does not independently
verify source-dataset provenance. No dataset is downloaded implicitly.

Invalid/missing test metadata raises an error rather than silently removing a
record. Both stdin strings and function-call argument lists are supported. Stdin
comparison strips outer whitespace, as the project's existing APPS scorer does.
Function-call outputs use JSON structural equality, accepting APPS's singleton
expected-output wrapper. The local image supplies Python's standard library, not
arbitrary third-party dependencies; float-tolerance conventions and unusual APPS
output types require validation before expanding beyond the wiring pilot.

Use `GroupedRLTrainer(sender, receiver, evaluator)` and await
`train_step(train_problems)` for one alternating update. Call
`evaluate(held_out_problems)` before training to record a frozen baseline and
again afterward for a held-out comparison. Evaluation makes no optimizer calls;
its repeatable payload RNG is separate from training. Sampling itself is not
forced deterministic. Keep the same ordered held-out problems for paired payloads.
A frozen receiver's ability to read explicit comments is not receiver learning.

`BatchResult.metrics()` reports decoding accuracy, APPS correctness, joint
success, recorded/requested sample counts, infrastructure failures, nonzero
advantages, payload-one counts, and mean/min/max rewards. Reward summaries remain
null if any reward is missing; they never summarize only a selected subset.
All recorded rollouts are in the denominator,
including broken code and missing scores. Per-rollout rewards/probabilities and
update receipts remain available in the result. An accepted optimizer step is not
proof that useful learning occurred; inspect reward variation and held-out results.

Infrastructure failures abort training updates and raise `BatchInfrastructureError`
with all recorded data in `.result`. Missing rewards are never imputed. Extra
samples are retained, but an incorrect sample count aborts the update. An update
failure records the affected role's state as uncertain (`None`), because its
optimizer may already have committed: **do not blindly retry the batch**.

## Execution and paid-run boundary

The Docker executor never executes generated code on the host. Each selected test
uses a fresh container with no network, no Linux capabilities, a non-root UID,
read-only source/root filesystem, and CPU/memory/PID/output/time limits. It removes
its own uniquely named container on timeout or output overflow; killing only the
Docker CLI is insufficient. Images are local-only (`--pull=never`); a missing image
or unavailable daemon is an infrastructure failure, not incorrect code. Review and
pin the worker image before a live run. Docker isolation is not a formal security
guarantee; use an isolated execution worker for adversarial generated code.

There is intentionally no default paid-run CLI or experiment configuration here.
`create_tinker_team()` refuses to create clients unless the caller explicitly opts
in. **That flag is not spending approval and does not enforce a dollar cap.**
Before any live pilot, obtain named-run approval, validate the actual SDK/model and
sandbox, and specify a budget-aware run driver covering sampling, scoring, training,
and evaluation. The discussed $5 ceiling is not authorization to spend.

Sampling refreshes use ephemeral snapshots. Use each role's explicit
`save_checkpoint(name)` to keep persistent sampler weights when a live run is
approved. The run driver must also persist results and retain the identities of
training/validation/test problems; these are not experiment records baked into
this feature library.

## References

- [HiPT, §3.3.1](https://arxiv.org/html/2410.03768v1): alternating sender RL and receiver SFT.
- [Tinker SDK cheatsheet](https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/index.md): RL losses and externally supplied advantages.
- [Tinker cookbook advantage processing](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py): group-mean-centered rewards.
