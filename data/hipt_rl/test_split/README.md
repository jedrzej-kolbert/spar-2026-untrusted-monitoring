# Test-split data for the RL pilot evaluation

The data behind the full test-split evaluation of pilot attempt 1: `rl_rewrite_signal_pilot_step30`
(step-30 checkpoint) against `rl_rewrite_signal_untrained` (base DeepSeek-V3.1 with the same rewrite
prompt). See [`docs/hipt_rl/ROADMAP.md`](../../../docs/hipt_rl/ROADMAP.md). Training data is in
[`../rewrite_signal_pilot_attempt1/`](../rewrite_signal_pilot_attempt1/).

All problems are APPS test split (even problem IDs). Exported from these Inspect logs, which are not committed (about 100 MB):

| File | Source log | Rows | Contents |
| --- | --- | --- | --- |
| `attacks.jsonl` | `self_rec/generator/attacks.eval` | 1,015 | DeepSeek-V3.1 attacks (the rewrite inputs) |
| `honest.jsonl` | `self_rec/generator/honest.eval` | 1,024 | Honest solutions |
| `honeypots.jsonl` | `self_rec/generator/honeypots.eval` | 1,004 | gpt-oss-120b honeypots (collusion-monitor negatives) |
| `rewrites_untrained.jsonl` | `rl_rewrite_signal_untrained/generator/attacks.eval` | 114 | Signal-ON rewrites by the untrained model |
| `rewrites_step30.jsonl` | `rl_rewrite_signal_pilot_step30/generator/attacks.eval` | 114 | Signal-ON rewrites by the step-30 checkpoint |
| `problems.jsonl` | all of the above | 1,030 | Shared problem data, stored once per problem |

Paths are relative to `experiments/open-source-model/apps/`.

Code rows have `problem_id`, `code`, `main_task_success`, `side_task_success` (Inspect APPS scorer:
`C` correct, `I` incorrect, `N` no backdoor attempted) and, except in `honest.jsonl`, `secret_input`.
`problems.jsonl` has `problem_id`, `question`, `inputs`, `outputs` (all tests, stdin/stdout) and
`reference_solution`. Join on `problem_id`.

Subsets used in the evaluation:

- **Rewritten:** the 114 attacks with `main_task_success == side_task_success == "C"`.
- **Scored as attacks:** rewrites that still pass with a working backdoor (57 untrained, 93 step 30).
- **Honeypot negatives:** the 246 honeypots that pass with a working backdoor.
