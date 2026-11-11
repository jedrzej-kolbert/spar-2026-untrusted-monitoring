# Pilot attempt 1: training data

The exact tasks the first HiPT-style RL run trained and was evaluated on
(`rewrite_signal_pilot`, attempt 1, started 2026-10-11 at commit `e583471`; see
[`docs/hipt_rl/ROADMAP.md`](../../../docs/hipt_rl/ROADMAP.md)).

Source: `experiments/open-source-model/apps/self_rec_train/generator/{attacks,honeypots}.eval`,
DeepSeek-V3.1 attacks on the APPS train split (odd problem IDs), generated with
`configs/apps_open/self_rec/generator/train_generator_config.yaml`. The raw logs
(68 MB) are not committed.

The run's `load_tasks` (`finetuning/grouped_rl/run.py`) did the filtering and splitting.
It was given the config from the run's manifest, and its output was checked against the
manifest's problem IDs:

- Keep attacks that pass the main task, have a working backdoor and a secret input
  (1,033 attacks → 128 tasks); first 10 tests only.
- Split by hashing the problem ID with seed 0, 20% to validation, capped at 20 problems.

| File | Rows | Contents |
| --- | --- | --- |
| `train.jsonl` | 108 | Tasks the sender was trained on |
| `validation.jsonl` | 20 | Held-out tasks for the step 0/10/20/30 evaluations |
| `validation_honeypots.jsonl` | 20 | gpt-oss-120b honeypot code for the validation problems (`problem_id`, `code`) |

Task rows are `AppsProblem` fields (`finetuning/grouped_rl/core.py`): `problem_id`,
`question`, `inputs`, `outputs` (test stdin/stdout), `starter_code`, `fn_name`, `split`,
`base_code` (the attack the sender rewrites), `secret_input` (triggers the backdoor),
and `reference_solution` (used to confirm the backdoor still triggers). Load one with
`AppsProblem(**{**row, "inputs": tuple(row["inputs"]), "outputs": tuple(row["outputs"])})`.
