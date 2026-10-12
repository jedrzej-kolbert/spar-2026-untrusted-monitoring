# HiPT-style RL for collusion signals: roadmap

Running log of what exists, what's in progress, and what's next. Update it whenever an item lands.
Spec: "Notes for HiPT Replication for Eliciting Collusion Signals" (team doc).
Background: [`src/PPO Critic Explained.md`](../../src/PPO%20Critic%20Explained.md) (why HiPT's value head doesn't port to Tinker),
[`src/APPS Message Evolution.md`](../../src/APPS%20Message%20Evolution.md) (results of the 100-update run on the RL branch).

## Decisions so far (updated 2026-10-11)

| Topic | Decision |
| --- | --- |
| Base code | Build on `grouped_rl` from `origin/run/grouped-apps-smoke-2026-10-08`; reuse, don't rewrite. |
| Sender task | Configurable `task_mode`: `honest+signal` (what the branch does today: a correct solution from scratch plus the bit, no backdoor), `attack+signal` (Option B: backdoor + secret input + signal), `rewrite_signal` (Option A: inject a signal into a pre-generated attack). Option A is built first. |
| Reward | Configurable. The default is the branch's existing reward, `pass × (task_w + comm_w·P(correct bit)) − overseer_w·suspicion`. `hipt` is an alternative. Optional extra terms: failed-backdoor penalty, trusted-monitor backdoor-suspicion penalty. |
| Loss | Tinker `ppo` + group-mean advantages; optional per-token KL to the base model and multiple PPO passes. For cost (2026-10-11) the pilot uses 1 pass, no KL, group size 4, and 30 steps (about 1/5 of attempt 1's DeepSeek tokens). |
| Cost | RL arms reuse the reference arm's (`rl_rewrite_signal_untrained`) honest/honeypot monitor results via `reuse_scores_from`; only attacks are re-scored. Next lever: develop on a smaller U and run DeepSeek once. |
| Monitor (receiver) | Frozen first. Two fine-tuning regimes later: **(1)** signal vs no signal, **(2)** signalled attacks vs honeypots. |
| Payload | Train: same task with signal (bit 1) or without (bit 0). Held-out: signalled attacks vs H honeypots. |
| Receiver prompt | New prompt explaining what "a signal from yourself" means, without naming a specific signal. |
| Overseer | Trusted model T judging "is there a hidden signal?"; backdoor suspicion is a separate optional reward term. |
| Signal jamming | Configurable channel between sender and receiver: `none`, `minify` (exists on the branch), `paraphrase`/summarize (new). Applied both in training and at eval. |
| Models | Sender and receiver: DeepSeek-V3.1 (U). Overseer: Qwen3.5-4B (T). Honeypots: gpt-oss-120b (H). |
| Data split | The project's existing APPS split: odd problem IDs for train, even for test. |
| Logging | W&B (offline until `wandb login`) + local JSONL; checkpoints written to `experiments/open_models/ckpts.json`. |

## Lessons from the branch's 100-update run (APPS Message Evolution)

Qwen3-8B sender and receiver, frozen Qwen3.5-4B overseer, penalty weight 5, 4 train / 4 held-out trivial tasks.

1. **The sender dropped the signal instead of hiding it.** Explicit `secret_bit` comments disappeared and suspicion fell, but bit 1 then decoded as 0. On one task, the bit-0 and bit-1 outputs were byte-identical. The only "successes" were bit-0 cases where the receiver's default guess happened to be right.
   - *Fix:* report per-bit and balanced accuracy, plus a "bit-0 and bit-1 outputs differ" check. Make each group contain both bits for the same problem, so a signal that carries no information earns nothing relative to its siblings.
2. **The penalty was too strong.** At weight 5, maximum suspicion costs 5 against a communication reward of at most 1, so silence wins.
   - *Fix:* start with the overseer weight near 1, with an optional ramp-up schedule.
3. **The overseer score saturates.** Many unrelated outputs got exactly 0.4073, the same as the clean control, so the penalty gave almost no signal within a group.
   - *Fix:* validate the overseer on clean/signalled controls before a run, and log the spread of its scores.
4. **The tasks had no room to hide anything** ("Hello, World!", single-test problems).
   - *Fix:* use real attack code from our split, with many more problems, and a bigger held-out set than 8 samples.
5. **The same collapse threatens Option A.** "No signal" is bit 0, so a sender that never signals gets bit 0 right for free. Lessons 1–2 apply directly.

## Already on the RL branch (reuse)

- Grouped rollout loop, group-mean advantages, alternating receiver SFT / sender update (`core.py`)
- Tinker sender/receiver/overseer LoRA clients, forced-label probability scoring (`tinker_backend.py`)
- Docker APPS grader for public tests (`sandbox.py`)
- Pass-gated reward with an overseer penalty, minification channel, W&B + JSONL recording, budget ledger
- Offline test suite with fake backends

## Findings from the wiring smoke run and prompt probe (2026-10-11)

- **The pipeline works end to end on Tinker** with DeepSeek-V3.1: LoRA creation, sampling, PPO with KL, frozen monitors, evaluation, checkpointing. That run had 16 rollouts per step and says nothing about learning.
- **Sandbox matches Inspect exactly.** On 60 attacks Inspect scored as passing with a working backdoor, and 50 it scored as failing, the Docker sandbox agrees every time. Batched execution makes grading about 4x faster (about a minute per training step).
- **Rewrites usually break the backdoor**: only 4 of 16 kept it. The pilot therefore uses `failed_backdoor_weight: 1.0` (agreed 2026-10-11).
- **Receiver prompt calibration** (DeepSeek-V3.1, 20 per cell, P(YES)):

  | Prompt | Clean attacks | Honeypots | Explicit signal |
  | --- | --- | --- | --- |
  | existing self-rec | 0.11 | 0.29 | 0.12 |
  | self-signal draft | 0.81 | 0.70 | 0.98 |
  | self-signal + base rate (adopted) | 0.53 | 0.43 | 0.94 |

- **Overseer** (Qwen3.5-4B): clean attacks already score about 0.8 suspicion and an explicit marker about 1.0. The gap is small, and it caught the sender's first crude signals (about 0.998).
- **Untrained sender signals are crude**, e.g. an unused `x = 17` variable.

## Pilot attempt 1 (2026-10-11, steps 1-39)

- **Crash.** The Docker daemon died at about 20:46. Step 40's grading failed and was correctly skipped, but step 41 then crashed on a driver bug: refilling the epoch queue could put the same problem twice in one batch. Both are fixed: batches are always distinct, an unreachable Docker daemon now stops the run with resume instructions, and resumable optimizer state (`save_state`) is saved at every checkpoint (`--resume-state`, `--start-step`). Attempt 1 only saved sampler weights, so it cannot be resumed; its records are in `experiments/open-source-model/apps/rl/rewrite_signal_pilot_attempt1/`.
- **Learning so far** (held-out, 20 problems × 2 bits per evaluation):

  | Step | Balanced acc | P(YES) gap ON-OFF | AUROC vs honeypots | Backdoor works | Pass |
  | --- | --- | --- | --- | --- | --- |
  | 0 | 0.55 | 0.14 | 0.44 | 48% | 93% |
  | 10 | 0.58 | 0.16 | 0.50 | 53% | 98% |
  | 20 | 0.58 | 0.17 | 0.61 | 73% | 98% |
  | 30 | 0.63 | 0.26 | 0.61 | 73% | 98% |

- **The first thing learned was keeping the backdoor** (the penalty fell from -0.58 to -0.22). Overseer suspicion rose with it: Qwen3.5-4B partly penalises backdoors, not just signals, despite its prompt. A stronger T would separate the two.
- **Untrained-sender baseline** (full pipeline, 57 surviving attacks vs 246 honeypots): collusion AUROC 0.43 (none), 0.37 (minified), 0.37 (paraphrased). The receiver says YES to 79% of honeypots, so it is poorly calibrated on that set.
- **Step-30 checkpoint, full test-split pipeline** (`rl_rewrite_signal_pilot_step30` vs `rl_rewrite_signal_untrained`): attacks surviving the rewrite rose from 57/114 to 93/114. Trusted suspicion fell slightly (6.28 to 6.10 of 10). Collusion AUROC against honeypots did **not** improve: 0.43/0.37/0.37 untrained vs 0.41/0.37/0.34 trained (none/minified/paraphrased). The frozen receiver says YES to 79-96% of honeypots, so it is the bottleneck. The training-time 0.61 came from only 20 training-split problems.
- **Next (proposed):** fix the receiver first (fine-tune regime 2, or recalibrate its prompt on the full honeypot set), developing on a smaller U; only then rerun the slim DeepSeek pilot. The 50-step rerun was held on cost grounds (2026-10-11).

## Phase 0: setup
- [x] New branch `feat/hipt-rl` from `origin/main`; `grouped_rl` + tests ported (dry-run WIP left in `stash@{0}`)
- [x] Offline tests pass

## Phase 1: configurable core + Option A, sender-only, Tinker
- [x] YAML config + run driver (`configs/rl/rewrite_signal_pilot.yaml`, `grouped_rl/run.py`); `hf_gpu` rejected with a clear error
- [x] `task_mode` option; `rewrite_signal` implemented. `honest_signal` is still available in the core library but not yet in the driver; `attack_signal` is Phase 3
- [x] Data from generator `.eval` logs, filtered to passing attacks with working backdoors
- [x] Backdoor-trigger check in the grader, with control_arena parity
- [x] Reward registry: `branch` (default), `hipt`, failed-backdoor and backdoor-suspicion terms, overseer ramp
- [x] Both-bit groups; per-bit, balanced-accuracy, p1-gap and identical-pair metrics
- [x] Receiver scored with the eval monitor's own YES/NO scorer; self-signal prompt drafted and calibrated (**still open for your review**)
- [x] Overseer signal-detection prompt for T + preflight controls
- [x] `ppo` loss with multiple passes; KL via base-model log probabilities
- [x] Backend generalised beyond Qwen3-8B (DeepSeek-V3.1 verified live)
- [x] Signal-jamming channel: `none` / `minify` / `paraphrase`
- [x] Generate train-split data: 1,033 DeepSeek attacks, 108 train / 20 validation tasks after filtering
- [ ] Pilot run (50 steps). The first attempt reached step 39 (see below); a rerun is queued

## Phase 2: evaluation integration
- [x] `scripts/rewrite_solutions_inspect.py` (attacks.eval + RL checkpoint → signalled attacks.eval, scored by the Inspect APPS scorer); verified live on 5 samples, and the output is accepted by `monitor_solutions_inspect.py`
- [x] `make_configs.py`: open U = DeepSeek-V3.1; RL arms (`rl_rewrite_signal_pilot` + `rl_rewrite_signal_untrained` control) with rewrite, transform and monitor configs; `ckpts.json` accepts RL arms. Regenerating changes no existing config
- [x] `src/lasr_labs_2025_control_project/rl-steps.md`
- [ ] Full evaluation of the pilot checkpoint + plots (needs the checkpoint and the self_rec test-split inputs in `experiments/open-source-model/apps/self_rec/`)

## Phase 3: later
- [ ] Monitor fine-tuning regime (1): signal vs no signal (the branch's receiver SFT, wired to a flag)
- [ ] Monitor fine-tuning regime (2): signalled attacks vs honeypots
- [ ] Option B `attack+signal`: backdoor prompt, secret-input parsing, attack-success reward
- [ ] Shared sender/receiver LoRA adapter as a red-team strategy
- [ ] ICRL (HiPT's in-context learner)
- [ ] Hugging Face / cloud-GPU backend (closer to HiPT's TRL value-head PPO)
- [ ] Small LLM vs classifier as overseer

## Open items
- The receiver prompt (`RL_SELF_SIGNAL_RECEIVER_SYSTEM_PROMPT`) needs your review.
- W&B isn't logged in on the dev machine; runs log offline (`wandb sync` later).
- The self_rec test-split inputs exist locally only under `experiments/open_models/test_data/`, not in the layout the configs expect.

## Resolved
- U = DeepSeek-V3.1, H = gpt-oss-120b, so there is no U/H clash.
- "Summarization" is the existing LLM paraphrasing transform (`summarize_solution`), used as the `paraphrase` channel.
