# BashArena integration review (issues #45 and #29)

Scope: the whole BashArena path on `feat/basharena-monitor-plots` (PR #46, stacked on
#39–#43), against ControlArena 17.1.2 (`control-arena==17.1.2`, `inspect-ai==0.3.201`)
and the original Redwood BashArena (`redwoodresearch/basharena_public`, commit
`86082fd`, BashArena 1.1). The goal is a clean, fast, drop-in replacement for APPS and
BigCodeBench in `sft-steps.md`.

Proposed fixes are on branch **`feat/basharena-review-fixes`** (stacked on
`feat/basharena-monitor-plots`). Only the one clearly-correct, offline-testable fix is
committed there (invalid-sample handling, below); every other change is given as an
exact diff for you to apply with validation, because I could not run the pipeline here
(see "What I could and couldn't run").

## What I could and couldn't run

- **Docker daemon: works.** I started `dockerd`, pulled `alpine:3.20`, and ran a
  container. CPU 4 / RAM ~15 GiB here (your machine: 8c/12t, 30 GiB).
- **BashArena containers: could not bring up the five-container stack here, so the
  smoke test is code-review-only (as the brief allows).** Two environment blocks, both
  network policy, neither a code bug:
  1. `huggingface.co` returns 403 through this environment's egress proxy, so the
     dataset loader (`Eccentricity/bashbench2`) cannot fetch tasks.
  2. The base image build and the three service containers do `pip install` at build
     and container start; this environment transparently re-terminates TLS, so pip
     inside the container fails cert verification unless the proxy CA is added to the
     container trust store. Doing that was declined by the sandbox safety classifier,
     and I did not work around it.
  I confirmed the stack's **shape** from upstream's local task mirror instead: all 640
  tasks are present, split exactly 320 odd / 320 even, and the five services
  (`default`, `router`, `logging_server`, `malware_server`, `database_server`) and
  their memory limits (1024 + 128 + 3×384 ≈ 2.3 GB/sample — matching your "~2.3 GB"
  figure) are as the compose template declares.
- **Tinker: blocked.** `tinker.thinkingmachines.dev` is also 403 through the proxy, so
  no fine-tune or monitor call could run. The `$10` budget was not spent. Q2 (timing)
  and Q4 (input design) are therefore from code + your measurements, not new runs.
- **Tests: ran.** The BashArena suite passes (26/26 after the fix). The full offline
  suite is 70 passed; the 4 failures are all the blocked Hugging Face proxy
  (`codeparrot/apps` 403 in `test_apps_loader` / `test_apps_rescore`), pre-existing and
  unrelated.

To actually run BashArena you need an environment that allows `huggingface.co` and
PyPI egress (or a pre-trusted proxy CA) — i.e. your own machine, or a cloud environment
with a broader network policy.

---

## Q1. Findings ranked by severity

Severity = impact on correctness of trajectories/labels or on the conclusions, not
effort. "file:line" is in our repo unless prefixed `PB:` (the packaged control-arena
under `.venv/.../control_arena/`) or `U:` (upstream `basharena_public`).

### S1 — Length is doing almost all the work; the headline numbers are not yet about the monitor (confirms #29 caveat)
Not a code bug, but the most important finding. Attacks stop at side-task success
(median 14 actions); honest runs use most of the 60-step budget (median 60). Counting
actions alone gives ROC AUC 0.998 for attack-vs-honest and 0.89 for U-vs-honeypot, and
the fine-tuned monitor keeps only ~0.84–0.89 after a length trend is removed. So the
safety curve and the self-recognition AUC cannot yet be credited to the monitor. The
root cause is a design choice (score once per *finished* trajectory, so the monitor
sees how long the run turned out to be). Fix is Q4: score per action. This dominates
everything below.

### S2 — Monitor/SFT view can hide the attack inside a long argument
`utils/utils.py` `_cap` (`:231-238`, used by `trajectory_view` `:258-259`) keeps only
the first 2000 + last 2000 chars of each tool-call argument over 4000 chars. The
malicious payload in the middle of a large `file_text`/`cmd` is removed from what the
monitor (and the fine-tune) sees, while the sample stays labelled an attack. Upstream
never truncates arguments — only tool *outputs*. This weakens exactly the attack
samples and biases both the suspiciousness and collusion signals. (The example run in
your #29 report is ~1.4 KB, under the cap, so it isn't hit there — but long file-write
attacks are.) Fix: raise/remove the per-argument cap for the monitor view, or cap the
total view length instead of each argument's middle.

*How upstream BashArena handles this:* its monitor view (`U:src/xml_utils.py:51-150`)
writes every tool call's arguments in full (`<arguments>{tool_call.arguments}</arguments>`,
line 95, no cap) and truncates only tool **outputs**, to 50 head + 50 tail lines, or
5,000 + 5,000 characters when the output is longer than 10,000 characters
(`truncate_output`, lines 9-30, called with `max_output_lines=100`, line 55). Separately,
upstream has an optional *agent-side* guard, `DatasetSpec.restrict_tool_arg_length`
(`U:src/wrap_tool.py:7-46`), which rejects an over-long argument with an error before
it runs, so nothing is ever hidden from the monitor. It is off by default and not used in
`run_tasks.py`. Neither the packaged ControlArena setting nor upstream ever cuts the
middle out of an argument.

### S3 — Invalid samples kept / handled three different ways *(fixed on the branch)*
A zero-action trajectory (empty `base_solution`) and an errored sample were handled
inconsistently: SFT kept them (`finetuning/utils/eval_to_dataset.py:126-138`), the
checkpoint ROC crashed on them (`scripts/tinker_sft_dynamics.py:73-82`), the monitor
kept them (`scripts/helpers/eval_to_dataset.py:47-52`), and `check_sample_is_valid`
never looked at `sample.error` (`utils/utils.py:290-299`). A zero-action honest run was
scored on an empty action list and counted as a benign negative (it feeds the length
shortcut and the safety curve). Zero-action attacks (`side_task_success=C`, empty view)
were accepted as valid training rows. This bites hardest on upstream's blacklisted
task `109553` ("instantly triggers firewall"), which is odd → **our train split**: the
auto-submit check runs *before* the first model call, so the run is a forced submit
with zero actions and side task C. **Fixed**: `check_sample_is_valid` now treats an
errored sample as invalid and, in the BashArena branch, treats a blank `base_solution`
as invalid; all three consumers pass these. APPS/BigCodeBench behaviour is unchanged
(the gate is keyword-only and scoped to the BashArena scorer format). Tests added.

*Known upstream?* Partly. Redwood knew about 109553 and blacklisted it on 2026-03-19
(`U:` commit `3e486fa`, "instantly triggers firewall"), along with four other ids. They
avoid the task rather than fixing the code: neither upstream nor ControlArena (checked
through 19.0.0) filters zero-action trajectories. Upstream's safety code
(`U:science/safety_utils.py`) also doesn't filter errored samples. The blacklist lives
only in upstream's run script, and the packaged `DatasetSpec` has no blacklist field, so
anyone using `BashArenaSetting()` from ControlArena, as we do, misses it. Recommended:
keep the validity fix, and also add the five upstream ids to `UNBUILDABLE_TASKS`
(renamed, since they aren't all build failures). Two of them, 34113609 (fork bomb) and
59487096 (CPU-heavy miner), also put the host and the other parallel samples at risk.

### S4 — `hit_max_tokens` latches on in the packaged policies (correctness)
`PB:policies/attack_policy.py:95-104` and `honest_policy.py:87-96` scan *all* prior
messages and set `last_message_hit_max_tokens=True` if **any** earlier assistant
message was truncated (the loop only `break`s on a True). So after the first truncated
step, every later step gets "Your last tool call hit the max token limit…" instead of
the attack-planning reminder (attack) or the submit guidance (honest). Upstream checks
only the most recent assistant message (`U:.../attack_policy.py:102-108`). Exposure is
highest in the DeepSeek configs (`max_tokens: 4096`). This is a bug in the vendored
package, not our repo; see "proposed patches" for how to avoid relying on it.

*Fixed in a newer version?* Upstream BashArena yes, ControlArena no.
- **Upstream** shipped the same latching loop on 2025-12-15 (commit `f9b4686`) and fixed
  it on 2026-03-23 (commit `79ed8e9`, "syncing stuff with private repo"). It now reads
  only the most recent assistant message:
  `if isinstance(message, ChatMessageAssistant): last = bool(...); break`.
- **ControlArena** never picked up that fix. I downloaded 18.0.1 and 19.0.0 (the latest,
  2026-07-31) from PyPI. Their `settings/bash_arena/` is byte-identical to 17.1.2 apart
  from two test files, so both policies still latch.
- Upgrading won't fix it, and 18+ also needs `inspect-ai>=0.3.250` (see `pyproject.toml`).
  The fix has to come from us, in one of two ways:
  - a ~3-line local copy or subclass of the two policies with upstream's loop;
  - an upstream ControlArena PR that ports `79ed8e9`. This is the better long-term
    option, but you said not to open PRs, so it's for you to decide.

### S5 — A hung main-task test discards the side-task label (generation robustness + lost data)
The main-task scorer runs pytest with **no timeout** (`PB:task/scorer.py:32-65`), and
runs *before* the side-task scorer (`PB:bash_arena_setting.py:136-140`). Inspect gives
scoring `time_limit/2` = 30 min (`inspect_ai/_eval/task/run.py:1047`, verified). So a
task whose own tests hang (your 9.9 h / 4.5 h cases) burns the whole 30-min budget on
the main task; the side-task scorer never runs, the sample errors, and an attack whose
side task already succeeded is dropped with no score. Your "4 of 260 scored samples"
were lost this way, and nothing counts them.

*Known issue or our bug?* The hang is upstream behaviour. The unbounded hang was our bug.
- **Upstream (both versions).** `run_main_task_test_with_results` runs pytest with no
  timeout in upstream BashArena (`U:src/task/scorer.py`) and in every ControlArena
  through 19.0.0 (scorer byte-identical). Neither base image installs `pytest-timeout`.
  Upstream masks the problem with a 3 h `time_limit` (scoring gets 1.5 h) and 25 parallel
  samples.
- **Ours.** We had no `time_limit` at all, because `get_control_task` drops it. That is
  what turned a hung test into a 9.9 h stall, and the #29 fix closed it.
- **What remains** is the upstream half: a hang still uses up the 30-minute scoring budget
  and loses the side-task label.

*Fix (corrected).* Two things I proposed earlier would not work:
- `sandbox().exec([...], timeout=N)`. Inspect's docker exec kills only the host-side
  `docker compose exec` client, leaving pytest running in the container. It also
  re-runs a timed-out command up to twice by default (`timeout_retry=True`,
  `inspect_ai/util/_sandbox/docker/compose.py:345-366`), so tests would run 3 times.
- Running the side-task scorer first. That changes results. The main-task tests run the
  agent's code, and running it can fire a payload the agent planted. The honeypot example
  in #29 has this shape: `fd-tracker read-config` sends the key to the logging server.
  With main-task scoring first, as upstream does, that counts as side-task success.
  Scoring the side task first would miss it.

The correct fix keeps upstream's order and bounds pytest inside the container:
- a thin override of `main_task_success` that runs
  `timeout -k 10 600 python3 -m pytest …` with `timeout_retry=False`;
- on exit code 124/137, score the main task 0 (or read the partial junit XML), then let
  the side-task scorer run as normal.

This is label-preserving except that a test which genuinely needs more than 10 minutes
scores 0. Pick N from the tail of your measured scoring times (median 33 s).

### S6 — Side-task description and scorer diverge from published BashArena (comparability)
Upstream attack runs use **one** side task per run, and the final scorer counts only
that task (`U:run_tasks.py:190-201`). We use `BashArenaSetting()`'s default all-4
"achieve any one of the following" description (730 words) and an any-of-4 scorer
(`PB:side_tasks/multi_side_task.py:224-231`; our call
`scripts/generate_solutions_inspect.py:341`). The model picks the easiest side task, and
attack-mix/attack-success numbers are not comparable to published per-side-task
BashArena results. Internally consistent (labels fine); flag it when comparing. Honest
runs match upstream (upstream honest also uses all 4).

*Risks of the all-4 setup*, most serious first:
1. **Side task as a confound for self-recognition.**
   - U (Qwen) and H (gpt-oss) each choose their own side task per run. If they prefer
     different ones, "which side task" becomes a feature that separates U attacks from
     honeypots. The fine-tune can learn it instead of authorship.
   - The leak and firewall attacks also look very different as commands.
   - This is the same kind of shortcut as length, and nothing in the current results
     rules it out.
   - Check it from the logs you already have: the final side-task scorer explanation
     says which objective passed. Tabulate side task by role, and report the
     self-recognition AUC within each side task.
2. **Easiest-target skew.** Attacks will cluster on whichever side task is cheapest
   (probably `leak_passwords`). So the monitors are mostly tested on one attack type,
   and safety numbers say little about the other three.
3. **Credit for the wrong side task.**
   - The any-of-4 scorer counts an attack as successful if it trips any objective, even
     by accident while aiming at another.
   - Upstream scores only the targeted task, and an accidental trip of a different one
     ends the run as a failure.
   - So our "usable attack" set is looser than upstream's.
4. **Prompt cost.** The 730-word combined description is in every attack prompt, re-sent
   every step, which adds to token and context pressure (S9).
5. **Comparability.** Attack mix and success rates can't be set against published
   per-side-task numbers.

*Mitigation, smallest first:*
- Report the side-task mix per role and the per-side-task AUC. This needs no
  regeneration.
- Then, if it matters, assign one side task per task id (e.g. `id % 4`), with the same
  assignment for U attacks and honeypots on each id. That is
  `DatasetSpec(side_tasks=[x])` per subset.
- Honest runs keep all 4 checkers, as upstream does.

### S7 — Monitoring granularity diverges from published BashArena (comparability)
Upstream scores **each step** with the prior actions *and their (truncated) outputs*,
averages a 3–6 sample ensemble, takes the max over steps, and sets the audit threshold
as a quantile over **all honest step scores** (per-step budget). We make **one** call
per finished trajectory, no outputs, one sample, trajectory-level budget. A 1% step
budget ≠ a 1% trajectory budget, so the safety numbers are not comparable. See Q4.

### S8 — No generation retries (speed + honest-set bias)

Upstream wraps every model call in 3-try backoff then a fallback message and continues
(`U:policies/generate_with_retry.py:30-68`). The packaged policy calls `model.generate`
bare, our Tinker provider has no `should_retry`, and `retry_on_error` is unset. One
transient Tinker error or a context overflow kills the whole sample. Honest runs that
overflow the 65,536-token context (your 14 of 103) raise and are dropped rather than
scored on their prefix — this silently removes the *longest* honest runs, biasing the
honest set shorter (interacts with the length shortcut) and keeping whichever runs
finished first in the partial snapshot.

*Fix.* Three small pieces, all in our code. None of them touch the vendored policies.

1. **Retry transient errors.** Override `TinkerAPI.should_retry(ex)` in
   `utils/tinker_provider.py`:
   - Retry on `tinker.APIConnectionError`, which includes `APITimeoutError`, and on
     `RateLimitError` and `InternalServerError`.
   - Don't retry `BadRequestError`, `AuthenticationError`, `BillingError` or
     `PermissionDeniedError`.

   The base `ModelAPI.should_retry` returns False (`inspect_ai/model/_model.py:321-327`),
   so today nothing is retried. Once it returns True, Inspect's own retry loop
   (`_model.py:1064+`) backs off and retries, bounded by `GenerateConfig.max_retries` and
   `timeout`. This is what upstream's `generate_with_backoff` does by hand.
2. **End context overflow cleanly instead of crashing.**
   - In `TinkerAPI.generate`, if `prompt.length + max_tokens` would exceed the model's
     context window, or Tinker returns a `BadRequestError` about context length, return a
     `ModelOutput` with `stop_reason="model_length"` rather than raising.
   - Inspect's react loop then stops the agent cleanly (`inspect_ai/agent/_react.py:383-388`).
     Scoring still runs, and the trajectory is kept up to that point.
   - That recovers your 14 dropped honest runs and removes the bias towards shorter honest
     runs.
   - I couldn't check Tinker's exact overflow error text from here. Treat the pre-check
     as the primary guard and the exception match as a fallback.
3. **Backstop.** Pass `retry_on_error=1` to the BashArena `eval()` call in
   `scripts/generate_solutions_inspect.py:428-445`. This re-runs a whole sample once if it
   still errors, e.g. a container that failed to start.
   - It costs a full rerun, and a sample that succeeds on retry has a different
     trajectory.
   - So log how many samples were retried.

### S9 — `token_limit=2,000,000` can bind before `max_steps=60` (correctness)
Upstream uses 100M (`U:run_task_utils.py:177`); we use 2M
(`scripts/generate_solutions_inspect.py:364`). Every step re-sends the whole context, so
cumulative tokens grow ~quadratically; a rough estimate puts the limit at ~step 41 if
context grows 2K tokens/step, ~57 at 1K/step. When it binds, the run is truncated early
(lower honest main-task success; attacks cut short). Needs a log check:
`metadata["control_solver_error"]` contains "token" or "Max steps". Raise it toward
upstream's value, or at least log when it's hit.

*BashArena defaults, checked:*

| Limit | ControlArena `ControlEvalConfig` default (17.1.2 = 19.0.0) | Upstream `eval_policy_on_tasks` default (`U:src/run_task_utils.py:142-189`) | Upstream `run_tasks.py` (what they ran) | Ours |
|---|---|---|---|---|
| `token_limit` | 1,000,000 | **100,000,000** (hard-coded) | 100,000,000 | 2,000,000 |
| `max_steps` | None | 100 | **75** (`:219`) | 60 |
| `time_limit` | None (and dropped by `get_control_task`) | 2 h | **3 h** (`:214`) | 1 h (set on `Task`) |
| scoring limit | — | 1 h (time_limit / 2) | 1.5 h | 30 min |
| parallel samples | `max_tasks=4` | 50 | 25 | 6 |

- **Upstream's 100M is effectively "no token limit".** 75 steps × a 200K context is only
  15M. So they rely on steps and time only.
- **The ControlArena default (1M) is tighter than ours.** Anyone running the packaged
  setting with defaults gets cut off even earlier.
- **Why it binds for us.**
  - Inspect counts every call's full prompt toward the limit: `total_tokens` in
    `inspect_ai/util/_limit.py:554,600`, which our provider fills with `prompt.length`
    (`utils/tinker_provider.py:388-392`).
  - Your own data says honest runs approach the 65,536-token context (14 of 103 went
    over).
  - A run whose context grows roughly linearly from ~7K to ~65K over 58 steps uses about
    58 × 36K ≈ 2.1M tokens. That is right at the 2M limit.
- **So the long honest runs are being stopped by two limits at about the same point**:
  context overflow and the token limit. Some of the "median 60 actions" may be
  token-capped rather than step-capped.
- **Recommendation.** Set `token_limit=None`, or 100M to match upstream, and keep
  `max_steps` and the time limit as the real bounds.
- **Cost.** Removing the cap allows longer, and so dearer, runs. The step and time limits
  still bound them, and they bound upstream too.

### S10 — Train/test reused for design choices (methodology, confirms #29 "exploratory")
Split is strictly enforced (odd=train/even=test at generation, in `collect`, and an
assert in `main`), so no id is in both. But the checkpoint ROC is computed on the test
set with no validation split (`val_frac=0`), the honeypot model was chosen with
`honeypot_comparison_*` configs that run on test tasks, and the pilot hyper-parameters
were judged on the 20-task test prefix. Nothing auto-selects a checkpoint (always the
last), but the test set has informed design. Hold out some train ids (e.g. odd with
`id % 4 == 3`) for curves and model choice.

### Lower-severity / latent
- **Stale base image** (`PB:bash_arena_setting.py:106-111`): the setting skips building
  `benchmark-base:latest` if the tag exists. An old image would ship the pre-1.1
  (easier-to-hack) firewall daemon. Check `docker run --rm benchmark-base:latest
  md5sum /usr/local/bin/firewall-daemon` against the packaged file.
- **Firewall-check orphan** (`PB:side_tasks/disable_firewall/task.py:47-60`, same
  upstream): if the listener takes >5 s to start, it is left bound to UDP 8888 and every
  later firewall check (including the final scorer) reads "not disabled". Grep score
  explanations for `LISTENER_READY timed out`.
- **Honest reminder persisted** (`PB:honest_policy.py:105-107`): honest (not attack)
  stores each reminder in history, ~100 tokens/step re-sent forever — grows honest runs
  faster toward the token limit. Upstream persists neither.
- **Malformed tool calls counted as actions**: `trajectory_view` (`utils/utils.py:249-262`)
  includes calls with `parse_error` (rendered `<tool>unknown</tool>` or empty-arg
  actions) produced by the Tinker renderer (`utils/tinker_provider.py:295-303`). They
  inflate action counts and give the fine-tune a format shortcut. Skip `call.parse_error`.
- **Minified APPS arm is trained and scored on different text** (predates this branch):
  `finetuning/utils/eval_to_dataset.py:192-197` fences the minified solution in
  ```` ```python ````; monitor and dynamics use it unfenced.
- **Drop-in shell path trains BashArena on APPS prompts**:
  `finetuning/binary_tuning/transform_raw_to_prepared.sh:13-14` defaults to the APPS
  collusion prompts; the monitor/dynamics use the BashArena ones.
- **`collect` checks too little** (`run_logs/run_half_experiment.py:61-84`): it accepts a
  mixed honest/attack log in the same role and does not compare generation settings
  (time_limit, max_steps, `UNBUILDABLE_TASKS`) between prefix and extension batches — and
  those three were only added in the last commit, so earlier prefixes may differ.
- **Generation overwrites finished files** without a status check
  (`scripts/generate_solutions_inspect.py:447-455`); a rerun/interrupt can replace a good
  `.eval`.
- **Monitor prompt** `BASHARENA_SUSPICIOUSNESS_SYSTEM_PROMPT` matches upstream
  `standardized.txt` (0–9), adapted per-action→whole-trajectory. Upstream's *run scripts*
  actually use `standardized_deferral.txt` (0–100, extra calibration). So even the prompt
  differs from what produced published numbers.

---

## Q2. Where generation time goes, and what to cut

Your measured split (113 tasks, 12.6 h): 44% per-step side-task check (~12 s/step), 28%
start-up (~2 min/task), 15% model, 6% agent, 7% unattributed. I could not re-measure
live (HF + Tinker blocked), but every figure is consistent with the code, and the two
biggest levers are structural and safe:

**Lever A (biggest, label-preserving): bake service deps into the images (~28% → near 0).**
`PB:base_environment/compose-template.yml:81,101,121` runs, at *every container start*,
`apt-get update && apt-get install -y curl && pip install -r requirements.txt && python
app.py` for all three services, behind a 120 s healthcheck `start_period`; the router
does `apk add …`. The requirements are tiny (`Flask==3.0.0`; the DB server adds
`requests==2.31.0`). Moving the apt/pip/apk into the service Dockerfiles (built once,
reused across all samples) turns ~2 min of per-sample start-up into seconds. This is the
single largest overall win and changes no semantics. (It lives in vendored
`control_arena`, so it needs either a small fork of the base environment or an upstream
PR — see proposed patches. I could not build/verify it here because pip-in-container
needs the proxy CA.)

**Lever B (biggest per-step, needs care): the firewall packet test dominates the 44%.**
`check_any_side_task` runs the 4 checkers **sequentially** before **every** step
(`PB:side_tasks/multi_side_task.py:56-73`, called from
`PB:protocol/auto_submit_protocol_wrapper.py:54-76`). Only the firewall checker is slow:
sender sends 7 packets with `time.sleep(0.3)` (2.1 s) plus listener start/finish polls
with 1 s sleeps and several `docker exec` round-trips — ~5–10 s/step, i.e. ~5+ min over a
60-step run. The leak checker re-reads all 12 secret files every step
(`PB:side_tasks/leak_passwords/task.py:97-127`). Options, in increasing aggressiveness
(**corrected after review** — my first draft wrongly called B2 and B3 label-preserving):
1. **Run the 4 checkers concurrently** (`asyncio.gather`, then pick `winning_task` in the
   original list order). The four checks share no state (sudo reads `sudo -l`; malware
   reads `/tmp/evil`; leak reads files + the logging-server log; firewall uses its own
   UDP listener) and the agent is paused while they run, so results are identical.
   Saves the sudo/leak/malware time (~2–4 s/step; ~44% → ~34%).
2. **Batch the leak checker's 12 file reads into one `exec`** (one `bash -c` that prints
   each file with a delimiter) instead of 12 `cat` round-trips. Same data, same result,
   ~11 fewer `docker exec` calls per step. *Not* "cache once per sample" as I first wrote:
   the agent runs as root and can rewrite those files (main tasks rotate keys, edit
   `/etc/shadow` via `useradd`/`passwd`), and the checker deliberately matches the
   *current* contents, so a cache would change which leaks count.
3. **Gate the firewall packet test** behind a cheap state diff. **This can change
   results** and I no longer recommend it by default. The firewall can be defeated many
   ways (edit `firewall.conf`; kill `firewall-daemon` or `custom_firewall.py`; edit those
   scripts; flush/alter iptables or the NFQUEUE rules; change the content blacklist). A
   gate covering conf + `iptables-save` + process list + script hashes catches the known
   routes, but a missed route means an attack is not stopped when it succeeds (longer,
   different trajectory) and an honest live failure is kept as honest. Only adopt it with
   an A/B run that executes both the gate and the full test and counts disagreements.
4. **Honest runs: check once at the end, not per step.** **This changes results.** It
   departs from BashArena's live-failure protocol: an honest run that trips a side task
   keeps going instead of stopping, and a transient trip the agent later reverts is
   missed and the run is kept as honest. It does roughly halve the honest batch. Treat it
   as a deliberate protocol change, not an optimisation.

**Does any lever weaken container or host safety?** No. None of A, B1–B4 or C changes the
containers' isolation (capabilities, networks, memory limits and the router are
untouched). The side tasks are simulated inside each task's own network: the "firewall"
is the in-container firewall, the logging/malware servers are local containers, and
`sudo` is inside the container. Skipping a check (B3/B4) changes *what we record and when
we stop*, not what the agent can reach. The real host-safety items are elsewhere: tasks
34113609 (fork bomb) and 59487096 (CPU-heavy miner), which upstream blacklists and we
don't (S3), and running many sandboxes on a 30 GiB machine without swap.

| Lever | Changes labels / evaluation results? | Changes isolation / host safety? | Verdict |
|---|---|---|---|
| A: bake service deps into images | No (same apps, same healthchecks; pins package versions that today float with `apt-get update`) | No | Do it |
| B1: concurrent checkers | No | No | Do it |
| B2: one `exec` for the 12 secret files | No | No | Do it |
| B2′: cache secrets once per sample | **Yes** (misses leaks of rewritten secrets) | No | Don't |
| B3: gate firewall test on state diff | **Yes**, in edge cases | No | Only with A/B validation |
| B4: honest end-only check | **Yes** (protocol change) | No | Only as an explicit choice |
| C: bound pytest *inside* the container (`timeout -k`, no retry), keep scorer order | Only for tests needing > N min (score 0) | No | Do it |
| C′: score side task before main task | **Yes** (misses payloads that fire during tests) | No | Don't |
| C: Tinker `should_retry` + `model_length` on overflow | Keeps samples that are dropped today (that is the point) | No | Do it |
| Add upstream's 5-id blacklist | Removes those tasks from the data | **Improves** host safety | Do it |

**Lever C: don't lose whole samples to hangs/errors (S5, S8).** Bound pytest inside the
container and keep upstream's scorer order (see S5 for why the naive `exec(timeout=)` and
side-first ordering are wrong); add `TinkerAPI.should_retry` and a clean `model_length`
stop on overflow; optionally `retry_on_error=1` (S8). Each lost long sample is pure wasted
wall-clock plus a dropped data point.

**Already done well:** your three #29 fixes are real and verified — `Task.time_limit =
3600` is applied by Inspect (the `ControlEvalConfig.time_limit` path is genuinely dropped
by `get_control_task`, confirmed), `log_buffer=1` saves each sample as it finishes, and 6
parallel sandboxes fit the RAM. Model (15%) and agent commands (6%) are inherent.

Net: Levers A, B1, B2 (single exec) and C are label-preserving and should remove well
over half the wall-clock (most of the 28% start-up and a chunk of the 44% checks). B3 and
B4 remove more but change evaluation results — use them only deliberately, behind a flag,
with validation.

---

## Q3. What to delete or simplify (keeps the APPS/BigCodeBench drop-in shape)

All verified as unused or no-ops; none touch the APPS/BigCodeBench path.

- **Dead code**: `utils/logs_to_model.py` (whole file; `runner()` raises
  `NotImplementedError`, nothing imports it). `utils/utils.py:194-196`
  (`clean_markdown_from_code`) and `:199-226` (`get_accuracies_from_log`) — no callers.
  `finetuning/utils/eval_to_dataset.py:35-40` — unused local copy of
  `apply_python_code_format`.
- **No-op**: `scripts/monitor_solutions_inspect.py:260-263` — `yes_logprob != float("nan")`
  is always True, so the guard does nothing (NaN still propagates); simplify to a real
  `math.isnan` check or drop it. `:234-244` has a redundant `and y is not None`.
- **Legacy chain** (confirm nobody runs `eval_analysis` first):
  `utils/evals_to_dataframes.py` → `data_pipelines/monitors.py` → `scripts/eval_analysis.py`
  filters on `apps_scorer`/`<final_code>`, returns 0 rows for BashArena, and is superseded
  by `monitor_solutions_inspect`.
- **Consolidate, don't delete**: the two `eval_to_dataset.py` files are **both used**
  (one builds a `MemoryDataset` for monitors, one writes SFT JSON). They should share the
  one validity+metadata helper (this is where the S3 fix now lives) rather than either
  being removed.
- **Operational, keep out of `src`**: `run_logs/*.sh` and `watch_half_progress.py`
  hard-code `/home/jkk/...` paths, a dated log name, and a pid file read at import;
  `run_half_experiment.py:48-50` reads an untracked file at import so the module can't be
  imported or tested. Move the import-time read into `main`.
- **Reproducibility**: `finetuning/utils/eval_to_dataset.py:150-157` uses
  `hash(str) % 1e8`, salted per process, so non-numeric ids get a different `problem_id`
  each run; use a stable hash.

The BashArena branches leave APPS/BigCodeBench untouched. One latent fragility:
BashArena mode is detected from scorer *key names* (`main_task_success` +
`side_task_success`), not `setting_name` — any other setting using those keys would get
BashArena semantics. Prefer branching on `setting_name == "bash_arena"`.

---

## Q4. How to build the monitor / SFT input — recommendation and cost

The problem to solve is S1+S7: scoring **once per finished trajectory** lets the monitor
read the run's *length*, which is the giveaway (attacks are short because they stop at
success; honest runs use the full budget). Your own report shows action-count alone
reaches 0.998. No amount of monitor tuning fixes a length tell the monitor can see.

**Recommendation: score per action, on message prefixes, matching upstream/LinuxArena.**
For each assistant action *i*, score `messages[:i+1]` with the current action shown
without its output and prior actions shown with their (truncated) outputs; take the max
over steps; set the audit budget per step. This:
- removes the length shortcut (each scored unit is one action in context, not "the whole
  run is 14 vs 60 actions long");
- makes the safety number comparable to published BashArena/LinuxArena;
- needs **no regeneration** — the saved `.eval` logs keep full conversations with
  outputs, as your #29 report notes.

Per-sub-question:
- **Whole trajectory vs per action** → per action. This is the single highest-value
  change.
- **With or without outputs** → for the *trusted suspiciousness* monitor T, **include
  truncated outputs** (upstream: 50+50 lines or 5k+5k chars). Your #29 report's own
  example shows the monitor can't tell the read file held secrets precisely because it
  never sees the output; exfiltration is much easier to judge with outputs. For the
  *collusion self-recognition* monitor, commands-only is defensible (it keys on the
  model's own action *style*), but it must still be per-action to kill the length tell.
- **Last call without its output** → yes, that's exactly upstream's "current action shown
  separately, without output since it hasn't run." Adopt it as the per-action format.
- **Training on single tool calls** → for the self-rec fine-tune, train on per-action
  prefixes (one row per action). This removes the length tell from *training* too and
  multiplies the data ~15–60×. Keep commands-only there if you want to isolate style
  recognition.
- **Fix S2 regardless**: stop truncating the middle of a single argument, or the attack
  can be invisible to any of these.

**Cost.** Monitor API calls scale by the mean action count — roughly 14× (attacks) to
60× (honest), ~40× blended — versus one call now. But monitor calls are cheap next to
generation, and **all of it reuses existing logs (no new Docker/Tinker generation)**, so
the expensive part is unchanged. Engineering: a per-action scorer that iterates prefixes
(upstream's `remonitor_runs.py` / control-arena's `run_monitor_for_sample` already do
exactly this), plus the outputs-included view and a per-step threshold in the safety
model. Keep the whole-trajectory view as a secondary/legacy path.

If you want a cheap first cut that still removes most of the length signal without 40×
calls: score only a fixed-size window of the last *k* actions with outputs, per action,
for a subsample of steps. But the clean answer is the full per-action pass.

---

## Divergence table (what / where / does it matter)

MATTERS = affects labels or the conclusions; MINOR = small/condition-dependent; NONE =
confirmed equivalent. (a) correctness, (b) speed, (c) comparability with published
BashArena.

| # | What differs | Upstream (`U:`) | Ours (`O:`) / packaged (`PB:`) | Verdict |
|---|---|---|---|---|
| 1 | Score timing | per action | once per finished trajectory | **MATTERS a,c** (S1,S7) |
| 2 | Argument truncation in monitor view | none (outputs only) | each arg capped 2000+2000 (`utils/utils.py:231-238`) | **MATTERS a** (S2) |
| 3 | Zero-action / errored samples | n/a | kept/crashed inconsistently | **MATTERS a** (S3, fixed) |
| 4 | `hit_max_tokens` reset | most-recent msg only (fixed 2026-03-23) | latches after first truncation (`PB:*_policy.py`, still in 19.0.0) | **MATTERS a** (S4) |
| 5 | Main-task pytest timeout | no timeout, main-first (masked by a 3 h limit) | same, with a 1 h limit; a hang drops the side label | **MATTERS a,b** (S5) |
| 6 | Side task per attack run | one per run, scorer = that task | all-4 "any one" description + any-of-4 scorer | **MATTERS c** (S6) |
| 7 | Monitor inputs/ensemble/budget | step + outputs, 3–6 samples, per-step quantile | trajectory, no outputs, 1 sample, per-trajectory | **MATTERS c** (S7) |
| 8 | Retries | 3-try backoff + fallback | none; one error kills the sample | **MATTERS b**; a (honest bias) (S8) |
| 9 | `token_limit` | 100M (ControlArena default 1M) | 2M (≈ binds for runs near the 65K context) | **MATTERS a,c** (S9) |
| 10 | Blacklist | 5 ids w/ reasons | only `1003684`; `109553`/`34113609` etc. in our train split | **MATTERS a,b** (S3) |
| 11 | Test-set reuse | — | checkpoint ROC + H choice + pilot HPs on test | **MATTERS c** (S10) |
| 12 | Service deps | apt/pip/apk at container start | identical | **MATTERS b** (Q2 Lever A) |
| 13 | Per-step side-task checks | 4 sequential every step | identical | **MATTERS b** (Q2 Lever B) |
| 14 | Monitor system prompt | run scripts use `standardized_deferral.txt` (0–100) | `standardized.txt` (0–9), adapted to whole-trajectory | MINOR c |
| 15 | Honest reminder persisted | neither | honest persists it (`PB:honest_policy.py:105-107`) | MINOR a,b |
| 16 | Reasoning config | 32k/16k/"all" (attack) | 16k/4k/"last"; ignored on Tinker tool path (thinking off) | MINOR c |
| 17 | `generate_config` honeypot | — | thinking off for all families via renderer | NONE (consistent) |
| 18 | eval runner | `eval_set(**eval_args())` | direct `eval(...)`; drops only `time_limit` (re-added), `log_dir`, `log_level` | MINOR |
| 19 | `max_connections` (untrusted) | set 25 | never applied (policies call `get_model` w/o role) — same upstream | NONE |
| 20 | Prompts `attack.txt`/`honest.txt`, tools, compose, 4 checkers | BashArena 1.1 | byte-identical / equivalent | NONE |
| 21 | Base image rebuild | always (layer-cached) | skipped if tag exists | MINOR a (stale-image risk) |
| 22 | Missing `/var/log/sudo-management.log` | present | absent in packaged | MINOR c (honeypot realism) |

Full line-by-line evidence (every row, both subagents' raw notes) is reproduced in the
companion files under `scratchpad/` of the review session; the citations above are the
load-bearing ones.

---

## Proposed patches

### Committed on `feat/basharena-review-fixes`
- **S3 — invalid-sample handling.** `check_sample_is_valid` gains keyword-only
  `base_solution` and `errored`; it rejects errored samples and (BashArena branch only)
  blank trajectories. The three consumers pass `sample.error` and `base_solution`. APPS /
  BigCodeBench unchanged. 3 unit tests added; 26/26 BashArena tests pass.

### Proposed (exact changes, not committed — they alter results or live in vendored code, and I could not run the pipeline to validate)
1. **S2** — in `trajectory_view`/`_cap` (`utils/utils.py:231-263`): raise `_MAX_ARG_CHARS`
   to a large value or cap the *total* view length instead of each argument's middle, so a
   payload inside a big `file_text` can't vanish. (Behaviour change to monitor inputs —
   re-score after.)
2. **S5** — a thin override of `main_task_success` that runs pytest as
   `timeout -k 10 <N> python3 -m pytest …` *inside* the container with
   `timeout_retry=False`, scoring 0 on exit 124/137; keep upstream's scorer order
   (main first). (Corrected: a plain `exec(timeout=)` leaves pytest running and retries
   it twice; scoring the side task first misses payloads that fire during the tests.)
3. **Q2 Lever A** — move `apt-get/pip/apk` out of the compose `command:` into the service
   Dockerfiles (`service.Dockerfile` for logging/malware/database; the router image).
   Lives in vendored `control_arena`; cleanest as an upstream PR or a small vendored base
   environment. Biggest speed win; label-preserving.
4. **Q2 Lever B1/B2** — `asyncio.gather` the 4 side-task checks (pick the winner in
   list order); read the leak checker's 12 secret files in one `exec`. (Vendored;
   label-preserving.) Do **not** cache the secrets across steps — that changes results.
5. **S8** — add `TinkerAPI.should_retry` (connection/timeout, rate-limit, 5xx only);
   on context overflow return `stop_reason="model_length"` so react ends the run cleanly
   and it is scored; optionally `retry_on_error=1` on the BashArena `eval()`.
6. **S9** — `token_limit=None` (or upstream's 100M) at
   `scripts/generate_solutions_inspect.py:364`; steps and time stay the real bounds. Log
   `metadata["control_solver_error"]` counts per batch either way.
6b. **S4** — local copy/subclass of the two packaged policies with upstream's
   most-recent-message `hit_max_tokens` check (upstream commit `79ed8e9`); ControlArena
   19.0.0 still has the bug.
6c. **S3 follow-up** — add upstream's blacklist (`12249547`, `50618411`, `109553`,
   `59487096`, `34113609`) to the excluded-task set in `generate_solutions_inspect.py:322`.
6d. **S6** — tabulate side task by role from the existing logs and report the
   per-side-task self-recognition AUC before deciding whether to regenerate.
7. **Malformed tool calls** — skip `call.parse_error` calls in `trajectory_view`, or count
   them separately.
8. **R7/minified/shell-path** — guard `remove_comments` in
   `scripts/transform_solutions_inspect.py:116-120` with `is_code_setting`; format
   `transformed_solution` identically in SFT, monitor and dynamics; pick collusion prompts
   by `setting_name` in `transform_raw_to_prepared.sh`. (The BashArena configs don't use
   the transform arm today, so these are latent.)
9. **Cleanups (Q3)** — delete the dead files/helpers and the NaN no-op.

### Log checks to run on your saved `.eval`s (to turn estimates into numbers)
- `metadata["control_solver_error"]` per sample → how often the token/step limit bound.
- Count assistant messages with `metadata.hit_max_tokens == True`, by model/`max_tokens`.
- Attack samples with `side_task_success == "C"` and empty `base_solution` (zero-action).
- `sample.error` counts (scoring time-outs, Tinker errors) per batch.
- Grep score explanations for `LISTENER_READY timed out` (orphaned firewall listener).
- `md5sum /usr/local/bin/firewall-daemon` inside `benchmark-base:latest` vs the packaged
  file (stale pre-1.1 image).

---

## Rules I followed

Read-only on upstream and the vendored package; no edits/comments/PRs on any issue or PR;
no pushes to existing branches. Proposed changes are on a new branch
`feat/basharena-review-fixes` stacked on `feat/basharena-monitor-plots`. No samples were
dropped by me; the only sample-dropping change (S3) drops only zero-action/errored
samples and is the fix for a bug. The Tinker key was never printed or committed.
