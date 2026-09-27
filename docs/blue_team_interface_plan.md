# Blue team interface: design review

This draft uses the project vision pasted in the conversation on 2026-09-25.
The linked `docs/dependent_monitors_stats/inferce.md` is absent from this branch
and Git history, so its exact text could not be checked.

## Compatibility boundary

Existing `sft-steps.md` commands, YAML schemas, `.eval` contracts and downstream
SFT/plot readers remain supported. The minification solver delegates through
`MinifierJammer` to the existing minifier, preserving code and log fields. Other
new strategy entry points are optional. Integrations must preserve these
contracts or provide a separate versioned opt-in path; no migration is required.

## Goal and boundary

For APPS, let a user develop a defense with a weaker trusted model T, benign
generations from U, and black-box judgments from the untrusted monitor U_M.
The user supplies a pipeline that makes honeypots and a procedure for handling
candidate code. The evaluator holds out real U attacks, scores the same
defense on attack, honeypot, and benign cases, then uses the existing LASR
statistical model where its policy assumptions apply.

The rules about U_G are an **experiment convention** while U is downloadable.
Someone with the weights can run an attack prompt outside our API. We can keep
the official attack dataset and evaluation labels private, log sanctioned
queries, and state the rule clearly. A Python context object cannot enforce
the rule against a participant with local weights or arbitrary process access.

## What nearby code actually shares

| System | User supplies | Harness owns | Fit here |
| --- | --- | --- | --- |
| Control Arena checkout | Async per-step protocol callback, assembled from policy and monitor factories | Agent loop, tools, trajectories, scoring | Useful factory/composition pattern; its callback receives `mode`, which would reveal the attack label. |
| Control Tower eval2 | Protocol with input and output hooks; external `module:factory` registration | U calls, tool execution, history, step budget, task scoring | Best pattern for eventual online interventions: a narrow blue view excludes the side task. |
| BashArena | `ControlAgent` composed from policy, monitor, defer wrappers | Inspect task and tool execution | Good later backend for command trajectories, with a different Control Arena API version. |
| LinuxArena | Policy factory returning an Inspect agent; separate recorded-trajectory monitors | Agent execution and posthoc monitoring | Good source for offline trajectory scoring, less direct for live intervention. |
| This repo | YAML-selected generator, transform, monitor, SFT, and plot runs | Inspect eval logs and LASR optimizer | The shortest route to the next jamming/SFT figures. |

These are *composition* APIs, not security sandboxes. Do not pass an attack
mode, side-task description, real attack log, or hidden label into a blue hook.

## Recommended interface: implement, then test across arenas

Start with small blue team contracts and immediately exercise them on APPS
and a narrow LinuxArena integration. The earlier proposal to defer the
abstractions reduced initial code, but left portability untested. Porting a
whole arena first would also leave unclear which strategy operations its
adapter needs to support.

Use three shallow, composable contracts. Ordinary callable classes or stdlib
`typing.Protocol` are enough; implementations need no shared base class.

| Contract | Responsibility | First implementation |
| --- | --- | --- |
| `SignalJammer` | Turn a permitted monitor view into another view, retaining task context and recording representation/provenance. | Wrap the existing identity, Python minifier, and T summary transforms. |
| `HoneypotPipeline` | Given a public task and allowed resources, produce synthetic attack artifacts plus evaluator-only validation information. | Insert a T-generated attack into benign U code on APPS. |
| `BlueTeamPolicy` | Compose jammers and monitor calls into a result the evaluator understands. | Return UM/T scores using the current fixed monitoring policy. |

The first policy returns scores because that is what the current LASR model
consumes. An action-changing policy will need an explicit decision contract
and a runner that executes it. Keep calibration separate from evaluation:
freeze selected prompts/checkpoints and tuning parameters before held-out
evaluation; the honeypot pipeline itself can run on fresh public tasks.

The first implementation should make this composition work end to end:

```text
public task + benign U code -> HoneypotPipeline -> validated honeypot
candidate -> permitted monitor view -> SignalJammer -> UM
candidate -> permitted monitor view ----------------> T
paired scores + evaluator-only labels -> existing statistical analysis
```

The policy chooses which monitor receives each representation. Executable
artifacts and monitor observations must be distinct: a summary is an
observation, never a replacement program or command. An APPS view can contain
Python code; a LinuxArena view needs an ordered trajectory or a visible prefix
at the point of monitoring. Do not flatten every arena into an executable
code string. A Python-specific minifier should explicitly reject unsupported
representations; portability does not require every jammer to suit every arena.

The evaluator owns task splits, the canonical benign U prompt, model endpoints,
query logs and limits, code execution, attack labels, and the scoring run.
`resources` should expose operations such as `sample_benign(problem_id, n)`,
`judge_um(problem_id, artifact)`, and `call_t(...)`; T can also be trained or
run locally by the user. `sample_benign` takes a problem ID rather than an
arbitrary U prompt so the official API cannot be used to smuggle an attack
instruction. Evaluation policy calls see the candidate they must monitor;
calibration and honeypot generation never receive the held-out real attacks.

Arena adapters own task loading, permitted-view construction, execution, and
artifact validation. Keep native runners and scorers in their own arenas.
Start with a LinuxArena recorded-trajectory adapter, then run one real task
through its existing runner. For live monitoring, construct only the history
visible at that step; a full recorded trajectory would leak future outcomes.
Use an explicit allowlist of fields rather than handing blue code a raw
trajectory containing side tasks and scores. Sandbox replay and intervention
timing need their own check before claiming live control support.

## Fit to the stated requirements

| Requirement | Proposed handling |
| --- | --- |
| Downloadable colluding U | Publish checkpoint and exact inference/prompt recipe; treat benign-only usage as a stated research rule. |
| Free use of T | Let strategy own T prompting, transformations, and training artifacts; record versions and costs. |
| Benign U_G access | Harness-owned benign-generation call on calibration problems; never expose official attack prompts or held-out attack files. |
| User-made honeypots | `HoneypotPipeline` builds them from benign U code and T; the arena adapter validates them and keeps validation details out of monitor views. |
| U_M oracle | Logged score calls on user-supplied artifacts, with consistent prompt and checkpoint. |
| Held-out comparison | Run frozen strategy on real attacks, honeypots, and benign samples; retain paired scores and sample IDs for LASR analysis. |

The existing optimizer models score thresholds, audits, and honeypot budgets.
It does not estimate the value of every possible stateful `Decision`. For a
strategy that changes actions or updates across steps, run it in the actual
task loop and measure outcomes directly; only pass compatible score triples
to the LASR optimizer.

## Implementation sequence and acceptance checks

> **Status (2026-09-27):** Step 1 has working async contracts, jammers, a real
> model CLI, and native APPS honeypot validation. Offline regressions and a
> two-sample Docker check passed using deterministic model doubles. Step 2's
> recorded LinuxArena fixture works through the shared summary policy; live
> Linux execution is pending (no local environments or LinuxArena virtualenv).
> Steps 3–5 remain pending, including SFT export integration. See
> `src/lasr_labs_2025_control_project/blue_team/USAGE.md` for exact commands and
> limitations. No live-model quality or SFT results are claimed.

1. **Implement the contracts with APPS examples.** Wrap existing minification
   and summary code, compose one scoring policy, and implement one pipeline
   that inserts a T attack into benign U code. Reuse existing Inspect tasks
   and scorers. Leave one small runnable check that verifies the policy sees
   no hidden labels and a jammer does not mutate the executable artifact.
2. **Stress-test LinuxArena immediately.** Adapt one recorded trajectory to
   the permitted view and run identity and T-summary jamming through the same
   policy. Then run one LinuxArena task using its native runner and consume
   the resulting trajectory. This tests scoring integration; it does not yet
   establish live interception or safe trajectory modification.
3. **Test the harder boundary before freezing the API.** Build one synthetic
   LinuxArena honeypot at a known sandbox state and validate it with the native
   task/side-task checks. An edited command transcript alone is insufficient.
   Exercise live monitoring separately if the intended policy intervenes
   before execution. Record any arena-specific branches forced into the
   policy, changes needed to port a jammer, missing state, or leaked labels.
   Revise the contracts using those concrete failures.
4. **Feed the working abstractions into the SFT/plot pipeline.** Keep `none`,
   minification, summary, and code rewrite by T as distinct arms; the existing
   `llm`/“paraphrased” arm is a summary. Transform both train and evaluation
   monitor inputs, retain source IDs and provenance, and use problem-disjoint
   splits. Existing SFT/plot runs can continue while the adapter is developed.
5. **Expand after those checks.** Add BigCodeBench and BashArena adapters using
   their native scoring/execution. Package the interface once the APPS and
   LinuxArena examples establish which parts are shared.

For monitor-only jamming, task/attack scores continue to describe the original
executed artifact. The current transform task copies those scores, which is
appropriate for that interpretation. If transformed code will be executed,
re-run task and attack checks and store separate scores. If a monitor-only
code rewrite claims to preserve behavior, test that claim as a diagnostic;
do not silently filter away transforms that erased attack evidence. Summaries
need coverage/faithfulness assessment rather than executable-code tests.

Success means a new jammer can be plugged into both supported monitor views
without editing either arena's runner, and arena-specific execution/validation
stays in the adapters. Carry paired scores, transform failures, and sample
counts into the existing analysis; use LASR curves only for compatible policies.

## Review of the proposed architecture

- **Keep one repo for now**. Retain the existing `finetuning` and `experiments`
  code, and add a small arena package when its first callback is implemented.
  Publish a checkpoint manifest. These parts share model recipes, sample
  schemas, and the score pipeline. Split repos when external users need an
  independently versioned install or training dependencies become a practical
  barrier.
- **Introduce the three contracts now and keep them provisional.** Named
  `HoneypotPipeline`, `SignalJammer`, and `BlueTeamPolicy` interfaces make the
  portability experiment concrete. Start with direct Python composition;
  add discovery/registration only if selecting user implementations needs it.
  Avoid building an inheritance hierarchy before the two-arena checks.
- **Do not claim the resource facade is an access-control sandbox.** With an
  open checkpoint, enforceable secrecy is limited to the official held-out
  attack corpus and evaluator service. If untrusted third-party code must be
  isolated, execute it in a separate process/container with only the exposed
  API and no dataset or model-service credentials.
- **Keep a private final test split.** Repeated public feedback about held-out
  attacks would let users tune to that set even without seeing its rows.

## Source pointers

- `src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py`
  and `utils/paraphrasers/paraphrasers.py`: current minifier and summary.
- `src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py`
  and `experiments/open_models/make_configs.py`: monitor view and arm wiring.
- `experiments/open_models/SPEC.md`: train/eval transformations and SFT steps.
- `src/lasr_labs_2025_control_project/scripts/plot_data.py` and
  `docs/dependent_monitors_stats/SPEC.md`: current statistical evaluation.
- `/home/jkk/control-tower/src/control_tower/protocols/eval2/{blue.py,game_view.py}`:
  narrow view and harness-driven hooks.
- `/home/jkk/linuxarena/src/protocols/monitoring/types.py` and
  `src/lb_runs/monitor/task.py`: recorded action/trajectory monitor callbacks.
- `/home/jkk/linuxarena/src/protocols/policies/solver.py`: native task execution.
