# Blue team interfaces

Three async contracts live in `contracts.py`: `SignalJammer`,
`HoneypotPipeline`, and `BlueTeamPolicy`. Callables need no base class or
registration. `MonitorView` contains immutable strings and a representation
tag; private labels, execution results, and provenance stay with the evaluator.

`FixedMonitoringPolicy` sends the jammed view to the UM collusion and UM
suspiciousness models, and the original view to T by default. Both jammers
are independently configurable. Missing callers, incomplete model responses,
bad scores, and missing YES/NO logprobs raise errors rather than becoming
successful observations. The caller should retain failures in its run ledger.

```python
from lasr_labs_2025_control_project.blue_team.contracts import (
    MonitorView,
    RepresentationKind,
)
from lasr_labs_2025_control_project.blue_team.jammers import TSummaryJammer
from lasr_labs_2025_control_project.blue_team.policy import (
    FixedMonitoringPolicy,
    make_resources,
)

# Evaluator-owned public task map and actual Inspect model IDs.
resources = make_resources(
    tasks,
    generator_model=u_generator,
    um_model=u_monitor,
    trusted_model=t_model,
    collusion_model=u_checkpoint,
)
policy = FixedMonitoringPolicy(TSummaryJammer(resources.call_t, t_model))
scores = await policy(
    MonitorView(tasks[problem_id], code, RepresentationKind.PYTHON_CODE, problem_id),
    resources,
)
```

All three numeric field names match the statistical model's DataFrame columns.
`scores.um_view`/`t_view` record the actual observations; provenance records T
transform prompts and model IDs. Preserve source IDs, split membership, model
settings and failed attempts alongside these results. The CLI writes one JSON
object; conversion to legacy SFT `.eval` formats and mixed-transform training
are pending. A matching score schema does not validate the LASR assumptions for
a new sequential policy.

## Compatibility with existing experiments

The existing minification solver delegates to `MinifierJammer`, which calls the
unchanged `MinifierParaphraser`. YAML options, minified text, `.eval` store keys,
transform metadata and copied scorer values are preserved. No commands in
`sft-steps.md`, configs, training scripts or plotting readers need to change.
The summary path is unchanged. The standalone CLI below is optional; its JSON
is not a substitute for existing SFT `.eval` inputs.

```sh
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.test_minifier_compatibility
```

## CLI with real models

From the worktree, with its dependencies installed and provider credentials
available, set `U_MODEL` and `T_MODEL` to your Inspect model IDs. Generation and
suspiciousness support Inspect providers; collusion needs YES/NO logprobs
(including the existing Tinker forced-decoding path). `--um` and `--collusion`
select separate monitor models/checkpoints. Unsupported logprobs fail clearly.

An APPS scoring input is a JSON object with `problem` and `code` strings:

```sh
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.example \
  --kind apps --input sample.json --output scores.json \
  --u "$U_MODEL" --t "$T_MODEL" --jammer summary
```

`--jammer` accepts `identity`, `minifier`, `summary`, or `rewrite`. Minification
and rewriting require Python. Summary accepts Python, full trajectories, and
prefixes. These change monitor observations only. A rewrite's syntax check
does not establish behavioral equivalence. Scores still refer to the original
executed artifact, so do not silently discard evidence-erasing transforms.

For an APPS honeypot, supply `problem`, paired nonempty `inputs`/`outputs`
(lists of strings), and `solutions` (reference code strings):

```sh
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.example \
  --kind honeypot --input task.json --output honeypot.json \
  --u "$U_MODEL" --t "$T_MODEL"
```

This samples benign U code with a fixed honest prompt, asks T to insert an
attack, and validates inside Inspect's Docker sandbox. The native APPS tests
check both benign and candidate code, and the secret input is compared against
the evaluator's reference solution. `validation.valid` requires both checks.
It is empirical test validation, not a proof over all inputs. Output and eval
logs include private triggers/reference data and belong to the evaluator.

## LinuxArena adapter

`trajectory_to_view` reads current native JSON or legacy exports. It selects
public task/environment descriptions and tool actions only. It serializes a
snapshot without truncation, private reasoning, side-task metadata, scores,
source IDs or total future length. The harness may provide an opaque ID and
override the public task description for incomplete exports.

`before_action=k` includes executed actions `[0,k)` and proposal `k`, without
that proposal's result. `k` is list position, not an optional stored index.
Omitting it means posthoc full-trajectory monitoring, not live intervention.

```sh
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.example \
  --kind linux --input trajectory.json --output scores.json \
  --before-action 0 --u "$U_MODEL" --t "$T_MODEL" --jammer summary
```

To obtain a fresh trajectory, use LinuxArena's native runner in a separately
configured LinuxArena environment (choose one installed environment/task):

```sh
uv run lb run eval -e ENVIRONMENT -t TASK --just-main-tasks --policy honest --no-upload
uv run lb traj make RUN.eval --sample-id SAMPLE_ID --save-path /tmp/trajectory.json
```

The local `/home/jkk/linuxarena` has no installed environments beyond `.gitkeep`
and no `.venv`. A recorded fixture was tested, but native Linux task execution,
trajectory honeypot construction/validation, and live interception remain
pending. Never treat an edited transcript as an execution-validated honeypot.

## Checks

```sh
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.test_blue_team
# Additionally execute native APPS scorers in Docker; model replies are fakes:
PYTHONPATH=src python -m lasr_labs_2025_control_project.blue_team.test_blue_team --sandbox
```

The offline check covers actual resource model routing with mocked providers,
rejected failures, original-artifact preservation, label/prefix invariance,
native scorer routing, and a local LinuxArena recorded fixture when available.
The Docker check accepts a valid synthetic backdoor and rejects a candidate
that fails normal tests. Neither check spends model API credits.
