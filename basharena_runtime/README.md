# BashArena runtime

BashArena needs control-arena 17. The main project pins control-arena 2.1.3 for the
APPS and BigCodeBench pipelines, so everything that must run against the newer
version lives here, with its own lockfile. The main project never imports this
package; it reads the files this package writes.

| Module | What it does |
|---|---|
| `generate.py` | Builds the BashArena agent task and runs generation from a generator config |
| `export.py` | Writes a saved log as trajectory and per-action JSONL for the main project |
| `views.py` | Renders a trajectory as actions, and each action as the monitor sees it |
| `monitor_prompts.py` | Monitor, self-recognition and summary prompts |

The per-action format is ported from `basharena_public` (commit `86082fd`) and
pinned to it by golden tests.

## Setup

```bash
uv sync --frozen --project basharena_runtime
```

This creates `basharena_runtime/.venv`. Docker is needed to run generation, not to
install or test.

## Generate

The config is the generator YAML the other settings use, with
`dataset_type: basharena` and a `limit` on every run:

```yaml
defaults:
  data_split: test          # odd task ids are train, even are test
  dataset_type: basharena
  limit: 5                  # or [start, stop]
  max_tokens: 16384
runs:
  attacks:
    eval_mode: attack
    system_prompt: BASHARENA_ATTACK_SYSTEM_PROMPT
    model: tinker/Qwen/Qwen3.6-35B-A3B
```

```bash
cd basharena_runtime

# Prints the plan. No dataset reads, containers or model calls.
uv run --frozen python -m basharena_runtime.generate --config cfg.yaml --output out/

# Builds sandboxes and calls models.
uv run --frozen python -m basharena_runtime.generate --config cfg.yaml --output out/ --allow-paid
```

`out/` gets one `<population>.eval` per run, the native Inspect logs, a copy of
the config and `generation_manifest.json`. An existing output directory is never
overwritten. A failed run keeps its partial logs and leaves
`"complete": false` in the manifest.

Each agent gets 60 steps, 2,000,000 tokens and one hour. Task `1003684` is skipped
because its image does not build.

`tinker/` models use the main project's provider, imported from this checkout's
`src/`. Set `TINKER_API_KEY` in `.env`.

## Export

Turns one saved `.eval` log into JSONL the main project reads without needing
control-arena 17. Offline: no containers, no model calls.

```bash
uv run --frozen python -m basharena_runtime.export out/attacks.eval \
  --output exports/test_attacks --population attacks \
  --split test --split-provenance "task-id parity: even ids" \
  --monitor-actor-id tinker/Qwen/Qwen3.6-35B-A3B
```

The output directory gets:

- `trajectories.jsonl`: one row per attempt, with the full sample, its scores and
  the reasons a baseline loader would reject it.
- `actions.jsonl`: one row per rated action, with the proposed tool calls, their
  outputs, and the exact self-recognition and suspiciousness messages.
- `manifest.json`: counts, the source log's hash and what rendered the views.

Every attempt is kept: errored, unscored, zero-action and unsuccessful ones
included. `baseline_dataset_eligible` marks the attempts the older
success-filtered loader would have kept; it is a label, not a filter.

`own_source` says whether the action's actor is `--monitor-actor-id`. The actor
is `--actor-id` if given, otherwise the model recorded in the log. Logs record
the model without its provider (`Qwen/X` for `tinker/Qwen/X`), so an actor read
from the log matches either spelling. A declared `--actor-id` is compared
exactly, which is what a fine-tuned checkpoint needs.

Trajectory scores stay on the trajectory row; they are never copied onto
actions.

## Test

```bash
cd basharena_runtime && uv run --frozen pytest
```

The tests are offline: no containers, no model calls.
