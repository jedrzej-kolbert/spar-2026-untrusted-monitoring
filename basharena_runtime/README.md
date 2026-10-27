# BashArena runtime

BashArena needs control-arena 17. The main project pins control-arena 2.1.3 for the
APPS and BigCodeBench pipelines, so generation, the one step that must run against
the newer version, lives here with its own lockfile.

`generate.py` builds the BashArena agent task and runs it from a generator config.
It writes ordinary `.eval` logs, which the main project's SFT and monitor steps read
directly: see the BashArena section of the
root `README.md`. The main project never imports
this package.

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

`tinker/` models use the main project's provider, and the trajectory view stored in
each log comes from the main project's `utils/basharena_views.py`; both are imported
from this checkout's `src/`. Set `TINKER_API_KEY` in `.env`.

## Test

```bash
cd basharena_runtime && uv run --frozen pytest
```

The tests are offline: no containers, no model calls.
