"""Human-legible names for W&B artifacts and runs.

Names are for reading at a glance; identity lives in exact artifact versions
(``name:vN``) and run ids. A name is an *identity*: re-running the same thing
logs a new version of the same name, while anything that changes what the
thing *is* (model, prompt, split, label, transform, monitor) changes the name.

Scheme
------
A generation is named::

    <setting>-<model>-<split>-<label>-<prompt>
    apps-qwen3_30b-test-attacks-causal_attack

Each later stage appends one ``.<stage>[-<model>-<prompt>]`` segment to its
input's name, so a name reads left to right as the file's history::

    apps-qwen3_30b-test-attacks-causal_attack.min
    apps-qwen3_30b-test-attacks-causal_attack.para-qwen3_5_4b-llm_summarize
    apps-qwen3_30b-test-attacks-causal_attack.min.mon-qwen3_30b-causal_collusion

Monitor segments name only the collusion monitor (the trusted and untrusted
backdoor monitors are in the run config). Stages that merge several inputs
break the chain: finetune runs are named after their first training input plus
``.ft-<checkpoint name>``, the model reference they log is ``ft-<alias>-<hash>``
(derivable from the model id alone, so monitor runs can find it), and analyse
runs are named after their condition. Run names are the output name plus the
first four characters of the run id.

Shortening
----------
Models keep their last path segment, lowercased, with date/version stamps and
filler tokens dropped (``tinker/Qwen/Qwen3-30B-A3B-Instruct-2507`` ->
``qwen3_30b``). Prompts are lowercased with ``_system_prompt`` / ``_prompt`` /
``_policy`` and the words ``untrusted`` / ``monitor`` removed
(``CAUSAL_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT`` -> ``causal_collusion``).
When the automatic result is unreadable, add an entry to ``MODEL_ALIASES`` or
``PROMPT_ALIASES`` below rather than special-casing the rules.

Allowed characters are ``[a-z0-9_-]`` inside a part, ``-`` between parts and
``.`` between stages (W&B forbids ``/`` and refs use ``:``). Names longer than
W&B's 128-character limit are truncated with a short hash suffix.
"""

from __future__ import annotations

import hashlib
import re

# Exact model string (as written in configs, provider prefix included) -> alias.
MODEL_ALIASES: dict[str, str] = {}

# Exact prompt registry name -> alias.
PROMPT_ALIASES: dict[str, str] = {}

SETTING_ALIASES = {"apps": "apps", "bigcodebench": "bcb"}

MAX_NAME_LENGTH = 128

_FILLER_TOKENS = {"instruct", "chat", "it", "preview", "latest"}
_PROMPT_SUFFIXES = ("_system_prompt", "_prompt", "_policy")
_PROMPT_DROP_WORDS = {"untrusted", "monitor"}


def clean(part: str) -> str:
    """Make one name part safe: lowercase, ``[a-z0-9_]`` only, no runs of ``_``."""
    part = re.sub(r"[^a-z0-9_]+", "_", part.lower())
    return re.sub(r"_+", "_", part).strip("_") or "x"


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def short_model(model: str) -> str:
    """Short alias for a model string (see module docstring)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if "tinker://" in model:
        # tinker://<run>/sampler_weights/<checkpoint name>: the checkpoint name
        # is chosen once by whoever trains it (--checkpoint-name).
        return f"ft_{clean(model.rstrip('/').rsplit('/', 1)[-1])}"
    if "ft:" in model:
        # openai/ft:<base>:<org>:<suffix>:<id>[:ckpt-step-N]
        fields = model.split("ft:", 1)[1].split(":")
        base = short_model(fields[0])
        suffix = clean(fields[2]) if len(fields) > 2 else short_hash(model, 6)
        step = next((f for f in fields[3:] if f.startswith("ckpt-step-")), None)
        alias = f"{base}_ft_{suffix}"
        return f"{alias}_{clean(step)}" if step else alias
    tokens = [t for t in re.split(r"[-_]", model.rsplit("/", 1)[-1].lower()) if t]
    kept: list[str] = []
    for token in tokens:
        if token.isdigit() and len(token) >= 4:
            break  # date/version stamp: drop it and everything after
        if token in _FILLER_TOKENS or re.fullmatch(r"a\d+b", token):
            continue  # filler words and MoE active-parameter counts
        kept.append(token.replace(".", "_"))
    return clean("_".join(kept) or model)


def short_prompt(prompt_name: str) -> str:
    """Short alias for a prompt registry name (see module docstring)."""
    if prompt_name in PROMPT_ALIASES:
        return PROMPT_ALIASES[prompt_name]
    name = prompt_name.lower()
    for suffix in _PROMPT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    words = [w for w in name.split("_") if w and w not in _PROMPT_DROP_WORDS]
    return clean("_".join(words) or prompt_name)


def fit(name: str) -> str:
    """Truncate to W&B's limit, keeping names unique with a hash suffix."""
    if len(name) <= MAX_NAME_LENGTH:
        return name
    suffix = "-" + short_hash(name)
    return name[: MAX_NAME_LENGTH - len(suffix)].rstrip(".-") + suffix


def generation_name(
    setting: str, model: str, split: str, label: str, prompt_name: str
) -> str:
    parts = [
        SETTING_ALIASES.get(setting, clean(setting)),
        short_model(model),
        clean(split),
        clean(label),
        short_prompt(prompt_name),
    ]
    return fit("-".join(parts))


def stage_segment(
    stage: str, model: str | None = None, prompt_name: str | None = None
) -> str:
    parts = [clean(stage)]
    if model:
        parts.append(short_model(model))
    if prompt_name:
        parts.append(short_prompt(prompt_name))
    return "-".join(parts)


def extend(input_name: str, segment: str) -> str:
    return fit(f"{input_name}.{segment}")


def model_ref_name(model_id: str) -> str:
    """Name of the artifact that records a fine-tuned model id."""
    return fit(f"ft-{short_model(model_id)}-{short_hash(model_id)}")


def condition_name(
    setting: str, collusion_strategy: str, transform: str, monitor_arm: str
) -> str:
    parts = [
        SETTING_ALIASES.get(setting, clean(setting)),
        clean(collusion_strategy),
        clean(transform),
        clean(monitor_arm),
    ]
    return fit(".".join(parts))


def run_name(name: str, run_id: str) -> str:
    return f"{name} · {run_id[:4]}"
