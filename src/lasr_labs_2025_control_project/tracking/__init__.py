"""Weights & Biases tracking for the pipeline stages.

Developer notes. Researchers only need the environment variables in the
top-level README; everything below is for people changing the code.

Model
-----
* One W&B run per stage invocation per label (finetune: per training job;
  analyse: per condition). ``job_type`` is the stage. Runs are opened by
  :class:`StageRun`; switching to one run per invocation would only change
  where the stage scripts construct it.
* Local files stay canonical ("mirror mode"). Any stage input may instead be
  ``wandb:<entity>/<project>/<name>:vN``; it is downloaded and linked exactly.
  Aliases (``:latest``) are rejected -- references are always exact versions.
* Lineage travels inside the files: each Inspect stage passes
  :meth:`StageRun.eval_metadata` to ``eval(metadata=...)``, so the output
  .eval header names its producing run and artifact. A downstream stage reads
  that header (:func:`resolve_input`), finds the artifact the producer logged
  with the same name and file digest, and declares it as an input. No sidecar
  files; paths and renames don't matter. Inputs without a header (produced
  with tracking disabled, or before tracking existed) are recorded by path
  and md5 instead.
* Fine-tuned models are recorded by a small ``model`` artifact holding only
  the id (``tinker://...`` or ``ft:...``), named by
  :func:`names.model_ref_name` so monitor runs can find it from the id alone.
* A run is *complete* when Inspect reports success and it produced
  min(limit, dataset size) samples. Only complete runs log their output as an
  artifact; others are tagged ``partial``, finish with exit code 1, and record
  why in ``summary.failure``, so nothing downstream can link to them.
* Every Inspect run records a count trail in its summary (dataset size,
  limit, samples, errored, valid, plus upstream samples dropped by the
  validity filter for transform/monitor), and per-scorer means/C-rates.

Failure modes
-------------
* ``WANDB_MODE=disabled``: no W&B calls; output headers still carry the name.
* W&B unreachable or no API key: the run logs offline with a loud warning
  (``wandb sync`` uploads it later) -- *unless* that would lose a link to a
  tracked input, which offline runs cannot declare and ``wandb sync`` cannot
  add afterwards. Then the stage refuses to start without ``--allow-unlinked``.
* An input whose header names a producer that can't be matched (failed or
  unsynced run, edited file) also needs ``--allow-unlinked``.

Naming conventions live in :mod:`.names`. Runs started by the pipeline
(``lasr_labs_2025_control_project.pipeline``) carry its merged config, the
node's key (``pipeline_node``) and a ``config`` artifact; tags are the
experiment name and the condition (collusion strategy, transform, monitor arm),
label, setting and split.
"""

from lasr_labs_2025_control_project.tracking import names
from lasr_labs_2025_control_project.tracking.core import (
    ALLOW_UNLINKED_HELP,
    PIPELINE_KEY,
    PipelineContext,
    ResolvedInput,
    StageFailed,
    StageRun,
    UnlinkedInputError,
    artifact_input,
    available,
    find_pipeline_run,
    header_complete,
    input_counts,
    is_ref,
    mode,
    prompt_digest,
    read_header,
    resolve_input,
)

__all__ = [
    "ALLOW_UNLINKED_HELP",
    "PIPELINE_KEY",
    "PipelineContext",
    "ResolvedInput",
    "StageFailed",
    "StageRun",
    "UnlinkedInputError",
    "artifact_input",
    "available",
    "find_pipeline_run",
    "header_complete",
    "input_counts",
    "input_name",
    "is_ref",
    "mode",
    "names",
    "prompt_digest",
    "read_header",
    "resolve_input",
]


def input_name(resolved: ResolvedInput, label: str) -> str:
    """The upstream name to extend, with a stable stand-in for untracked files."""
    return resolved.name or f"untracked-{label}-{names.short_hash(resolved.md5)}"
