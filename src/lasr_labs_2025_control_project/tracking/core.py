"""W&B run lifecycle, input lineage, and eval-log bookkeeping for pipeline stages.

See the package docstring (``tracking/__init__.py``) for the design overview.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import logging
import math
import os
import re
import secrets
import string
from collections import defaultdict
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from lasr_labs_2025_control_project.tracking import names

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "untrusted-monitoring"
REF_PREFIX = "wandb:"
HEADER_KEY = "tracking"  # key in an .eval log's eval.metadata
PIPELINE_KEY = "pipeline"  # the producing pipeline node, in the same metadata
ALLOW_UNLINKED_HELP = (
    "Continue even if links to tracked inputs cannot be recorded in W&B "
    "(W&B unreachable, or an input came from a failed or unsynced run)."
)
_EXACT_VERSION = re.compile(r":v\d+$")
_INIT_TIMEOUT_SECONDS = 30


class UnlinkedInputError(RuntimeError):
    """Raised when a tracked input cannot be linked and --allow-unlinked is unset."""


class StageFailed(RuntimeError):
    """An Inspect run ended without success; its log is kept where it was saved."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def mode() -> str:
    """'disabled', 'offline' or 'online', from WANDB_MODE (wandb's own variable)."""
    value = os.environ.get("WANDB_MODE", "online").strip().lower()
    if value in ("disabled", "off"):
        return "disabled"
    if value in ("offline", "dryrun"):
        return "offline"
    return "online"


def available() -> bool:
    """Whether the W&B API is reachable (online, with a key)."""
    return _api() is not None


def is_ref(path: str | Path) -> bool:
    return str(path).startswith(REF_PREFIX)


def jsonable(obj: Any) -> Any:
    """Dataclasses / enums / paths / numpy scalars -> plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item) and not isinstance(obj, type):  # numpy scalar
        return item()
    return obj


def prompt_digest(text: str) -> str:
    """Short digest of resolved prompt text, to spot edits behind a stable name."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _md5_b64(path: Path) -> str:
    """Base64 MD5 of a file: how W&B identifies an artifact's files."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _has_api_key() -> bool:
    """Whether W&B credentials are configured (``wandb.Api`` refuses without)."""
    import wandb
    from wandb.errors import UsageError

    try:
        wandb.Api(timeout=_INIT_TIMEOUT_SECONDS)
    except UsageError:
        return False
    return True


def _run_id() -> str:
    return "".join(
        secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8)
    )


def _project() -> str:
    return os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT


# ---------------------------------------------------------------------------
# W&B API access (lazy; None when there is no key or no connection)
# ---------------------------------------------------------------------------

_api_cache: dict[str, Any] = {}


def _api() -> Any:
    if "api" in _api_cache:
        return _api_cache["api"]
    api = None
    if mode() == "online":
        try:
            import wandb

            if _has_api_key():
                api = wandb.Api(timeout=_INIT_TIMEOUT_SECONDS)
                api.viewer  # forces a round-trip, so "unreachable" shows up here
        except Exception as exc:  # network, auth, server errors
            logger.warning("W&B API unavailable: %s", exc)
            api = None
    if api is not None:
        _api_cache["api"] = api
    return api


def _entity(api: Any) -> str:
    return os.environ.get("WANDB_ENTITY") or api.default_entity


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResolvedInput:
    """A stage input: where it is on disk and how it links into W&B lineage.

    ``artifact`` is set when the exact upstream artifact version is known.
    ``tracked`` means the input claims a tracked producer (via its .eval header
    or a ``wandb:`` ref); a tracked input without an artifact is *unlinked*,
    and ``note`` says why.
    """

    source: str
    local_path: Path
    name: Optional[str]
    md5: str
    tracked: bool = False
    artifact: Any = None
    note: Optional[str] = None

    @property
    def unlinked(self) -> bool:
        return self.tracked and self.artifact is None

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "md5": self.md5,
            "artifact": self.artifact.qualified_name if self.artifact else None,
            "note": self.note,
        }


def read_header(path: str | Path) -> dict[str, Any]:
    """The tracking block an upstream stage wrote into an .eval header ({} if none)."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path), header_only=True)
    return dict((log.eval.metadata or {}).get(HEADER_KEY) or {})


def resolve_input(path_or_ref: str | Path) -> ResolvedInput:
    """Resolve a local path or ``wandb:<entity>/<project>/<name>:vN`` ref.

    Refs are downloaded (W&B's artifact cache) and link exactly. Local .eval
    files link through the producer info in their header; files without one
    are recorded by path and hash.
    """
    source = str(path_or_ref)
    if is_ref(source):
        ref = source[len(REF_PREFIX) :]
        if not _EXACT_VERSION.search(ref):
            raise ValueError(
                f"{source}: W&B refs must name an exact version (…:vN); "
                "aliases such as :latest are not accepted"
            )
        api = _api()
        if api is None:
            raise RuntimeError(f"cannot download {source}: W&B is unavailable")
        artifact = api.artifact(ref)
        files = sorted(p for p in Path(artifact.download()).rglob("*") if p.is_file())
        if len(files) != 1:
            raise ValueError(f"{source}: expected exactly one file, found {len(files)}")
        return ResolvedInput(
            source=source,
            local_path=files[0],
            name=artifact.metadata.get("name") or artifact.name.split(":")[0],
            md5=_md5_b64(files[0]),
            tracked=True,
            artifact=artifact,
        )

    local = Path(source)
    resolved = ResolvedInput(
        source=source, local_path=local, name=None, md5=_md5_b64(local)
    )
    header = read_header(local) if local.suffix == ".eval" else {}
    resolved.name = header.get("name")
    if not header.get("run_id"):
        return resolved  # produced without tracking: recorded by path + hash
    resolved.tracked = True
    api = _api()
    if api is None:
        resolved.note = "W&B is unavailable, so the producing run can't be looked up"
        return resolved
    run_path = "/".join(
        [header.get("entity") or _entity(api), header["project"], header["run_id"]]
    )
    try:
        run = api.run(run_path)
        candidates = list(run.logged_artifacts())
    except Exception:
        resolved.note = (
            f"producing run {run_path} is not in W&B (if it ran offline, "
            "`wandb sync` it first)"
        )
        return resolved
    for artifact in candidates:
        entry_digests = {e.digest for e in artifact.manifest.entries.values()}
        if (
            artifact.name.split(":")[0] == header.get("name")
            and resolved.md5 in entry_digests
        ):
            resolved.artifact = artifact
            return resolved
    resolved.note = (
        f"producing run {run_path} logged no artifact matching this file "
        "(the run failed or was partial, or the file was changed afterwards)"
    )
    return resolved


def artifact_input(artifact: Any) -> ResolvedInput:
    """A logged (committed) artifact as an input to declare on another run."""
    return ResolvedInput(
        source=f"{REF_PREFIX}{artifact.qualified_name}",
        local_path=Path(artifact.name),
        name=artifact.name.split(":")[0],
        md5="",
        tracked=True,
        artifact=artifact,
    )


def is_finetuned_model(model: str) -> bool:
    return "tinker://" in model or "ft:" in model


def resolve_model(model: str) -> Any:
    """The model-reference artifact a finetune run logged for ``model``, or None."""
    if not is_finetuned_model(model):
        return None
    api = _api()
    if api is None:
        return None
    name = names.model_ref_name(model)
    try:
        versions = api.artifacts("model", f"{_entity(api)}/{_project()}/{name}")
        return next((a for a in versions if a.metadata.get("model_id") == model), None)
    except Exception:
        return None  # never trained under tracking; recorded by id only


# ---------------------------------------------------------------------------
# eval-log bookkeeping
# ---------------------------------------------------------------------------


def _score_rates(samples: Sequence[Any]) -> dict[str, float]:
    """Mean of numeric scores and the C-rate of C/I scores, per scorer field."""
    values: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for scorer, score in (sample.scores or {}).items():
            value = getattr(score, "value", score)
            fields = value.items() if isinstance(value, dict) else [(None, value)]
            for field, v in fields:
                key = f"{scorer}/{field}" if field else scorer
                if v in ("C", "I"):
                    values[key].append(1.0 if v == "C" else 0.0)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not math.isnan(float(v)):
                        values[key].append(float(v))
    return {k: sum(v) / len(v) for k, v in values.items() if v}


def eval_counts(log: Any, limit: Optional[int]) -> dict[str, Any]:
    """Count trail and completeness for a finished Inspect run.

    Complete means the run succeeded and produced every sample it was asked
    for: min(limit, dataset size). The dataset size is whatever the stage's
    dataset filters left (split / id range / validity of upstream samples).
    """
    from lasr_labs_2025_control_project.utils.utils import check_sample_is_valid

    samples = list(log.samples or [])
    dataset_samples = log.eval.dataset.samples
    expected = (
        dataset_samples
        if limit is None or dataset_samples is None
        else min(limit, dataset_samples)
    )
    counts = {
        "status": log.status,
        "dataset_samples": dataset_samples,
        "limit": limit,
        "expected_samples": expected,
        "samples": len(samples),
        "errored_samples": sum(1 for s in samples if getattr(s, "error", None)),
        "valid_samples": sum(1 for s in samples if check_sample_is_valid(s.scores)),
    }
    counts["complete"] = log.status == "success" and len(samples) == expected
    return counts


def input_counts(input_path: Path, dataset_size: int) -> dict[str, int]:
    """Upstream samples in, and how many the validity filter dropped."""
    from inspect_ai.log import read_eval_log

    header = read_eval_log(str(input_path), header_only=True)
    total = header.results.total_samples if header.results else dataset_size
    return {
        "input_samples": total,
        "input_dropped_invalid": total - dataset_size,
    }


def header_complete(header: Any, limit: Optional[int]) -> bool:
    """``eval_counts``' completeness rule, from an .eval header alone."""
    results = header.results
    dataset_samples = header.eval.dataset.samples
    if header.status != "success" or results is None or dataset_samples is None:
        return False
    expected = dataset_samples if limit is None else min(limit, dataset_samples)
    return results.completed_samples == expected


@dataclasses.dataclass
class PipelineContext:
    """The pipeline invocation a stage run belongs to.

    Every stage run of one invocation logs the merged config in its run config
    and uses the same ``config`` artifact (identical files are one version).
    """

    name: str
    config: Mapping[str, Any]
    config_files: Sequence[Path] = ()
    tags: Sequence[str] = ()


def find_pipeline_run(node_hash: str, key: Mapping[str, Any]) -> Any:
    """A complete W&B run that produced the pipeline node with this hash, or None.

    The run's stored key must equal ``key`` exactly; a mismatch is a hash
    collision and raises rather than reusing the wrong output.
    """
    api = _api()
    if api is None:
        raise RuntimeError("reuse=wandb needs W&B access (WANDB_MODE online and a key)")
    runs = api.runs(
        f"{_entity(api)}/{_project()}",
        filters={"config.pipeline_hash": node_hash, "summary_metrics.complete": True},
        order="-created_at",  # a forced rerun replaces earlier runs of the node
        lazy=False,  # lazy runs (wandb >= 0.22) carry their config as a raw JSON string
    )
    for run in runs:
        stored = (run.config.get("pipeline_node") or {}).get("key")
        if jsonable(stored) != jsonable(key):
            raise RuntimeError(
                f"W&B run {run.path} has pipeline hash {node_hash} but a different "
                "key: a hash collision. Rename or delete one of them."
            )
        return run
    return None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class StageRun:
    """One W&B run for one stage invocation on one label (or one condition).

    Use as a context manager. Inside it:

    * ``eval_metadata()`` goes into ``inspect_ai.eval(metadata=...)`` so the
      output .eval carries its producer;
    * ``finish_eval(log, save_path, limit)`` logs the count trail and, if the
      run is complete, the output file as an artifact;
    * ``log`` / ``summary`` / ``log_artifact`` / ``finish`` for stages that
      are not Inspect runs (finetune, analyse).

    With ``WANDB_MODE=disabled`` every method is a no-op except
    ``eval_metadata``, which still records the output name so downstream
    naming keeps working.
    """

    def __init__(
        self,
        stage: str,
        name: str,
        *,
        config: Mapping[str, Any],
        tags: Iterable[str] = (),
        inputs: Sequence[ResolvedInput] = (),
        models: Iterable[str] = (),
        allow_unlinked: bool = False,
        pipeline: Optional[PipelineContext] = None,
        node: Optional[Mapping[str, Any]] = None,
    ):
        self.stage = stage
        self.name = name
        self.config = dict(config)
        self.pipeline = pipeline
        self.node = dict(node) if node else None
        if pipeline is not None:
            self.config["pipeline"] = pipeline.config
            tags = [*tags, *pipeline.tags]
        if self.node is not None:
            self.config["pipeline_node"] = self.node
            self.config["pipeline_hash"] = self.node["hash"]
        self.tags = list(dict.fromkeys(t for t in tags if t))
        self.inputs = list(inputs)
        self.models = list(models)
        self.allow_unlinked = allow_unlinked
        self.run: Any = None
        self.offline = False
        self._finished = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "StageRun":
        current = mode()
        if current == "disabled":
            return self
        unlinked = [i for i in self.inputs if i.unlinked]
        if unlinked and not self.allow_unlinked:
            raise UnlinkedInputError(
                "Cannot record links to these inputs:\n"
                + "\n".join(f"  {i.source}: {i.note}" for i in unlinked)
                + "\nFix the cause, or pass --allow-unlinked to continue without them."
            )
        if current == "online":
            self.run = self._init("online")
        if self.run is None:
            linkable = [i for i in self.inputs if i.artifact is not None]
            if linkable and not self.allow_unlinked:
                raise UnlinkedInputError(
                    "W&B is offline/unreachable, so links to these inputs can't be "
                    "recorded (and `wandb sync` can't add them later):\n"
                    + "\n".join(f"  {i.source}" for i in linkable)
                    + "\nRestore W&B access, or pass --allow-unlinked to continue."
                )
            if current == "online":
                logger.warning(
                    "\n%s\nW&B IS UNREACHABLE: logging this run OFFLINE. Upload it "
                    "later with `wandb sync wandb/offline-run-*`.\n%s",
                    "!" * 72,
                    "!" * 72,
                )
            self.run = self._init("offline")
            self.offline = True
            for inp in self.inputs:
                inp.artifact = None  # offline runs cannot declare artifact inputs
        self._declare_inputs()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if not self._finished:
            reason = f"{exc_type.__name__}: {exc}" if exc_type else None
            self.finish(complete=exc_type is None, reason=reason)

    def _init(self, init_mode: Literal["online", "offline"]) -> Any:
        import wandb

        if init_mode == "online" and not _has_api_key():
            logger.warning("No W&B API key found (run `wandb login`).")
            return None
        run_id = _run_id()
        try:
            return wandb.init(
                project=_project(),
                job_type=self.stage,
                name=names.run_name(self.name, run_id),
                id=run_id,
                config=jsonable(self.config),
                tags=self.tags,
                mode=init_mode,
                settings=wandb.Settings(init_timeout=_INIT_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            if init_mode == "offline":
                raise
            logger.warning("W&B init failed: %s", exc)
            return None

    def _declare_inputs(self) -> None:
        if self.run is None:
            return
        for inp in self.inputs:
            if inp.artifact is not None:
                self.run.use_artifact(inp.artifact)
        models_linked = {}
        for model in self.models:
            artifact = None if self.offline else resolve_model(model)
            if artifact is not None:
                self.run.use_artifact(artifact)
            models_linked[model] = artifact.qualified_name if artifact else None
        if (
            self.pipeline is not None
            and self.pipeline.config_files
            and not self.offline
        ):
            import wandb

            config = wandb.Artifact(
                names.fit(f"config-{names.clean(self.pipeline.name)}"), type="config"
            )
            for path in self.pipeline.config_files:
                config.add_file(str(path))
            self.run.use_artifact(config)
        self.run.config.update(
            {
                "inputs": [i.record() for i in self.inputs],
                "finetuned_models": {
                    m: v for m, v in models_linked.items() if is_finetuned_model(m)
                },
            },
            allow_val_change=True,
        )

    # -- logging -------------------------------------------------------------

    def eval_metadata(self) -> dict[str, Any]:
        run = self.run
        metadata: dict[str, Any] = {
            HEADER_KEY: {
                "name": self.name,
                "stage": self.stage,
                "run_id": run.id if run else None,
                "project": run.project if run else None,
                "entity": (run.entity or None) if run else None,
            }
        }
        if self.node is not None:
            metadata[PIPELINE_KEY] = self.node
        return metadata

    def log(self, data: Mapping[str, Any]) -> None:
        if self.run is not None:
            self.run.log(jsonable(data))

    def summary(self, data: Mapping[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(jsonable(data))

    def define_curve(self, y: str, x: str) -> None:
        """Plot metric ``y`` against logged metric ``x`` (W&B's step_metric)."""
        if self.run is not None:
            self.run.define_metric(x, hidden=True)
            self.run.define_metric(y, step_metric=x)

    def log_artifact(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        type: str,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        wait: bool = False,
    ) -> Any:
        """Log files as an artifact; returns it (None when not logging).

        With ``wait``, block until it is committed, so another run can then
        declare it as an input (``artifact_input``).
        """
        if self.run is None:
            return None
        import wandb

        artifact_name = names.fit(name or self.name)
        pipeline_hash = {"pipeline_hash": self.node["hash"]} if self.node else {}
        artifact = wandb.Artifact(
            artifact_name,
            type=type,
            metadata=jsonable(
                {"name": artifact_name, **pipeline_hash, **(metadata or {})}
            ),
        )
        for file in [path] if isinstance(path, (str, Path)) else path:
            artifact.add_file(str(file))
        logged = self.run.log_artifact(artifact)
        if wait and not self.offline:
            logged.wait()
        return logged

    def log_roc(
        self, key: str, curves: Mapping[str, Sequence[tuple[float, float]]], title: str
    ) -> None:
        """ROC curves as one W&B ROC chart, drawn from (fpr, tpr) points.

        A table chart keeps every point; a metric plotted against another
        metric keeps one point per x value, which turns an empirical ROC's
        vertical steps into diagonals.
        """
        if self.run is None:
            return
        import wandb
        from wandb.plot import plot_table

        rows: list[Any] = [
            [label, float(fpr), float(tpr)]
            for label, points in curves.items()
            for fpr, tpr in sorted(points)
        ]
        table = wandb.Table(columns=["class", "fpr", "tpr"], data=rows)
        self.run.log(
            {
                key: plot_table(
                    vega_spec_name="wandb/area-under-curve/v0",
                    data_table=table,
                    fields={"x": "fpr", "y": "tpr", "class": "class"},
                    string_fields={
                        "title": title,
                        "x-axis-title": "False positive rate",
                        "y-axis-title": "True positive rate",
                    },
                )
            }
        )

    def log_model_ref(
        self, model_id: str, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Record a fine-tuned model id (as monitors and the pipeline use it)."""
        if self.run is None:
            return
        import json
        import tempfile

        record = {"model_id": model_id, **(metadata or {})}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(jsonable(record), indent=2))
            self.log_artifact(
                path, type="model", name=names.model_ref_name(model_id), metadata=record
            )

    def log_image(self, key: str, path: str | Path) -> None:
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Image(str(path))})

    def finish_eval(
        self,
        log: Any,
        save_path: Path,
        *,
        limit: Optional[int],
        metadata: Optional[Mapping[str, Any]] = None,
        extra_counts: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record the count trail; log the output .eval only if the run is complete.

        Raises StageFailed (after recording) if Inspect reports no success.
        """
        counts = {**eval_counts(log, limit), **(extra_counts or {})}
        rates = {f"scores/{k}": v for k, v in _score_rates(log.samples or []).items()}
        self.summary({**counts, **rates})
        if counts["complete"]:
            self.log_artifact(
                save_path,
                type="eval-log",
                metadata={**(metadata or {}), **counts},
            )
            self.finish(complete=True)
        else:
            self.finish(
                complete=False,
                reason=f"status={counts['status']}, "
                f"{counts['samples']}/{counts['expected_samples']} samples",
            )
        if log.status != "success":
            raise StageFailed(
                f"{self.name}: eval ended with status {log.status!r}; "
                f"log saved to {save_path}"
            )
        return counts

    def finish(self, *, complete: bool, reason: Optional[str] = None) -> None:
        self._finished = True
        if self.run is None:
            return
        self.run.summary["complete"] = complete
        if not complete:
            self.run.tags = tuple(self.run.tags or ()) + ("partial",)
            if reason:
                self.run.summary["failure"] = reason
        self.run.finish(exit_code=0 if complete else 1)
        self.run = None
