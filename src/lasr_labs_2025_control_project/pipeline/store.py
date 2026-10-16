"""Where node outputs live, whether they're done, and fetching them from W&B.

Each node writes one folder, ``<experiments_dir>/<setting>/<stage>/<name>-<hash>/``:

* generate / transform / monitor: ``<label>.eval``; the node's key is in the
  log header (``eval.metadata["pipeline"]``, beside the W&B lineage block);
* finetune: ``model.json``, the model id and the node's key;
* analyse: the analysis files (see ``scripts/plot_data.py``), the key in each
  payload's ``pipeline`` record.

An output counts as done only if it is complete (for .eval files: Inspect
reported success and every expected sample is there) and its stored key equals
the node's key. A stored key that differs under the same hash is a collision
and stops the run; an incomplete output is redone.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lasr_labs_2025_control_project.pipeline.graph import Node, canonical
from lasr_labs_2025_control_project.tracking import (
    PIPELINE_KEY,
    find_pipeline_run,
    header_complete,
)

EVAL_STAGES = ("generate", "transform", "monitor")
MODEL_FILE = "model.json"


class CollisionError(RuntimeError):
    """An output folder holds a different node's output under the same hash."""


@dataclass(frozen=True)
class Output:
    """A node's output: a file or folder, or (finetune) a model id."""

    path: Optional[str] = None
    model: Optional[str] = None


class Store:
    def __init__(self, experiments_dir: str | Path, setting: str):
        self.root = Path(experiments_dir) / setting

    def folder(self, node: Node) -> Path:
        return self.root / node.stage / f"{node.name}-{node.hash}"

    def output_path(self, node: Node) -> Path:
        folder = self.folder(node)
        if node.stage in EVAL_STAGES:
            return folder / f"{node.info['label']}.eval"
        if node.stage == "finetune":
            return folder / MODEL_FILE
        return folder / "analysis_data.json.gz"

    # -- done? ---------------------------------------------------------------

    def done(self, node: Node) -> Optional[Output]:
        """The node's output if it already exists and is complete."""
        if node.external is not None:
            if node.stage == "finetune":
                return Output(model=node.external)
            return Output(path=node.external)
        path = self.output_path(node)
        if not path.exists():
            return None
        stored, complete, output = self._read(node, path)
        if (
            not isinstance(stored, dict)
            or stored.get("hash") != node.hash
            or canonical(stored.get("key")) != canonical(node.key)
        ):
            raise CollisionError(
                f"{path} was not produced by node {node.id} (hash {node.hash}): "
                "its stored key differs. That is a hash collision or a hand-edited "
                "file; inspect it, and delete the folder to recompute."
            )
        return output if complete else None

    def _read(self, node: Node, path: Path) -> tuple[Any, bool, Output]:
        if node.stage in EVAL_STAGES:
            from inspect_ai.log import read_eval_log

            header = read_eval_log(str(path), header_only=True)
            stored = (header.eval.metadata or {}).get(PIPELINE_KEY)
            return (
                stored,
                header_complete(header, node.params.get("limit")),
                Output(path=str(path)),
            )
        if node.stage == "finetune":
            data = json.loads(path.read_text())
            return data.get(PIPELINE_KEY), True, Output(model=data["model"])
        from lasr_labs_2025_control_project.scripts.plot_data import (
            bootstrap_path,
            payload_path,
            read_replicates,
        )
        from lasr_labs_2025_control_project.utils.numerical_pipeline.results_io import (
            load_analysis_payload,
        )

        stored = load_analysis_payload(path).get(PIPELINE_KEY) or {}
        folder = path.parent
        variants = node.params["variants"]
        replicates = node.params["bootstrap"]["replicates"]
        complete = all(
            payload_path(folder, i, v).exists() for i, v in enumerate(variants)
        ) and all(
            replicates == 0
            or len(read_replicates(bootstrap_path(folder, v))) >= replicates
            for v in variants
        )
        return stored, complete, Output(path=str(folder))

    def write_model(self, node: Node, model: str) -> Output:
        path = self.output_path(node)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": model, PIPELINE_KEY: node.record()}, indent=2) + "\n"
        )
        return Output(model=model)

    # -- W&B -----------------------------------------------------------------

    def fetch(self, node: Node) -> Optional[Output]:
        """Download a matching complete output logged to W&B, if there is one.

        The producing run is found by the node's hash and its stored key is
        checked exactly (see ``tracking.find_pipeline_run``). The output is
        copied into this node's folder, so it counts as done from then on and
        downstream stages link to the producer's artifact.
        """
        if node.external is not None:
            return None
        run = find_pipeline_run(node.hash, node.key)
        if run is None:
            return None
        artifacts = list(run.logged_artifacts())
        folder = self.folder(node)
        if node.stage == "finetune":
            # The final model's reference (OpenAI also logs its checkpoints, with a step).
            refs = [
                a for a in artifacts if a.type == "model" and "step" not in a.metadata
            ]
            if not refs:
                return None
            return self.write_model(node, refs[0].metadata["model_id"])
        wanted = "analysis" if node.stage == "analyse" else "eval-log"
        matches = [a for a in artifacts if a.type == wanted]
        if not matches:
            return None
        downloaded = Path(matches[0].download())
        folder.mkdir(parents=True, exist_ok=True)
        if node.stage in EVAL_STAGES:
            (source,) = [p for p in downloaded.rglob("*") if p.is_file()]
            shutil.copy(source, self.output_path(node))
        else:
            for source in downloaded.iterdir():
                shutil.copy(source, folder / source.name)
        return self.done(node)
