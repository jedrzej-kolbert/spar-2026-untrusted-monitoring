"""Reading W&B's local run files (``wandb/run-*/``), to re-log a run elsewhere.

A run that logged to one W&B server or project keeps its full record locally
in ``run-<id>.wandb``: a log of protobuf records in W&B's block format (32 KiB
blocks, each record framed by a 7-byte header: checksum, length, type). wandb
no longer ships a Python reader for it, so this is a small one; it only reads
what re-logging needs (history rows and the final summary).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

_BLOCK = 32768
_HEADER = 7
_FILE_HEADER = 7  # ":W&B" + magic + version
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4


@dataclass
class LocalRun:
    path: Path
    #: history rows in logging order, {metric: value}
    history: list[dict[str, Any]] = field(default_factory=list)
    #: the run's last summary values
    summary: dict[str, Any] = field(default_factory=dict)
    #: when the run started (ISO 8601, from the run folder name), or None
    started: Optional[str] = None


def _records(path: Path) -> list[Any]:
    from wandb.proto import wandb_internal_pb2 as pb

    data = path.read_bytes()
    if data[:4] != b":W&B":
        raise ValueError(f"{path} is not a W&B run file")
    records, buffer, pos = [], b"", _FILE_HEADER
    while pos + _HEADER <= len(data):
        left = _BLOCK - pos % _BLOCK
        if left < _HEADER:  # the rest of a block too small for a header is padding
            pos += left
            continue
        _, length, kind = struct.unpack("<IHB", data[pos : pos + _HEADER])
        pos += _HEADER
        chunk = data[pos : pos + length]
        pos += length
        if kind in (_FULL, _FIRST):
            buffer = chunk
        elif kind in (_MIDDLE, _LAST):
            buffer += chunk
        else:
            continue
        if kind in (_FULL, _LAST):
            record = getattr(pb, "Record")()  # generated protobuf class
            record.ParseFromString(buffer)
            records.append(record)
            buffer = b""
    return records


def _items(items: Any) -> dict[str, Any]:
    return {
        (item.key or "/".join(item.nested_key)): json.loads(item.value_json)
        for item in items
    }


def read_local_run(run_dir: Path) -> LocalRun:
    (path,) = run_dir.glob("*.wandb")
    run = LocalRun(path=run_dir)
    for record in _records(path):
        kind = record.WhichOneof("record_type")
        if kind == "history":
            run.history.append(_items(record.history.item))
        elif kind == "summary":
            run.summary.update(_items(record.summary.update))
    # run-YYYYMMDD_HHMMSS-<id>
    stamp = run_dir.name.split("-")[1]
    run.started = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
    return run


def find_local_run(
    pipeline_hash: str, wandb_dir: Path = Path("wandb")
) -> Optional[Path]:
    """The newest local run folder whose config records this pipeline node."""
    found = []
    for config in wandb_dir.glob("run-*/files/config.yaml"):
        text = config.read_text()
        if pipeline_hash not in text:
            continue
        value = (yaml.safe_load(text) or {}).get("pipeline_hash") or {}
        if isinstance(value, dict) and value.get("value") == pipeline_hash:
            found.append(config.parent.parent)
    return max(found, key=lambda p: p.name) if found else None
