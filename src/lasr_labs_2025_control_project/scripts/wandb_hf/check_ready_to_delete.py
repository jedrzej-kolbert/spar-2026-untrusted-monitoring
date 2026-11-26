"""Read-only: which stored .eval artifact versions can be deleted from W&B.

A stored (non-reference) `eval-log` version is READY when:
  1. a version in the same collection has `metadata.migrated_from` == its name (the copy);
  2. the copy downloads from Hugging Face (`tracking.download_artifact`, with the HF login)
     and its MD5 equals the original's;
  3. the original's producer run lists the copy among its outputs (pipeline reuse and
     local-.eval lineage go through the producer's logged artifacts);
  4. every run that used the original also uses the copy.
Exact `wandb:…:vN` refs to a deleted original resolve to the copy (`tracking.resolve_input`).
`latest` on an original is reported: move it to the copy before deleting.

usage: python check_ready_to_delete.py ENTITY/PROJECT [--json OUT]
"""

import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import wandb

from lasr_labs_2025_control_project.tracking import download_artifact

project = sys.argv[1]
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
api = wandb.Api(timeout=120)
versions = [
    a
    for c in api.artifact_type("eval-log", project).collections()
    for a in c.artifacts()
]
copies = {
    a.metadata.get("migrated_from"): a
    for a in versions
    if a.metadata.get("migrated_from")
}
originals = [
    a
    for a in versions
    if not a.metadata.get("migrated_from")
    and not any(e.ref for e in a.manifest.entries.values())
]

report = []
with tempfile.TemporaryDirectory() as scratch:
    os.chdir(scratch)  # download_artifact writes to ./artifacts
    for a in originals:
        problems = []
        copy = copies.get(a.qualified_name)
        if copy is None:
            problems.append("not migrated")
        else:
            (entry,) = a.manifest.entries.values()
            try:
                (f,) = [p for p in download_artifact(copy).rglob("*") if p.is_file()]
                md5 = base64.b64encode(hashlib.md5(f.read_bytes()).digest()).decode()
                if md5 != entry.digest:
                    problems.append("HF copy MD5 differs")
                f.unlink()
            except Exception as exc:
                problems.append(f"HF download failed: {exc}")
            producer = a.logged_by()
            if producer and copy.qualified_name not in {
                x.qualified_name for x in producer.logged_artifacts()
            }:
                problems.append(f"producer {producer.id} does not list the copy")
            missing = {r.id for r in a.used_by()} - {r.id for r in copy.used_by()}
            if missing:
                problems.append(f"consumers not linked to the copy: {sorted(missing)}")
        report.append(
            dict(
                original=a.qualified_name,
                copy=copy.qualified_name if copy else None,
                size=a.size,
                latest="latest" in a.aliases,
                problems=problems,
            )
        )
        status = "READY" if not problems else "NOT READY: " + "; ".join(problems)
        print(
            f"{a.name:80s} {status}"
            + ("  (holds `latest`)" if "latest" in a.aliases and not problems else ""),
            flush=True,
        )

ready = [r for r in report if not r["problems"]]
print(
    f"\n{len(ready)}/{len(report)} stored .eval versions ready to delete "
    f"({sum(r['size'] for r in ready) / 1e6:.0f} MB of {sum(r['size'] for r in report) / 1e6:.0f} MB)"
)
if out:
    Path(out).write_text(json.dumps(report, indent=1))
