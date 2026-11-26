"""Full backup of one W&B project, with what is needed to recreate it.

Per run (<out>/runs/<run id>/):
  run.json        id, name, state, group, job type, tags, notes, config, summary, created_at, user
  history.jsonl   every logged row (scan_history, with _step)
  system.jsonl    system-metrics stream
  files/          the run's files (logs, media, tables, config.yaml, ...)
  logged.json     every artifact version the run logged: type, version, aliases,
                  description, metadata, digest and full manifest (refs included)
  used.json       every artifact version the run used (input lineage, any project)
  artifacts/<name>_<version>/  the logged files, MD5-checked against the manifest
                  (entries that only reference external URLs are recorded, not downloaded)
Project: <out>/views.json (saved workspaces and reports).

Re-run to resume: runs with a DONE marker are skipped (running runs are always refetched).

usage: python backup_full.py ENTITY/PROJECT OUT_DIR
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import wandb

path, out = sys.argv[1], Path(sys.argv[2]).resolve()
entity, project = path.split("/")
api = wandb.Api(timeout=120)
VIEWS = """
query V($entityName: String, $name: String, $viewType: String) {
  project(name: $name, entityName: $entityName) {
    allViews(viewType: $viewType) {
      edges { node { id name displayName type description spec createdAt updatedAt user { username } } }
    }
  }
}"""


def md5(p: Path) -> str:
    return base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()


def describe(a) -> dict:
    return dict(
        qualified=a.qualified_name,
        name=a.name,
        type=a.type,
        version=a.version,
        aliases=a.aliases,
        description=a.description,
        metadata=a.metadata,
        digest=a.digest,
        size=a.size,
        created_at=a.created_at,
        manifest={
            k: dict(digest=e.digest, ref=e.ref, size=e.size, extra=e.extra)
            for k, e in a.manifest.entries.items()
        },
    )


out.mkdir(parents=True, exist_ok=True)
views = []
for view_type in ("project-view", "runs", "runs/draft"):
    r = api.__dict__["_service_api"].execute_graphql(
        VIEWS, variables={"entityName": entity, "name": project, "viewType": view_type}
    )
    views += [e["node"] for e in r["project"]["allViews"]["edges"]]
(out / "views.json").write_text(json.dumps(views, indent=1))
print("views", len(views), flush=True)

bad = []
for r in api.runs(path, per_page=200):
    d = out / "runs" / r.id
    if (d / "DONE").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(
        id=r.id,
        name=r.name,
        state=r.state,
        group=r.group,
        job_type=r.job_type,
        tags=r.tags,
        notes=r.notes,
        created_at=r.created_at,
        url=r.url,
        user=getattr(r.user, "username", None),
        config=r.config,
        summary=r.summary._json_dict,
    )
    (d / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    rows = 0
    with open(d / "history.jsonl", "w") as f:
        for row in r.scan_history():
            f.write(json.dumps(row, default=str) + "\n")
            rows += 1
    with open(d / "system.jsonl", "w") as f:
        for row in r.history(stream="system", pandas=False, samples=1_000_000):
            f.write(json.dumps(row, default=str) + "\n")
    for fl in r.files():
        fl.download(root=str(d / "files"), replace=True, exist_ok=True)
    logged, run_bad = [], len(bad)
    for a in r.logged_artifacts():
        info = describe(a)
        stored = {k: m for k, m in info["manifest"].items() if not m["ref"]}
        if stored:
            folder = d / "artifacts" / f"{a.name.split(':')[0]}_{a.version}"
            for k in stored:
                a.get_entry(k).download(root=str(folder))
            for k, m in stored.items():
                if md5(folder / k) != m["digest"]:
                    bad.append(f"{r.id} {a.name} {k}")
        logged.append(info)
    (d / "logged.json").write_text(json.dumps(logged, indent=1, default=str))
    used = [dict(qualified=a.qualified_name, type=a.type) for a in r.used_artifacts()]
    (d / "used.json").write_text(json.dumps(used, indent=1))
    if r.state != "running" and len(bad) == run_bad:  # a mismatch is refetched on rerun
        (d / "DONE").write_text(
            f"{rows} history rows, {len(logged)} logged, {len(used)} used\n"
        )
    print(
        r.id, r.name, rows, "rows", len(logged), "logged", len(used), "used", flush=True
    )
print("MD5 mismatches:", bad or "none")
print("ALL DONE" if not bad else "INCOMPLETE (rerun to refetch)", path)
sys.exit(1 if bad else 0)
