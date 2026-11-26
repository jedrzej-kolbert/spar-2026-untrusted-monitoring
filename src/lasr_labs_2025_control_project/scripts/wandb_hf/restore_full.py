"""Recreate a W&B project from a backup_full.py backup.

For each backed-up run, oldest first: a run with the same id, name, group, job type, tags,
notes and config; its history replayed at the original steps (tables and images rebuilt from
the backed-up files); its summary; its other saved files; and every artifact version it
logged (same name, type, description, metadata, aliases; stored files re-uploaded, URL
references re-recorded as is). Then each run's inputs are linked (artifacts from this project
map to their recreated versions; other projects' artifacts are linked as they are), and the
run is closed with its original state (finished, or failed for failed/crashed).

Progress goes to <target>.restored.json in the backup folder; a rerun skips finished runs.
Not recreatable: created_at/runtime timestamps, system metrics charts, the runs' original
console log as a live stream (it is re-uploaded as a file), saved views (views.json is kept
for reference only).

usage: python restore_full.py BACKUP_DIR ENTITY/TARGET_PROJECT [run id ...] [--apply]
       (dry run without --apply; no run ids = all runs)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import wandb
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry

args = [a for a in sys.argv[1:] if a != "--apply"]
apply = "--apply" in sys.argv
backup, target, only = Path(args[0]), args[1], set(args[2:])
entity, project = target.split("/")
state_file = backup / f"{project}.restored.json"
done = (
    json.loads(state_file.read_text())
    if state_file.exists()
    else {"runs": [], "artifacts": {}}
)
SKIP_TYPES = {
    "run_table",
    "job",
    "code",
    "wandb-history",
    "wandb-events",
}  # recreated by the replay, or not ours
AUTO_FILES = {
    "config.yaml",
    "wandb-summary.json",
    "wandb-metadata.json",
    "requirements.txt",
}
SETTINGS = wandb.Settings(
    silent=True,
    console="off",
    save_code=False,
    disable_code=True,
    disable_git=True,
    disable_job_creation=True,
    x_disable_meta=True,
    x_disable_stats=True,
    x_disable_machine_info=True,
    x_save_requirements=False,
)


def save_state():
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(done, indent=1))
    tmp.replace(state_file)


def media(value, files: Path):
    """Rebuild a logged table or image from its backed-up file; None if not media."""
    if not isinstance(value, dict) or "path" not in value:
        return None
    path = files / value["path"]
    if value.get("_type") == "table-file":
        t = json.loads(path.read_text())
        return wandb.Table(columns=t["columns"], data=t["data"])
    if value.get("_type") == "image-file":
        return wandb.Image(str(path))
    return None


def clean(row: dict, files: Path) -> dict:
    out = {}
    for k, v in row.items():
        if k.startswith("_"):
            continue
        m = media(v, files)
        if m is not None:
            out[k] = m
        elif not (isinstance(v, dict) and v.get("_type")):  # other media refs: dropped
            out[k] = v
    return out


runs = []
for d in sorted((backup / "runs").iterdir()):
    meta = json.loads((d / "run.json").read_text())
    if not only or meta["id"] in only:
        runs.append((meta["created_at"], d, meta))
runs.sort(key=lambda r: r[0])
print(f"{len(runs)} runs to restore into {target}" + ("" if apply else " (dry run)"))

one = os.environ.get("RESTORE_ONE") == "1"
if apply and not one:
    # One process per run: W&B carries a run's group into the next run created in the same
    # process when the next one has none (seen even after wandb.teardown()).
    for _, d, meta in runs:
        if meta["id"] not in done["runs"]:
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    __file__,
                    str(backup),
                    target,
                    meta["id"],
                    "--apply",
                ],
                env={**os.environ, "RESTORE_ONE": "1"},
                check=True,
            )
    done = json.loads(state_file.read_text())

for _, d, meta in runs:
    if meta["id"] in done["runs"] or (apply and not one):
        continue
    logged = [
        a
        for a in json.loads((d / "logged.json").read_text())
        if a["type"] not in SKIP_TYPES
    ]
    rows = [json.loads(line) for line in open(d / "history.jsonl")]
    print(
        meta["id"],
        meta["name"],
        len(rows),
        "rows",
        len(logged),
        "artifacts",
        flush=True,
    )
    if not apply:
        continue
    files = d / "files"
    run = wandb.init(
        entity=entity,
        project=project,
        id=meta["id"],
        name=meta["name"],
        group=meta["group"],
        job_type=meta["job_type"],
        tags=meta["tags"],
        notes=meta["notes"],
        config=meta["config"],
        settings=SETTINGS,
    )
    for row in rows:
        step = row.get("_step")
        values = clean(row, files)
        if values:
            run.log(values, step=step)
    run.summary.update(clean(meta["summary"], files))
    for f in files.rglob("*") if files.exists() else []:
        rel = f.relative_to(files)
        # media/ is re-logged with the history; artifact/ names are reserved by W&B (an upload
        # including one is rejected as a batch, media files with it)
        if (
            f.is_file()
            and rel.parts[0] not in ("media", "artifact")
            and str(rel) not in AUTO_FILES
        ):
            run.save(str(f), base_path=str(files), policy="now")
    for a in sorted(logged, key=lambda a: int(a["version"][1:])):
        art = wandb.Artifact(
            a["name"].split(":")[0],
            type=a["type"],
            description=a["description"],
            metadata=a["metadata"],
        )
        folder = d / "artifacts" / f"{a['name'].split(':')[0]}_{a['version']}"
        for path, m in a["manifest"].items():
            if m["ref"]:
                art.manifest.add_entry(
                    ArtifactManifestEntry(
                        path=path,
                        ref=m["ref"],
                        digest=m["digest"],
                        size=m["size"],
                        extra=m["extra"],
                    )
                )
            else:
                art.add_file(str(folder / path), name=path)
        new = run.log_artifact(art, aliases=[x for x in a["aliases"] if x != "latest"])
        new.wait()
        done["artifacts"][a["qualified"]] = new.qualified_name
    run.finish(exit_code=0 if meta["state"] == "finished" else 1)
    done["runs"].append(meta["id"])
    save_state()

if apply and not one:  # inputs, once every artifact exists
    api = wandb.Api(timeout=120)
    linked = set(done.setdefault("linked", []))
    for _, d, meta in runs:
        if meta["id"] in linked:
            continue
        r = api.run(f"{target}/{meta['id']}")
        for u in json.loads((d / "used.json").read_text()):
            if u["type"] in SKIP_TYPES:
                continue
            try:
                r.use_artifact(
                    api.artifact(done["artifacts"].get(u["qualified"], u["qualified"]))
                )
            except wandb.errors.CommError as e:
                print("  input not found:", meta["id"], u["qualified"], str(e)[:80])
        done["linked"].append(meta["id"])
        save_state()
    print(
        "restored",
        len(done["runs"]),
        "runs,",
        len(done["artifacts"]),
        "artifact versions",
    )
