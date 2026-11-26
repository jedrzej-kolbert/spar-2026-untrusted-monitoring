"""Recreate a W&B project from a backup_full.py backup.

For each backed-up run, oldest first: a run with the same id, name, group, job type, tags,
notes and config; its history replayed at the original steps (tables, images and histograms
rebuilt from the backup); its summary; its other saved files; and the artifact versions it
created (same name, type, description, metadata, aliases; stored files re-uploaded, URL
references re-recorded as is). Then the links: every other run that listed a version as an
output gets it attached as one, and every run's inputs are linked (artifacts of this project
map to their recreated versions; other projects' artifacts are linked as they are). Runs
close with their original state (finished, or failed for failed/crashed).

Exact refs (`name:vN`) must keep pointing at the same data, so the dry run predicts every
recreated version number and `--apply` refuses if one would differ from the original
(override with --allow-renumber, e.g. for a partial test restore). Values that can't be
rebuilt are listed per run, not dropped silently.

Progress goes to <target project>.restored.json in the backup folder; a rerun skips runs
already restored and retries links that failed. The script exits non-zero if any link
failed. Not recreatable: timestamps, system-metric charts, saved views (views.json is kept).

usage: python restore_full.py BACKUP_DIR ENTITY/TARGET_PROJECT [run id ...] [--apply]
       [--allow-renumber]      (dry run without --apply; no run ids = all runs)
"""

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import wandb
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry

flags = {"--apply", "--allow-renumber"}
args = [a for a in sys.argv[1:] if a not in flags]
apply, allow_renumber = "--apply" in sys.argv, "--allow-renumber" in sys.argv
backup, target, only = Path(args[0]).resolve(), args[1], set(args[2:])
entity, project = target.split("/")
state_file = backup / f"{project}.restored.json"
done = (
    json.loads(state_file.read_text())
    if state_file.exists()
    else {"runs": [], "artifacts": {}, "linked": []}
)
# recreated by the history replay, or W&B's own bookkeeping
SKIP_TYPES = {"run_table", "job", "code", "wandb-history", "wandb-events"}
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
    """Rebuild a logged table, image or histogram; None if it isn't one of those."""
    kind = value.get("_type")
    if kind == "table-file":
        t = json.loads((files / value["path"]).read_text())
        return wandb.Table(columns=t["columns"], data=t["data"])
    if kind == "image-file":
        return wandb.Image(str(files / value["path"]))
    if kind == "histogram":
        return wandb.Histogram(np_histogram=(value["values"], value["bins"]))
    return None


def clean(row: dict, files: Path, dropped: set) -> dict:
    out = {}
    for k, v in row.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "_type" in v:
            m = media(v, files)
            if m is None:
                dropped.add(f"{k} ({v['_type']})")
                continue
            v = m
        out[k] = v
    return out


runs = []
for d in sorted((backup / "runs").iterdir()):
    meta = json.loads((d / "run.json").read_text())
    if not only or meta["id"] in only:
        logged = [
            a
            for a in json.loads((d / "logged.json").read_text())
            if a["type"] not in SKIP_TYPES
        ]
        runs.append((meta["created_at"], d, meta, logged))
runs.sort(key=lambda r: r[0])

# Each version is created once, by the first run (in restore order) that lists it; later
# runs that list it (e.g. a producer given a migrated copy) get it attached as an output.
creator, outputs = {}, collections.defaultdict(list)
for _, _, meta, logged in runs:
    for a in sorted(logged, key=lambda a: int(a["version"][1:])):
        creator.setdefault(a["qualified"], meta["id"])
        if creator[a["qualified"]] != meta["id"]:
            outputs[meta["id"]].append(a["qualified"])
count, renumbered = collections.Counter(), []
for q, rid in creator.items():  # insertion order = creation order
    name, version = q.rsplit("/", 1)[-1].split(":")
    if f"v{count[name]}" != version:
        renumbered.append(f"{name}:{version} -> v{count[name]}")
    count[name] += 1

one = os.environ.get("RESTORE_ONE") == "1"
if not one:
    print(
        f"{len(runs)} runs, {len(creator)} artifact versions -> {target}"
        + ("" if apply else " (dry run)")
    )
    if renumbered:
        print(
            f"{len(renumbered)} versions would get a different number:", renumbered[:10]
        )
        if apply and not allow_renumber:
            sys.exit(
                "refusing: exact refs would point at other data (--allow-renumber)"
            )
if apply and not one:
    # One process per run: W&B carries a run's group into the next run created in the same
    # process when the next one has none (seen even after wandb.teardown()).
    for _, d, meta, _ in runs:
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

for _, d, meta, logged in runs:
    if meta["id"] in done["runs"] or (apply and not one):
        continue
    mine = [a for a in logged if creator[a["qualified"]] == meta["id"]]
    rows = [json.loads(line) for line in open(d / "history.jsonl")]
    print(
        meta["id"], meta["name"], len(rows), "rows", len(mine), "artifacts", flush=True
    )
    if not apply:
        continue
    files, dropped = d / "files", set()
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
        values = clean(row, files, dropped)
        if values:
            run.log(values, step=row.get("_step"))
    run.summary.update(clean(meta["summary"], files, dropped))
    for f in files.rglob("*") if files.exists() else []:
        rel = f.relative_to(files)
        # media/ is re-logged with the history; artifact/ names are reserved by W&B (an
        # upload including one is rejected as a batch, media files with it)
        if (
            f.is_file()
            and rel.parts[0] not in ("media", "artifact")
            and str(rel) not in AUTO_FILES
        ):
            run.save(str(f), base_path=str(files), policy="now")
    for a in sorted(mine, key=lambda a: int(a["version"][1:])):
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
    if dropped:
        print("  not restorable, left out:", sorted(dropped))
        done.setdefault("dropped", {})[meta["id"]] = sorted(dropped)
    done["runs"].append(meta["id"])
    save_state()

if apply and not one:  # links, once every artifact exists
    api = wandb.Api(timeout=120)
    failed = 0
    for _, d, meta, _ in runs:
        if meta["id"] in done["linked"]:
            continue
        r = api.run(f"{target}/{meta['id']}")
        errors = []
        for q in outputs[meta["id"]]:
            try:
                r.log_artifact(api.artifact(done["artifacts"][q]), aliases=[])
            except Exception as e:
                errors.append(f"output {q}: {str(e)[:80]}")
        for u in json.loads((d / "used.json").read_text()):
            if u["type"] in SKIP_TYPES:
                continue
            try:
                r.use_artifact(
                    api.artifact(done["artifacts"].get(u["qualified"], u["qualified"]))
                )
            except Exception as e:
                errors.append(f"input {u['qualified']}: {str(e)[:80]}")
        if errors:  # not marked linked: a rerun retries this run
            failed += 1
            print("  link failed:", meta["id"], errors)
        else:
            done["linked"].append(meta["id"])
            save_state()
    print(
        "restored",
        len(done["runs"]),
        "runs,",
        len(done["artifacts"]),
        "artifact versions;",
        failed,
        "runs with failed links",
    )
    sys.exit(1 if failed else 0)
