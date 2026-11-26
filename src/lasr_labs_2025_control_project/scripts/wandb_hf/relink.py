"""Step 5: for each stored .eval artifact version, add a version that only references its
pinned Hugging Face copy. Old versions stay in place (deleting them is a separate step).

- One `hf-migration` run in the project commits the reference versions. No existing run is
  resumed: the original producer gets the new version as an output, and each run that read
  the old version gets it as an input, through the public API (tested on a throwaway project:
  state, summary, config, history, files and tags unchanged).
- Each entry references the commit-pinned HF URL (not the expiring CDN URL a redirect leads
  to), with the ETag and size read through an authenticated GET; the file's MD5 goes into
  the metadata (`md5`), with `migrated_from` naming the original version.
- `latest` is put back on the original version that held it (logging moves it).
- Every new version is resolved the way the pipeline does (`tracking.resolve_input`, which
  downloads HF references with the HF login) and MD5-checked against the original.
- Rerunnable: an existing version whose metadata names `migrated_from` is reused, and
  progress goes to inventory.json by atomic replace.

usage: python relink.py <backup dir> [qualified name ...]   (no names = all)
"""

import json
from datetime import date
import os
import sys
import tempfile
from pathlib import Path

import requests
import wandb
from huggingface_hub import get_token
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry

hf_session = requests.Session()
hf_session.headers["Authorization"] = f"Bearer {get_token()}"
root = Path(sys.argv[1]).resolve()  # the resolve step changes directory
inv = json.load(open(root / "inventory.json"))
E, P = inv[0]["qualified"].split("/")[:2]
chosen = set(sys.argv[2:]) or {x["qualified"] for x in inv if x["n_files"] == 1}
todo = sorted(
    (x for x in inv if x["qualified"] in chosen),
    key=lambda x: (x["name"], int(x["version"][1:])),
)
assert all(x["n_files"] == 1 for x in todo), (
    "only single-file artifacts can be migrated"
)
assert all(x.get("hf_ok") for x in todo), (
    "every chosen file must pass the HF round trip"
)
api = wandb.Api(timeout=120)


def save():
    tmp = root / "inventory.json.tmp"
    json.dump(inv, open(tmp, "w"), indent=1)
    os.replace(tmp, root / "inventory.json")


def existing_copy(x):
    for a in api.artifact_collection(x["type"], f"{E}/{P}/{x['name']}").artifacts():
        if a.metadata.get("migrated_from") == x["qualified"]:
            return a
    return None


def live_holder(key):
    """The version that should hold `latest`, read live (the inventory may be stale): the
    current holder, or its original if a copy holds it, or else the newest original."""
    versions = list(api.artifact_collection(key[0], f"{E}/{P}/{key[1]}").artifacts())
    held = [a for a in versions if "latest" in a.aliases]
    if held:
        return held[0].metadata.get("migrated_from") or held[0].qualified_name
    originals = [a for a in versions if not a.metadata.get("migrated_from")]
    newest = max(originals, key=lambda a: int(a.version[1:]), default=None)
    return newest.qualified_name if newest else None


holder = {key: live_holder(key) for key in {(x["type"], x["name"]) for x in todo}}

# 1. commit reference versions from one migration run
pending = [x for x in todo if not x.get("new_qualified")]
for x in pending:
    if (a := existing_copy(x)) is not None:
        x["new_qualified"] = a.qualified_name
pending = [x for x in pending if not x.get("new_qualified")]
if pending:
    with wandb.init(
        entity=E,
        project=P,
        name=f"hf-migration-{date.today()}",
        job_type="hf-migration",
        notes="Adds versions of .eval artifacts that reference their copies on "
        f"huggingface.co/datasets/{pending[0]['hf_url'].split('/')[4]}/"
        f"{pending[0]['hf_url'].split('/')[5]}. Originals are untouched.",
        settings=wandb.Settings(silent=True),
    ) as run:
        for x in pending:
            entry = x["files"][0]
            artifact = wandb.Artifact(
                x["name"],
                type=x["type"],
                metadata={
                    **x["metadata"],
                    "migrated_from": x["qualified"],
                    "md5": entry["digest"],
                    "hf_url": x["hf_url"],
                },
            )
            # W&B's HTTP check can't log in to a private repo, so read the ETag and size
            # ourselves and keep the pinned URL (not the expiring redirect target).
            with hf_session.get(x["hf_url"], stream=True, timeout=120) as rsp:
                rsp.raise_for_status()
                etag, size = rsp.headers["etag"], int(rsp.headers["content-length"])
            assert size == entry["size"], (x["qualified"], size)
            artifact.manifest.add_entry(
                ArtifactManifestEntry(
                    path=entry["path"],
                    ref=x["hf_url"],
                    digest=etag.strip('"'),
                    size=size,
                    extra={"etag": etag},
                )
            )
            logged = run.log_artifact(artifact)
            logged.wait()
            x["new_qualified"] = logged.qualified_name
            save()
            print("logged", x["qualified"], "->", x["new_qualified"], flush=True)
save()

# 2. lineage: producer output, consumer inputs (no run is resumed)
for x in todo:
    new = api.artifact(x["new_qualified"])
    producer = api.run(f"{E}/{P}/{x['producer']}")
    if x["new_qualified"] not in {
        a.qualified_name for a in producer.logged_artifacts()
    }:
        producer.log_artifact(new, aliases=[])
    for c in x["consumers"]:
        if c not in x.setdefault("relinked", []):
            api.run(f"{E}/{P}/{c}").use_artifact(new)
            x["relinked"].append(c)
    save()

# 3. aliases: in every touched collection, `latest` back where it was before step 1
for key in holder:
    for a in api.artifact_collection(key[0], f"{E}/{P}/{key[1]}").artifacts():
        holds = "latest" in a.aliases
        if a.qualified_name == holder.get(key) and not holds:
            a.aliases.append("latest")
            a.save()
        elif a.qualified_name != holder.get(key) and holds:
            a.aliases.remove("latest")
            a.save()

# 4. resolve every new version the way the pipeline does (tracking.resolve_input, which
#    fetches HF references with huggingface_hub), in a scratch folder, MD5 against the original
from lasr_labs_2025_control_project.tracking import resolve_input  # noqa: E402

with tempfile.TemporaryDirectory() as scratch:
    os.chdir(scratch)
    for x in todo:
        a = api.artifact(x["new_qualified"])
        (entry,) = a.manifest.entries.values()
        assert entry.ref == x["hf_url"], (x["new_qualified"], entry.ref)
        resolved = resolve_input(f"wandb:{x['new_qualified']}")
        x["resolve_ok"] = resolved.md5 == x["files"][0]["digest"]
        resolved.local_path.unlink()
save()
done = [x for x in todo if x.get("resolve_ok")]
print(
    f"{len(done)}/{len(todo)} new versions resolve through the pipeline with matching MD5"
)
sys.exit(0 if len(done) == len(todo) else 1)
