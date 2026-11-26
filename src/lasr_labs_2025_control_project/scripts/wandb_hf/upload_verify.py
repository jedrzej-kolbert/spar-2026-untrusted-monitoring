"""Steps 3-4: upload chosen backed-up .eval files to the private HF dataset repo in one commit,
then download each back from the pinned commit and check its MD5 against W&B's.
Uses the cached `hf auth login` token (it needs write access to the org).

usage: python upload_verify.py <backup dir> <HF dataset repo id> [qualified name ...]   (no names = all not yet uploaded)
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

root, REPO = Path(sys.argv[1]), sys.argv[2]
inv = json.load(open(root / "inventory.json"))
PREFIX = inv[0]["qualified"].split("/")[1]  # W&B project name
chosen = set(sys.argv[3:]) or {x["qualified"] for x in inv}
todo = [x for x in inv if x["qualified"] in chosen and not x.get("hf_ok")]
assert all(x["md5_ok"] for x in todo), "backup has MD5 mismatches"


def save():
    tmp = root / "inventory.json.tmp"
    json.dump(inv, open(tmp, "w"), indent=1)
    os.replace(tmp, root / "inventory.json")


api = HfApi()
assert api.dataset_info(REPO).private, f"{REPO} must be private"
if todo:
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=root,
        path_in_repo=PREFIX,
        allow_patterns=[x["local"] for x in todo],
        commit_message=f"{len(todo)} untrusted-runs .eval artifacts copied from W&B",
    )
    print("commit", commit.oid)
    for x in todo:
        x["hf_url"] = (
            f"https://huggingface.co/datasets/{REPO}/resolve/{commit.oid}/{PREFIX}/{x['local']}"
        )
        path = hf_hub_download(
            REPO,
            f"{PREFIX}/{x['local']}",
            repo_type="dataset",
            revision=commit.oid,
            force_download=True,
        )
        md5 = base64.b64encode(hashlib.md5(Path(path).read_bytes()).digest()).decode()
        x["hf_ok"] = md5 == x["files"][0]["digest"]
    save()
print(
    f"{sum(bool(x.get('hf_ok')) for x in inv if x['qualified'] in chosen)}/{len(chosen)} on HF with matching MD5"
)
