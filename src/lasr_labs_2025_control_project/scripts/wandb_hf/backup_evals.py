"""Download every .eval artifact in inventory.json to <dir>/<name>/<version>/ and verify
its MD5 against W&B's manifest.

usage: python backup_evals.py DIR   (DIR holds inventory.json from inventory_evals.py)
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import wandb

root = Path(sys.argv[1])
inv = json.load(open(root / "inventory.json"))
api = wandb.Api(timeout=120)
bad = 0
for x in inv:
    folder = root / x["name"] / x["version"]
    entry = x["files"][0]
    path = folder / entry["path"]
    if not path.exists():
        api.artifact(x["qualified"]).download(root=str(folder))
    md5 = base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
    x["local"] = str(path.relative_to(root))
    x["md5_ok"] = md5 == entry["digest"]
    bad += not x["md5_ok"]
json.dump(inv, open(root / "inventory.json", "w"), indent=1)
print(len(inv), "downloaded;", bad, "MD5 mismatches")
