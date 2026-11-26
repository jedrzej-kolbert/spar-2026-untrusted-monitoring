"""Compare restored runs with the originals, field by field.

usage: python check_restore.py ORIGINAL_ENTITY/PROJECT RESTORED_ENTITY/PROJECT RUN_ID ...
       (VERSIONLESS=1 compares artifact names without versions, for partial restores)
"""

import json
import math
import os
import sys
import tempfile

import wandb

SRC, DST = sys.argv[1], sys.argv[2]
SKIP = {"run_table", "job", "code", "wandb-history", "wandb-events"}
AUTO = {"config.yaml", "wandb-summary.json", "wandb-metadata.json", "requirements.txt"}
api = wandb.Api(timeout=120)


def scalars(d):
    return {
        k: v
        for k, v in d.items()
        if not k.startswith("_") and not (isinstance(v, dict) and "_type" in v)
    }


def same(a, b):
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return a == b


def table(run, value):
    with tempfile.TemporaryDirectory() as d:
        t = json.load(open(run.file(value["path"]).download(root=d).name))
    return t["columns"], t["data"]


def short(
    q,
):  # with VERSIONLESS=1, compare names only (partial restores renumber versions)
    name = q.split("/")[-1]
    return name.split(":")[0] if os.environ.get("VERSIONLESS") else name


for rid in sys.argv[3:]:
    o, n = api.run(f"{SRC}/{rid}"), api.run(f"{DST}/{rid}")
    problems = []
    for f in ("name", "group", "job_type", "tags", "notes"):
        if getattr(o, f) != getattr(n, f):
            problems.append(f"{f}: {getattr(o, f)!r} != {getattr(n, f)!r}")
    if dict(o.config) != dict(n.config):
        problems.append("config differs")
    want_state = "finished" if o.state == "finished" else "failed"
    if n.state != want_state:
        problems.append(f"state {o.state} -> {n.state}")
    so, sn = scalars(o.summary._json_dict), scalars(n.summary._json_dict)
    diff = [k for k in so if not same(so[k], sn.get(k))]
    if diff:
        problems.append(f"summary keys differ: {diff[:5]}")
    ho, hn = list(o.scan_history()), list(n.scan_history())
    if [r["_step"] for r in ho] != [r["_step"] for r in hn]:
        problems.append(f"history steps differ ({len(ho)} vs {len(hn)} rows)")
    else:
        bad = [
            (r1["_step"], k)
            for r1, r2 in zip(ho, hn)
            for k, v in scalars(r1).items()
            if not same(v, r2.get(k))
        ]
        if bad:
            problems.append(f"history values differ: {bad[:5]}")
        tables = 0
        for r1, r2 in zip(ho, hn):
            for k, v in r1.items():
                if isinstance(v, dict) and v.get("_type") == "table-file":
                    tables += 1
                    if table(o, v) != table(n, r2[k]):
                        problems.append(f"table {k} at step {r1['_step']} differs")
        images = sum(
            1
            for r in hn
            for v in r.values()
            if isinstance(v, dict) and v.get("_type") == "image-file"
        )
        want_images = sum(
            1
            for r in ho
            for v in r.values()
            if isinstance(v, dict) and v.get("_type") == "image-file"
        )
        if images != want_images:
            problems.append(f"images {want_images} -> {images}")
    lo = sorted(
        (
            short(a.qualified_name),
            a.type,
            a.digest,
            json.dumps(a.metadata, sort_keys=True),
        )
        for a in o.logged_artifacts()
        if a.type not in SKIP
    )
    ln = sorted(
        (
            short(a.qualified_name),
            a.type,
            a.digest,
            json.dumps(a.metadata, sort_keys=True),
        )
        for a in n.logged_artifacts()
        if a.type not in SKIP
    )
    if lo != ln:
        problems.append(
            f"logged artifacts differ: {[x[:2] for x in lo]} vs {[x[:2] for x in ln]}"
        )
    uo = sorted(
        short(a.qualified_name) for a in o.used_artifacts() if a.type not in SKIP
    )
    un = sorted(
        short(a.qualified_name) for a in n.used_artifacts() if a.type not in SKIP
    )
    if uo != un:
        problems.append(f"inputs differ: {uo} vs {un}")

    def keep(name):
        return name not in AUTO and not name.startswith(("media/", "artifact/"))

    files_orig = {f.name for f in o.files() if keep(f.name)}
    files_new = {f.name for f in n.files() if keep(f.name)}
    if files_orig - files_new:
        problems.append(f"files missing: {sorted(files_orig - files_new)[:5]}")
    print(
        rid,
        o.name[:40],
        f"| {len(ho)} rows, {len(lo)} artifacts, {len(uo)} inputs |",
        "OK" if not problems else problems,
    )
