"""List every artifact version in a W&B project holding .eval files, with producer and
consumers (walks the runs: per-version used_by() lookups are too slow).

usage: python inventory_evals.py ENTITY/PROJECT OUT_DIR/inventory.json
"""

import json
import sys

import wandb

api = wandb.Api(timeout=60)
runs = list(api.runs(sys.argv[1]))
print(len(runs), "runs", flush=True)
arts, consumers = {}, {}
for i, r in enumerate(runs):
    for a in r.logged_artifacts():
        if a.qualified_name in arts:
            continue
        entries = a.manifest.entries
        evals = {k: v for k, v in entries.items() if k.endswith(".eval")}
        if evals:
            arts[a.qualified_name] = dict(
                qualified=a.qualified_name,
                name=a.name.split(":")[0],
                version=a.version,
                type=a.type,
                aliases=a.aliases,
                metadata=a.metadata,
                n_files=len(entries),
                size=a.size,
                is_ref=any(v.ref for v in entries.values()),
                producer=r.id,
                files=[
                    dict(path=k, digest=v.digest, size=v.size) for k, v in evals.items()
                ],
            )
    for a in r.used_artifacts():
        consumers.setdefault(a.qualified_name, []).append(r.id)
    if i % 25 == 0:
        print(i, len(arts), flush=True)
for q, a in arts.items():
    a["consumers"] = consumers.get(q, [])
json.dump(list(arts.values()), open(sys.argv[2], "w"), indent=1)
total = sum(x["size"] for x in arts.values())
print(len(arts), "eval artifact versions,", round(total / 1e6), "MB")
