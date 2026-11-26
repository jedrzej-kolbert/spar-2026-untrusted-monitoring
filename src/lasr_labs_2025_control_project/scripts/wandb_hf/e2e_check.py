"""End-to-end check, in a throwaway W&B project and a throwaway private HF dataset repo
(both deleted at the end), of the code paths in this folder's README:

  (b) a pipeline stage (`tracking.StageRun`) with HF_EVAL_REPO set logs its .eval as a
      reference to Hugging Face, and the pipeline resolves it back byte for byte;
  (c) a stored .eval version, after a migrated copy exists and the stored version is
      DELETED, still resolves through its old exact ref (`wandb:…:v0`) to the copy.

Free (W&B and HF only). Needs W&B access to ENTITY and an HF login that can create
private dataset repos in HF_NAMESPACE.

usage: python e2e_check.py ENTITY HF_NAMESPACE
"""

import base64
import hashlib
import os
import secrets
import sys
import tempfile
from pathlib import Path

import wandb
from huggingface_hub import HfApi

entity, namespace = sys.argv[1], sys.argv[2]
tag = secrets.token_hex(3)
project, repo = f"hf-e2e-{tag}", f"{namespace}/hf-e2e-{tag}"
os.environ.update(WANDB_ENTITY=entity, WANDB_PROJECT=project, WANDB_MODE="online")
from lasr_labs_2025_control_project import tracking  # noqa: E402  (reads the env above)

hf, api = HfApi(), wandb.Api(timeout=120)


def md5(p: Path) -> str:
    return base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()


hf.create_repo(repo, repo_type="dataset", private=True)
results = {}
try:
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        eval_file = Path(tmp) / "out.eval"
        eval_file.write_bytes(os.urandom(200_000))

        # (b) new run logs to HF
        os.environ["HF_EVAL_REPO"] = repo
        with tracking.StageRun("generate", "e2e-gen", config={}) as stage:
            logged = stage.log_artifact(eval_file, type="eval-log", wait=True)
        new = api.artifact(logged.qualified_name)
        (entry,) = new.manifest.entries.values()
        resolved = tracking.resolve_input(f"wandb:{new.qualified_name}")
        results["b: stored on HF, not W&B"] = bool(entry.ref) and entry.ref.startswith(
            f"https://huggingface.co/datasets/{repo}/resolve/"
        )
        results["b: resolves byte for byte"] = resolved.md5 == md5(eval_file)

        # (c) stored version -> migrated copy -> delete stored -> old ref resolves to copy
        del os.environ["HF_EVAL_REPO"]
        with tracking.StageRun("generate", "e2e-old", config={}) as stage:
            old = stage.log_artifact(eval_file, type="eval-log", wait=True)
        old_ref = old.qualified_name
        with wandb.init(entity=entity, project=project, job_type="hf-migration") as run:
            copy = wandb.Artifact(
                old.name.split(":")[0],
                type="eval-log",
                metadata={**old.metadata, "migrated_from": old_ref},
            )
            tracking.add_hf_references(copy, [eval_file], repo, f"{project}/migrated")
            copy = run.log_artifact(copy)
            copy.wait()
        api.artifact(old_ref).delete(delete_aliases=True)
        resolved = tracking.resolve_input(f"wandb:{old_ref}")
        results["c: deleted ref resolves to the copy"] = (
            resolved.artifact is not None
            and resolved.artifact.metadata.get("migrated_from") == old_ref
        )
        results["c: copy bytes match"] = resolved.md5 == md5(eval_file)
finally:
    for r in api.runs(f"{entity}/{project}"):
        r.delete(delete_artifacts=True)
    hf.delete_repo(repo, repo_type="dataset")

for check, ok in results.items():
    print(("PASS " if ok else "FAIL ") + check)
sys.exit(0 if results and all(results.values()) else 1)
