# W&B storage: `.eval` files on Hugging Face, full backup and restore

The team W&B project is near its storage cap. `.eval` outputs (the bulk of it) move to the
private HF dataset repo `spar-untrusted-monitoring-collusion-org/untrusted-runs`; W&B keeps
artifact versions that reference them, so lineage, `reuse=wandb` and `wandb:` refs keep
working. Everything here is free (W&B and HF only).

Setup: W&B access to the team, and an HF login that can read the org repo
(`hf auth login`, or `HF_TOKEN` in `.env`; the pipeline loads `.env`). Writing (logging new
runs, migrating) also needs write access. Run scripts from a scratch folder: W&B writes
`./wandb/` and downloads go to `./artifacts/`.

## (a) Can `untrusted-runs` be restored safely?

- `backup_full.py ENTITY/PROJECT OUT` saves every run (details, config, summary, state,
  full history, system metrics, files), every artifact version it logged (files MD5-checked
  against W&B, metadata, aliases, manifest; URL references recorded as is), every input it
  used, and the saved views. The 2026-11-26 backup of `untrusted-runs` (200 runs, 0 MD5
  mismatches) is on the team Drive: `Experiments/W&B backups (2026-11-21)/wandb-untrusted-runs-full-2026-11-26.zip`
  (556 MB, MD5 `aa5d2b738a00900f3ec2f7f2ec25a4a5`), with these scripts. `backup_full.py` exits
  non-zero and leaves a run unfinished (refetched on rerun) if any file fails its MD5 check.
- `restore_full.py BACKUP ENTITY/PROJECT [run ids] [--apply]` recreates runs oldest first,
  one process per run (W&B otherwise carries a run's group into the next run). Each artifact
  version is created once, by the first run that lists it, and attached as an output to the
  others (a migrated copy is listed by both its producer and the migration run). The dry run
  predicts every recreated version number; `--apply` refuses if an original `name:vN` would
  get another number (`--allow-renumber` overrides, for partial test restores). For the full
  backup: 143 versions, none renumbered. Values it can't rebuild are listed, not dropped
  silently; a run is only marked linked when all its links succeed (rerun retries), and the
  script exits non-zero otherwise.
- Verify: restore some runs into a throwaway project and compare with the originals:
  ```
  R="0840ivcw l1n8dla1 1pz613fr z7v7zusn dezansvr ma6z71bn 1ua4wa8w"
  python restore_full.py untrusted-runs-full s184361/restore-test $R --apply --allow-renumber
  VERSIONLESS=1 python check_restore.py spar-untrusted-monitoring-collusion/untrusted-runs s184361/restore-test $R
  ```
  Expected: `OK` for each (name, group, job type, tags, notes, config, state, summary, every
  history row, every logged key including media, tables, image count, artifacts by
  type/digest/metadata, inputs, files). A run
  just restored can show `0 rows` for a minute while W&B processes its history; rerun.
  Run ids cover a game run with tables, an analysis run with images, a failed run, a model
  run, an `.eval` producer and its consumer, the HF migration run (reference artifacts) and a
  producer that lists a migrated copy it didn't create.
  `VERSIONLESS=1` because a partial restore renumbers versions; a full restore logs them in
  creation order. Not restorable: original timestamps, system-metric charts, saved views
  (kept in `views.json`).

## (b) Do new runs log `.eval` files to Hugging Face?

With `HF_EVAL_REPO=spar-untrusted-monitoring-collusion-org/untrusted-runs` set, every
`eval-log` artifact a stage logs (`tracking.StageRun.log_artifact`) is uploaded to
`<W&B project>/<artifact name>/<run id>/` in that repo and logged as a reference to the
commit-pinned URL (`tracking.add_hf_references`). The repo must be private (`.eval` files hold
transcripts); if it isn't, or the upload fails, the file is stored in W&B as before (warning
in the log). Without `HF_EVAL_REPO`, nothing changes.

Verify: `python e2e_check.py <W&B entity> spar-untrusted-monitoring-collusion-org` runs a
real `StageRun` in a throwaway W&B project and throwaway private HF repo (both deleted at
the end; W&B keeps the empty project). Expected:
```
PASS b: stored on HF, not W&B
PASS b: resolves byte for byte
PASS c: deleted ref resolves to the copy
PASS c: copy bytes match
```
Unit tests: `pytest -m "not slow and not mc" tests/test_tracking.py`.

## (c) Are the stored versions ready to be removed from W&B?

Migration (per project, rerunnable, deletes nothing):
1. `inventory_evals.py ENTITY/PROJECT DIR/inventory.json`: every stored `.eval` version
   (not migrated copies), its producer and consumers; a rerun keeps earlier progress.
2. `backup_evals.py DIR`: download them, MD5-check.
3. `upload_verify.py DIR HF_REPO [versions]`: one HF commit, download back, MD5-check.
   Steps 2-4 exit non-zero on any MD5 mismatch. Only single-file artifacts are migrated
   (all 87 in `untrusted-runs` are); others are refused.
4. `relink.py DIR [versions]`: for each original `name:vN`, a new version of the same
   collection referencing the pinned HF URL, metadata `migrated_from=<original>` and `md5`;
   logged by one `hf-migration-<date>` run, attached to the original producer as an output
   and to every consumer as an input through the public API (no run is resumed); `latest`
   put back where it was, read live before the step (not from the inventory); every copy resolved through `tracking.resolve_input` and
   MD5-checked.

After an original is deleted: an exact `wandb:…:vN` ref to it resolves to its copy,
pipeline reuse finds the copy among the producer's outputs, and a local `.eval` matches the
copy by the MD5 in its metadata (all in `tracking/core.py`, tested in `test_tracking.py` and
`e2e_check.py`).

Verify readiness (read-only): `python check_ready_to_delete.py ENTITY/PROJECT`. A version is
READY when its copy exists, the copy's HF bytes match the original's MD5, the producer lists
the copy, and every consumer of the original also uses the copy. State at the end of
2026-11-26: all 87 migrated; 82 originals deleted from W&B (~275 MB); 5 kept
because 13 runs in `collusion-rl-v2` still read only the originals. Before deleting a READY
original that holds `latest`, move `latest` to its copy. Deleting is not automated here.
