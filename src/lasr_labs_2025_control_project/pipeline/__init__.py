"""One entrypoint for the whole pipeline, configured with Hydra.

Developer notes. Researchers only need the README's "Running the pipeline".

Config
------
``conf/config.yaml`` holds every default; ``conf/experiment/*.yaml`` holds
run-specific overrides (``# @package _global_``), and the groups
``setting/`` (dataset), ``models/`` (U/T/H) and ``limits/`` are swapped
whole. ``schema.py`` types every key; Hydra rejects unknown keys and wrong
types when composing. The merged config of a run is ``<run dir>/.hydra/``
(``config.yaml``, ``overrides.yaml``), and every W&B stage run of that
invocation logs it as its run config and uses it as one ``config`` artifact.

The three sections map to the paper's phases: ``elicitation`` (fine-tuning U
into a collusion monitor), ``measurement`` (test-split generation, transforms,
monitors) and ``optimization`` (solving the game).

Graph
-----
``graph.py`` turns a config into nodes: generate -> transform -> finetune ->
monitor -> game (one per condition: the solved game and its bootstrap) ->
analysis (one per experiment: the figures, drawn from all its games), one node
per output. A node's key is its parameters plus
its inputs' hashes; its hash names its folder (``store.py``), so outputs that
two experiments share (honest and honeypot generations, H's transforms, an
additive run's causal checkpoint) are computed once and reused, and a changed
setting reruns only the nodes it reaches. ``measurement.share_generations:
false`` puts the collusion strategy into generation keys (the paper's closed
runs generated per strategy). Stored keys are compared in full before reuse.

Seeds: one ``seed`` in the config. Every stochastic node has it in its key and
derives its own seed from its hash (``Node.seed``), so each stage is seeded
independently of the others. Seeded: Inspect sampling (generation, LLM
paraphrase, suspiciousness monitors; retries offset the seed), SFT shuffles
and label permutation, BashArena action sampling, bootstrap resamples and the
game solver. Not seedable: OpenAI/Gemini sampling (seeds are best-effort),
Tinker LoRA initialisation, the APPS/BigCodeBench sandbox under load
(timeouts), and BashArena agents' environments.

Running
-------
``run.py`` walks back from the analysis: anything done locally, given in
``inputs`` (a path, ``wandb:<entity>/<project>/<name>:vN``, or a model id),
or found on W&B as a complete run with the same key (``reuse=wandb``, the
default; skipped when W&B is offline) stops the walk. ``force`` marks nodes
and everything downstream of them as not done; they rerun from an emptied
folder. Nodes of stages not in ``stages`` must already exist. A plan that
would call paid models only prints itself unless ``allow_paid=true``.
``execute.py`` runs a node by calling the stage functions in ``scripts/`` and
``finetuning/`` under a ``tracking.StageRun``.

Importing (``ingest.py``) turns existing outputs into steps: loose .eval files
are matched to the steps of a declared experiment by their contents, then
copied in with the step's key and logged through the same per-stage tracker
builders as ``execute.py`` (so imported and executed runs look alike).
Pipeline run folders are re-logged from the store, with fine-tune curves read
back from the local W&B run files (``tracking/local_runs.py``).

Adding a setting: a ``conf/setting/<name>.yaml`` (prompts, view, ids), a
``DatasetType`` and task in ``scripts/generate_solutions_inspect.py`` (or, for
a separate runtime like BashArena, a branch in ``execute.run_generate``).
Adding an option: a schema field, then use it in ``graph.py`` so it enters
the right nodes' keys.
"""
