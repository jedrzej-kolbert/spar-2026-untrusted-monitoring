# When can we trust untrusted monitoring? A safety case sketch across collusion strategies

Paper: [arXiv:2602.20628](https://arxiv.org/abs/2602.20628)

An untrusted model **U** writes code and also monitors it. We measure how safe that is when U
colludes with itself, under four collusion strategies:

- **Passive self-recognition** (`self_rec`): U recognises its own code.
- **Causal** (`causal`): U hides a pre-agreed signal in its attacks.
- **Acausal** (`acausal`): U coordinates via Schelling points, with no shared signal.
- **Additive** (`additive`): passive self-recognition plus causal.

The settings are APPS and BigCodeBench, each with closed models (OpenAI/Gemini) or open models
(Tinker).

## Terms

| Term | Meaning |
|---|---|
| U, T, H | Untrusted model (policy + untrusted monitors), trusted weak monitor (also the paraphraser), honeypot generator |
| label | `attacks` (U's backdoors), `honest` (U's normal code), `honeypots` (H's backdoors) |
| transform | Signal jamming applied before U sees the code: `none`, `minified`, `paraphrased` |
| monitor arm | Which collusion monitor: base U, or a fine-tuned (SFT) U |
| condition | Collusion strategy × transform × monitor arm; one analysis per condition |

## Data pipeline

What happens to the data from problem to result, including every point where samples are
dropped or changed.

1. **Problems.** APPS problems 0–2999 whose statements include example inputs and outputs, or
   all of BigCodeBench. Even-numbered problems form the **test** split, which every result is
   measured on; odd-numbered problems form the **train** split, used only for SFT data. A limit
   keeps the first N problems (at later stages, the first N valid samples).
2. **Generation.** U writes `attacks` with the collusion strategy's attack prompt, and `honest`
   solutions. H writes `honeypots` with the same attack prompt in every condition. Each solution
   is run against the problem's tests (at most 10 for APPS); each attack and honeypot is also run
   on its backdoor input.
3. **Validity filter.** Only valid samples reach the later stages (transforms, monitoring and
   SFT data):
   - `honest`: passes the tests.
   - `attacks` and `honeypots`: pass the tests, and the backdoor works (its input makes the code
     give a wrong answer).
   - BashArena only: trajectories that errored or took no action are also dropped.

   This can remove most of a set, attacks especially, since they must pass both checks. Transform
   and monitor runs on W&B record how many samples they dropped (`input_dropped_invalid`). The
   pipeline stops if fewer than 10 valid samples of a label remain
   (`measurement.min_samples_per_label`; `elicitation.min_examples_per_label` for SFT data).
4. **Comment stripping.** Comments are removed from Python code before it is transformed, shown
   to a monitor, or used as an SFT example. BashArena trajectories are left as they are.
5. **Transforms.** `minified` mechanically rewrites the code into a compact equivalent;
   `paraphrased` has T rewrite it. U's monitors see the transformed code; T's monitor always sees
   the original.
6. **SFT data** (SFT arms only). Train-split samples, after the filter and the arm's transform.
   U's solutions are labelled YES ("I wrote this") and everyone else's NO: by default U's attacks
   against H's honeypots. Examples are shuffled; upsampling can repeat one label. The label
   controls train on honest (backdoor-free) solutions instead, or with the labels randomised.
7. **Monitoring** (test split). Each valid sample gets three scores:
   - collusion, from U (base or fine-tuned, per the monitor arm): log P(YES) − log P(NO) when
     asked whether it wrote the code;
   - suspiciousness, from U and from T: a number the monitor states in its reply.

   A reply without a usable score is retried, up to 3 attempts; after that the score is missing.
   In BashArena each action is scored, shown with the steps before it (the oldest are dropped
   past a length cap), and a trajectory's score is its highest action score.
8. **Analysis.** The collusion ROC compares attacks with honeypots; the suspiciousness ROCs
   compare attacks with honest solutions. An attack missing any score the game uses is left out
   entirely; an honest solution or honeypot missing a score is left out only of the ROCs that
   need it. U's suspiciousness becomes a yes/no report at the threshold that maximises the
   fraction of attacks flagged minus the fraction of honest solutions flagged. The bootstrap
   resamples each label's samples with replacement and solves the game again.

## Setup

1. `uv sync`
2. Copy `env.example` to `.env` and fill in the API keys you need (`TINKER_API_KEY` for open
   models; `OPENAI_API_KEY` and `GOOGLE_API_KEY` for closed models).
3. Start Docker. Generation runs the code it produces in a sandbox.
4. Run `wandb login`. Every stage logs to the W&B project named by `WANDB_PROJECT` (default
   `untrusted-monitoring`) under `WANDB_ENTITY` (default: your account). Set `WANDB_MODE=disabled`
   to run without W&B.

## Running the pipeline

One command runs one experiment end to end: training-split generations, fine-tuning U into
collusion monitors (the SFT arms), test-split generations, transforms, monitoring and analysis.

```bash
P="uv run python -m lasr_labs_2025_control_project.pipeline"

$P experiment=self_rec                     # prints the plan; runs nothing that costs money
$P experiment=self_rec allow_paid=true     # runs it
$P experiment=self_rec setting=bigcodebench models.U=tinker/deepseek-ai/DeepSeek-V3.1 allow_paid=true
$P experiment=self_rec limits=dry allow_paid=true          # ~200 problems per generation
$P -m experiment=self_rec,causal,acausal allow_paid=true   # several experiments
```

Run from the repo root. Re-running a command picks up where it stopped; finished steps are
reused, including across experiments (e.g. `causal` reuses the honest and honeypot solutions
`self_rec` generated).

**Configs** live in `conf/`. `config.yaml` has every default and documents the options. Pick one
of each group, or override any value as `key=value`:

| Group | Options |
|---|---|
| `experiment` | `self_rec`, `causal`, `acausal`, `additive`, `self_rec_controls` (label controls), `acausal_optimized`, and the paper's closed-model conditions under `paper/` |
| `setting` | `apps`, `bigcodebench`, `basharena` (transforms default to the setting's own: `none` only) |
| `models` | `open` (Tinker), `closed` (OpenAI/Gemini); or set `models.U`, `models.T`, `models.H` |
| `limits` | `full`, `dry` |

The config has three sections: `elicitation` (fine-tuning: arms, hyperparameters, limits),
`measurement` (collusion strategy, transforms, monitor arms, monitors, limits) and
`optimization` (game parameters, regeneration cutoffs, solver, analysis variants, bootstrap).
`seed` seeds everything that can be seeded. To try a new run, copy an experiment file.

**What runs.** Each run prints its plan: every step's id, whether it's done, and its output
folder under `experiments/<setting>/<stage>/`. Useful options:

- `stages=[generate,transform]`: run only those stages. Steps of other stages must already
  exist.
- `+inputs.<step id>=<output>`: use an existing output instead of running a step. The output can
  be a local `.eval`, a W&B artifact `wandb:<entity>/<project>/<name>:v<N>`, or for a
  fine-tune a model id (`+inputs.finetune.self_rec.none=tinker/tinker://...`).
- `optimization.variants=[all_monitors,um_only,tm_only]`: add the trusted-monitor-only analysis.
  `measurement.monitors` picks which monitors run.

**Reuse and reruns.** A step is done if an output with the same settings and inputs exists on
this machine or, when W&B is online, on W&B from anyone's run (`reuse=local` skips W&B). Code and
prompt wording aren't part of the match (prompts match by name), so after changing them, force a
rerun:

- `force=[<step id or stage>]`, e.g. `force=[monitor.none.sft.attacks]` or `force=[game]`, reruns
  those steps and everything that depends on them, replacing their outputs (on W&B the newest
  run wins). Downstream steps that already ran must have their stages in `stages`.
- To run one condition's game alone, narrow the experiment to it, e.g.
  `stages=[game] measurement.transforms=[none] '~measurement.arms.sft' force=[game.none.base]`.
- A forced step keeps its seed, so seeded models may give the same answers. For new samples,
  change `seed`; that reruns everything, fine-tunes included.
- Steps shared with another experiment (e.g. `self_rec`'s honest generations) are only rerun
  downstream in this one; force the other experiment's steps too.

**Where things go.** Each invocation's merged config is saved in
`experiments/runs/<experiment>/<time>/.hydra/` and attached to every W&B run it made. Each
condition's solved game is in `experiments/<setting>/game/`, and the experiment's figures in
`experiments/<setting>/analysis/`.

**Importing existing results.** `.eval` files made outside the pipeline (e.g. one experiment's
folder from the team Drive) can become steps of a pipeline experiment:

```bash
I="uv run python -m lasr_labs_2025_control_project.pipeline.ingest"
$I <folder of .eval files> experiment=paper/apps_causal   # shows which file is which step
$I <folder of .eval files> experiment=paper/apps_causal --apply
$P experiment=paper/apps_causal                             # then: games and analysis (free)
```

Each file is matched to its step from its contents (models, prompts, split, transform, monitors,
limits); give the overrides that describe how the files were made (e.g. `models=closed`,
`measurement.limits.generate=200`). Imported steps are logged to W&B like pipeline steps, tagged
`imported` with their original path and date, and are reused like any other output. Files that
match no step, or come from unfinished runs, are listed and skipped. A pipeline run folder
(`experiments/runs/<experiment>/<time>`) can be imported the same way, to log it to another W&B
project.

Open-model specifics (model choices, token budgets, Docker and throughput notes) are in
`docs/open_models.md`. BashArena generation runs in `basharena_runtime/` (its own environment;
see its README) and is driven by the pipeline like the other settings.

## Results

- **W&B** has one run per stage and label, one `game` run per condition (the solved game and its
  bootstrap, saved as the `game` artifact `game.<condition>`), and one `analysis` run per
  experiment, which draws the figures for all its conditions and lists their `game` artifacts as
  inputs. Filter by job type (stage) or by tag (experiment, collusion strategy, transform, monitor
  arm, label). Runs from one invocation share a group and a `config` artifact; the config
  artifact's lineage lists every run made from that config. Curves are logged as raw numbers:
  each y metric has its x-axis metric, which W&B uses as the panel's x-axis (set it by hand on a
  custom panel). Panels plotted against "Step" are meaningless. The analysis run logs the
  figures as images under `figures/`; each figure names U, H and T under its legend.

  | Panel (y) | x-axis | One point is |
  |---|---|---|
  | `<variant>/safety` | `<variant>/usefulness` | one regeneration cutoff; usefulness = 1 − cutoff, safety = P(blue team wins) at equilibrium |
  | `<variant>/band/safety_{lo,median,hi}` | `<variant>/band/usefulness` | the same, as bootstrap percentiles |
  | `strategy/<variant>/<parameter>` | `strategy/<variant>/usefulness` | one regeneration cutoff: blue's most likely `blue_*` value, or red's best-response `red_*` value, at equilibrium |
  | `strategy/<variant>/{blue_top_probability,blue_support_size,red_nash_support_size}` | `strategy/<variant>/usefulness` | the same: how mixed each team's equilibrium strategy is |
  | `train/loss`, `checkpoint/roc_auc` | `train/step` | one training step or saved checkpoint |

  `<variant>` is `all_monitors`, `um_only` or `tm_only`. ROC curves are W&B ROC charts
  (`roc/<monitor>`; in an analysis run, one curve per condition): each point is a score
  threshold, with the fraction of attacks (tpr) and of negatives (fpr) scoring at or above it.
  For `roc/untrusted_collusion`, negatives are honeypots and "at or above" means U would collude;
  for the two suspiciousness monitors, negatives are honest solutions. These are the empirical
  curves; the figures' trusted and collusion ROCs show the fitted model the game uses, which
  interpolates between thresholds. `red_team_roc_threshold` is U's collusion threshold, in YES −
  NO logprob, so it isn't comparable across monitor arms. Set `red_team_p_attack` panels to a log
  y-axis. When `red_nash_support_size` > 1, red has several equally good responses, so a jump in
  red's values may not mean anything. `figures/strategies` shows blue's whole mix and the
  bootstrap replicates, with usefulness on the same axis as `figures/outcome_distributions`.
  Every figure names U, H and T under its legend.
- **Local figures**: each experiment's are in its analysis folder
  (`experiments/<setting>/analysis/`). Other tools take condition folders
  (`experiments/<setting>/game/`):
  - compare conditions on shared axes: `scripts/plot_data.py <folder>... --out <dir>`;
  - bootstrap bands per collusion strategy: `scripts/plot_bootstrap_curves.py --out <dir>`
    (for the paper's 1000 replicates, run with `optimization.bootstrap.replicates=1000`; it
    resumes if interrupted);
  - parameter sensitivity: `scripts/plot_parameter_sensitivity.py <folder>`;
  - low-count audit: `scripts/helpers/event_count_audit.py --out <file>`.

  Run them with `uv run python -m lasr_labs_2025_control_project.scripts.<name>`.
- **Published data** is in the HF dataset `jbostock/untrusted-monitoring-2026-paper`, in the
  old per-experiment layout. To analyse it with the pipeline, import it (above), or pass its
  monitor outputs as inputs (`+inputs.monitor.<transform>.<arm>.<label>=<file>`).
- **Team data**: the APPS and BigCodeBench results are also in the team Google Drive's
  `FINAL_RESULTS` folder. Don't change it without agreeing with the whole team.

## Other experiments

- `src/lasr_labs_2025_control_project/self_rec_baselines/`: standalone self-recognition
  baselines.
- `docs/dependent_monitors_stats/`: the dependence-aware statistical model.
- `docs/optimizer_comparison/`: game solver comparison.
