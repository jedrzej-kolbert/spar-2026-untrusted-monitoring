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
   and monitor runs on W&B record how many samples they dropped (`input_dropped_invalid`).
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

## Configs

Configs are generated. Don't edit the YAML; edit the spec in `experiments/open_models/make_configs.py`
(model roles are in `ROLE_MODELS`), then regenerate:

```bash
python experiments/open_models/make_configs.py --model-class open   # or closed, or omit for both
```

Configs land in `src/lasr_labs_2025_control_project/configs/<setting>[_open]/<collusion strategy>/`
under `generator/`, `transforms/` and `monitors/`. Outputs go to
`experiments/<model class>/<setting>/<experiment>/`. To pilot, add `limit: N` to a config's
`defaults`.

Any input path, whether an `eval_path` in a config or a finetune `--…-eval` flag, can be a W&B
artifact instead: `wandb:<entity>/<project>/<artifact name>:v<N>`.

## Running the pipeline

Run from the repo root. Each stage takes a config:

```bash
C=src/lasr_labs_2025_control_project/configs/apps_open
S=src/lasr_labs_2025_control_project/scripts

uv run $S/generate_solutions_inspect.py  --config $C/self_rec/generator/generator_config.yaml
uv run $S/transform_solutions_inspect.py --config $C/self_rec/transforms/minified_config.yaml
uv run $S/monitor_solutions_inspect.py   --config $C/self_rec/monitors/base/minified_monitor_config.yaml
uv run $S/plot_data.py                   --config $C/self_rec/monitors/base/minified_monitor_config.yaml
```

- **Generate** before the other collusion strategies. In the open tree, `self_rec` generates the
  `honest` and `honeypots` sets that every strategy shares.
- **Transform** once per transform (`minified_config.yaml`, `paraphrased_config.yaml`).
- **Monitor** once per condition (`monitors/**`).
- **Analyse** (`plot_data.py`) once per condition. It solves the game and bootstraps it
  (`--bootstrap-replicates`, default 2; 0 skips). Figures and analysis files go to the config's
  `results/` folder.

### Fine-tuned collusion monitors (SFT arms)

1. Generate training data on the train split, then transform it for the minified and paraphrased
   arms:
   ```bash
   uv run $S/generate_solutions_inspect.py  --config $C/self_rec/generator/train_generator_config.yaml
   uv run $S/transform_solutions_inspect.py --config $C/self_rec/transforms/minified_config_train.yaml
   uv run $S/transform_solutions_inspect.py --config $C/self_rec/transforms/paraphrased_config_train.yaml
   ```
   Causal arms use `causal/…` instead. The honest-label control arms use
   `self_rec/generator/honest_train_generator_config.yaml`.
2. Fine-tune U on its attacks (labelled YES) against H's honeypots (labelled NO). Name the
   checkpoint after its arm (`self_rec_base`, `self_rec_minified`, `causal_paraphrased`, … — see
   `SFT_ARMS` in `make_configs.py`):
   ```bash
   E=experiments/open-source-model/apps/self_rec_train
   uv run python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
     --train-eval $E/transforms/minified/attack_transformed.eval \
     --train-eval $E/transforms/minified/honeypots_transformed.eval \
     --transformed --checkpoint-name self_rec_minified \
     --target-model <Tinker model id of U, without the tinker/ prefix> \
     --base-model <the same id>
   ```
   - Leave out `--transformed` for the `none` arm, and pass `generator/*.eval` instead.
   - To track ROC AUC across training, add `--checkpoint-every-steps N --attacks-eval <held-out attacks.eval> --honeypots-eval <held-out honeypots.eval>`.
   - Closed models use `finetuning/openai_finetune.py` with the same `--train-eval` flags plus
     `--suffix <arm>`.
   - Label-ablation controls train on a hand-prepared JSONL (`--train-jsonl`); see
     `finetuning/binary_tuning/README.md`.
3. Add the printed `tinker://…` path to `experiments/open_models/ckpts.json` under the arm's
   name. Then regenerate the configs, and monitor and analyse with the `monitors/sft/` configs.

Open-model specifics (model choices, token budgets, Docker and throughput notes) are in
`experiments/open_models/SPEC.md`.

### BashArena

BashArena runs the same steps on agent trajectories. Each rated action is one example, shown
with every earlier step and its tool output, exactly as the per-action monitor sees it.
Paraphrasing, minification and checkpoint learning dynamics aren't available for it yet.

1. Generate configs with `make_configs.py --model-class open --settings basharena
   --basharena-limit N` (N tasks per population and split; default 5).
2. Generate in its own environment, once per split; without `--allow-paid` it only prints the plan:
   ```bash
   cd basharena_runtime
   uv run --frozen python -m basharena_runtime.generate --allow-paid \
     --config ../src/lasr_labs_2025_control_project/configs/basharena_open/self_rec/generator/train_generator_config.yaml \
     --output ../experiments/open-source-model/basharena/self_rec/generator/train
   ```
   and the same with `base_generator_config.yaml` and `.../generator/test`.
3. Prepare per-action training data, then fine-tune with `--train-jsonl` and
   `--checkpoint-name basharena_self_rec_base`:
   ```bash
   G=experiments/open-source-model/basharena/self_rec/generator
   bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh $G $G/raw --action-views
   bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh $G/raw <U model id> \
     --system-prompt BASHARENA_SELF_REC_ACTION_SYSTEM_PROMPT --user-prompt BASHARENA_ACTION_USER_PROMPT
   ```
4. Add the checkpoint to `ckpts.json`, regenerate with the same `--basharena-limit`, and monitor
   with `configs/basharena_open/self_rec/monitors/{base,sft}/`. A trajectory's score is the
   maximum over its actions.

Long trajectories can exceed a model's context: `defaults.action_view.max_history_chars` in the
monitor config and `ACTION_MAX_HISTORY_CHARS` for `transform_logs_to_raw.sh` drop the oldest
steps; use the same value in both.

## Results

- **W&B** has one run per stage and label, one `game` run per condition (the solved game and its
  bootstrap, saved as the `game` artifact `game.<condition>`), and one `analysis` run per
  `plot_data.py` invocation, which draws the figures for all its conditions and lists their `game`
  artifacts as inputs.
  Filter by job type (stage) or by tag (collusion strategy, transform, monitor arm, label). Set
  `WANDB_RUN_GROUP=<name>` to group a batch of runs. Curves are logged as raw numbers: each y
  metric has its x-axis metric, which W&B uses as the panel's x-axis (set it by hand on a
  custom panel). Panels plotted against "Step" are meaningless. The analysis run logs the
  figures as images under `figures/`; each figure names U, H and T under its legend.

  | Panel (y) | x-axis | One point is |
  |---|---|---|
  | `<variant>/safety` | `<variant>/usefulness` | one regeneration cutoff; usefulness = 1 − cutoff, safety = P(blue team wins) at equilibrium |
  | `<variant>/band/safety_{lo,median,hi}` | `<variant>/band/usefulness` | the same, as bootstrap percentiles |
  | `strategy/<variant>/<parameter>` | `strategy/<variant>/usefulness` | one regeneration cutoff: blue's most likely `blue_*` value, or red's best-response `red_*` value, at equilibrium |
  | `strategy/<variant>/{blue_top_probability,blue_support_size,red_nash_support_size}` | `strategy/<variant>/usefulness` | the same: how mixed each team's equilibrium strategy is |
  | `train/loss`, `checkpoint/roc_auc` | `train/step` | one training step or saved checkpoint |

  `<variant>` is `all_monitors` or `um_only`. ROC curves are W&B ROC charts (`roc/<monitor>`; in an
  analysis run, one curve per condition): each point is a score threshold, with the fraction of
  attacks (tpr) and of negatives (fpr) scoring at or above it. For `roc/untrusted_collusion`,
  negatives are honeypots and "at or above" means U would collude; for the two suspiciousness
  monitors, negatives are honest solutions. These are the empirical curves; the figures' trusted
  and collusion ROCs show the fitted model the game uses, which interpolates between thresholds.
  `red_team_roc_threshold` is U's collusion threshold, in YES − NO logprob, so it isn't
  comparable across monitor arms. Set `red_team_p_attack` panels to a log y-axis. When
  `red_nash_support_size` > 1, red has several equally good responses, so a jump in red's values
  may not mean anything. `figures/strategies` shows blue's whole mix and the bootstrap
  replicates, with usefulness on the same axis as `figures/outcome_distributions`.
- **Local figures** for the paper's groupings come from `scripts/run_all_grouped_plotting.sh`. To
  analyse every monitor config in a tree, run `scripts/plot_everything.sh <config dir>`. For big
  resumable bootstrap sweeps use `scripts/bootstrap.py`, then `scripts/plot_bootstrap_curves.py`.
  Parameter sensitivity: `scripts/plot_parameter_sensitivity.py --config <monitor config>`.
- **Published data** is in the HF dataset `jbostock/untrusted-monitoring-2026-paper` (same layout
  as `experiments/`).
- **Team data**: the APPS and BigCodeBench results are also in the team Google Drive's
  `FINAL_RESULTS` folder. Don't change it without agreeing with the whole team.

## Other experiments

- `src/lasr_labs_2025_control_project/self_rec_baselines/`: standalone self-recognition
  baselines.
- `docs/dependent_monitors_stats/`: the dependence-aware statistical model.
- `docs/optimizer_comparison/`: game solver comparison.
