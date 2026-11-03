#!/usr/bin/env bash
# Test scoring of one final strategy or checkpoint, the same way as the SFT judges:
#   test.sh rewrite <label> <source> [limit]   rewrite the 309 working test attacks (paid)
#       source: strategy=<icrl_result.json or wandb: ref to its strategy artifact>
#               model=<tinker:// sender checkpoint> | none (rewrite with no strategy)
#   test.sh score <label> [pipeline overrides]  monitors, game and analysis on the rewrites
#       (base judge, paraphrased and no jamming). Prints the plan only, unless given
#       allow_paid=true. Honeypot and honest scores are the V4.1 runs' own, so AUROC and
#       safety compare directly with the base and SFT judges.
#   test.sh report <label> <entity/project/run_id>  write the games' test scores into the
#       RL run's W&B page (test_paraphrased/, test_none/), free
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/collusion-rl-v2/env.sh
here=experiments/collusion-rl-v2
V41=/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps
X=/home/jkk/spar-2026-untrusted-monitoring/experiments/exports/attack_SFT_full_test_batch64
U='tinker/deepseek-ai/DeepSeek-V4.1-Flash:peft:262144'
step=$1 label=$2
shift 2
folder="$here/test/$label"

case $step in
rewrite)
  source=$1 limit=${2:-null} model=$U strategy=() system=RL_SELF_REC_SENDER_SYSTEM_PROMPT
  case $source in
    strategy=*) strategy=(strategy_result "${source#strategy=}") ;;
    model=*) model=${source#model=} ;;
    none) ;;
    *) echo "unknown source $source" >&2; exit 1 ;;
  esac
  mkdir -p "$folder"
  uv run python - "$folder" "$label" "$model" "$system" "$limit" "${strategy[@]}" <<'EOF'
import sys
from pathlib import Path

import yaml

folder, label, model, system, limit, *strategy = sys.argv[1:]
rewrite = {
    # Inspect's provider prefix: tinker/<base model> or tinker/tinker://<checkpoint>.
    "model": model if model.startswith("tinker/") else f"tinker/{model}",
    "system_prompt": system,
    "user_prompt": "RL_SELF_REC_SENDER_USER_PROMPT",
    "signal": "RL_SIGNAL_ON",
    "max_tokens": 3072,
}
if strategy:
    rewrite[strategy[0]] = strategy[1]
attacks = (
    "/home/jkk/spar-2026-untrusted-monitoring/.claude/worktrees/v41-pilot/experiments/apps/"
    "generate/apps-262144_ft_a1611f-test-attacks-self_rec_subtle_attack-6f5c6a86b195/attacks.eval"
)
config = {
    "experiment_folder": str(Path(folder).resolve()),
    "experiment_name": "collusion-rl-v2",
    "defaults": {"rewrite": rewrite},
    "runs": {label: {"eval_path": attacks, "save_path": "attacks.eval",
                     "limit": None if limit == "null" else int(limit)}},
}
Path(folder, "rewrite.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
EOF
  INSPECT_LOG_DIR="$folder/logs" uv run python -m \
    lasr_labs_2025_control_project.scripts.rewrite_solutions_inspect --config "$folder/rewrite.yaml"
  ;;
score)
  M=$V41/monitor
  base=262144_ft_a1611f-self_rec_collusion
  exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
    "models.U='$U'" "name=rlv2_$label" models.generation_max_tokens=65536 \
    '~measurement.arms.sft' 'stages=[transform,monitor,game,analysis]' \
    'measurement.transforms=[none,paraphrased]' \
    optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
    "+inputs.generate.test.attacks=$folder/attacks.eval" \
    "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
    "+inputs.monitor.none.base.honest=$M/apps-262144_ft_a1611f-test-honest-honest.mon-$base-57a5bb885f16/honest.eval" \
    "+inputs.monitor.none.base.honeypots=$M/untracked-honeypots.mon-$base-0e73330bf758/honeypots.eval" \
    "+inputs.monitor.paraphrased.base.honest=$M/apps-262144_ft_a1611f-test-honest-honest.para-qwen3_5_4b-llm_summarize.mon-$base-8062ab61d0cd/honest.eval" \
    "+inputs.monitor.paraphrased.base.honeypots=$M/untracked-honeypots.para-qwen3_5_4b-llm_summarize.mon-$base-3ef184bddf22/honeypots.eval" \
    "wandb_tags=[collusion-rl-v2,$label]" "$@"
  ;;
report)
  # This label's games: their attacks input is a monitor output of this label's rewrite.
  games=()
  for game in experiments/apps/game/apps.self_rec.*.base-*; do
    if zcat "$game/analysis_data.json.gz" | grep -q "rewrite-collusion-rl-v2-$label"; then
      channel=${game#experiments/apps/game/apps.self_rec.}
      games+=(--game "${channel%%.*}=$game")
    fi
  done
  [ ${#games[@]} -gt 0 ] || { echo "no game for $label; run score first" >&2; exit 1; }
  exec uv run python -m lasr_labs_2025_control_project.finetuning.grouped_rl.report_test \
    --run "$1" --rewrites "$folder/attacks.eval" "${games[@]}"
  ;;
*) echo "step must be rewrite, score or report" >&2; exit 1 ;;
esac
