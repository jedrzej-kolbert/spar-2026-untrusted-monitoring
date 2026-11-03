#!/usr/bin/env bash
# Test scoring of one final strategy or checkpoint, the same way as the SFT judges:
#   test.sh rewrite <label> <source> [limit]   rewrite the 309 working test attacks (paid)
#       source: strategy=<icrl_result.json or wandb: ref to its strategy artifact>
#               model=<tinker:// sender checkpoint> | none (rewrite with no strategy)
#   test.sh score <label> [pipeline overrides]  monitors, game and analysis on the rewrites
#       (base judge, paraphrased and no jamming). Prints the plan only, unless given
#       allow_paid=true. Honeypots and honest solutions are the V4.1 runs' own, but they
#       are scored again here: Tinker's V4.1 scores on 3 Nov differ from the stored
#       28 Oct ones for the same prompt and code (e.g. YES - NO -1.75 vs -0.21), so
#       stored scores are not mixed with new ones. The pipeline scores them once per
#       project and reuses them for every later label.
#   test.sh report <label> <entity/project/run_id>  write the games' test scores into the
#       RL run's W&B page (test/, paraphrased_* and no_jamming_*), free
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
    "max_tokens": 4096,  # the same cap as every arm's search and training
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
  exec uv run python -m lasr_labs_2025_control_project.pipeline experiment=self_rec \
    "models.U='$U'" "name=rlv2_$label" models.generation_max_tokens=65536 \
    '~measurement.arms.sft' 'stages=[transform,monitor,game,analysis]' \
    'measurement.transforms=[none,paraphrased]' \
    optimization.bootstrap.replicates=200 optimization.bootstrap.chunk_size=100 \
    "+inputs.generate.test.attacks=$folder/attacks.eval" \
    "+inputs.generate.test.honeypots=$X/self_rec_attack_test/generator/honeypots.eval" \
    "+inputs.generate.test.honest=$V41/generate/apps-262144_ft_a1611f-test-honest-honest-267d38248808/honest.eval" \
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
