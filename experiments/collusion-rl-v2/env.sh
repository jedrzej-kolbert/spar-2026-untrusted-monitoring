# Sourced by the launchers: the team's W&B project, with the ~/.netrc key (the .env key
# cannot write to the team, and load_dotenv would otherwise pick it up).
# WANDB_PROJECT may be set beforehand, e.g. to collusion-rl-v2-smoke for a wiring test.
export WANDB_ENTITY=spar-untrusted-monitoring-collusion WANDB_PROJECT=${WANDB_PROJECT:-collusion-rl-v2}
WANDB_API_KEY=$(awk '$0 ~ /machine api.wandb.ai/ {f=1} f && $1 == "password" {print $2; exit}' ~/.netrc)
export WANDB_API_KEY
[ -n "$WANDB_API_KEY" ] || { echo "no api.wandb.ai key in ~/.netrc" >&2; exit 1; }
