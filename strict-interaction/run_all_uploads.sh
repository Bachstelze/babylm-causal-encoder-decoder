#!/usr/bin/env bash
# Sequentially upload all experiments' checkpoints to the HF Hub.
# The upload_checkpoints.py script is idempotent: it skips revisions that
# are already complete on the Hub, so re-running this is safe.
set -u  # (no -e: we want to continue past a failing experiment)

cd "$(dirname "$0")"

# experiment_folder:repo_name
EXPERIMENTS=(
  "train_100m:BabyLM-2026-Baseline-Strict-Interaction"
  "train_10m:BabyLM-2026-Baseline-Strict-Small-Interaction"
)

LOGDIR=upload_logs
mkdir -p "$LOGDIR"

for entry in "${EXPERIMENTS[@]}"; do
  exp="${entry%%:*}"
  repo_name="${entry##*:}"
  echo "============================================================"
  echo "[$(date '+%F %T')] starting upload for: $exp -> $repo_name"
  echo "============================================================"
  log="$LOGDIR/${exp}.log"
  if uv run python upload_checkpoints.py "$exp" --repo-name "$repo_name" 2>&1 | tee "$log"; then
    echo "[$(date '+%F %T')] OK: $exp"
  else
    echo "[$(date '+%F %T')] FAILED: $exp (see $log)"
  fi
  # Small pause between experiments to be gentle on the Hub.
  sleep 5
done

echo "============================================================"
echo "[$(date '+%F %T')] all uploads attempted."
echo "============================================================"
