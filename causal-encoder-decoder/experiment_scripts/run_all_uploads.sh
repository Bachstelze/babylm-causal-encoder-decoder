#!/usr/bin/env bash
# Sequentially upload all experiments' checkpoints to the HF Hub.
# The upload_checkpoints.py script is idempotent: it skips revisions that
# are already complete on the Hub, so re-running this is safe.
#
# Usage: run_all_uploads.sh <hf-org>
set -u  # (no -e: we want to continue past a failing experiment)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-org>" >&2
  exit 2
fi
ORG="$1"

cd "$(dirname "$0")/.."

EXPERIMENTS=(
  BabyLM-2026-Strict
  BabyLM-2026-Strict-Small
  babylm-nld
  babylm-zho
  en_nld_equal
  en_nld_zho_equal
  en_zho_equal
  nld_zho_equal
)

LOGDIR=upload_logs
mkdir -p "$LOGDIR"

for exp in "${EXPERIMENTS[@]}"; do
  echo "============================================================"
  echo "[$(date '+%F %T')] starting upload for: $exp"
  echo "============================================================"
  log="$LOGDIR/${exp}.log"
  if uv run python upload_checkpoints.py "$exp" --org "$ORG" 2>&1 | tee "$log"; then
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
