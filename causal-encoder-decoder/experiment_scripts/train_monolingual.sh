#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
PROJECT="babylm_2026_gpt2"

declare -A WORDS_PER_EPOCH=(
    [BabyLM-2026-Strict]=100000000
    [BabyLM-2026-Strict-Small]=10000000
    [babylm-nld]=100000000
    [babylm-zho]=100000000
)

for ds in BabyLM-2026-Strict BabyLM-2026-Strict-Small babylm-nld babylm-zho; do
    echo ""
    echo "=========================================="
    echo "  Training: $ds"
    echo "=========================================="
    $PYTHON training.py \
        --dataset "$ds" \
        --words_per_epoch "${WORDS_PER_EPOCH[$ds]}" \
        --experiment_name "$ds" \
        --use_wandb \
        --wandb_project_name "$PROJECT" \
        --wandb_experiment_name "$ds"
done

echo ""
echo "All monolingual runs complete."
