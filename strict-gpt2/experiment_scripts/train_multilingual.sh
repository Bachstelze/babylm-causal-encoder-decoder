#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
PROJECT="babylm_2026_gpt2"

for ds in en_nld_equal en_zho_equal nld_zho_equal en_nld_zho_equal; do
    echo ""
    echo "=========================================="
    echo "  Training: $ds"
    echo "=========================================="
    $PYTHON training.py \
        --dataset "$ds" \
        --words_per_epoch 100000000 \
        --experiment_name "$ds" \
        --use_wandb \
        --wandb_project_name "$PROJECT" \
        --wandb_experiment_name "$ds"
done

echo ""
echo "All multilingual runs complete."
