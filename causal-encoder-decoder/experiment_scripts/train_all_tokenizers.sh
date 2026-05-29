#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="train_tokenizer.py"

DATASETS=(
    # Monolingual
    BabyLM-2026-Strict
    BabyLM-2026-Strict-Small
    babylm-nld
    babylm-zho
    # Multilingual
    en_nld_equal
    en_zho_equal
    nld_zho_equal
    en_nld_zho_equal
)

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Training tokenizer for: $ds"
    echo "=========================================="
    $PYTHON $SCRIPT "$ds"
done

echo ""
echo "All tokenizers trained."
