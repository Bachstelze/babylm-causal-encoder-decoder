#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="clean_dataset.py"

DATASETS=(
    BabyLM-2026-Strict
    BabyLM-2026-Strict-Small
    babylm-nld
    babylm-zho
)

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Cleaning: $ds"
    echo "=========================================="
    $PYTHON $SCRIPT "$ds"
done

echo ""
echo "All datasets cleaned."
