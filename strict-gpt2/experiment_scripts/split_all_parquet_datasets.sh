#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="split_parquet_by_category.py"

DATASETS=(
    babylm-nld
    babylm-zho
)

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Splitting: $ds"
    echo "=========================================="
    $PYTHON $SCRIPT "$ds"
done

echo ""
echo "All parquet datasets split."
