#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

DATASETS=(
    BabyLM-community/babylm-nld
    BabyLM-community/babylm-zho
    BabyLM-community/BabyLM-2026-Strict
    BabyLM-community/BabyLM-2026-Strict-Small
)

for repo in "${DATASETS[@]}"; do
    name="${repo##*/}"
    echo ""
    echo "=========================================="
    echo "  Downloading: $repo -> $DATA_DIR/$name"
    echo "=========================================="
    huggingface-cli download "$repo" --repo-type dataset --local-dir "$DATA_DIR/$name"
done

echo ""
echo "All datasets downloaded into $DATA_DIR/"
