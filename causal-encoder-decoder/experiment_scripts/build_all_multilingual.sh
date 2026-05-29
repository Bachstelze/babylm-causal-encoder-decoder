#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="build_multilingual_dataset.py"

# 2-language datasets: 100M / 2 = 50M words per language
$PYTHON $SCRIPT --languages en nld  --words-per-lang 50000000 --output data/en_nld_equal
$PYTHON $SCRIPT --languages nld zho --words-per-lang 50000000 --output data/nld_zho_equal
$PYTHON $SCRIPT --languages en zho  --words-per-lang 50000000 --output data/en_zho_equal

# 3-language dataset: 100M / 3 = 33333333 words per language
$PYTHON $SCRIPT --languages en nld zho --words-per-lang 33333333 --output data/en_nld_zho_equal

echo ""
echo "All multilingual datasets built."
