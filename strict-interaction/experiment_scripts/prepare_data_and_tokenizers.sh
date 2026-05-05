#!/bin/bash
set -e

DATA_ROOT=data/text_data

python prepare_data.py --data_root="${DATA_ROOT}"

python train_tokenizer.py \
    --input_dir="${DATA_ROOT}/clean_train_10M" \
    --output_dir=tokenizers/10m

python train_tokenizer.py \
    --input_dir="${DATA_ROOT}/clean_train_100M" \
    --output_dir=tokenizers/100m
