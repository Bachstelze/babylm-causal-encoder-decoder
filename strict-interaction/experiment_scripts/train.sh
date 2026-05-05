#!/bin/bash
set -e

# Train 10M
python interactive_training.py \
    --dataset_size=10m \
    --num_rounds=2 \
    --num_warmup_steps=200 \
    --num_training_steps=20000 \
    --experiment_name=train_10m \
    --use_wandb \
    --wandb_project_name=babylm_interaction_2026 \
    --wandb_experiment_name=train_10m \
    --save_generated_data

# Train 100M
python interactive_training.py \
    --dataset_size=100m \
    --num_rounds=20 \
    --num_warmup_steps=2000 \
    --num_training_steps=200000 \
    --experiment_name=train_100m \
    --use_wandb \
    --wandb_project_name=babylm_interaction_2026 \
    --wandb_experiment_name=train_100m \
    --save_generated_data
