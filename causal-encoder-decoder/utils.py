# File: utils.py
# --------------
# Minor utility functions

import argparse
import os
import random

import numpy as np
import torch
import yaml


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datapoint_length",
        type=int,
        help="The length of a datapoint, regardless of the underlying dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset folder name under data/ (e.g. BabyLM-2026-Strict, en_nld_equal)",
    )
    parser.add_argument(
        "--words_per_epoch",
        type=int,
        help="Number of words per epoch (e.g. 100000000 for 100M, 10000000 for 10M)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--n_epochs", type=int, help="Max number of epochs to train for a given round"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")

    parser.add_argument(
        "--learning_rate", type=float, help="The learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, help="The weight decay for training"
    )
    parser.add_argument(
        "--gradient_clip_norm", type=float, help="Gradient clipping value, if used"
    )

    # Encoder-decoder model architecture parameters
    parser.add_argument(
        "--from_pretrained",
        type=bool,
        help="Whether to load from a pretrained checkpoint",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Path or name of pretrained model"
    )
    parser.add_argument(
        "--decoder_start_token_id", type=int, help="Token ID for decoder start"
    )
    parser.add_argument(
        "--source_ratio",
        type=float,
        help="Fraction of datapoint used as encoder source (default: 0.5)",
    )

    # Encoder parameters
    parser.add_argument("--encoder_n_layer", type=int, help="Number of encoder layers")
    parser.add_argument(
        "--encoder_n_embd", type=int, help="Encoder embedding dimension"
    )
    parser.add_argument(
        "--encoder_n_head", type=int, help="Number of encoder attention heads"
    )
    parser.add_argument(
        "--encoder_n_inner", type=int, help="Encoder feed-forward inner dimension"
    )
    parser.add_argument(
        "--encoder_activation_function",
        type=str,
        help="Encoder activation function (e.g. gelu_new, relu, gelu)",
    )

    # Decoder parameters
    parser.add_argument("--decoder_n_layer", type=int, help="Number of decoder layers")
    parser.add_argument(
        "--decoder_n_embd", type=int, help="Decoder embedding dimension"
    )
    parser.add_argument(
        "--decoder_n_head", type=int, help="Number of decoder attention heads"
    )
    parser.add_argument(
        "--decoder_n_inner", type=int, help="Decoder feed-forward inner dimension"
    )
    parser.add_argument(
        "--decoder_activation_function",
        type=str,
        help="Decoder activation function (e.g. gelu_new, relu, gelu)",
    )

    # Experiment hyperparameters
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--base_folder",
        type=str,
        help="The name of the folder holding all experimentation data",
    )
    parser.add_argument(
        "--experiment_name", type=str, help="The name of the current experiment"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="If set, we will use wandb to log"
    )
    parser.add_argument(
        "--wandb_project_name", type=str, help="The project name for wandb"
    )
    parser.add_argument(
        "--wandb_experiment_name", type=str, help="The experiment name for wandb"
    )

    args = parser.parse_args()
    config = construct_config(args)
    return config


def setup_experiment(cfg):
    # Set the seed for reproducibility
    if cfg["seed"] == -1:
        cfg["seed"] = random.randint(0, 1000000)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make the relevant folders for the current experiment
    cfg["expdir"] = os.path.join(cfg["base_folder"], cfg["experiment_name"])
    cfg["checkpoint_dir"] = os.path.join(cfg["expdir"], "checkpoints")
    cfg["logdir"] = os.path.join(cfg["expdir"], "logging")
    mkdir(cfg["expdir"])
    mkdir(cfg["checkpoint_dir"])
    mkdir(cfg["logdir"])

    with open(os.path.join(cfg["logdir"], "exp_cfg.yaml"), "w") as cfg_file:
        yaml.dump(cfg, cfg_file)


def setup_wandb(cfg):
    wandb_input = {
        "name": cfg["wandb_experiment_name"],
        "project": cfg["wandb_project_name"],
    }
    import wandb

    wandb.init(**wandb_input)


def load_yaml(filepath):
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data


def construct_config(args):
    base_path = os.path.join("config.yaml")
    cfg = load_yaml(base_path)

    # Iterate over arguments and replace new arguments with defaults in the config
    args_dict = args.__dict__
    for key, value in args_dict.items():
        if value is None:
            continue
        cfg[key] = value

    return cfg
