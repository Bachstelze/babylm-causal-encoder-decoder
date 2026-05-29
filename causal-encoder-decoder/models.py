# File: models.py
# ---------------
# All functions related to loading and saving encoder-decoder models

import os

import torch
import transformers
from transformers import EncoderDecoderConfig, EncoderDecoderModel, GPT2Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from utils import mkdir

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


## INITIALIZATION ##


def _build_encoder_config(cfg):
    """Create an encoder config based on the experiment configuration."""
    dataset_name = cfg["dataset"]
    base_config = GPT2Config.from_pretrained(f"./configs/{dataset_name}")

    base_config.is_decoder = False
    base_config.add_cross_attention = False
    base_config.n_layer = cfg.get("encoder_n_layer", 6)
    base_config.n_embd = cfg.get("encoder_n_embd", 768)
    base_config.n_head = cfg.get("encoder_n_head", 12)
    base_config.n_inner = cfg.get("encoder_n_inner", None)
    base_config.activation_function = cfg.get("encoder_activation_function", "gelu_new")
    base_config.use_cache = False

    return base_config


def _build_decoder_config(cfg):
    """Create a decoder config based on the experiment configuration."""
    dataset_name = cfg["dataset"]
    base_config = GPT2Config.from_pretrained(f"./configs/{dataset_name}")

    base_config.is_decoder = True
    base_config.add_cross_attention = True
    base_config.n_layer = cfg.get("decoder_n_layer", 6)
    base_config.n_embd = cfg.get("decoder_n_embd", 768)
    base_config.n_head = cfg.get("decoder_n_head", 12)
    base_config.n_inner = cfg.get("decoder_n_inner", None)
    base_config.activation_function = cfg.get("decoder_activation_function", "gelu_new")
    base_config.use_cache = True

    return base_config


def initialize_model_and_optimizers(cfg):
    student = initialize_model(cfg)
    optimizer = initialize_optimizer(cfg, student)
    scheduler = initialize_scheduler(cfg, student, optimizer)
    return student, optimizer, scheduler


def initialize_model(cfg):
    """Create an EncoderDecoderModel with independently configurable encoder and decoder."""
    dataset_name = cfg["dataset"]

    if cfg.get("from_pretrained", False):
        model_name_or_path = cfg.get("model_name_or_path", None)
        if model_name_or_path is None:
            model_name_or_path = f"./configs/{dataset_name}"
        student = EncoderDecoderModel.from_pretrained(model_name_or_path)
    else:
        encoder_config = _build_encoder_config(cfg)
        decoder_config = _build_decoder_config(cfg)

        encdec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )
        student = EncoderDecoderModel(config=encdec_config)

    # Handle special token IDs
    special_cfg = GPT2Config.from_pretrained(f"./configs/{dataset_name}")
    if cfg.get("decoder_start_token_id") is not None:
        student.config.decoder_start_token_id = cfg["decoder_start_token_id"]
    else:
        student.config.decoder_start_token_id = special_cfg.bos_token_id

    student.config.eos_token_id = special_cfg.eos_token_id
    student.config.pad_token_id = special_cfg.eos_token_id  # GPT-2 uses eos as pad
    student.config.bos_token_id = special_cfg.bos_token_id

    student = student.to(DEVICE)
    return student


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def initialize_optimizer(cfg, student):
    lr = cfg["learning_rate"]
    decay_parameters = get_parameter_names(student, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in student.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in student.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999)
    )

    return optimizer


def initialize_scheduler(cfg, student, optimizer):
    num_training_steps = cfg["num_training_steps"]
    num_warmup_steps = cfg["num_warmup_steps"]
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


## SAVING AND LOADING ##
def save_epoch_checkpoint(student, optimizer, scheduler, epoch, checkpoint_dir):
    # Open a folder for the round
    folder = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    mkdir(folder)

    # Save the metrics and model
    torch.save(optimizer.state_dict(), os.path.join(folder, "latest_optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(folder, "latest_scheduler.pt"))

    # Better for Hugging Face models — saves config + weights
    student.save_pretrained(os.path.join(folder, "student"))

    # Also keep the raw state_dict for compatibility
    torch.save(student.state_dict(), os.path.join(folder, "latest_student.pt"))
