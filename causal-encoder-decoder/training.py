# File: training.py
# -----------------------------
# Main script for pretraining an encoder-decoder model

import os
from time import time

import torch
import torch.nn as nn
import wandb
from data_utils import load_babylm_data
from models import initialize_model_and_optimizers, save_epoch_checkpoint
from tqdm import tqdm
from utils import get_config, setup_experiment, setup_wandb


def compute_checkpoint_steps(epoch_steps, words_per_epoch, n_epochs):
    """Compute global step numbers at which to save checkpoints.
    Checkpoints at: every 1M words until 10M, every 10M until 100M, every 100M until 1B."""
    milestones = []
    # 1M, 2M, ..., 10M
    milestones.extend(
        range(1_000_000, min(10_000_001, words_per_epoch * n_epochs + 1), 1_000_000)
    )
    # 10M, 20M, ..., 100M
    milestones.extend(
        range(10_000_000, min(100_000_001, words_per_epoch * n_epochs + 1), 10_000_000)
    )
    # 100M, 200M, ..., 1B
    total_words = words_per_epoch * n_epochs
    milestones.extend(
        range(100_000_000, min(1_000_000_001, total_words + 1), 100_000_000)
    )

    steps_per_word = epoch_steps / words_per_epoch
    max_step = epoch_steps * n_epochs - 1
    checkpoint_steps = {}
    for words in sorted(set(milestones)):
        step = min(int(words * steps_per_word), max_step)
        if step > 0:
            label = (
                f"{words // 1_000_000}M"
                if words < 1_000_000_000
                else f"{words // 1_000_000_000}B"
            )
            checkpoint_steps[step] = label

    return checkpoint_steps


def full_train_loop(cfg, model, optimizer, scheduler, dataloader):
    start_time = time()
    epoch_size = len(dataloader)
    words_per_epoch = cfg["words_per_epoch"]
    checkpoint_steps = compute_checkpoint_steps(
        epoch_size, words_per_epoch, cfg["n_epochs"]
    )
    print(
        f"Epoch size: {epoch_size} steps ({words_per_epoch // 1_000_000}M words/epoch)"
    )
    print(f"Intermediate checkpoints at: {', '.join(checkpoint_steps.values())}")

    for epoch in range(cfg["n_epochs"]):
        torch.cuda.empty_cache()

        tr_metrics = train_epoch(
            cfg,
            model,
            optimizer,
            scheduler,
            dataloader,
            epoch,
            epoch_size,
            start_time,
            checkpoint_steps,
        )
        print(f"Epoch {epoch}; train loss: {tr_metrics['loss']}")
        metric_path = os.path.join(cfg["logdir"], f"epoch_{epoch}_metrics.pth")
        torch.save(tr_metrics, metric_path)

        checkpoint_dir = cfg["checkpoint_dir"]
        save_epoch_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir)


def train_epoch(
    cfg,
    model,
    optimizer,
    scheduler,
    dataloader,
    epoch,
    epoch_size,
    start_time,
    checkpoint_steps,
):
    model.train()
    total_loss = 0
    total_tokens = 0
    temp_loss = 0
    temp_tokens = 0

    device = model.device

    for train_step, batch in enumerate(tqdm(dataloader)):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # EncoderDecoderModel automatically creates decoder_input_ids from labels
        # (shifted right + decoder_start_token_id prefix) and trains causally.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        if cfg["gradient_clip_norm"] != -1:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Count non-masked tokens for perplexity
        num_tokens = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        temp_loss += loss.item() * num_tokens
        temp_tokens += num_tokens

        if cfg["use_wandb"] and (train_step % 10 == 0 and train_step > 0):
            steps = epoch_size * epoch + train_step
            wandb_train_epoch(temp_loss / temp_tokens, steps, start_time)
            temp_loss = 0
            temp_tokens = 0

        # Intermediate checkpoint saving at word-count milestones
        global_step = epoch * epoch_size + train_step
        if global_step in checkpoint_steps:
            label = checkpoint_steps[global_step]
            print(f"\n  Saving checkpoint at {label} words (step {global_step})")
            save_epoch_checkpoint(
                model, optimizer, scheduler, label, cfg["checkpoint_dir"]
            )

    return {"loss": total_loss / total_tokens}


def wandb_train_epoch(loss, step, start_time):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {
        "train_metrics/time_elapsed": time_elapsed,
        "train_metrics/batch_train_loss": loss,
    }
    wandb.log(curr_dict, step=step)


def main():
    cfg = get_config()

    setup_experiment(cfg)
    if cfg["use_wandb"]:
        setup_wandb(cfg)
    print("Env init")

    # Load data first to determine training steps
    dataloader = load_babylm_data(cfg)
    epoch_steps = len(dataloader)
    cfg["num_training_steps"] = epoch_steps * cfg["n_epochs"]
    cfg["num_warmup_steps"] = int(cfg["num_training_steps"] * cfg["warmup_ratio"])
    print(
        f"Training steps: {cfg['num_training_steps']} "
        f"({epoch_steps}/epoch x {cfg['n_epochs']} epochs, "
        f"{cfg['num_warmup_steps']} warmup)"
    )

    # Load the model and optimizers
    model, optimizer, scheduler = initialize_model_and_optimizers(cfg)
    print("Models loaded")
    print(
        f"  Encoder layers: {model.config.encoder.n_layer}, "
        f"embd: {model.config.encoder.n_embd}, "
        f"heads: {model.config.encoder.n_head}"
    )
    print(
        f"  Decoder layers: {model.config.decoder.n_layer}, "
        f"embd: {model.config.decoder.n_embd}, "
        f"heads: {model.config.decoder.n_head}"
    )
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    full_train_loop(cfg, model, optimizer, scheduler, dataloader)


if __name__ == "__main__":
    main()
