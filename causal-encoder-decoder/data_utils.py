# File: data_utils.py
# -------------------
# Function for dataset loading, construction and saving + collation functions
# Adapted for encoder-decoder models with source-target pairs.

import math
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def load_texts_from_dir(data_dir: Path) -> list[tuple[str, str]]:
    """Load all training texts from a dataset directory.
    Returns list of (source_name, text) pairs.
    Handles both .txt and .parquet files."""
    texts = []

    for f in sorted(data_dir.glob("*.train.txt")):
        source = f.stem.replace(".train", "")
        text = f.read_text()
        texts.append((source, text))

    for f in sorted(data_dir.glob("*.train.parquet")):
        source = f.stem.replace(".train", "")
        df = pd.read_parquet(f)
        text = "\n".join(df["text"].tolist())
        texts.append((source, text))

    return texts


class FullEncoderDecoderDataset(Dataset):
    def __init__(self, cfg):
        dataset_name = cfg["dataset"]

        # Load the tokenizer
        self.processor = AutoTokenizer.from_pretrained(f"./tokenizers/{dataset_name}")
        self.model_bos = self.processor.bos_token_id
        self.model_eos = self.processor.eos_token_id
        self.model_pad = self.processor.pad_token_id or self.model_eos

        # Source length ratio: what fraction of the datapoint is used as source context
        self.source_ratio = cfg.get("source_ratio", 0.5)

        # Load and tokenize each source file
        self.data = []
        data_dir = Path("data") / dataset_name
        texts = load_texts_from_dir(data_dir)

        for source_name, all_text in texts:
            print(f"Opened {source_name} ({len(all_text):,} chars)")

            # Process full text into tokens (no special tokens; bos/eos added in __getitem__)
            tokenized_dataset = self.processor(
                text=[all_text], add_special_tokens=False
            )["input_ids"][0]
            print(f"Tokenized {source_name}; {len(tokenized_dataset):,} tokens total")

            # Chunk and add (reserve 2 tokens for bos/eos on target side)
            chunk_size = cfg["datapoint_length"] - 2
            num_chunks = math.ceil(len(tokenized_dataset) / chunk_size)
            for curr_chunk in tqdm(range(num_chunks)):
                start = curr_chunk * chunk_size
                end = (curr_chunk + 1) * chunk_size
                chunk_tokens = tokenized_dataset[start:end]
                self.data.append(chunk_tokens)
            print(f"Chunked {source_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return (source_input_ids, target_input_ids) where target includes bos/eos."""
        chunk = self.data[idx]

        # Split into source (context) and target (continuation) portions
        split_point = max(1, int(len(chunk) * self.source_ratio))
        source_tokens = chunk[:split_point]
        target_tokens = [self.model_bos] + chunk[split_point:] + [self.model_eos]

        return (
            torch.LongTensor(source_tokens),
            torch.LongTensor(target_tokens),
        )


## General utilities ##
def load_babylm_data(cfg):
    dataset_name = cfg["dataset"]
    cache_dir = Path("data/cached_train")
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir / f"train_encdec_{dataset_name}.pkl"

    if filename.exists():
        with open(filename, "rb") as f:
            full_babylm_dset = pickle.load(f)
    else:
        full_babylm_dset = FullEncoderDecoderDataset(cfg)
        with open(filename, "wb") as f:
            pickle.dump(full_babylm_dset, f)

    collate_fn = get_collate_fn(full_babylm_dset.model_pad)
    dataloader = DataLoader(
        full_babylm_dset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        source_batch = [item[0] for item in batch]
        target_batch = [item[1] for item in batch]

        # Pad source sequences
        source_input_ids = pad_sequence(
            source_batch, padding_value=pad_token_id, batch_first=True
        )
        source_attention_mask = (source_input_ids != pad_token_id).long()

        # Pad target sequences (full: bos + continuation + eos)
        labels = pad_sequence(
            target_batch, padding_value=pad_token_id, batch_first=True
        ).clone()

        # Mask padding positions in labels with -100 so loss ignores them.
        # EncoderDecoderModel internally handles decoder_input_ids (shift + decoder_start_token_id)
        # and computes cross-entropy between shifted logits and these labels.
        labels[labels == pad_token_id] = -100

        return {
            "input_ids": source_input_ids,
            "attention_mask": source_attention_mask,
            "labels": labels,
        }

    return collate_fn
