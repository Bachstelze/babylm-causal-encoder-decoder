"""
Train a BPE tokenizer on a dataset folder under data/.

Usage:
    python train_tokenizer.py <dataset_folder>

Example:
    python train_tokenizer.py en_nld_equal
    python train_tokenizer.py BabyLM-2026-Strict

Handles both .txt and .parquet files (extracts the 'text' column).
Saves the tokenizer to tokenizers/<dataset_folder>/ and a matching
GPT-2 config to configs/<dataset_folder>/config.json.
"""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, Regex, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def collect_training_files(input_dir: Path):
    """Collect .txt files and convert .parquet files to temporary .txt files.
    Returns (list of file paths, list of temp files to clean up)."""
    txt_files = sorted(str(f) for f in input_dir.glob('*.txt') if f.stem != 'README')
    temp_files = []

    for pf in sorted(input_dir.glob('*.parquet')):
        df = pd.read_parquet(pf)
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write('\n'.join(df['text'].tolist()))
        tmp.close()
        txt_files.append(tmp.name)
        temp_files.append(tmp.name)
        print(f"  Extracted {len(df)} rows from {pf.name}")

    return txt_files, temp_files


def train_tokenizer(dataset_name: str, vocab_size: int = 16384,
                    data_root: Path = Path('data'),
                    tokenizer_root: Path = Path('tokenizers'),
                    config_root: Path = Path('configs')):
    input_dir = data_root / dataset_name
    output_dir = tokenizer_root / dataset_name
    config_dir = config_root / dataset_name

    print(f"Training tokenizer for '{dataset_name}'")
    print(f"  Input:     {input_dir}")
    print(f"  Tokenizer: {output_dir}")
    print(f"  Config:    {config_dir}")

    files, temp_files = collect_training_files(input_dir)
    if not files:
        raise FileNotFoundError(f'No .txt or .parquet files found in {input_dir}')

    print(f"  Training on {len(files)} files:")
    for f in files:
        print(f"    {f}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFKC(),
        normalizers.Replace(Regex(r"\n"), "\n "),
        normalizers.Replace(Regex(r" *\n"), "\n"),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
            behavior="isolated",
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        pre_tokenizers.Split(Regex(r".{1,24}"), behavior="isolated"),
    ])

    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    tokenizer.train(files, trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(),
        decoders.Strip(" ", left=1, right=0),
        decoders.Replace("\n ", "\n"),
    ])

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=1024,
    )
    tok.save_pretrained(output_dir)
    print(f"  Saved tokenizer to {output_dir} (vocab size: {tok.vocab_size})")

    # Save GPT-2 config
    config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 1,
        "embd_pdrop": 0.1,
        "eos_token_id": 2,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_inner": None,
        "n_layer": 12,
        "n_positions": 1024,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.1,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
                "do_sample": True,
                "max_length": 50
            }
        },
        "use_cache": True,
        "vocab_size": vocab_size,
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")

    # Clean up temp files
    for tmp in temp_files:
        Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on a dataset folder.')
    parser.add_argument('dataset', help='Dataset folder name under data/ (e.g. en_nld_equal)')
    parser.add_argument('--vocab_size', type=int, default=16384)
    args = parser.parse_args()
    train_tokenizer(args.dataset, args.vocab_size)
