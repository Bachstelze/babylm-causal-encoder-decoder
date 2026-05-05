"""Train BPE tokenizers matching the existing tokenizer architecture."""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, Regex, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(input_dir: Path, output_dir: Path, vocab_size: int = 16384):
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

    files = sorted(str(f) for f in input_dir.iterdir() if f.is_file() and f.suffix == ".txt")
    print(f"Training on {len(files)} files from {input_dir}:")
    for f in files:
        print(f"  {f}")

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save via PreTrainedTokenizerFast to get all config files
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=1024,
    )
    tok.save_pretrained(output_dir)
    print(f"Saved tokenizer to {output_dir} (vocab size: {tok.vocab_size})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=16384)
    args = parser.parse_args()
    train_tokenizer(args.input_dir, args.output_dir, args.vocab_size)
