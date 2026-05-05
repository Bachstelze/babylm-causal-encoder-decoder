"""Download BabyLM-2026 datasets from HuggingFace and run baby_llama cleaning."""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

from baby_llama_clean import (
    cleanup_aochildes,
    cleanup_bnc_spoken,
    cleanup_gutenberg,
    cleanup_open_subtitles,
    cleanup_simple_wikipedia,
    cleanup_switchboard,
)

DATASETS = {
    "train_100M": "BabyLM-community/BabyLM-2026-Strict",
    "train_10M": "BabyLM-community/BabyLM-2026-Strict-Small",
}

CLEANUP_FUNCTIONS = {
    "childes": cleanup_aochildes,
    "bnc_spoken": cleanup_bnc_spoken,
    "gutenberg": cleanup_gutenberg,
    "open_subtitles": cleanup_open_subtitles,
    "simple_wiki": cleanup_simple_wikipedia,
    "switchboard": cleanup_switchboard,
}


def download(data_root: Path):
    for split, repo_id in DATASETS.items():
        target = data_root / split
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target),
        )


def clean(data_root: Path, seq_length: int):
    for split in DATASETS.keys():
        input_dir = data_root / split
        output_dir = data_root / f"clean_{split}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_files = [f for f in input_dir.iterdir() if f.is_file() and ".train" in f.suffixes]
        for file in train_files:
            name_key = file.name.split(".")[0]
            if name_key not in CLEANUP_FUNCTIONS:
                print(f"Skipping {file.name} (no cleanup function for '{name_key}')")
                continue
            text = file.read_text()
            cleaned = CLEANUP_FUNCTIONS[name_key](text, seq_length)
            (output_dir / file.name).write_text(cleaned)
            print(f"Cleaned {file.name} ({len(text)} -> {len(cleaned)}) in {split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/text_data"))
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_clean", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download(args.data_root)
    if not args.skip_clean:
        clean(args.data_root, args.seq_length)
