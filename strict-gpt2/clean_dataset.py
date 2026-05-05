"""
Clean a BabyLM dataset (parquet or plain text).

Usage:
    python clean_dataset.py <dataset_name>

Example:
    python clean_dataset.py babylm-nld
    python clean_dataset.py babylm-zho
    python clean_dataset.py BabyLM-2026-Strict
    python clean_dataset.py BabyLM-2026-Strict-Small

Reads per-source .train.parquet or .train.txt files from data/<dataset_name>/,
applies source-appropriate cleaning to the text, and overwrites the originals.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# --- Punctuation sets (Latin + CJK fullwidth) ---

LATIN_PUNCT = r'.,;!?'
CJK_PUNCT = r'。，；！？'
ALL_PUNCT = LATIN_PUNCT + CJK_PUNCT


# --- Cleaning functions ---
# Each takes a string and returns a cleaned string.

def cleanup_extra_spaces(text):
    """Collapse multiple spaces and remove spaces before punctuation."""
    text = re.sub(r'[ \t\u00A0]+', ' ', text)
    text = re.sub(rf'[ \t\u00A0]([{ALL_PUNCT}])', r'\1', text)
    return text


def cleanup_wiki(text):
    """Clean wiki-style text: normalize paragraph breaks, strip stray HTML."""
    text = re.sub(r'<[^>]+>', '', text)  # strip HTML tags
    text = cleanup_extra_spaces(text)
    return text


def cleanup_speech(text):
    """Clean speech transcripts (CHILDES-style)."""
    return cleanup_extra_spaces(text)


def cleanup_subtitles(text):
    """Clean subtitle text: remove subtitle credits in any language."""
    # Match lines containing "subtitle" (English) or common CJK credit patterns
    text = re.sub(r'^.*subtitle.*$\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^.*字幕.*$\n?', '', text, flags=re.MULTILINE)  # Chinese "subtitles"
    text = re.sub(r'^.*ondertitel.*$\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)  # Dutch "subtitles"
    return cleanup_extra_spaces(text)


def cleanup_books(text):
    """Clean book text: strip stray HTML, normalize spaces."""
    text = re.sub(r'<[^>]+>', '', text)
    text = cleanup_extra_spaces(text)
    return text


def cleanup_news(text):
    """Clean news text: normalize spaces."""
    return cleanup_extra_spaces(text)


def cleanup_educational(text):
    """Clean educational text: strip HTML, normalize spaces."""
    text = re.sub(r'<[^>]+>', '', text)
    text = cleanup_extra_spaces(text)
    return text


# --- Source -> cleanup function mapping ---

CLEANUP_FUNCTIONS = {
    # Parquet datasets (NLD, ZHO, etc.)
    'child-wiki': cleanup_wiki,
    'child-directed-speech': cleanup_speech,
    'child-available-speech': cleanup_speech,
    'subtitles': cleanup_subtitles,
    'padding-opensubtitles': cleanup_subtitles,
    'child-books': cleanup_books,
    'child-news': cleanup_news,
    'educational': cleanup_educational,
    # Text datasets (Strict, Strict-Small)
    'simple_wiki': cleanup_wiki,
    'childes': cleanup_speech,
    'bnc_spoken': cleanup_speech,
    'switchboard': cleanup_speech,
    'open_subtitles': cleanup_subtitles,
    'gutenberg': cleanup_books,
}


def clean_parquet_files(input_dir, output_dir, files):
    for pf in files:
        source = pf.name.replace('.train.parquet', '')
        cleanup_fn = CLEANUP_FUNCTIONS.get(source)

        if cleanup_fn is None:
            print(f'  WARNING: no cleanup function for source "{source}", copying as-is')
            cleanup_fn = lambda x: x

        df = pd.read_parquet(pf)
        original_chars = df['text'].str.len().sum()
        df['text'] = df['text'].apply(cleanup_fn)
        cleaned_chars = df['text'].str.len().sum()

        out_path = output_dir / pf.name
        df.to_parquet(out_path, index=False)
        print(f'  Cleaned {pf.name}: {original_chars:,} -> {cleaned_chars:,} chars ({len(df)} rows)')


def clean_text_files(input_dir, output_dir, files):
    for tf in files:
        source = tf.name.replace('.train.txt', '')
        cleanup_fn = CLEANUP_FUNCTIONS.get(source)

        if cleanup_fn is None:
            print(f'  WARNING: no cleanup function for source "{source}", copying as-is')
            cleanup_fn = lambda x: x

        text = tf.read_text()
        original_len = len(text)
        cleaned_text = cleanup_fn(text)
        cleaned_len = len(cleaned_text)

        out_path = output_dir / tf.name
        out_path.write_text(cleaned_text)
        print(f'  Cleaned {tf.name}: {original_len:,} -> {cleaned_len:,} chars')


def clean_dataset(dataset_name: str, data_root: Path = Path('data')):
    input_dir = data_root / dataset_name

    parquet_files = sorted(input_dir.glob('*.train.parquet'))
    text_files = sorted(input_dir.glob('*.train.txt'))

    if parquet_files:
        clean_parquet_files(input_dir, input_dir, parquet_files)
    elif text_files:
        clean_text_files(input_dir, input_dir, text_files)
    else:
        raise FileNotFoundError(f'No .train.parquet or .train.txt files found in {input_dir}')

    print(f'Done. Files overwritten in {input_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean a BabyLM parquet dataset.')
    parser.add_argument('dataset', help='Dataset folder name under data/ (e.g. babylm-nld)')
    args = parser.parse_args()
    clean_dataset(args.dataset)
