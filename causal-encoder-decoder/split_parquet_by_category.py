"""
Split a HuggingFace parquet dataset (e.g. babylm-nld, babylm-zho) into
per-category .train.parquet files.

Reads the consolidated parquet shards from data/<dataset_name>/data/,
groups rows by the 'category' column, and writes one
data/<dataset_name>/<category>.train.parquet file per category.

Usage:
    python split_parquet_by_category.py <dataset_name>

Example:
    python split_parquet_by_category.py babylm-nld
    python split_parquet_by_category.py babylm-zho
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def split_dataset(dataset_name: str, data_root: Path = Path('data')):
    dataset_dir = data_root / dataset_name
    shards_dir = dataset_dir / 'data'

    if not shards_dir.exists():
        raise FileNotFoundError(f'No shards directory found at {shards_dir}')

    df = pd.read_parquet(shards_dir)
    print(f'Loaded {len(df):,} rows from {shards_dir}/')

    if 'category' not in df.columns:
        raise ValueError(f"No 'category' column in {dataset_name}")

    for cat in sorted(df['category'].unique()):
        subset = df[df['category'] == cat]
        fname = cat.replace(' ', '_') + '.train.parquet'
        out_path = dataset_dir / fname
        subset.to_parquet(out_path, index=False)
        print(f'  {fname}: {len(subset):,} rows')

    shutil.rmtree(shards_dir)
    print(f'Done. Wrote {df["category"].nunique()} files to {dataset_dir}/ '
          f'and removed {shards_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a parquet dataset by category.')
    parser.add_argument('dataset', help='Dataset folder name under data/ (e.g. babylm-nld)')
    args = parser.parse_args()
    split_dataset(args.dataset)
