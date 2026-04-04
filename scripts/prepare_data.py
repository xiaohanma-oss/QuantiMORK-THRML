#!/usr/bin/env python3
"""Download Tiny Shakespeare and prepare tokenized data for training.

Creates the data files that PC-Transformers' dataloader expects:
  vendor/PC-Transformers/data_preparation/data/tiny_shakespear/{train,validation,test}.csv
  vendor/PC-Transformers/data_preparation/encoded/{train,valid,test}.pt
  vendor/PC-Transformers/data_preparation/tokenizer.json
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "PC-Transformers"
DATA_DIR = VENDOR_DIR / "data_preparation" / "data" / "tiny_shakespear"
ENCODED_DIR = VENDOR_DIR / "data_preparation" / "encoded"

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare():
    """Download raw Tiny Shakespeare text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "input.txt"

    if not raw_path.exists():
        print(f"Downloading Tiny Shakespeare → {raw_path}")
        urlretrieve(SHAKESPEARE_URL, raw_path)
    else:
        print(f"Already exists: {raw_path}")

    return raw_path


def split_data(raw_path):
    """Split into train/validation/test CSV files (80/10/10)."""
    with open(raw_path, "r") as f:
        text = f.read()

    n = len(text)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train.csv": text[:train_end],
        "validation.csv": text[train_end:val_end],
        "test.csv": text[val_end:],
    }

    for name, content in splits.items():
        path = DATA_DIR / name
        with open(path, "w") as f:
            f.write(content)
        print(f"  {name}: {len(content)} chars")


def tokenize():
    """Run PC-Transformers' tokenizer to produce encoded .pt files."""
    # Add vendor to path so we can import their modules
    sys.path.insert(0, str(VENDOR_DIR))
    sys.path.insert(0, str(VENDOR_DIR / "data_preparation"))

    from prepare_tokens import build_tokenizer, encode_and_save

    tokenizer = build_tokenizer()
    encode_and_save(tokenizer)
    print("Tokenization complete.")


def main():
    print("=== Preparing Tiny Shakespeare data ===")
    raw_path = download_shakespeare()
    split_data(raw_path)
    tokenize()
    print("=== Done ===")


if __name__ == "__main__":
    main()
