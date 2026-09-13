"""Byte-level dataset loading and batching."""
import os
import torch

from .config import config

FALLBACK = ("the little cat saw the sun. the sun was big and warm. "
            "the cat ran to the hill. the hill was green. ")

data = None  # set by load_dataset() — call it once before get_batch()


def load_dataset(filepath=None):
    """Load dataset from file or use default (Option 2, 3, 4)."""
    global data

    if filepath:
        filepath = filepath.strip()

        if not os.path.exists(filepath):
            print(f"  ✗ File not found: {filepath}")
            print(f"  Current directory: {os.getcwd()}")
            print(f"  Available .txt files: {', '.join([f for f in os.listdir('.') if f.endswith('.txt')]) or '(none)'}")
            return False

        if not os.path.isfile(filepath):
            print(f"  ✗ Not a file: {filepath}")
            return False

        try:
            text = open(filepath, encoding="utf-8", errors="ignore").read()
            data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
            config.data_file = filepath
            print(f"  ✓ Loaded {filepath}")
            print(f"  ✓ {len(text):,} chars, {len(data):,} bytes")
            return True
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return False
    else:
        # task1 Part 2: prefer the real ~5MB corpus over the 49.5KB built-in
        # fallback whenever it's sitting right there — using the tiny
        # fallback by default while scaling the model up would make the
        # "grow the corpus proportionally" half of Part 2 silently not
        # happen even though the real data was already on disk.
        if os.path.exists("train.txt"):
            text = open("train.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "train.txt"
            print(f"✓ loaded train.txt ({len(text):,} chars)")
        elif os.path.exists("carbide_training_dataset.txt"):
            text = open("carbide_training_dataset.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "carbide_training_dataset.txt"
            print(f"✓ loaded carbide_training_dataset.txt ({len(text):,} chars)")
        else:
            text = FALLBACK * 500
            config.data_file = "[fallback]"
            print("✓ using built-in fallback sample")

        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        print(f"✓ {len(data):,} bytes total\n")
        return True


def get_batch(batch=None, seq_len=None):
    if batch is None:
        batch = config.batch_size
    if seq_len is None:
        seq_len = config.seq_len
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,))
    xs = torch.stack([data[i : i + seq_len] for i in ix])
    ys = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return xs, ys
