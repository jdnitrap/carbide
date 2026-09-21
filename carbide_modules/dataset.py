"""Byte-level dataset loading and batching."""
import os
import torch

from .config import config

FALLBACK = ("the little cat saw the sun. the sun was big and warm. "
            "the cat ran to the hill. the hill was green. ")

data = None  # set by load_dataset() — call it once before get_batch()


def _build_layer2_vocab(path):
    """Build/persist the Layer-2 word vocabulary from the corpus. Every load
    path needs this: without it every word is UNK and Layer 2 is one row. A
    failure is reported, not swallowed."""
    try:
        from . import training
        from .layers import build_word_vocab
        if training.ensure_word_capacity():           # room for this dataset's new words
            print("\u2713 Word table grown to make room for new words")
        n = len(build_word_vocab(path))
        print(f"\u2713 Layer-2 word vocab {n} entries")
    except Exception as e:
        print(f"  ! Layer-2 word vocab not built ({type(e).__name__}: {e}); all words will be UNK")


def load_dataset(filepath=None):
    """Load dataset from file or use default (Option 2, 3, 4)."""
    global data

    if filepath:
        filepath = filepath.strip()

        if not os.path.exists(filepath):
            print(f"  \u2717 File not found: {filepath}")
            print(f"  Current directory: {os.getcwd()}")
            print(f"  Available .txt files: {', '.join([f for f in os.listdir('.') if f.endswith('.txt')]) or '(none)'}")
            return False

        if not os.path.isfile(filepath):
            print(f"  \u2717 Not a file: {filepath}")
            return False

        try:
            text = open(filepath, encoding="utf-8", errors="ignore").read()
            data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
            config.data_file = filepath
            print(f"  \u2713 Loaded {filepath}")
            print(f"  \u2713 {len(text):,} chars, {len(data):,} bytes")
            _build_layer2_vocab(filepath)
            return True
        except Exception as e:
            print(f"  \u2717 Error loading file: {e}")
            return False
    else:
        if os.path.exists("train.txt"):
            text = open("train.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "train.txt"
            print(f"\u2713 loaded train.txt ({len(text):,} chars)")
        elif os.path.exists("carbide_training_dataset.txt"):
            text = open("carbide_training_dataset.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "carbide_training_dataset.txt"
            print(f"\u2713 loaded carbide_training_dataset.txt ({len(text):,} chars)")
        else:
            text = FALLBACK * 500
            config.data_file = "[fallback]"
            print("\u2713 using built-in fallback sample")

        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        print(f"\u2713 {len(data):,} bytes total\n")
        _build_layer2_vocab(config.data_file if config.data_file and os.path.exists(config.data_file) else None)
        return True


holdout_start = None  # set by set_holdout(): training batches never touch data[holdout_start:]


def set_holdout(fraction=0.05):
    """Reserve the last `fraction` of the data for measuring (not training), so a held-out loss
    means something. Automatic model growth uses this as its gate."""
    global holdout_start
    holdout_start = int(len(data) * (1.0 - fraction)) if data is not None and fraction else None
    return holdout_start


def get_batch(batch=None, seq_len=None):
    if batch is None:
        batch = config.batch_size
    if seq_len is None:
        seq_len = config.seq_len
    hi = holdout_start if holdout_start is not None else len(data)
    ix = torch.randint(0, hi - seq_len - 1, (batch,))
    xs = torch.stack([data[i : i + seq_len] for i in ix])
    ys = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return xs, ys


def get_holdout_batch(generator, batch=None, seq_len=None):
    """A batch from the reserved tail (needs set_holdout first); `generator` makes it repeatable."""
    if holdout_start is None:
        raise RuntimeError("no held-out slice reserved -- call dataset.set_holdout() first")
    batch = batch or config.batch_size
    seq_len = seq_len or config.seq_len
    ix = torch.randint(holdout_start, len(data) - seq_len - 1, (batch,), generator=generator)
    return (torch.stack([data[i : i + seq_len] for i in ix]),
            torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]))
