"""Causality regression tests for the three-layer stack.

An autoregressive next-byte model may only use bytes <= t to predict byte
t+1. Layer 2 used to give every letter of a word the id of the WHOLE word, so
byte 't' of "the" already knew the word was "the" -- the model was handed the
answer it was being trained to predict, and generation (which only has the
partial word) disagreed with training. These tests pin that down, using a real
(non-empty) word vocabulary so they cannot pass vacuously with every word UNK.

Run: python3 tests/test_causality.py   (from the repo root)
"""
import contextlib
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import layers as L
from carbide_modules.incremental import build_state, incremental_step

VOCAB_WORDS = ["the", "cat", "sat", "on", "mat", "did", "it", "run", "yes", "he", "she",
               "not", "was", "and", "a", "dog", "big", "warm", "sun", "hill", "green"]
TEXT = ("The cat sat on the mat. Did it run? Yes, it did. The sun was big and warm! "
        "She did not run to the green hill. A dog sat. The cat ran. ") * 6


@contextlib.contextmanager
def real_vocab(words=VOCAB_WORDS):
    """Point layers at a throwaway word_vocab.json so the test never depends on
    (or writes) a committed vocab file."""
    old_path = L.WORD_VOCAB_PATH
    with tempfile.TemporaryDirectory() as d:
        L.WORD_VOCAB_PATH = os.path.join(d, "word_vocab.json")
        if words is not None:
            with open(L.WORD_VOCAB_PATH, "w") as f:
                json.dump({"words": words}, f)
        L._reset_vocab_cache()
        try:
            yield
        finally:
            L.WORD_VOCAB_PATH = old_path
            L._reset_vocab_cache()


def enc(s):
    return torch.tensor([[ord(c) for c in s]])


def test_prefix_is_causal_in_every_mode():
    with real_vocab():
        assert L.word_id_for("the") != L.UNK_ID, "vocab not in effect -- test would be vacuous"
        torch.manual_seed(0)
        model = L.Carbide(d_model=32, n_layers=2, d_state=8).eval()
        a = TEXT[:60]
        for mode in L.MODES:
            for cut in (2, 9, 20, 33, 47):
                b = a[:cut] + "qzx" + a[cut + 3:]  # same prefix, different later bytes
                with torch.no_grad():
                    ya, yb = model(enc(a), mode=mode), model(enc(b), mode=mode)
                diff = float((ya[0, :cut] - yb[0, :cut]).abs().max())
                assert diff < 1e-5, (
                    f"mode={mode!r}: logits at positions < {cut} changed by {diff:.3e} when only "
                    f"LATER bytes changed -- the model is reading the future")


def test_word_is_known_only_once_its_delimiter_arrives():
    with real_vocab():
        s = L.strips(enc("the cat."))
        assert s["word_str"][0] == ["", "", "", "the", "the", "the", "the", "cat"], s["word_str"][0]
        the, cat = L.word_id_for("the"), L.word_id_for("cat")
        assert s["word_ids"][0].tolist() == [0, 0, 0, the, the, the, the, cat]


def test_incremental_decode_matches_full_forward_on_the_layered_model():
    """tests/test_incremental_decode.py builds the OLD mdbe.Carbide, so it never
    exercised the LayerStack path. Lengths straddle the 255-byte history window."""
    with real_vocab():
        torch.manual_seed(0)
        model = L.Carbide(d_model=32, n_layers=2, d_state=8).eval()
        for n in (1, 5, 60, 300):
            seq = enc(TEXT[:n])[0]
            with torch.no_grad():
                full = model(seq.unsqueeze(0), mode="full")[0, -1]
            state = build_state(model, seq[:-1].tolist())
            inc, _ = incremental_step(model, int(seq[-1]), state)
            diff = float((full - inc).abs().max())
            assert diff < 1e-4, f"L={n}: incremental decode differs from full forward by {diff:.3e}"


def test_vocab_built_after_first_use_is_picked_up():
    """word_vocab() used to cache an EMPTY vocabulary forever if it ran before
    word_vocab.json existed (e.g. generate from a loaded checkpoint before a
    dataset load), leaving every word UNK for the rest of the process."""
    with real_vocab(words=None):
        assert L.word_id_for("the") == L.UNK_ID  # no vocab yet -> UNK
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("the the the cat cat sat")
        try:
            L.build_word_vocab(f.name)
        finally:
            os.unlink(f.name)
        assert L.word_id_for("the") != L.UNK_ID, "vocab built later was ignored (stale empty cache)"


def test_load_dataset_with_explicit_path_builds_the_vocab():
    """load_dataset(filepath) (the menu route) used to return before the vocab
    was built; only the default-file route built it."""
    from carbide_modules import dataset
    from carbide_modules.config import config
    import io
    old_file = config.data_file
    with real_vocab(words=None):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("the cat sat on the mat. " * 30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                assert dataset.load_dataset(filepath=f.name)
            assert os.path.exists(L.WORD_VOCAB_PATH), "explicit-path load did not build word_vocab.json"
            assert L.word_id_for("the") != L.UNK_ID
        finally:
            os.unlink(f.name)
            config.data_file = old_file


if __name__ == "__main__":
    test_prefix_is_causal_in_every_mode()
    test_word_is_known_only_once_its_delimiter_arrives()
    test_incremental_decode_matches_full_forward_on_the_layered_model()
    test_vocab_built_after_first_use_is_picked_up()
    test_load_dataset_with_explicit_path_builds_the_vocab()
    print("ok")
