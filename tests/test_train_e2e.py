"""End to end: load data -> train -> save -> reload -> generate -> resume, for both
model kinds, on a fresh directory (no carbide_checkpoints/ or mdbe_snapshots/ yet,
like a fresh clone). Also checks that a checkpoint carries its own word vocabulary
and that vocabulary ids stay stable as datasets are added.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import dataset, generation, layers, training  # noqa: E402
from carbide_modules.config import config  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "carbide_training_dataset.txt")


def _sandbox(kind):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "corpus.txt")
    with open(CORPUS, "rb") as f, open(path, "wb") as out:
        out.write(f.read(120_000))
    layers.WORD_VOCAB_PATH = os.path.join(tmp, "word_vocab.json")
    layers._reset_vocab_cache()
    config.d_model, config.n_layers, config.d_state = 16, 1, 8
    config.seq_len, config.batch_size = 32, 4
    config.checkpoint_dir = os.path.join(tmp, "ckpts")      # deliberately not created
    config.mdbe_snapshot_dir = os.path.join(tmp, "snaps")   # deliberately not created
    config.model_kind = kind
    training.model = training.opt = None
    assert dataset.load_dataset(path)
    return tmp


def _train(n):
    return [float(training.train_step()) for _ in range(n)]


def _roundtrip(kind):
    tmp = _sandbox(kind)
    training.init_model()
    losses = _train(50)
    assert sum(losses[-5:]) < sum(losses[:5]), f"{kind}: loss should fall"
    training.save_checkpoint()
    assert os.path.exists(os.path.join(config.checkpoint_dir, "carbide_ckpt.pt")), "checkpoint dir not created"
    training._maybe_snapshot_mdbe(config.mdbe_snapshot_interval)
    assert os.listdir(config.mdbe_snapshot_dir), "snapshot dir not created"
    before = [p.detach().clone() for p in training.model.parameters()]
    training.model = training.opt = None
    config.model_kind = "beside" if kind == "layered" else "layered"  # the checkpoint's own kind must win
    assert training.load_checkpoint()
    assert training._kind_of(training.model) == kind and config.model_kind == kind
    assert all(torch.equal(a, b) for a, b in zip(before, training.model.parameters())), "weights changed on reload"
    out = generation.generate("the ", n_bytes=30, temperature=0.8, top_k=10)
    assert isinstance(out, str) and out.startswith("the ") and len(out) > 4, out
    _train(5)  # resume after reload
    return tmp


def test_beside_model_end_to_end():
    _roundtrip("beside")


def test_layered_model_end_to_end():
    _roundtrip("layered")


def test_layered_checkpoint_carries_its_vocabulary():
    _roundtrip("layered")
    saved = list(layers.word_vocab()[0])
    assert saved
    training.save_checkpoint()
    # someone rebuilds the vocabulary differently between save and load
    layers._write_vocab_file(list(reversed(saved)))
    layers._reset_vocab_cache()
    training.model = training.opt = None
    assert training.load_checkpoint()
    assert layers._read_vocab_file() == saved, "checkpoint's vocabulary should be restored"
    assert os.path.exists(layers.WORD_VOCAB_PATH + ".bak"), "the conflicting file should be backed up"
    assert layers.word_id_for(saved[0]) == 1


def test_word_vocab_is_append_only():
    tmp = _sandbox("beside")
    first = layers._read_vocab_file()
    assert first and len(first) <= layers.WORD_VOCAB_SIZE - 1 - layers.WORD_VOCAB_HEADROOM
    second_corpus = os.path.join(tmp, "second.txt")
    with open(second_corpus, "w") as f:
        f.write("zzalpha zzbeta zzgamma " * 40)
    grown = layers.build_word_vocab(second_corpus)
    assert grown[:len(first)] == first, "existing word ids must never move"
    assert set(grown[len(first):]) == {"zzalpha", "zzbeta", "zzgamma"}
    assert layers.build_word_vocab(second_corpus) == grown, "re-running must be idempotent"


def test_unknown_model_kind_is_rejected():
    try:
        training._build_model("nope")
    except ValueError:
        return
    raise AssertionError("an unknown model kind should raise")
