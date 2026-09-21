"""The graph-fed model: reads the compiled graph table through causal word ids, so it must
(a) never see the future, (b) decode one byte at a time exactly like a full forward pass."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import layers  # noqa: E402
from carbide_modules.graphmem import N_GRAPH_COLS  # noqa: E402
from carbide_modules.incremental import IncrementalState, incremental_step  # noqa: E402

WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "a", "big", "red"]


def _model(mode="l1_l2_graph"):
    layers.WORD_VOCAB_PATH = os.path.join(tempfile.mkdtemp(), "word_vocab.json")
    layers._write_vocab_file(WORDS)
    layers._reset_vocab_cache()
    torch.manual_seed(0)
    m = layers.Carbide(d_model=16, n_layers=2, d_state=8, graph_cols=N_GRAPH_COLS, default_mode=mode)
    table = torch.zeros(layers.WORD_VOCAB_SIZE, N_GRAPH_COLS)
    table[1:len(WORDS) + 1] = torch.rand(len(WORDS), N_GRAPH_COLS)   # every known word has graph features
    m.set_graph_table(table)
    return m.eval()


def _bytes(text):
    return torch.tensor([[ord(c) for c in text]])


def test_graph_modes_never_see_the_future():
    m = _model()
    a, b = "the cat sat on the mat.", "the cat sat on a big dog"     # identical up to and including "the cat sat on "
    n = len("the cat sat on ")
    for mode in layers.GRAPH_MODES:
        with torch.no_grad():
            ya, yb = m(_bytes(a), mode=mode), m(_bytes(b), mode=mode)
        assert float((ya[0, :n] - yb[0, :n]).abs().max()) < 1e-6, f"{mode} leaked future bytes"


def test_the_graph_actually_changes_the_output():
    m = _model()
    with torch.no_grad():
        with_graph = m(_bytes("the cat sat on the mat."))
        m.layers.graph_table.zero_()
        without = m(_bytes("the cat sat on the mat."))
    assert float((with_graph - without).abs().max()) > 1e-4, "graph features should matter"


def test_incremental_decode_matches_the_full_forward_for_graph_modes():
    text = "the cat sat on a big red dog. a dog ran on the mat."
    for mode in layers.GRAPH_MODES:
        m = _model(mode)
        with torch.no_grad():
            full = m(_bytes(text))[0]
            state = IncrementalState(m)
            worst = 0.0
            for i, byte in enumerate(text.encode()):
                logits, state = incremental_step(m, byte, state)
                worst = max(worst, float((logits - full[i]).abs().max()))
        assert worst < 1e-4, f"{mode}: incremental decode drifted by {worst}"


def test_graph_modes_are_refused_by_a_model_without_graph_columns():
    m = layers.Carbide(d_model=16, n_layers=1, d_state=8)
    try:
        m(_bytes("the cat"), mode="l1_graph")
    except ValueError:
        return
    raise AssertionError("a model without graph_cols must refuse graph modes")


def test_a_table_that_does_not_fit_is_refused():
    m = _model()
    try:
        m.set_graph_table(torch.zeros(4, N_GRAPH_COLS + 1))
    except ValueError:
        return
    raise AssertionError("wrong-width table must be refused")
