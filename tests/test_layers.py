import torch
from carbide_modules.layers import (
    NUM_SENTENCE_DIMS, STRIP_WIDTH, strips, word_id_for, Carbide,
)


def test_strips_shapes():
    x = torch.tensor([[ord(c) for c in "the cat."]])
    s = strips(x)
    T = x.shape[1]
    assert s["flags"].shape == (1, T, 6)
    assert s["strip"].shape[-1] == STRIP_WIDTH
    assert s["l3"].shape[-1] == NUM_SENTENCE_DIMS
    assert s["word_str"][0][0] == "the"
    assert s["word_ids"][0, 0] == s["word_ids"][0, 1]


def test_carbide_modes_run():
    m = Carbide(d_model=32, n_layers=1, d_state=8)
    x = torch.tensor([[ord(c) for c in "The cat sat."]])
    for mode in ("full", "l1_l2", "l1", "flags_only", "no_constraints", "plain_embedding"):
        y = m(x, mode=mode)
        assert y.shape == (1, x.shape[1], 256)
    y.sum().backward()


def test_known_word_or_unk():
    assert isinstance(word_id_for("the"), int)
    assert word_id_for("the") >= 0


def test_layerstack_on_model():
    m = Carbide(d_model=32, n_layers=1, d_state=8)
    assert hasattr(m, "layers")
    x = torch.tensor([[ord(c) for c in "Hi."]])
    m(x, mode="full")
    assert m.last_strips is not None
    assert m.blocks[0].constraint_proj.in_features == STRIP_WIDTH
