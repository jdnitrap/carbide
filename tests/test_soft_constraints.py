"""Soft grammar scores + hard byte flags + flags_only ablation mask."""
import torch

import inspect

from carbide_modules import mdbe
from carbide_modules.mdbe import (
    BYTE_IDENTITY_COLUMN_NAMES,
    CONSTRAINT_COLUMN_NAMES,
    Carbide,
    NUM_CONSTRAINTS,
    all_constraints,
    mdbe_constraints,
)

# Commit a2f87a7 ("Soft grammar confidences ... flags_only ablation") added only THIS
# test file; the mdbe.py code it describes (the CONF_* confidences and
# all_constraints(..., flags_only=)) was never committed. Importing them by name made the
# whole file fail at import and hid the tests that do pass. Resolve them here so each
# dependent test fails on its own, saying what is missing.
CONF_CLOSED_POS = getattr(mdbe, "CONF_CLOSED_POS", None)
CONF_DISCOVERED = getattr(mdbe, "CONF_DISCOVERED", None)
HAS_FLAGS_ONLY = "flags_only" in inspect.signature(all_constraints).parameters


def _require_soft_confidences():
    assert CONF_CLOSED_POS is not None and CONF_DISCOVERED is not None, (
        "soft grammar confidences (CONF_CLOSED_POS / CONF_DISCOVERED) are not implemented in "
        "carbide_modules/mdbe.py -- commit a2f87a7 added this test but not the code")


def _bytes_of(s: str):
    return torch.tensor([[b for b in s.encode("ascii")]])


def test_six_flags_are_hard_bits():
    x = torch.arange(256).view(1, 256)
    flags = mdbe_constraints(x)
    assert flags.shape[-1] == 6
    assert set(flags.unique().tolist()) <= {0.0, 1.0}
    # '7' is 55, digit; 'A' is 65, upper letter; space is 32
    assert flags[0, 55, BYTE_IDENTITY_COLUMN_NAMES.index("is_digit")] == 1
    assert flags[0, 65, BYTE_IDENTITY_COLUMN_NAMES.index("is_alpha")] == 1
    assert flags[0, 65, BYTE_IDENTITY_COLUMN_NAMES.index("is_upper")] == 1
    assert flags[0, 32, BYTE_IDENTITY_COLUMN_NAMES.index("is_space")] == 1


def test_article_is_soft_not_one():
    _require_soft_confidences()
    x = _bytes_of(" the ")
    cols = all_constraints(x)[0]
    # last letter of 'the' is the byte before the trailing space
    article_i = CONSTRAINT_COLUMN_NAMES.index("ARTICLE")
    none_i = CONSTRAINT_COLUMN_NAMES.index("PART_OF_SPEECH:NONE")
    # position of 'e' in " the "
    e_pos = 3
    art = cols[e_pos, article_i].item()
    none = cols[e_pos, none_i].item()
    assert abs(art - CONF_CLOSED_POS) < 1e-5
    assert abs(none - (1.0 - CONF_CLOSED_POS)) < 1e-5
    assert 0.0 < art < 1.0


def test_discovered_cluster_is_weak():
    _require_soft_confidences()
    # 'alice' is in DISCOVERED_PREPOSITION_LIKE in the committed json
    x = _bytes_of(" alice ")
    cols = all_constraints(x)[0]
    name = "PREPOSITION_LIKE"
    if name not in CONSTRAINT_COLUMN_NAMES:
        return
    i = CONSTRAINT_COLUMN_NAMES.index(name)
    e_pos = len(" alice") - 1  # last letter of alice, before trailing space
    val = cols[e_pos, i].item()
    assert val in (0.0, CONF_DISCOVERED) or abs(val - CONF_DISCOVERED) < 1e-5
    if val > 0:
        assert abs(val - CONF_DISCOVERED) < 1e-5


def test_flags_only_zeroes_grammar_keeps_facts():
    assert HAS_FLAGS_ONLY, ("all_constraints has no flags_only argument -- not implemented in "
                            "carbide_modules/mdbe.py (see a2f87a7)")
    x = _bytes_of(" the 7")
    full = all_constraints(x)
    flags = all_constraints(x, flags_only=True)
    assert torch.equal(full[..., :NUM_CONSTRAINTS], flags[..., :NUM_CONSTRAINTS])
    assert torch.all(flags[..., NUM_CONSTRAINTS:] == 0)
    assert full[..., NUM_CONSTRAINTS:].abs().sum() > 0


def test_forward_modes_same_shape():
    model = Carbide(d_model=32, n_layers=1, d_state=8)
    x = _bytes_of("Hello 7")
    logits = {m: model(x, mode=m) for m in
              ("full", "flags_only", "no_constraints", "plain_embedding")}
    shape = logits["full"].shape
    assert shape[-1] == 256
    for v in logits.values():
        assert v.shape == shape


if __name__ == "__main__":
    test_six_flags_are_hard_bits()
    test_article_is_soft_not_one()
    test_discovered_cluster_is_weak()
    test_flags_only_zeroes_grammar_keeps_facts()
    test_forward_modes_same_shape()
    print("ok")
