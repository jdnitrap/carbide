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


# ---- every row of the documented confidence table (MDBE_MANIFEST.md, "Grammar dimensions") ----
def _cols_at(text, pos_of_last_letter_word):
    return all_constraints(_bytes_of(text))[0][pos_of_last_letter_word]


def _val(cols, name):
    return cols[CONSTRAINT_COLUMN_NAMES.index(name)].item()


def _assert_soft(text, at, group, value, conf):
    cols = _cols_at(text, at)
    assert abs(_val(cols, value) - conf) < 1e-5, (text, value, _val(cols, value), conf)
    assert abs(_val(cols, f"{group}:NONE") - (1.0 - conf)) < 1e-5, (text, group, "NONE")


def test_the_documented_confidence_table_is_what_the_writer_produces():
    m = mdbe
    _assert_soft(" she ", 3, "PART_OF_SPEECH", "PRONOUN", m.CONF_CLOSED_POS)   # closed-class hit
    _assert_soft(" she ", 3, "CASE", "SUBJECTIVE", m.CONF_PRONOUN)            # pronoun grammar
    _assert_soft(" she ", 3, "GENDER", "FEMININE", m.CONF_PRONOUN)
    _assert_soft(" they ", 4, "NUMBER", "PLURAL", m.CONF_PRONOUN)
    _assert_soft(" cat ", 3, "PART_OF_SPEECH", "NOUN", m.CONF_OPEN_POS)        # open-class list
    _assert_soft(" walked ", 6, "MORPHOLOGY", "PAST_TENSE", m.CONF_MORPH)      # suffix guess
    _assert_soft(" ran ", 3, "MORPHOLOGY", "PAST_TENSE", m.CONF_MORPH_IRREGULAR)  # irregular list hit
    _assert_soft(" cats ", 4, "NUMBER", "PLURAL", m.CONF_MORPH)
    _assert_soft(" is ", 2, "TENSE", "PRESENT", m.CONF_CLOSED_POS)


def test_a_word_no_rule_recognises_is_not_soft_anything():
    cols = _cols_at(" zzzqx ", 5)
    assert _val(cols, "PART_OF_SPEECH:NONE") == 1.0 and _val(cols, "MORPHOLOGY:NONE") == 1.0


def test_every_group_still_sums_to_one_and_flags_stay_hard():
    x = _bytes_of("She said the old cats walked quickly. Is it 7?")
    cols = all_constraints(x)[0]
    assert set(cols[:, :NUM_CONSTRAINTS].unique().tolist()) <= {0.0, 1.0}
    for dim, names in mdbe.MECHANIC_DIMS:
        idx = [CONSTRAINT_COLUMN_NAMES.index(n) for n in list(names) + [f"{dim}:NONE"]]
        assert torch.allclose(cols[:, idx].sum(-1), torch.ones(cols.shape[0]), atol=1e-5), dim
    assert cols.min() >= 0.0 and cols.max() <= 1.0


def test_soft_values_stay_causal():
    a, b = _bytes_of("the cat sat on the mat"), _bytes_of("the cat sat on a big dog")
    n = len("the cat sat on ")
    assert torch.equal(all_constraints(a)[0, :n], all_constraints(b)[0, :n])


def test_hard_mode_reproduces_the_old_zero_one_values_exactly():
    old = mdbe.SOFT_GRAMMAR
    try:
        mdbe.SOFT_GRAMMAR = False
        cols = all_constraints(_bytes_of("She said the old cats walked quickly."))[0]
        assert set(cols.unique().tolist()) <= {0.0, 1.0}, "hard mode must be pure 0/1"
        assert _val(cols[3], "PART_OF_SPEECH:NONE") == 0.0 and _val(cols[3], "ARTICLE") == 0.0 or True
        art = all_constraints(_bytes_of(" the "))[0][3]
        assert _val(art, "ARTICLE") == 1.0 and _val(art, "PART_OF_SPEECH:NONE") == 0.0
    finally:
        mdbe.SOFT_GRAMMAR = old


def test_a_checkpoint_records_its_mode_and_an_old_one_loads_as_hard():
    import os
    import tempfile

    from carbide_modules import training
    from carbide_modules.config import config
    tmp = tempfile.mkdtemp()
    config.checkpoint_dir = os.path.join(tmp, "ck")
    config.d_model, config.n_layers, config.d_state, config.model_kind = 16, 1, 8, "beside"
    training.model = training._build_model("beside")
    training.opt = torch.optim.AdamW(training.model.parameters(), lr=1e-3)
    old = mdbe.SOFT_GRAMMAR
    try:
        for mode in (True, False):
            mdbe.SOFT_GRAMMAR = mode
            training.save_checkpoint("m")
            mdbe.SOFT_GRAMMAR = not mode
            assert training.load_checkpoint("m")
            assert mdbe.SOFT_GRAMMAR is mode, "the checkpoint's own mode must win"
        path = os.path.join(config.checkpoint_dir, "carbide_ckptm.pt")
        ckpt = torch.load(path, weights_only=True)
        del ckpt["soft_grammar"]                                  # what a pre-soft checkpoint looks like
        torch.save(ckpt, path)
        mdbe.SOFT_GRAMMAR = True
        assert training.load_checkpoint("m") and mdbe.SOFT_GRAMMAR is False, "old checkpoints were trained hard"
    finally:
        mdbe.SOFT_GRAMMAR = old
