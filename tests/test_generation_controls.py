"""Sampling controls: defaults are unchanged, seeds repeat, and each control does what it says."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import generation, training  # noqa: E402
from carbide_modules.config import config  # noqa: E402
from carbide_modules.incremental import IncrementalState, incremental_step  # noqa: E402


def _tiny_model():
    config.d_model, config.n_layers, config.d_state = 16, 1, 8
    torch.manual_seed(0)
    training.model = training._build_model("beside").eval()


def _original_generate(prompt, n_bytes, temperature, top_k):
    """The sampling loop exactly as it was before the new controls (short generations only)."""
    model, out = training.model, bytearray(prompt.encode())
    state, logits = IncrementalState(model), None
    for b in out:
        logits, state = incremental_step(model, b, state)
    for _ in range(n_bytes):
        adj = logits / temperature
        kth = adj.topk(top_k).values[-1]
        x = int(torch.multinomial(F.softmax(adj.masked_fill(adj < kth, float("-inf")), -1), 1))
        out.append(x)
        logits, state = incremental_step(model, x, state)
    return out.decode("utf-8", errors="ignore")


def test_defaults_reproduce_the_original_sampling_exactly():
    _tiny_model()
    torch.manual_seed(7)
    old = _original_generate("the ", 40, 0.8, 20)
    torch.manual_seed(7)
    assert generation.generate("the ", n_bytes=40, temperature=0.8, top_k=20) == old


def test_a_seed_makes_generation_repeatable():
    _tiny_model()
    a = generation.generate("the ", n_bytes=40, seed=3)
    b = generation.generate("the ", n_bytes=40, seed=3)
    c = generation.generate("the ", n_bytes=40, seed=4)
    assert a == b and a != c


def test_temperature_zero_is_greedy_and_ignores_the_seed():
    _tiny_model()
    a = generation.generate("the ", n_bytes=30, temperature=0, seed=1)
    b = generation.generate("the ", n_bytes=30, temperature=0, seed=2)
    assert a == b


def test_a_tiny_top_p_collapses_to_the_single_best_byte():
    _tiny_model()
    greedy = generation.generate("the ", n_bytes=30, temperature=0)
    nucleus = generation.generate("the ", n_bytes=30, temperature=1.0, top_k=256, top_p=1e-6, seed=5)
    assert nucleus == greedy


def test_repetition_penalty_moves_probability_off_recent_bytes():
    logits = torch.zeros(256)
    logits[65] = 5.0
    logits[66] = 4.0
    pick = lambda pen: generation.sample_byte(logits, [65, 65, 65], 0, 20, 1.0, pen, 0)
    assert pick(1.0) == 65 and pick(3.0) == 66


def test_no_repeat_ngram_blocks_the_byte_that_would_repeat_a_run():
    logits = torch.zeros(256)
    logits[3] = 9.0                                  # 3 would complete the repeated run 1,2,3
    out = [1, 2, 3, 7, 1, 2]
    assert generation.sample_byte(logits, out, 0, 20, 1.0, 1.0, 0) == 3
    assert generation.sample_byte(logits, out, 0, 20, 1.0, 1.0, 3) != 3


def test_presets_apply_and_explicit_values_override_them():
    _tiny_model()
    for name in generation.PRESETS:
        assert isinstance(generation.generate("the ", n_bytes=10, preset=name, seed=0), str)
    same = generation.generate("the ", n_bytes=20, preset="wild", temperature=0, seed=1)
    assert same == generation.generate("the ", n_bytes=20, temperature=0, top_k=60, top_p=0.98,
                                       repetition_penalty=1.12, no_repeat_ngram=12, seed=9)
    try:
        generation.generate("the ", preset="nope")
    except ValueError:
        return
    raise AssertionError("an unknown preset must raise")
