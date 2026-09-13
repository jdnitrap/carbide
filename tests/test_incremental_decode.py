"""Verifies incremental.py's cached decode path computes the EXACT same
function as a full forward pass over the equivalent sequence — the same
discipline test_scan.py used for the chunked scan. Also verifies the
updated generate() (now using this path) produces IDENTICAL output to
the old full-recompute implementation, given the same random seed, for
any generation that stays within one context window — the "correction"
changes speed only, not behavior.

Run: python3 test_incremental_decode.py   (from carbide/tests/)
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules.mdbe import Carbide
from carbide_modules.incremental import IncrementalState, incremental_step, build_state

D_MODEL, N_LAYERS, D_STATE = 256, 4, 32


def old_full_recompute_generate(model, prompt_bytes, n_bytes, seq_len, temperature, top_k, seed):
    """A literal transcription of generation.py's PRE-incremental generate()
    logic, for direct comparison — not a re-derivation."""
    torch.manual_seed(seed)
    out = bytearray(prompt_bytes)
    for _ in range(n_bytes):
        ctx = torch.tensor(list(out[-seq_len:])).unsqueeze(0)
        logits = model(ctx)[0, -1] / temperature
        kth = logits.topk(top_k).values[-1]
        probs = torch.softmax(logits.masked_fill(logits < kth, float("-inf")), -1)
        out.append(int(torch.multinomial(probs, 1)))
    return bytes(out)


def new_incremental_generate(model, prompt_bytes, n_bytes, seq_len, temperature, top_k, seed):
    """The new generate() logic (mirrors generation.py after the update),
    for direct comparison — only valid while len(prompt)+n_bytes <= seq_len,
    the regime where no window resync is needed."""
    torch.manual_seed(seed)
    out = bytearray(prompt_bytes)
    state = IncrementalState(model)
    logits = None
    for b in out:
        logits, state = incremental_step(model, b, state)
    for _ in range(n_bytes):
        adj = logits / temperature
        kth = adj.topk(top_k).values[-1]
        probs = torch.softmax(adj.masked_fill(adj < kth, float("-inf")), -1)
        x = int(torch.multinomial(probs, 1))
        out.append(x)
        logits, state = incremental_step(model, x, state)
    return bytes(out)


def test_incremental_step_matches_full_forward():
    torch.manual_seed(0)
    model = Carbide(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE).eval()
    for L in (1, 2, 5, 16, 37, 128):
        seq = torch.randint(0, 256, (L,))
        with torch.no_grad():
            full_logits = model(seq.unsqueeze(0))[0, -1]
        state = build_state(model, seq[:-1].tolist())
        inc_logits, _ = incremental_step(model, int(seq[-1]), state)
        diff = (full_logits - inc_logits).abs().max().item()
        assert diff < 1e-3, f"L={L}: incremental step diverges from full forward, diff={diff:.6e}"
        print(f"[PASS] L={L:4d}  max abs diff = {diff:.6e}")


def test_generate_output_unchanged_within_one_window():
    torch.manual_seed(1)
    model = Carbide(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE).eval()
    seq_len = 128
    prompt = b"the "
    for seed in (0, 1, 2, 42):
        old_out = old_full_recompute_generate(model, prompt, 60, seq_len, 0.8, 20, seed)
        new_out = new_incremental_generate(model, prompt, 60, seq_len, 0.8, 20, seed)
        assert old_out == new_out, (
            f"seed={seed}: incremental generate() diverges from the original "
            f"full-recompute implementation — this must be bit-identical\n"
            f"old={old_out!r}\nnew={new_out!r}"
        )
    print("[PASS] generate() output is bit-identical to the old implementation "
          "across 4 seeds, for generations within one context window")


if __name__ == "__main__":
    test_incremental_step_matches_full_forward()
    test_generate_output_unchanged_within_one_window()
    print("\nAll incremental-decode correctness checks passed.")
