"""Verifies SelectiveSSM's chunked parallel scan (task1 Part 1) computes the
EXACT same function — forward values and gradients — as the plain
sequential loop it replaced. Not a smoke test: compares real torch
autograd gradients directly against a from-scratch sequential reference
implementation, across sequence lengths that exercise every edge case in
the chunking logic (T < chunk_size, T == chunk_size, T not divisible by
chunk_size, T == 1).

Run: python3 test_scan.py   (from carbide/tests/, or `python3 -m tests.test_scan` from carbide/)
"""
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules.mdbe import SelectiveSSM


def sequential_reference(ssm: SelectiveSSM, x: torch.Tensor) -> torch.Tensor:
    """The exact pre-Part1 algorithm, using the SAME module's parameters,
    for direct comparison — not a re-derivation, a literal transcription
    of what SelectiveSSM.forward used to do before the chunked scan."""
    Bsz, T, d = x.shape
    delta = F.softplus(ssm.W_delta(x))
    A = -torch.exp(ssm.A_log)
    a_bar = torch.exp(delta.unsqueeze(-1) * A)
    b_bar = delta.unsqueeze(-1) * ssm.W_B(x).unsqueeze(2)
    h = torch.zeros(Bsz, d, ssm.d_state, device=x.device)
    ys = []
    for t in range(T):
        h = a_bar[:, t] * h + b_bar[:, t]
        y_t = (ssm.W_C(x[:, t]).unsqueeze(1) * h).sum(-1)
        ys.append(y_t)
    return torch.stack(ys, dim=1) + ssm.D * x


def compare(T, chunk_size, B=2, d_model=6, d_state=4, seed=0, atol=1e-8, dtype=torch.float64):
    # float64 by default: at float32, W_C's gradient legitimately differs by
    # up to ~1e-4 between formulations because Cx=W_C(x) is now one batched
    # matmul instead of T separate ones — mathematically identical, but BLAS
    # accumulates the reduction in a different order, so float32 rounding
    # differs (confirmed: that same diff shrinks to ~1e-13 in float64, right
    # at the precision floor — i.e. accumulation-order noise, not a bug).
    torch.manual_seed(seed)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    ssm = SelectiveSSM(d_model, d_state=d_state, chunk_size=chunk_size)

    x1 = torch.randn(B, T, d_model, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    out_new = ssm.forward(x1)                     # chunked scan (the real, shipped code path)
    out_ref = sequential_reference(ssm, x2)        # plain loop, same parameters

    fwd_diff = (out_new - out_ref).abs().max().item()

    loss_new = (out_new ** 2).sum()
    loss_ref = (out_ref ** 2).sum()
    loss_new.backward()
    loss_ref.backward()

    grad_diff_x = (x1.grad - x2.grad).abs().max().item()

    # isolate parameter grads per formulation with fresh backward passes
    ssm.zero_grad()
    x1b = x1.detach().clone().requires_grad_(True)
    (ssm.forward(x1b) ** 2).sum().backward()
    grads_new = {n: p.grad.clone() for n, p in ssm.named_parameters()}

    ssm.zero_grad()
    x2b = x2.detach().clone().requires_grad_(True)
    (sequential_reference(ssm, x2b) ** 2).sum().backward()
    grads_ref = {n: p.grad.clone() for n, p in ssm.named_parameters()}

    max_param_diff = 0.0
    worst = None
    for n in grads_new:
        diff = (grads_new[n] - grads_ref[n]).abs().max().item()
        if diff > max_param_diff:
            max_param_diff = diff
            worst = n

    ok = fwd_diff < atol and grad_diff_x < atol and max_param_diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"{status} T={T:4d} chunk_size={chunk_size:3d} dtype={str(dtype).split('.')[-1]:9s} | "
          f"fwd_diff={fwd_diff:.2e} x_grad_diff={grad_diff_x:.2e} max_param_diff={max_param_diff:.2e} (worst={worst})")
    torch.set_default_dtype(prev_dtype)
    return ok


def main():
    cases = [
        (1, 16), (1, 1),
        (5, 16),      # T < chunk_size
        (16, 16),     # T == chunk_size exactly
        (17, 16),     # T just over one chunk
        (37, 16),     # not divisible by chunk_size
        (128, 16),    # the real default seq_len
        (128, 1),     # chunk_size=1 degenerates to the old per-step loop
        (128, 200),   # chunk_size > T
        (200, 32),
    ]
    all_ok = True
    print("-- primary correctness gate: float64, tight tolerance --")
    for T, cs in cases:
        all_ok &= compare(T, cs, dtype=torch.float64, atol=1e-8)

    print("\n-- sanity check at the precision actually used for training (float32) --")
    for T, cs in cases:
        # looser: float32 batched-vs-looped matmul reductions genuinely
        # differ at this level from accumulation order alone (see compare()).
        all_ok &= compare(T, cs, dtype=torch.float32, atol=2e-3)

    print("\nALL SCAN CHECKS PASSED" if all_ok else "\nSOME SCAN CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
