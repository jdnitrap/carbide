"""MDBE embedding and the Selective-SSM model architecture."""
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CONSTRAINTS = 6


def mdbe_constraints(bytes_seq: torch.Tensor) -> torch.Tensor:
    f = bytes_seq.float()
    cols = []
    cols.append(((f >= 65) & (f <= 90)) | ((f >= 97) & (f <= 122)))
    cols.append((f >= 48) & (f <= 57))
    cols.append((f >= 65) & (f <= 90))
    cols.append(((f >= 33) & (f <= 47)) | ((f >= 58) & (f <= 64)) |
                ((f >= 91) & (f <= 96)) | ((f >= 123) & (f <= 126)))
    cols.append((f == 32) | (f == 9) | (f == 10) | (f == 13))
    lead = ((f >= 0) & (f < 0x80)) | ((f >= 0xC0) & (f <= 0xFF))
    cols.append(lead)
    return torch.stack(cols, dim=-1).float()


def export_mdbe_table(model: "Carbide", filepath: str) -> None:
    """Writes the current learned embedding + constraint flags for all 256
    bytes to `filepath`. Shared by menu_save() (one-shot, on demand) and the
    periodic training-time snapshots (task1 refinement #4) so both produce
    the exact same format and can be diffed against each other directly."""
    rows = []
    for b in range(256):
        byte_tensor = torch.tensor([[b]])
        learned = model.mdbe.base(byte_tensor)[0, 0].tolist()
        live = mdbe_constraints(byte_tensor)[0, 0].tolist()
        char_repr = repr(chr(b)) if 32 <= b < 127 else ""
        rows.append([b, char_repr] + [f"{v:.4f}" for v in learned]
                    + [f"{v:.0f}" for v in live])
    header = (["byte", "character"]
              + [f"cell_{i}" for i in range(model.mdbe.base.embedding_dim)]
              + ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"])
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
        csv.writer(fh).writerows(rows)


class MDBE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.base = nn.Embedding(256, d_model)
        self.proj = nn.Linear(d_model + NUM_CONSTRAINTS, d_model)

    def forward(self, bytes_seq, live=True):
        learned = self.base(bytes_seq)
        cols = mdbe_constraints(bytes_seq) if live \
               else torch.zeros(*bytes_seq.shape, NUM_CONSTRAINTS)
        return self.proj(torch.cat([learned, cols], dim=-1))


class SelectiveSSM(nn.Module):
    """Selective SSM with a chunked-parallel scan (task1 Part 1).

    The recurrence h_t = a_bar_t*h_{t-1} + b_bar_t is unchanged — this is
    still the same sequential state-space scan, just computed with fewer
    Python-level loop iterations. A naive fully-parallel formulation would
    use a cumulative-product/division trick, but a_bar = exp(delta*A) with
    A = -exp(A_log) (A_log ~ Uniform(1,16)) routinely underflows to exactly
    0 in float32, which would make a division-based scan emit NaN. Instead:
    split T into chunks of `chunk_size`; within a chunk, do a real
    sequential loop (safe, no division) but batched across ALL chunks at
    once (folded into the batch dim); carry state across chunks with a
    second, much shorter sequential loop (num_chunks steps). Net effect:
    ~T Python iterations become ~chunk_size + T/chunk_size (e.g. 128 -> ~24
    at the default chunk_size=16). See gradcheck-style verification in
    tests/test_scan.py comparing this directly against the plain loop.
    """

    def __init__(self, d_model, d_state=16, chunk_size=16):
        super().__init__()
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.A_log = nn.Parameter(torch.log(
            torch.exp(torch.empty(d_model, d_state).uniform_(1, 16))))
        self.W_delta = nn.Linear(d_model, d_model)
        self.W_B = nn.Linear(d_model, d_state)
        self.W_C = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        Bsz, T, d = x.shape
        delta = F.softplus(self.W_delta(x))                       # (B,T,d)
        A = -torch.exp(self.A_log)                                 # (d,ds)
        a_bar = torch.exp(delta.unsqueeze(-1) * A)                  # (B,T,d,ds)
        b_bar = delta.unsqueeze(-1) * self.W_B(x).unsqueeze(2)       # (B,T,d,ds)

        h = self._chunked_scan(a_bar, b_bar)                        # (B,T,d,ds)

        Cx = self.W_C(x)                                            # (B,T,ds) — position-wise, same as per-t
        y = (Cx.unsqueeze(2) * h).sum(-1)                            # (B,T,d)
        return y + self.D * x

    def _chunked_scan(self, a_bar, b_bar):
        Bsz, T, d, ds = a_bar.shape
        C = min(self.chunk_size, T)
        num_chunks = math.ceil(T / C)
        Tp = num_chunks * C
        pad = Tp - T
        if pad > 0:
            a_bar = torch.cat([a_bar, a_bar.new_ones(Bsz, pad, d, ds)], dim=1)
            b_bar = torch.cat([b_bar, b_bar.new_zeros(Bsz, pad, d, ds)], dim=1)

        a_c = a_bar.reshape(Bsz * num_chunks, C, d, ds)
        b_c = b_bar.reshape(Bsz * num_chunks, C, d, ds)

        # Within-chunk recurrence: real sequential loop, but batched across
        # every chunk simultaneously — this is the piece that used to be a
        # T-iteration loop and is now only a C-iteration one.
        h_local = a_c.new_zeros(Bsz * num_chunks, d, ds)
        h_locals = []
        for t in range(C):
            h_local = a_c[:, t] * h_local + b_c[:, t]
            h_locals.append(h_local)
        h_local = torch.stack(h_locals, dim=1)                       # (B*nc,C,d,ds)
        P = torch.cumprod(a_c, dim=1)                                # (B*nc,C,d,ds) — pure product, safe even if it hits 0

        h_local = h_local.reshape(Bsz, num_chunks, C, d, ds)
        P = P.reshape(Bsz, num_chunks, C, d, ds)

        # Carry state across chunks — sequential, but only num_chunks steps.
        carry = a_bar.new_zeros(Bsz, d, ds)
        carries = [carry]
        for i in range(num_chunks - 1):
            carry = P[:, i, -1] * carries[-1] + h_local[:, i, -1]
            carries.append(carry)
        carry_stack = torch.stack(carries, dim=1)                    # (B,nc,d,ds)

        h_full = P * carry_stack.unsqueeze(2) + h_local              # (B,nc,C,d,ds)
        return h_full.reshape(Bsz, Tp, d, ds)[:, :T]


class Block(nn.Module):
    """task1 refinement #1: the 6 constraint flags are re-injected here via
    a small additive projection, not just consumed once by MDBE.proj at the
    input. MDBE.proj immediately blends learned+defined columns together, so
    without this the flags' traceability doesn't survive past layer 1 — the
    whole point of hand-coding them was to keep a legible "is this byte a
    digit/space/etc." signal available, and that only holds if later layers
    can still see it directly, not just a linearly-mixed trace of it."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.constraint_proj = nn.Linear(NUM_CONSTRAINTS, d_model, bias=False)

    def forward(self, x, constraint_cols):
        h = self.norm(x) + self.constraint_proj(constraint_cols)
        return x + self.ssm(h)


class LocalByteConv(nn.Module):
    """task1 refinement #2: a short CAUSAL depthwise 1D conv over a few
    neighboring bytes, right after MDBE and before the SSM stack, so the
    network gets a head start on merging bytes into word-like chunks
    instead of having to learn that unassisted in its first 1-2 SSM layers
    — the same job a BPE tokenizer would otherwise do for free.

    Causal is not optional: this is an autoregressive next-byte predictor,
    so padding is LEFT-only (kernel only ever sees the current position and
    the `kernel_size-1` positions before it). A non-causal conv would leak
    future bytes into predicting the current one — inflated training loss,
    broken generation, since generate() only ever has past bytes to give it.
    Depthwise (groups=d_model): local temporal mixing per channel, cheap;
    channel mixing already happens elsewhere via the Linear layers.
    """

    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)

    def forward(self, x):
        xt = x.transpose(1, 2)                            # (B,d_model,T)
        xt = F.pad(xt, (self.kernel_size - 1, 0))          # left-pad only — causal
        return self.conv(xt).transpose(1, 2)               # (B,T,d_model)


MODES = ("full", "no_constraints", "plain_embedding")


class Carbide(nn.Module):
    def __init__(self, d_model=64, n_layers=2, d_state=16):
        super().__init__()
        self.mdbe = MDBE(d_model)
        self.local_conv = LocalByteConv(d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_state=d_state) for _ in range(n_layers)])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)

    def forward(self, bytes_seq, mode="full"):
        live = (mode == "full")
        if mode == "plain_embedding":
            x = self.mdbe.base(bytes_seq)
        else:
            x = self.mdbe(bytes_seq, live=live)

        x = x + self.local_conv(x)

        constraint_cols = mdbe_constraints(bytes_seq) if live \
            else torch.zeros(*bytes_seq.shape, NUM_CONSTRAINTS, device=bytes_seq.device)
        for block in self.blocks:
            x = block(x, constraint_cols)

        return self.head(self.head_norm(x))
