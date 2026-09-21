"""L1 hex / L2 word / L3 sentence strips.

Layer 1 rows = byte IDs. Layer 2 rows = words built from those bytes.
Layer 3 is a packed sentence slot + sentence dimensions.
Named strips are kept; values are ingested into the mixer / SSM.
"""
import csv
import json
import os
import re
from collections import Counter

import torch
import torch.nn as nn

from .mdbe import (
    LANGUAGE_MECHANICS_NAMES,
    NUM_CONSTRAINTS,
    NUM_LANGUAGE_MECHANICS,
    TOTAL_CONSTRAINTS,
    all_constraints,
    mdbe_constraints,
)

WORD_VOCAB_SIZE = 2048
UNK_ID = 0
WORD_VOCAB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "word_vocab.json"))

SENTENCE_DIM_NAMES = [
    "DECLARATIVE", "INTERROGATIVE", "IMPERATIVE",
    "NEGATED", "HAS_OBJECT", "MULTI_CLAUSE",
]
NUM_SENTENCE_DIMS = len(SENTENCE_DIM_NAMES)
STRIP_WIDTH = TOTAL_CONSTRAINTS + NUM_SENTENCE_DIMS

L1_NAMES = ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"]
L2_GRAMMAR_NAMES = list(LANGUAGE_MECHANICS_NAMES)
L3_NAMES = list(SENTENCE_DIM_NAMES)


def _is_alpha_byte(v):
    return (65 <= v <= 90) or (97 <= v <= 122)


def _lower_byte(v):
    return v + 32 if 65 <= v <= 90 else v


def build_word_vocab(corpus_path=None, max_size=WORD_VOCAB_SIZE - 1):
    """Top-N lowercase words + reserved UNK=0. Persists word_vocab.json."""
    if os.path.exists(WORD_VOCAB_PATH):
        with open(WORD_VOCAB_PATH) as f:
            words = json.load(f).get("words") or []
        if words:
            return words[:max_size]
    words = []
    if corpus_path and os.path.exists(corpus_path):
        text = open(corpus_path, "rb").read(3_000_000).decode("ascii", errors="replace").lower()
        counts = Counter(re.findall(r"[a-z]+", text))
        words = [w for w, _ in counts.most_common(max_size)]
        with open(WORD_VOCAB_PATH, "w") as f:
            json.dump({"words": words}, f)
    return words


_VOCAB_WORDS = None
_WORD_TO_ID = None


def word_vocab():
    global _VOCAB_WORDS, _WORD_TO_ID
    if _VOCAB_WORDS is None:
        _VOCAB_WORDS = build_word_vocab()
        _WORD_TO_ID = {w: i + 1 for i, w in enumerate(_VOCAB_WORDS[:WORD_VOCAB_SIZE - 1])}
    return _VOCAB_WORDS, _WORD_TO_ID


def word_id_for(token: str) -> int:
    _, table = word_vocab()
    return table.get(token.lower(), UNK_ID)


def spans_from_bytes(bytes_seq, history=None):
    """Per-position word string and sentence id. Causal. history = prior bytes."""
    B, T = bytes_seq.shape
    if history is None:
        history = bytes_seq.new_zeros(B, 0)
    H = history.shape[1]
    full = torch.cat([history, bytes_seq], dim=1)
    word_ids = torch.zeros(B, T, dtype=torch.long)
    sent_ids = torch.zeros(B, T, dtype=torch.long)
    word_str = [[""] * T for _ in range(B)]
    for b in range(B):
        chars = full[b].tolist()
        sid = 0
        buf = []
        start = 0
        for i, v in enumerate(chars):
            if _is_alpha_byte(int(v)):
                if not buf:
                    start = i
                buf.append(chr(_lower_byte(int(v))))
            else:
                if buf:
                    tok = "".join(buf)
                    wid = word_id_for(tok)
                    for j in range(start, i):
                        if j >= H:
                            t = j - H
                            word_ids[b, t] = wid
                            word_str[b][t] = tok
                    buf = []
                if i >= H:
                    t = i - H
                    sent_ids[b, t] = sid
                if int(v) in (46, 63, 33, 10):
                    sid += 1
                continue
            if i >= H:
                t = i - H
                sent_ids[b, t] = sid
        if buf:
            tok = "".join(buf)
            wid = word_id_for(tok)
            end = full.shape[1]
            for j in range(start, end):
                if j >= H:
                    t = j - H
                    if t < T:
                        word_ids[b, t] = wid
                        word_str[b][t] = tok
    return word_ids, sent_ids, word_str


def sentence_columns(cols, sent_ids):
    """L3 values: max of selected grammar hints over the current sentence so far."""
    names = LANGUAGE_MECHANICS_NAMES
    def idx(name):
        return names.index(name) if name in names else None

    gram = cols[..., NUM_CONSTRAINTS:]
    B, T, _ = gram.shape
    out = gram.new_zeros(B, T, NUM_SENTENCE_DIMS)
    i_neg = idx("negation")
    i_obj = idx("OBJECT")
    i_int = idx("INTERROGATIVE")
    i_imp = idx("IMPERATIVE")
    i_q = idx("pragmatics")
    for b in range(B):
        start = 0
        for t in range(T):
            if t and int(sent_ids[b, t]) != int(sent_ids[b, t - 1]):
                start = t
            sl = gram[b, start:t + 1]
            inter = 0.0
            if i_int is not None:
                inter = float(sl[:, i_int].max())
            if i_q is not None:
                inter = max(inter, float(sl[:, i_q].max()))
            imp = float(sl[:, i_imp].max()) if i_imp is not None else 0.0
            neg = float(sl[:, i_neg].max()) if i_neg is not None else 0.0
            obj = float(sl[:, i_obj].max()) if i_obj is not None else 0.0
            multi = 1.0 if (t - start) > 48 else 0.0
            decl = max(0.0, 1.0 - inter - imp)
            out[b, t, 0] = decl
            out[b, t, 1] = inter
            out[b, t, 2] = imp
            out[b, t, 3] = neg
            out[b, t, 4] = obj
            out[b, t, 5] = multi
    return out


def causal_pack(h, sent_ids):
    B, T, D = h.shape
    out = torch.zeros_like(h)
    for b in range(B):
        run = h.new_zeros(D)
        n = 0
        for t in range(T):
            if t and int(sent_ids[b, t]) != int(sent_ids[b, t - 1]):
                run = h.new_zeros(D)
                n = 0
            run = run + h[b, t]
            n += 1
            out[b, t] = run / n
    return out


def strips(bytes_seq, history=None):
    """Single API: named L1/L2/L3 values + ids. Used by train, generate, export."""
    cols = all_constraints(bytes_seq, history=history)
    flags = cols[..., :NUM_CONSTRAINTS]
    grammar = cols[..., NUM_CONSTRAINTS:]
    word_ids, sent_ids, word_str = spans_from_bytes(bytes_seq, history=history)
    l3 = sentence_columns(cols, sent_ids)
    strip = torch.cat([flags, grammar, l3], dim=-1)
    return {
        "cols": cols,
        "flags": flags,
        "grammar": grammar,
        "l3": l3,
        "strip": strip,
        "word_ids": word_ids,
        "sent_ids": sent_ids,
        "word_str": word_str,
    }


class LayerStack(nn.Module):
    """Ingest L1 values + L2 word row + L3 packed slot into d_model."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.byte = nn.Embedding(256, d_model)
        self.word = nn.Embedding(WORD_VOCAB_SIZE, d_model)
        self.l1_proj = nn.Linear(NUM_CONSTRAINTS, d_model, bias=False)
        self.l2_proj = nn.Linear(NUM_LANGUAGE_MECHANICS, d_model, bias=False)
        self.l3_proj = nn.Linear(NUM_SENTENCE_DIMS, d_model, bias=False)
        self.pack_proj = nn.Linear(d_model, d_model, bias=False)
        self.mix = nn.Linear(d_model, d_model)

    def forward(self, bytes_seq, history=None, mode="full", pack_state=None):
        s = strips(bytes_seq, history=history)
        flags, grammar, l3 = s["flags"], s["grammar"], s["l3"]
        word_ids = s["word_ids"].to(bytes_seq.device)
        sent_ids = s["sent_ids"].to(bytes_seq.device)

        l1 = self.byte(bytes_seq)
        if mode in ("full", "l1", "l1_l2", "l1_l2_l3", "flags_only"):
            l1 = l1 + self.l1_proj(flags)
        l2 = self.word(word_ids)
        if mode in ("full", "l1_l2", "l1_l2_l3"):
            l2 = l2 + self.l2_proj(grammar)
        x = l1
        if mode in ("full", "l1_l2", "l1_l2_l3"):
            x = x + l2
        if mode in ("full", "l1_l2_l3"):
            if pack_state is None:
                packed = causal_pack(l2, sent_ids)
            else:
                run, n, last_sid = pack_state
                packed = torch.zeros_like(l2)
                for t in range(l2.shape[1]):
                    sid = int(sent_ids[0, t])
                    if last_sid is not None and sid != last_sid:
                        run = l2.new_zeros(run.shape)
                        n = 0
                    run = run + l2[0, t]
                    n += 1
                    packed[0, t] = run / n
                    last_sid = sid
                pack_state[0] = run
                pack_state[1] = n
                pack_state[2] = last_sid
            x = x + self.pack_proj(packed) + self.l3_proj(l3)
        if mode == "flags_only":
            x = self.byte(bytes_seq) + self.l1_proj(flags)
        if mode == "no_constraints":
            x = self.byte(bytes_seq)
        s["pack_state"] = pack_state
        return self.mix(x), s


def named_values(s, batch=0, pos=0):
    """Traceable dump: each named column next to its number at one byte."""
    rows = []
    for i, name in enumerate(L1_NAMES):
        rows.append(("L1", name, float(s["flags"][batch, pos, i])))
    for i, name in enumerate(L2_GRAMMAR_NAMES):
        rows.append(("L2", name, float(s["grammar"][batch, pos, i])))
    for i, name in enumerate(L3_NAMES):
        rows.append(("L3", name, float(s["l3"][batch, pos, i])))
    return rows


def export_word_table(model, filepath, sample_text="the cat sat."):
    """Dump Layer-2 word rows + grammar + pinned L3 tags for a sample."""
    bt = torch.tensor([[ord(c) if ord(c) < 256 else 32 for c in sample_text]])
    s = strips(bt)
    seen = {}
    rows = []
    for t, w in enumerate(s["word_str"][0]):
        if not w or w in seen:
            continue
        seen[w] = True
        gram = s["grammar"][0, t].tolist()
        l3 = s["l3"][0, t].tolist()
        rows.append([w, int(s["word_ids"][0, t])]
                    + [f"{v:.4f}" for v in gram]
                    + [f"{v:.4f}" for v in l3])
    header = ["word", "word_id"] + L2_GRAMMAR_NAMES + L3_NAMES
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
        csv.writer(fh).writerows(rows)


def export_sentence_table(model, filepath, sample_text="The cat sat. Did it run?"):
    """Dump Layer-3 sentence dimensions for each sentence in a sample."""
    bt = torch.tensor([[ord(c) if ord(c) < 256 else 32 for c in sample_text]])
    s = strips(bt)
    rows = []
    last = None
    acc = []
    for t, ch in enumerate(sample_text):
        sid = int(s["sent_ids"][0, t])
        if last is None:
            last = sid
        if sid != last:
            l3 = s["l3"][0, t - 1].tolist()
            rows.append(["".join(acc).strip()] + [f"{v:.4f}" for v in l3])
            acc = []
            last = sid
        acc.append(ch)
    if acc:
        l3 = s["l3"][0, -1].tolist()
        rows.append(["".join(acc).strip()] + [f"{v:.4f}" for v in l3])
    header = ["sentence"] + L3_NAMES
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
        csv.writer(fh).writerows(rows)


MODES = ("full", "l1_l2_l3", "l1_l2", "l1", "flags_only", "no_constraints", "plain_embedding")


class Block(nn.Module):
    """Re-inject named L1/L2/L3 strip values into each SSM block."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        from .mdbe import SelectiveSSM
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.constraint_proj = nn.Linear(STRIP_WIDTH, d_model, bias=False)

    def forward(self, x, constraint_cols):
        if constraint_cols.shape[-1] != self.constraint_proj.in_features:
            pad = self.constraint_proj.in_features - constraint_cols.shape[-1]
            if pad > 0:
                constraint_cols = torch.cat(
                    [constraint_cols,
                     constraint_cols.new_zeros(*constraint_cols.shape[:-1], pad)],
                    dim=-1)
            else:
                constraint_cols = constraint_cols[..., :self.constraint_proj.in_features]
        h = self.norm(x) + self.constraint_proj(constraint_cols)
        return x + self.ssm(h)


class Carbide(nn.Module):
    """Selective SSM with the three-layer MDBE stack on the front."""

    def __init__(self, d_model=64, n_layers=2, d_state=16):
        super().__init__()
        from .mdbe import MDBE, LocalByteConv
        self.mdbe = MDBE(d_model)
        self.layers = LayerStack(d_model)
        self.mdbe.base = self.layers.byte
        self.local_conv = LocalByteConv(d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_state=d_state) for _ in range(n_layers)])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
        self.last_strips = None

    def forward(self, bytes_seq, mode="full", history=None):
        if mode == "plain_embedding":
            x = self.layers.byte(bytes_seq)
            constraint_cols = torch.zeros(
                *bytes_seq.shape, STRIP_WIDTH, device=bytes_seq.device)
            self.last_strips = None
        elif mode == "no_constraints":
            x = self.layers.mix(self.layers.byte(bytes_seq))
            constraint_cols = torch.zeros(
                *bytes_seq.shape, STRIP_WIDTH, device=bytes_seq.device)
            self.last_strips = None
        else:
            use = "full" if mode in ("full", "l1_l2_l3") else mode
            x, s = self.layers(bytes_seq, history=history, mode=use)
            self.last_strips = s
            constraint_cols = s["strip"]
            if mode in ("flags_only", "l1"):
                constraint_cols = constraint_cols.clone()
                constraint_cols[..., NUM_CONSTRAINTS:] = 0
            elif mode == "l1_l2":
                constraint_cols = constraint_cols.clone()
                constraint_cols[..., -NUM_SENTENCE_DIMS:] = 0

        x = x + self.local_conv(x)
        for block in self.blocks:
            x = block(x, constraint_cols)
        return self.head(self.head_norm(x))
