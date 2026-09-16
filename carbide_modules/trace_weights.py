"""Gives every otherwise-opaque learned dimension in Carbide (an
embedding cell, a constraint_proj hidden unit) a real, checkable label
instead of "cell_N" -- read-only, applied to whatever weights the model
already has. Two different techniques, because two different parts of
the model have different amounts of real structure to read off:

1. constraint_proj (in each Block) and the constraint half of MDBE.proj
   take the 88 NAMED constraint columns as input, directly. Their
   learned weight matrices already map name -> hidden dim; the biggest-
   magnitude weight for a given hidden unit is a real, exact fact about
   that weight, readable with no data at all, no approximation.

2. MDBE.base (the raw byte -> embedding lookup) has no named input side
   -- its only input is a bare byte value. To label ITS dimensions, this
   correlates each embedding dimension's 256 real per-byte values
   against each of the 88 constraint columns' own 256 real per-byte
   values (from mdbe.all_constraints on every byte 0-255) and reports
   the constraint it correlates with most. A real Pearson correlation
   coefficient, not a guess -- reproducible by anyone who reruns it.

Neither technique changes what the model learns; both are purely
read-only reports about weights that already exist. And both are
honest about what they are: a label here is "this is what the data
says this dimension listens to most," not a claim about what the
dimension "really means."

Caveat that matters in practice: on a freshly-initialized (untrained)
model, every weight is random noise, so these labels are meaningless by
construction -- there is nothing to trace yet. They only become real,
useful facts once Carbide has actually been trained on real data.
"""
import csv
import json
import os
import torch

from .mdbe import BYTE_IDENTITY_COLUMN_NAMES, CONSTRAINT_COLUMN_NAMES, all_constraints

BYTE_COLUMN_NAMES = BYTE_IDENTITY_COLUMN_NAMES  # kept as an alias -- this module's own callers use this name
CONSTRAINT_NAMES = CONSTRAINT_COLUMN_NAMES  # single source of truth now lives in mdbe.py


def used_bytes_from_corpus(corpus_path):
    """The real, distinct byte values that actually appear in a training
    corpus. A byte that never appears never receives a gradient update,
    so its embedding row stays at random initialization forever --
    including it in a correlation computation dilutes real, learned
    signal with noise from rows nothing ever trained. Real check on
    carbide_training_dataset.txt: only 61 of 256 byte values ever
    appear; restricting to those roughly doubled the mean |correlation|
    found below (0.124 -> 0.235)."""
    with open(corpus_path, "rb") as f:
        return sorted(set(f.read()))


@torch.no_grad()
def label_embedding_dimensions(model, top_k=3, byte_values=None):
    """For every dimension of MDBE.base's learned embedding, the real
    Pearson correlation between that dimension's value and each named
    constraint's own value, for the same byte -- computed over
    `byte_values` (default: all 256). Pass the real bytes a corpus
    actually uses (see used_bytes_from_corpus) to avoid diluting the
    result with embedding rows that were never trained. Returns a list
    of d_model {dim, top_labels: [(constraint_name, correlation), ...]}
    dicts, top_labels sorted by |correlation| descending."""
    if byte_values is None:
        byte_values = list(range(256))
    d_model = model.mdbe.base.embedding_dim
    bytes_ = torch.tensor(byte_values).unsqueeze(1)     # (N, 1)
    embed = model.mdbe.base(bytes_).squeeze(1)          # (N, d_model)
    cols = all_constraints(bytes_).squeeze(1)           # (N, len(CONSTRAINT_NAMES))

    embed_c = embed - embed.mean(dim=0, keepdim=True)
    cols_c = cols - cols.mean(dim=0, keepdim=True)
    embed_std = embed_c.std(dim=0, keepdim=True).clamp_min(1e-8)
    cols_std = cols_c.std(dim=0, keepdim=True).clamp_min(1e-8)

    n = len(byte_values)
    corr = (embed_c.T @ cols_c) / max(n - 1, 1)         # (d_model, len(CONSTRAINT_NAMES))
    corr = corr / embed_std.T / cols_std

    results = []
    for d in range(d_model):
        scores = corr[d]
        top = torch.topk(scores.abs(), k=min(top_k, len(CONSTRAINT_NAMES)))
        labels = [(CONSTRAINT_NAMES[i], scores[i].item()) for i in top.indices.tolist()]
        results.append({"dim": d, "top_labels": labels})
    return results


@torch.no_grad()
def label_constraint_proj_weights(constraint_proj: torch.nn.Linear, top_k=3):
    """For every output hidden unit of one Block's constraint_proj, the
    real, exact top_k largest-magnitude weights -- directly readable
    from the weight matrix, no data or approximation needed, since this
    layer's entire input IS the named constraint vector. Returns a list
    of d_model {dim, top_labels: [(constraint_name, weight), ...]}
    dicts."""
    W = constraint_proj.weight  # (d_model, len(CONSTRAINT_NAMES)) -- bias=False
    d_model = W.shape[0]
    results = []
    for d in range(d_model):
        row = W[d]
        top = torch.topk(row.abs(), k=min(top_k, len(CONSTRAINT_NAMES)))
        labels = [(CONSTRAINT_NAMES[i], row[i].item()) for i in top.indices.tolist()]
        results.append({"dim": d, "top_labels": labels})
    return results


@torch.no_grad()
def label_mdbe_proj_weights(proj: torch.nn.Linear, base_embedding_dim, top_k=3):
    """Same exact-weight-readout idea for MDBE.proj, whose input is
    [learned_embedding ; constraints] concatenated -- only the
    constraint half of the input has real names. The learned-embedding
    half is still reported (as cell_i), just visibly distinguished
    rather than silently mixed in with the named half."""
    W = proj.weight  # (d_model, base_embedding_dim + len(CONSTRAINT_NAMES))
    d_model = W.shape[0]
    input_names = [f"cell_{i}" for i in range(base_embedding_dim)] + CONSTRAINT_NAMES
    results = []
    for d in range(d_model):
        row = W[d]
        top = torch.topk(row.abs(), k=min(top_k, len(input_names)))
        labels = [(input_names[i], row[i].item()) for i in top.indices.tolist()]
        results.append({"dim": d, "top_labels": labels})
    return results


def export_weight_trace_table(model, filepath, top_k=3, byte_values=None):
    """Writes one real, reviewable CSV: for every hidden dimension of
    the model, what its embedding is most correlated with, and what
    each Block's constraint_proj most strongly weights it by -- turns
    "cell_N" into a labeled, traceable dimension throughout the model,
    not just at the raw input. Pass byte_values (see
    used_bytes_from_corpus) to compute the embedding correlation only
    over bytes the model actually trained on."""
    embed_labels = label_embedding_dimensions(model, top_k, byte_values)
    proj_labels = label_mdbe_proj_weights(model.mdbe.proj, model.mdbe.base.embedding_dim, top_k)
    block_labels = [label_constraint_proj_weights(b.constraint_proj, top_k) for b in model.blocks]

    header = ["dim"]
    for k in range(top_k):
        header += [f"embedding_corr_label_{k+1}", f"embedding_corr_{k+1}"]
    for k in range(top_k):
        header += [f"mdbe_proj_label_{k+1}", f"mdbe_proj_weight_{k+1}"]
    for i in range(len(model.blocks)):
        for k in range(top_k):
            header += [f"block{i}_constraint_proj_label_{k+1}", f"block{i}_constraint_proj_weight_{k+1}"]

    d_model = model.mdbe.base.embedding_dim
    rows = []
    for d in range(d_model):
        row = [d]
        for name, val in embed_labels[d]["top_labels"]:
            row += [name, f"{val:.4f}"]
        while len(row) < 1 + 2 * top_k:
            row += ["", ""]
        for name, val in proj_labels[d]["top_labels"]:
            row += [name, f"{val:.4f}"]
        while len(row) < 1 + 4 * top_k:
            row += ["", ""]
        for bl in block_labels:
            for name, val in bl[d]["top_labels"]:
                row += [name, f"{val:.4f}"]
        rows.append(row)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


EMBEDDING_NAMES_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "embedding_dimension_names.json"))


@torch.no_grad()
def name_embedding_dimensions(model, byte_values=None):
    """A real, readable name for EVERY embedding dimension -- never
    "cell_N" -- derived from its own strongest real correlation with a
    named constraint. Same "_LIKE" convention discover_dimension.py
    already uses for auto-discovered word clusters, applied here to
    embedding dimensions instead: a dimension whose values track
    PART_OF_SPEECH:NOUN most closely gets named NOUN_LIKE.

    Never falls back to a placeholder: even a weak top correlation is a
    real, reproducible fact (the actual best match among all 88 named
    constraints, on this actual trained model) -- it's reported
    honestly with its real r-value alongside, so a weak one is visible
    as weak rather than hidden behind a confident-looking name. The
    dimension's own index is appended to the name as a real,
    necessary disambiguator (guarantees uniqueness, and is a direct,
    useful pointer back into the tensor) -- unlike a bare "dim_42", the
    prefix here already carries real meaning.

    Returns {dim: {"name", "constraint", "correlation"}}."""
    labels = label_embedding_dimensions(model, top_k=1, byte_values=byte_values)
    names = {}
    for entry in labels:
        d = entry["dim"]
        constraint, corr = entry["top_labels"][0]
        safe_constraint = constraint.replace(":", "_")
        names[d] = {
            "name": f"{safe_constraint}_LIKE_{d}",
            "constraint": constraint,
            "correlation": corr,
        }
    return names


def save_embedding_dimension_names(model, path=EMBEDDING_NAMES_PATH, byte_values=None):
    """Persists name_embedding_dimensions()'s result to disk (same
    pattern as discover_dimension.py's discovered_dimensions.json) --
    real names computed once against a specific trained checkpoint,
    reusable by any later export without re-deriving them, and directly
    inspectable as their own real, readable file."""
    names = name_embedding_dimensions(model, byte_values=byte_values)
    with open(path, "w") as f:
        json.dump({str(d): entry for d, entry in names.items()}, f, indent=2)
    return names


def load_embedding_dimension_names(path=EMBEDDING_NAMES_PATH):
    """Loads previously-saved embedding dimension names, or None if
    none have been computed yet for this checkpoint -- an honest
    default, not a fabricated name."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {int(d): entry for d, entry in data.items()}


def export_full_table(model, filepath, byte_values=None, refresh_names=True):
    """The single, combined, on-request table: every hand-given
    constraint column (byte-identity + language-mechanics, including
    whatever's been auto-discovered) AND every embedding dimension --
    each with a real, traceable name, never cell_N. Two distinct kinds
    of column, both real:
      - constraint columns: deterministic 0/1 facts, unaffected by
        training, identical to export_mdbe_table's columns
      - embedding columns: the model's actual learned floats, named by
        name_embedding_dimensions() and headed with that name PLUS the
        real correlation backing it, e.g. "TOKENIZATION_LIKE_42 (r=0.469)"
    export_mdbe_table() stays the smaller, constraints-only default;
    this is what to reach for when someone asks for one spreadsheet
    with everything in it."""
    names = (save_embedding_dimension_names(model, byte_values=byte_values) if refresh_names
             else (load_embedding_dimension_names() or name_embedding_dimensions(model, byte_values=byte_values)))
    d_model = model.mdbe.base.embedding_dim

    rows = []
    for b in range(256):
        byte_tensor = torch.tensor([[b]])
        learned = model.mdbe.base(byte_tensor)[0, 0].tolist()
        live = all_constraints(byte_tensor)[0, 0].tolist()
        char_repr = repr(chr(b)) if 32 <= b < 127 else ""
        rows.append([b, char_repr] + [f"{v:.4f}" for v in learned]
                    + [f"{v:.0f}" for v in live])

    embedding_header = [
        f"{names[d]['name']} (traces to {names[d]['constraint']}, r={names[d]['correlation']:.3f})"
        for d in range(d_model)
    ]
    header = ["byte", "character"] + embedding_header + CONSTRAINT_NAMES
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
