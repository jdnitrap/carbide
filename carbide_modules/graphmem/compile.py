"""Compile the graph into the fast, fixed-width table Carbide actually reads.

The graph is the editable, growing memory. Carbide never walks it inside a
training loop (Python graph walks are slow); it reads one row per word:

    [ POS (10) | semantic class (64) | discovered dimensions (32) | domain (2) ]

Row i+1 is the i-th word of Carbide's vocabulary (row 0 = UNK, all zeros), so the
table lines up with the Layer-2 word ids. Widths never change: unused class and
dimension slots are reserved headroom that later datasets fill, so adding
knowledge never changes the model's input width. Every column has a name.
"""
import torch
import torch.nn as nn

from .store import CORE, N_DIM_SLOTS, REL_HAS_CLASS, REL_HAS_DIM, REL_HAS_POS, REL_IN_DOMAIN, SOURCE_DICTIONARY, SOURCE_IMPORT

# Same ten tags as carbide_modules.mdbe.POS_NAMES (a test keeps them equal)
POS_NAMES = ("NOUN", "VERB", "ADJECTIVE", "ADVERB", "ARTICLE", "PRONOUN",
             "PREPOSITION", "CONJUNCTION", "AUXILIARY_VERB", "NUMERAL")
OPEN_CLASS = ("NOUN", "VERB", "ADJECTIVE", "ADVERB")
N_CLASS_SLOTS = 64          # WordNet has 45 lexicographer classes; the rest is headroom
DOMAIN_NAMES = ("DOMAIN_TERM", "DOMAIN_CONF")
N_POS, N_DOM = len(POS_NAMES), len(DOMAIN_NAMES)
CLASS_OFF = N_POS
DIM_OFF = CLASS_OFF + N_CLASS_SLOTS
DOM_OFF = DIM_OFF + N_DIM_SLOTS
N_GRAPH_COLS = DOM_OFF + N_DOM


class CompiledTable:
    def __init__(self, table, class_names, dim_names, words):
        self.table, self.class_names, self.dim_names, self.words = table, class_names, dim_names, words

    def column_names(self):
        cls = [f"CLASS:{n}" for n in self.class_names] + \
              [f"CLASS:unused_{i}" for i in range(len(self.class_names), N_CLASS_SLOTS)]
        dims = [f"DIM:{n}" for n in self.dim_names] + \
               [f"DIM:unused_{i}" for i in range(len(self.dim_names), N_DIM_SLOTS)]
        return [f"POS:{p}" for p in POS_NAMES] + cls + dims + list(DOMAIN_NAMES)

    def row_for(self, word):
        """Named, readable values for one word (for reports)."""
        if word not in self.words:
            return {}
        row = self.table[self.words.index(word) + 1]
        return {n: round(float(v), 4) for n, v in zip(self.column_names(), row) if float(v) != 0.0}

    def save(self, path):
        torch.save({"table": self.table, "class_names": self.class_names, "dim_names": self.dim_names,
                    "words": self.words}, path)

    @staticmethod
    def load(path):
        d = torch.load(path, weights_only=True)
        return CompiledTable(d["table"], d["class_names"], d["dim_names"], d["words"])


def compile_for_vocab(store, words, rows, modules=None):
    """Table of shape (rows, N_GRAPH_COLS) aligned to `words` (row i+1 = words[i]).
    Only accepted edges from enabled modules count. Dictionary and imported edges
    are trusted; learned edges (text/carbide) must clear the store's
    accept_confidence policy. `rows` is Carbide's word-table size."""
    if len(words) + 1 > rows:
        raise ValueError(f"{len(words)} words do not fit in {rows} rows")
    active = set(modules) if modules is not None else set(store.modules())
    active.add(CORE)
    row_of = {w: i + 1 for i, w in enumerate(words)}
    class_names = [r[0] for r in store.db.execute("SELECT text FROM nodes WHERE kind = 'CLASS' ORDER BY id")]
    if len(class_names) > N_CLASS_SLOTS:
        raise ValueError(f"{len(class_names)} semantic classes exceed {N_CLASS_SLOTS} slots; raise N_CLASS_SLOTS")
    class_slot = {n: i for i, n in enumerate(class_names)}
    dims = store.dimensions()
    dim_slot = {d["name"]: d["slot"] for d in dims}
    dim_names = [None] * (max(dim_slot.values()) + 1 if dim_slot else 0)
    for name, slot in dim_slot.items():
        dim_names[slot] = name
    dim_names = [n or f"slot_{i}" for i, n in enumerate(dim_names)]

    table = torch.zeros(rows, N_GRAPH_COLS)
    floor = store.policy("accept_confidence")
    marks = ",".join("?" * len(active))
    q = (f"SELECT ns.text, nd.text, nd.kind, e.rel, e.weight * e.confidence * e.adjust, e.confidence * e.adjust "
         f"FROM edges e JOIN nodes ns ON ns.id = e.src JOIN nodes nd ON nd.id = e.dst "
         f"WHERE e.status = 'accepted' AND ns.kind = 'WORD' AND e.module IN ({marks}) "
         f"AND (e.source IN (?, ?) OR e.rel = ? OR e.confidence * e.adjust >= ?)")
    # A discovered dimension's column value IS its (deliberately weak) confidence, so
    # HAS_DIM edges are never gated by the accept floor -- the model learns how much to trust them.
    for src, dst, dkind, rel, eff, conf in store.db.execute(
            q, (*sorted(active), SOURCE_DICTIONARY, SOURCE_IMPORT, REL_HAS_DIM, floor)):
        r = row_of.get(src)
        if r is None:
            continue
        col = None
        if rel == REL_HAS_POS and dkind == "POS" and dst in POS_NAMES:
            col = POS_NAMES.index(dst)
        elif rel == REL_HAS_CLASS and dst in class_slot:
            col = CLASS_OFF + class_slot[dst]
        elif rel == REL_HAS_DIM and dst in dim_slot:
            col = DIM_OFF + dim_slot[dst]
        elif rel == REL_IN_DOMAIN:
            table[r, DOM_OFF] = 1.0
            table[r, DOM_OFF + 1] = max(float(table[r, DOM_OFF + 1]), float(conf))
            continue
        if col is not None:
            table[r, col] = max(float(table[r, col]), float(eff))
    pos = table[:, :N_POS]
    table[:, :N_POS] = pos / pos.sum(-1, keepdim=True).clamp(min=1.0)  # a word's POS shares sum to at most 1
    return CompiledTable(table, class_names, dim_names, list(words))


def grow_embedding(emb, new_rows):
    """A larger nn.Embedding whose first rows are exactly the old ones, so nothing
    already learned changes. New rows start at the old table's scale. The caller
    must re-register the parameter with its optimizer (Adam state is per-parameter)."""
    if new_rows < emb.num_embeddings:
        raise ValueError("an embedding table can only grow")
    if new_rows == emb.num_embeddings:
        return emb
    new = nn.Embedding(new_rows, emb.embedding_dim, padding_idx=emb.padding_idx)
    with torch.no_grad():
        std = float(emb.weight.std()) if emb.num_embeddings > 1 else 0.02
        new.weight.normal_(0.0, max(std, 1e-3))
        new.weight[:emb.num_embeddings] = emb.weight
    return new


def sync_embedding(store, emb):
    """Automatic growth: if the graph's capacity has outgrown the word table, grow
    the table to match. Returns (embedding, grew)."""
    cap = store.capacity()
    if emb.num_embeddings >= cap:
        return emb, False
    return grow_embedding(emb, cap), True
