"""Every adjustable knob of Carbide and its graph memory, in one registry, so the shell's `set`
command and the TUI's settings panel are the same code path with the same validation.

Each setting says where its value survives a restart:
  file   carbide_settings.json -- written automatically on every change
  graph  the graph database's own policy table -- these knobs are also retuned by the graph itself
"""
import json
import math
import os

import torch

from . import generation, training
from .config import config

SETTINGS_PATH = "carbide_settings.json"
STRUCTURAL = ("model.kind", "model.d_model", "model.n_layers", "model.d_state")  # changing one rebuilds the model


class Setting:
    def __init__(self, name, kind, where, desc, get, set_, lo=None, hi=None, nullable=False, choices=None):
        self.name, self.kind, self.where, self.desc = name, kind, where, desc
        self.get, self.set_, self.lo, self.hi, self.nullable, self.choices = get, set_, lo, hi, nullable, choices


def _cfg(attr):
    return (lambda: getattr(config, attr)), (lambda v: setattr(config, attr, v))


def _gen(attr, default):
    def get():
        return getattr(config, attr, default)

    def set_(v):
        setattr(config, attr, v)
    return get, set_


def _structural(attr):
    def set_(v):
        setattr(config, attr, v)
        training.model = training.opt = None        # rebuilt on the next training step, like the menu does
    return (lambda: getattr(config, attr)), set_


def _soft_grammar():
    from . import mdbe

    def set_(v):
        mdbe.SOFT_GRAMMAR = v
    return (lambda: mdbe.SOFT_GRAMMAR), set_


def _policy(name):
    def _store():
        if not os.path.exists(config.graph_db):
            return None
        from .graphmem import GraphStore
        return GraphStore(config.graph_db)

    def get():
        st = _store()
        if st is None:
            return None
        try:
            return st.policy(name)
        finally:
            st.close()

    def set_(v):
        st = _store()
        if st is None:
            raise ValueError(f"no graph database at {config.graph_db!r} (build one: python -m carbide_modules.graphmem build-core)")
        try:
            st.set_policy(name, v, reason="set by user")
            st.commit()
        finally:
            st.close()
    return get, set_


def _lr_pair():
    return (lambda: config.learning_rate), training.set_learning_rate


def _device():
    def set_(v):
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("no CUDA GPU is available on this machine")
        old = config.device_pref
        config.device_pref = v
        try:
            training.move_to_device()   # moves any live model + optimizer state; no-op before one exists
        except Exception:
            config.device_pref = old    # never leave device_pref pointing at a device that isn't usable
            raise
    return (lambda: config.device_pref), set_


def _build():
    S = []

    def add(name, kind, where, desc, pair, **kw):
        S.append(Setting(name, kind, where, desc, pair[0], pair[1], **kw))

    add("model.kind", "choice", "file", "beside = best measured | layered = L1/L2/L3 | graph = reads graph memory",
        _structural("model_kind"), choices=training.MODEL_KINDS)
    add("model.d_model", "int", "file", "embedding width (rebuilds the model)", _structural("d_model"), lo=4, hi=2048)
    add("model.n_layers", "int", "file", "SSM blocks (rebuilds the model)", _structural("n_layers"), lo=1, hi=32)
    add("model.d_state", "int", "file", "state size per SSM (rebuilds the model)", _structural("d_state"), lo=2, hi=256)
    add("model.soft_grammar", "bool", "file", "grammar columns hold rule confidences (on) or hard 1.0 (off); a checkpoint keeps the mode it trained on",
        _soft_grammar())
    add("train.device", "choice", "file", "auto = CUDA if present else CPU | cuda/cpu = force one (moves the live model, no retrain needed)",
        _device(), choices=("auto", "cuda", "cpu"))
    add("train.batch_size", "int", "file", "sequences per step", _cfg("batch_size"), lo=1, hi=256)
    add("train.seq_len", "int", "file", "context window in bytes", _cfg("seq_len"), lo=8, hi=8192)
    add("train.learning_rate", "float", "file", "AdamW learning rate (applies immediately)", _lr_pair(), lo=1e-6, hi=1.0)
    add("train.autosave_interval", "int", "file", "steps between automatic checkpoints",
        _cfg("checkpoint_autosave_interval"), lo=1, hi=10_000_000)
    add("train.auto_grow", "bool", "file", "add depth when loss plateaus; kept only if held-out loss improves (off = never)",
        _gen("train_auto_grow", False))
    add("train.max_layers", "int", "file", "auto-grow never goes past this many blocks", _gen("train_max_layers", 8), lo=1, hi=64)
    add("train.growth_min_gain", "float", "file", "loss must improve less than this over 100 steps to count as a plateau",
        _gen("train_growth_min_gain", 0.01), lo=0.0, hi=1.0)
    add("train.snapshot_interval", "int", "file", "steps between MDBE table snapshots",
        _cfg("mdbe_snapshot_interval"), lo=1, hi=10_000_000)
    add("gen.temperature", "float", "file", "sampling temperature (0 = greedy)", _gen("gen_temperature", generation.DEFAULTS[0]), lo=0.0, hi=5.0)
    add("gen.top_k", "int", "file", "sample only from the k most likely bytes", _gen("gen_top_k", generation.DEFAULTS[1]), lo=1, hi=256)
    add("gen.top_p", "float", "file", "nucleus: smallest set reaching this probability", _gen("gen_top_p", generation.DEFAULTS[2]), lo=0.01, hi=1.0)
    add("gen.repetition_penalty", "float", "file", "discourage recently used bytes (1 = off)",
        _gen("gen_repetition_penalty", generation.DEFAULTS[3]), lo=1.0, hi=2.0)
    add("gen.no_repeat_ngram", "int", "file", "forbid repeating a run of this many bytes (0 = off; use 8-24)",
        _gen("gen_no_repeat_ngram", generation.DEFAULTS[4]), lo=0, hi=256)
    add("gen.length", "int", "file", "bytes to generate", _gen("gen_length", 200), lo=1, hi=5000)
    add("gen.seed", "int", "file", "fixed seed for repeatable text (off = random)", _gen("gen_seed", None),
        lo=0, hi=2**31 - 1, nullable=True)
    add("graph.promote_min_count", "int", "graph", "times a word must appear before it gets a node", _policy("promote_min_count"), lo=1, hi=1000)
    add("graph.accept_confidence", "float", "graph", "learned edges below this stay out of Carbide's table", _policy("accept_confidence"), lo=0.05, hi=0.99)
    add("graph.prune_confidence", "float", "graph", "weak learned edges below this get pruned", _policy("prune_confidence"), lo=0.0, hi=0.5)
    add("graph.prune_after_runs", "int", "graph", "runs a weak learned edge survives before pruning", _policy("prune_after_runs"), lo=1, hi=1000)
    add("graph.capacity_block", "int", "graph", "word-table rows added per growth step", _policy("capacity_block"), lo=64, hi=65536)
    return S


REGISTRY = _build()
BY_NAME = {s.name: s for s in REGISTRY}


def resolve(query):
    if query in BY_NAME:
        return BY_NAME[query]
    hits = [s for s in REGISTRY if s.name.split(".", 1)[1] == query]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ValueError(f"{query!r} is ambiguous: {', '.join(s.name for s in hits)}")
    raise ValueError(f"no setting {query!r} -- `set` lists them all")


def parse(s, raw):
    text = str(raw).strip().lower()
    if s.nullable and text in ("off", "none", "null"):
        return None
    if s.kind == "bool":
        if text in ("on", "true", "yes", "1"):
            return True
        if text in ("off", "false", "no", "0"):
            return False
        raise ValueError(f"{s.name} takes on/off")
    if s.kind == "choice":
        if text not in s.choices:
            raise ValueError(f"{s.name} must be one of: {', '.join(s.choices)}")
        return text
    try:
        v = int(text) if s.kind == "int" else float(text)
    except ValueError:
        raise ValueError(f"{s.name} takes {'an integer' if s.kind == 'int' else 'a number'}"
                         f"{' or off' if s.nullable else ''}, got {raw!r}") from None
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError(f"{s.name} must be finite")
    if (s.lo is not None and v < s.lo) or (s.hi is not None and v > s.hi):
        raise ValueError(f"{s.name} must be between {s.lo:g} and {s.hi:g}, got {v:g}")
    return v


def fmt(v):
    if v is None:
        return "off"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _read_file():
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def set_setting(query, raw):
    s = resolve(query)
    v = parse(s, raw)
    old = s.get()
    s.set_(v)
    msg = f"{s.name} = {fmt(v)}  (was {fmt(old)})"
    if s.where == "file":
        data = _read_file()
        data[s.name] = v
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        msg += "  [saved to carbide_settings.json]"
    else:
        msg += "  [stored in the graph database]"
    if s.name in STRUCTURAL:
        msg += "  -- model will be rebuilt on the next training step"
    return msg


def apply_file_settings():
    """Re-apply carbide_settings.json at startup. Returns notes for anything skipped."""
    notes = []
    for name, value in _read_file().items():
        try:
            s = BY_NAME.get(name)
            if s is None or s.where != "file":
                raise ValueError("not a persisted setting")
            s.set_(parse(s, value))
        except ValueError as e:
            notes.append(f"carbide_settings.json: ignored {name}: {e}")
    return notes


def snapshot():
    return [{"name": s.name, "kind": s.kind, "where": s.where, "desc": s.desc, "value": s.get(),
             "nullable": s.nullable, "choices": list(s.choices) if s.choices else None} for s in REGISTRY]


def listing():
    width = max(len(s.name) for s in REGISTRY)
    lines = [f"{s.name:<{width}}  {fmt(s.get()):<9} [{s.where:<5}] {s.desc}" for s in REGISTRY]
    return "\n".join(lines + ["", "set <name> <value> | set <name> | file = auto-saved, graph = in the graph database"])


def show(query):
    s = resolve(query)
    rng = f", {s.lo:g}..{s.hi:g}" if s.lo is not None and s.hi is not None else ""
    ch = f", one of {'/'.join(s.choices)}" if s.choices else ""
    return f"{s.name} = {fmt(s.get())}  [{s.where}{', nullable' if s.nullable else ''}{rng}{ch}] {s.desc}"
