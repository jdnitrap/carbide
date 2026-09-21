"""Lets Carbide propose AND name genuinely NEW dimensions for itself
from real training data, fully automatically -- no human names or
confirms anything. The one non-negotiable rule: every discovered column
still gets a real, readable label, never a placeholder like dim_0 or
DIMENSION_1.

Auto-added columns sit BESIDE the six hex flags. They start as
status=proposed with low confidence. After training, fix_dimensions()
can lock word lists and raise confidence.
"""
import json
import os
import re
from collections import Counter, defaultdict

from .mdbe import _pos_scan, _POS_LOOKUP_ORDER

DISCOVERED_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "discovered_dimensions.json"))

_PLACEHOLDER_NAME_RE = re.compile(r"^DIM(ENSION)?[\s_-]?\d+$", re.IGNORECASE)


def _is_covered(word: str) -> bool:
    return _pos_scan(word) is not None


def _context_signatures(words, include_word):
    sigs = defaultdict(Counter)
    for i in range(1, len(words) - 1):
        w = words[i]
        if not include_word(w):
            continue
        prev_pos = _pos_scan(words[i - 1])
        next_pos = _pos_scan(words[i + 1])
        sigs[w][(prev_pos, next_pos)] += 1
    return sigs


def _read_corpus_words(corpus_path, sample_bytes):
    with open(corpus_path, "rb") as f:
        text = f.read(sample_bytes).decode("ascii", errors="replace").lower()
    return re.findall(r"[a-z]+", text)


def mine_context_signatures(corpus_path, sample_bytes=2_000_000):
    words = _read_corpus_words(corpus_path, sample_bytes)
    return _context_signatures(words, lambda w: not _is_covered(w))


def reference_category_signatures(corpus_path, sample_bytes=2_000_000):
    words = _read_corpus_words(corpus_path, sample_bytes)
    by_category = defaultdict(Counter)
    for i in range(1, len(words) - 1):
        cat = _pos_scan(words[i])
        if cat is None:
            continue
        prev_pos = _pos_scan(words[i - 1])
        next_pos = _pos_scan(words[i + 1])
        by_category[cat][(prev_pos, next_pos)] += 1
    return {cat: counter.most_common(1)[0][0] for cat, counter in by_category.items() if counter}


def propose_dimensions(corpus_path, sample_bytes=2_000_000, min_freq=50, min_cluster_size=10):
    sigs = mine_context_signatures(corpus_path, sample_bytes)
    by_dominant_sig = defaultdict(list)
    for word, counter in sigs.items():
        total = sum(counter.values())
        if total < min_freq:
            continue
        dominant_sig, _ = counter.most_common(1)[0]
        by_dominant_sig[dominant_sig].append((word, total))

    proposals = []
    for sig, word_counts in by_dominant_sig.items():
        if len(word_counts) < min_cluster_size:
            continue
        word_counts.sort(key=lambda wc: -wc[1])
        proposals.append({
            "signature": sig,
            "words": [w for w, _ in word_counts],
            "total_occurrences": sum(c for _, c in word_counts),
        })
    proposals.sort(key=lambda p: -p["total_occurrences"])
    return proposals


def _signature_label(signature):
    prev_pos, next_pos = signature
    prev_part = f"AFTER_{prev_pos}" if prev_pos else "AT_START"
    next_part = f"BEFORE_{next_pos}" if next_pos else "AT_END"
    return f"{prev_part}_{next_part}_WORDS"


def name_cluster(signature, reference_signatures):
    for category, ref_sig in reference_signatures.items():
        if ref_sig == signature:
            value_name = f"{category}_LIKE"
            return f"DISCOVERED_{value_name}", value_name
    value_name = _signature_label(signature)
    return f"DISCOVERED_{value_name}", value_name


def _validate_real_name(name: str, kind: str):
    if not name or not name.strip():
        raise ValueError(f"{kind} name cannot be empty.")
    if _PLACEHOLDER_NAME_RE.match(name.strip()):
        raise ValueError(
            f"{kind} name {name!r} looks like a placeholder (dim_0 / DIMENSION_1 / "
            "Dim1 style), not a real, readable name -- refusing to persist it."
        )


def _persist(dimension_name: str, value_name: str, words, path, status="proposed",
             confidence=0.20, extra=None):
    _validate_real_name(dimension_name, "Dimension")
    _validate_real_name(value_name, "Value")
    discovered = {}
    if os.path.exists(path):
        with open(path) as f:
            discovered = json.load(f)
    key = dimension_name.strip().upper()
    prior = discovered.get(key) or {}
    if prior.get("status") == "fixed" and status == "proposed":
        status = "fixed"
        confidence = float(prior.get("confidence", confidence))
    entry = {
        "value_name": value_name.strip().upper(),
        "words": sorted(set(w.lower() for w in words)),
        "status": status,
        "confidence": float(confidence),
    }
    if extra:
        entry.update(extra)
    discovered[key] = entry
    with open(path, "w") as f:
        json.dump(discovered, f, indent=2)
    _record_in_graph(key, entry)
    return discovered[key]


def _record_in_graph(key, entry):
    """If a graph database exists, keep the dimension there too: permanent slot, provenance (which
    run, when, how confident), and it survives restarts. Never breaks discovery itself."""
    try:
        from .config import config
        if not os.path.exists(config.graph_db):
            return
        from .graphmem import GraphStore, ingest
        with GraphStore(config.graph_db) as store:
            ingest.record_dimension(store, key, entry)
    except Exception as e:  # noqa: BLE001 -- the JSON file is still the source Carbide loads from
        print(f"  ! could not record {key} in the graph ({type(e).__name__}: {e})")


def load_discovered(path=DISCOVERED_PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fix_dimensions(corpus_path, model=None, sample_bytes=2_000_000,
                   min_keep_freq=20, path=DISCOVERED_PATH):
    """Lock/prune auto-added columns after they have been seen in training.
    Columns stay BESIDE the six hex flags.
    """
    discovered = load_discovered(path)
    if not discovered:
        return []

    words = _read_corpus_words(corpus_path, sample_bytes)
    freq = Counter(words)
    report = []

    for dim_name, entry in list(discovered.items()):
        value_name = entry["value_name"]
        old_words = list(entry.get("words") or [])
        kept = [w for w in old_words if freq.get(w, 0) >= min_keep_freq]
        if len(kept) < 3:
            kept = old_words[:10]
        status = "fixed" if len(kept) >= 8 else "proposed"
        confidence = 0.35 if status == "fixed" else 0.20
        extra = {"fixed_from_corpus": True, "word_count": len(kept)}

        if model is not None:
            try:
                delta = _column_sensitivity(model, dim_name)
                extra["loss_delta"] = round(delta, 6)
                if delta >= 0.002:
                    status = "fixed"
                    confidence = min(0.55, 0.20 + 10.0 * delta)
                elif delta < 0.0005:
                    status = "proposed"
                    confidence = 0.15
            except Exception as exc:
                extra["sensitivity_error"] = str(exc)

        _persist(dim_name, value_name, kept, path, status=status,
                 confidence=confidence, extra=extra)
        report.append({
            "dimension_name": dim_name,
            "value_name": value_name,
            "status": status,
            "confidence": confidence,
            "words": kept,
        })
    return report


def _column_sensitivity(model, dim_name, batches=4, seq_len=128):
    import torch
    import torch.nn.functional as F
    from . import dataset
    from .mdbe import CONSTRAINT_COLUMN_NAMES, all_constraints

    if dim_name not in CONSTRAINT_COLUMN_NAMES:
        return 0.0
    idx = CONSTRAINT_COLUMN_NAMES.index(dim_name)
    model.eval()
    deltas = []
    with torch.no_grad():
        for _ in range(batches):
            xb, yb = dataset.get_batch(seq_len=seq_len)
            live = model(xb, mode="full")
            loss_live = F.cross_entropy(live.reshape(-1, 256), yb.reshape(-1))
            cols = all_constraints(xb).clone()
            cols[..., idx] = 0
            learned = model.mdbe.base(xb)
            x0 = model.mdbe.proj(torch.cat([learned, cols], dim=-1))
            x0 = x0 + model.local_conv(x0)
            for block in model.blocks:
                x0 = block(x0, cols)
            loss_off = F.cross_entropy(
                model.head(model.head_norm(x0)).reshape(-1, 256),
                yb.reshape(-1),
            )
            deltas.append((loss_off - loss_live).item())
    model.train()
    return sum(deltas) / max(len(deltas), 1)


def discover(corpus_path, sample_bytes=2_000_000, min_freq=50, min_cluster_size=10,
             path=DISCOVERED_PATH):
    proposals = propose_dimensions(corpus_path, sample_bytes, min_freq, min_cluster_size)
    reference = reference_category_signatures(corpus_path, sample_bytes)

    used_dimension_names = set()
    results = []
    for p in proposals:
        dim_name, value_name = name_cluster(p["signature"], reference)
        if dim_name in used_dimension_names:
            disambiguator = p["words"][0].upper()
            dim_name = f"{dim_name}_{disambiguator}"
            value_name = f"{value_name}_{disambiguator}"
        used_dimension_names.add(dim_name)
        entry = _persist(dim_name, value_name, p["words"], path)
        results.append({"dimension_name": dim_name, **entry})
    return results
