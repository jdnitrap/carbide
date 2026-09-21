"""Dump domain data into a graph module -- no hand-editing, all automatic.

Three kinds of dump, each one recorded run:
  triples   subject, relation, object[, confidence]   (CSV / TSV / JSONL)
  glossary  term, definition                          (CSV / TSV / "term: definition")
  text      raw prose -> discovers the terms that are specific to this domain

Imported facts are trusted as data (source `import`) but scoped to their module,
never override the core dictionary, and can be switched off with the module.
Text statistics are 'learned' (source `text`): they only reach Carbide once their
confidence clears the store's accept_confidence policy.
"""
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter

from .store import CORE, REL_HAS_DIM, REL_IN_DOMAIN, SOURCE_CARBIDE, SOURCE_IMPORT, SOURCE_TEXT

WORD_RE = re.compile(r"[a-z]+")
FORMATS = ("triples", "glossary", "text")


def _norm(s):
    return " ".join(str(s).lower().split())


def _kind(text):
    return "TERM" if " " in text else "WORD"   # multi-word terms live in the graph; Carbide's word table is single words


def _mark_domain(store, term, module, confidence, run, source=SOURCE_IMPORT, weight=1.0):
    store.add_edge(term, module, REL_IN_DOMAIN, source=source, module=module, weight=weight,
                   confidence=confidence, run=run, src_kind=_kind(term), dst_kind="DOMAIN")


def ingest_triples(store, module, rows, *, confidence=0.9, run=None):
    stats = {"added": 0, "invalid": 0, "already_in_core": 0}
    for row in rows:
        row = tuple(row)
        if len(row) < 3 or not all(str(x).strip() for x in row[:3]):
            stats["invalid"] += 1
            continue
        s, rel, o = _norm(row[0]), str(row[1]).strip().upper().replace(" ", "_"), _norm(row[2])
        conf = confidence
        if len(row) > 3 and str(row[3]).strip():
            try:
                conf = min(1.0, max(0.0, float(row[3])))
            except ValueError:
                stats["invalid"] += 1
                continue
        sk, ok = _kind(s), _kind(o)
        sn, on = store.node_id(s, sk), store.node_id(o, ok)
        if sn and on and store.db.execute("SELECT 1 FROM edges WHERE src=? AND dst=? AND rel=? AND module=?",
                                          (sn, on, rel, CORE)).fetchone():
            stats["already_in_core"] += 1
            continue
        store.add_edge(s, o, rel, source=SOURCE_IMPORT, module=module, confidence=conf, run=run,
                       src_kind=sk, dst_kind=ok)
        _mark_domain(store, s, module, conf, run)
        _mark_domain(store, o, module, conf, run)
        stats["added"] += 1
    return stats


def ingest_glossary(store, module, pairs, *, confidence=0.9, run=None):
    stats = {"added": 0, "invalid": 0}
    for row in pairs:
        if len(row) < 2 or not str(row[0]).strip() or not str(row[1]).strip():
            stats["invalid"] += 1
            continue
        term = _norm(row[0])
        store.add_node(term, _kind(term), note=str(row[1]).strip())
        _mark_domain(store, term, module, confidence, run)
        stats["added"] += 1
    return stats


def ingest_text(store, module, text, *, run=None):
    """Term discovery from raw prose. A word becomes a node once it has been seen
    `promote_min_count` times; it becomes a DOMAIN term when it is far more frequent
    in this domain's text than in the general corpus (specificity = log10 of the
    frequency ratio / 3, clamped to 0..1). Afterwards the graph retunes its own
    thresholds from how many tokens it did not know yet."""
    tokens = WORD_RE.findall(text.lower())
    n = len(tokens)
    counts = Counter(tokens)
    store.add_module(module)
    store.add_tokens(module, n)
    promote = int(store.policy("promote_min_count"))
    unknown = sum(c for w, c in counts.items() if store.node_id(w) is None)
    new_words = 0
    for w, c in counts.items():
        node = store.node_id(w)
        if node is None:
            if c < promote:
                continue
            node = store.add_node(w)
            new_words += 1
        store.bump_count(module, node, c)
    dom_tokens = max(store.tokens(module), 1)
    gen_tokens = max(store.tokens(CORE), 1)
    terms = 0
    for w in counts:
        node = store.node_id(w)
        if node is None or store.count(module, node) < promote:
            continue
        rate_dom = store.count(module, node) / dom_tokens
        rate_gen = (store.count(CORE, node) + 1) / (gen_tokens + 1)
        spec = min(1.0, max(0.0, math.log10(rate_dom / rate_gen) / 3.0))
        if spec >= 0.2:
            _mark_domain(store, w, module, spec, run, source=SOURCE_TEXT, weight=spec)
            terms += 1
    store.ensure_capacity(store.max_node_id() + 1)
    unk_rate = unknown / n if n else 0.0
    changes = store.autotune({"unk_rate": unk_rate})
    return {"tokens": n, "new_words": new_words, "domain_terms": terms, "unk_rate": round(unk_rate, 4),
            "policy_changes": [f"{c[0]}->{c[1]}" for c in changes]}


def read_rows(path, fmt):
    """Parse a dump file into rows for the ingest_* functions."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if fmt == "text":
        return raw
    if path.endswith((".jsonl", ".ndjson")):
        rows = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if fmt == "triples":
                rows.append((d.get("s") or d.get("subject"), d.get("r") or d.get("relation"),
                             d.get("o") or d.get("object"), d.get("c") or d.get("confidence") or ""))
            else:
                rows.append((d.get("term"), d.get("definition")))
        return rows
    if fmt == "glossary" and "\t" not in raw.split("\n", 1)[0] and ": " in raw.split("\n", 1)[0]:
        return [tuple(line.split(": ", 1)) for line in raw.splitlines() if ": " in line]
    try:
        dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",\t|;")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(raw), dialect) if r]
    if rows and fmt == "triples" and len(rows[0]) > 1 and rows[0][1].strip().lower() in ("relation", "rel", "predicate", "r"):
        rows = rows[1:]
    return rows


def dump(store, module, path, fmt, *, confidence=0.9):
    """One recorded run: read `path` and add it to `module` (created if new)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    with open(path, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    data = read_rows(path, fmt)
    store.add_module(module)
    with store.run(f"ingest-{fmt}", module, note=path, source_sha=sha) as run:
        if fmt == "triples":
            stats = ingest_triples(store, module, data, confidence=confidence, run=run)
        elif fmt == "glossary":
            stats = ingest_glossary(store, module, data, confidence=confidence, run=run)
        else:
            stats = ingest_text(store, module, data, run=run)
        store.ensure_capacity(store.max_node_id() + 1)
    stats["sha"] = sha[:10]
    return stats


def record_dimension(store, name, entry, *, module="discovered"):
    """Save ONE discovered dimension (a discovered_dimensions.json entry) into the graph as its own
    recorded run. Re-recording keeps its slot; a dimension already fixed is not demoted."""
    store.add_module(module, kind="discovered")
    with store.run("discover", module, note=name) as run:
        slot = store.add_dimension(name, entry["value_name"], entry.get("words", []), module=module,
                                   source=SOURCE_CARBIDE, run=run, status=entry.get("status", "proposed"),
                                   confidence=float(entry.get("confidence", 0.2)),
                                   detail={k: v for k, v in entry.items() if k not in ("words", "value_name")})
        store.ensure_capacity(store.max_node_id() + 1)
    return slot


def import_discovered_json(store, path, *, module="discovered"):
    """Move Carbide's discovered_dimensions.json into the graph: each dimension gets
    a permanent slot and provenance, and keeps working after a restart."""
    with open(path) as f:
        data = json.load(f)
    store.add_module(module, kind="discovered")
    with store.run("import-dimensions", module, note=path) as run:
        for name, entry in data.items():
            store.add_dimension(name, entry["value_name"], entry.get("words", []), module=module,
                                source=SOURCE_CARBIDE, run=run, status=entry.get("status", "proposed"),
                                confidence=float(entry.get("confidence", 0.2)),
                                detail={"imported_from": path})
        store.ensure_capacity(store.max_node_id() + 1)
    return {"dimensions": len(data)}
