"""SQLite-backed knowledge graph: Carbide's long-term memory.

This is NOT a dynamical graph (no Hebbian ticks, no k-WTA). It is typed,
evidence-weighted facts that are meant to be (a) built automatically, (b)
grown over time as new datasets arrive, and (c) read by Carbide through a
compiled lookup table (see compile.py), never walked inside a training loop.

Rules the store enforces itself, so growth can be automatic without being
unsafe:
  * node ids are stable and only grow (AUTOINCREMENT: an id is never reused,
    even if a row is deleted) -- Carbide's word table is indexed by them;
  * dictionary edges are ground truth: nothing can overwrite or delete them.
    A learned signal may only scale one down (`adjust`, floor 0.05);
  * every automatic change (capacity growth, policy retuning, pruning, a
    rejected run) is written to the `events` table;
  * a run that fails its gate is marked rejected, not erased.
"""
import json
import math
import sqlite3
import time
from contextlib import contextmanager

REL_HAS_POS = "HAS_POS"
REL_HAS_CLASS = "HAS_CLASS"
REL_IS_A = "IS_A"
REL_SYNONYM = "SYNONYM"
REL_ANTONYM = "ANTONYM"
REL_PART_OF = "PART_OF"
REL_FOLLOWS = "FOLLOWS"
REL_IN_DOMAIN = "IN_DOMAIN"
REL_HAS_DIM = "HAS_DIM"
RELATIONS = (REL_HAS_POS, REL_HAS_CLASS, REL_IS_A, REL_SYNONYM, REL_ANTONYM,
             REL_PART_OF, REL_FOLLOWS, REL_IN_DOMAIN, REL_HAS_DIM)  # imports may add their own names

SOURCE_DICTIONARY = "dictionary"   # ground truth, immutable
SOURCE_IMPORT = "import"           # data the user dumped into a domain module
SOURCE_TEXT = "text"               # statistics extracted from raw text
SOURCE_CARBIDE = "carbide"         # proposed by the model, gated

CORE = "core"
UNK_ROW = 0  # row 0 of every compiled table is reserved for unknown words
N_DIM_SLOTS = 32  # reserved columns for discovered dimensions: the model's input width never changes

# name: (default, lo, hi). Everything the graph retunes about itself stays inside these bounds.
DEFAULT_POLICY = {
    "capacity_block": (1024, 64, 65536),      # rows added at a time when the word table must grow
    "promote_min_count": (3, 1, 1000),        # times a word must be seen in a dump before it becomes a node
    "accept_confidence": (0.6, 0.05, 0.99),   # learned edges below this stay out of the compiled table
    "prune_confidence": (0.15, 0.0, 0.5),     # learned, still-proposed edges below this get pruned...
    "prune_after_runs": (3, 1, 1000),         # ...once they are this many runs old
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS policy(name TEXT PRIMARY KEY, value REAL NOT NULL, lo REAL NOT NULL, hi REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, module TEXT NOT NULL, note TEXT,
    source_sha TEXT, created REAL NOT NULL, status TEXT NOT NULL DEFAULT 'open', gate TEXT);
CREATE TABLE IF NOT EXISTS modules(
    name TEXT PRIMARY KEY, kind TEXT NOT NULL, tokens INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL, enabled INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS nodes(
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL, note TEXT,
    UNIQUE(text, kind));
CREATE TABLE IF NOT EXISTS edges(
    src INTEGER NOT NULL REFERENCES nodes(id), dst INTEGER NOT NULL REFERENCES nodes(id),
    rel TEXT NOT NULL, module TEXT NOT NULL, source TEXT NOT NULL,
    weight REAL NOT NULL, confidence REAL NOT NULL, adjust REAL NOT NULL DEFAULT 1.0,
    evidence INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'accepted',
    run INTEGER REFERENCES runs(id),
    PRIMARY KEY (src, dst, rel, module, source));
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_module ON edges(module);
CREATE INDEX IF NOT EXISTS edges_run ON edges(run);
CREATE TABLE IF NOT EXISTS dimensions(
    name TEXT PRIMARY KEY, value_name TEXT NOT NULL, slot INTEGER NOT NULL UNIQUE,
    module TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL,
    run INTEGER REFERENCES runs(id), created REAL NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS term_counts(
    module TEXT NOT NULL, node INTEGER NOT NULL REFERENCES nodes(id), count INTEGER NOT NULL,
    PRIMARY KEY (module, node));
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
"""


class GraphStore:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        for name, (value, lo, hi) in DEFAULT_POLICY.items():
            self.db.execute("INSERT OR IGNORE INTO policy(name, value, lo, hi) VALUES (?,?,?,?)",
                            (name, value, lo, hi))
        self.db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('capacity', ?)",
                        (str(DEFAULT_POLICY["capacity_block"][0]),))
        self.db.execute("INSERT OR IGNORE INTO modules(name, kind, created) VALUES (?, 'core', ?)",
                        (CORE, time.time()))
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()

    # -- audit log ----------------------------------------------------------
    def log(self, kind, **detail):
        self.db.execute("INSERT INTO events(time, kind, detail) VALUES (?,?,?)",
                        (time.time(), kind, json.dumps(detail, sort_keys=True)))

    def events(self, limit=20, kind=None):
        q = "SELECT id, time, kind, detail FROM events"
        args = []
        if kind:
            q += " WHERE kind = ?"
            args.append(kind)
        rows = self.db.execute(q + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
        return [{"id": r[0], "time": r[1], "kind": r[2], "detail": json.loads(r[3])} for r in rows]

    # -- policy: the graph's own bounded, logged knobs -----------------------
    def policy(self, name):
        return self.db.execute("SELECT value FROM policy WHERE name = ?", (name,)).fetchone()[0]

    def set_policy(self, name, value, reason=""):
        """Clamp to the knob's bounds and log. Returns (old, new), or None if nothing changed."""
        old, lo, hi = self.db.execute("SELECT value, lo, hi FROM policy WHERE name = ?", (name,)).fetchone()
        new = min(hi, max(lo, value))
        if math.isclose(new, old):
            return None
        self.db.execute("UPDATE policy SET value = ? WHERE name = ?", (new, name))
        self.log("policy", name=name, old=old, new=new, reason=reason)
        return old, new

    def autotune(self, metrics):
        """Retune thresholds from what the last run measured. Heuristic, one step
        at a time, always inside the bounds in DEFAULT_POLICY, always logged.
          unk_rate   -- share of ingested tokens that had no node yet
          prune_rate -- share of learned edges the last prune removed"""
        changes = []
        unk, pr = metrics.get("unk_rate"), metrics.get("prune_rate")
        if unk is not None:
            cur = self.policy("promote_min_count")
            if unk > 0.05:
                changes.append(self.set_policy("promote_min_count", cur - 1, f"unk_rate {unk:.3f} > 0.05: learn words sooner"))
            elif unk < 0.01:
                changes.append(self.set_policy("promote_min_count", cur + 1, f"unk_rate {unk:.3f} < 0.01: be stricter"))
        if pr is not None:
            cur = self.policy("accept_confidence")
            if pr > 0.5:
                changes.append(self.set_policy("accept_confidence", cur + 0.05, f"prune_rate {pr:.2f} > 0.5: learned edges are noisy"))
            elif pr < 0.1:
                changes.append(self.set_policy("accept_confidence", cur - 0.05, f"prune_rate {pr:.2f} < 0.1: room to accept more"))
        return [c for c in changes if c is not None]

    # -- capacity: how many rows Carbide's word table needs ------------------
    def max_node_id(self):
        row = self.db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'nodes'").fetchone()
        return int(row[0]) if row else 0

    def capacity(self):
        return int(self.db.execute("SELECT value FROM meta WHERE key = 'capacity'").fetchone()[0])

    def ensure_capacity(self, min_rows):
        """Grow capacity in whole blocks, with one block of headroom, so the
        embedding table is not resized on every new word. Never shrinks."""
        cap = self.capacity()
        if min_rows <= cap:
            return cap
        block = int(self.policy("capacity_block"))
        new = (math.ceil(min_rows / block) + 1) * block
        self.db.execute("UPDATE meta SET value = ? WHERE key = 'capacity'", (str(new),))
        self.log("grow_capacity", old=cap, new=new, needed=min_rows)
        return new

    # -- runs (one per ingestion / teaching pass) -----------------------------
    @contextmanager
    def run(self, kind, module=CORE, note=None, source_sha=None):
        cur = self.db.execute("INSERT INTO runs(kind, module, note, source_sha, created) VALUES (?,?,?,?,?)",
                              (kind, module, note, source_sha, time.time()))
        run_id = cur.lastrowid
        try:
            yield run_id
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def settle_run(self, run_id, passed, metrics=None):
        """The measured accept gate. A failed run's learned edges are kept for
        audit but marked rejected, so the compiled table ignores them. Dictionary
        edges are never touched."""
        self.db.execute("UPDATE runs SET status = ?, gate = ? WHERE id = ?",
                        ("accepted" if passed else "rejected", json.dumps(metrics or {}), run_id))
        n = 0
        if not passed:
            n = self.db.execute("UPDATE edges SET status = 'rejected' WHERE run = ? AND source != ?",
                                (run_id, SOURCE_DICTIONARY)).rowcount
        self.log("settle_run", run=run_id, passed=bool(passed), edges_rejected=n, metrics=metrics or {})
        self.db.commit()
        return n

    # -- modules --------------------------------------------------------------
    def add_module(self, name, kind="domain"):
        self.db.execute("INSERT OR IGNORE INTO modules(name, kind, created) VALUES (?,?,?)",
                        (name, kind, time.time()))

    def set_module_enabled(self, name, enabled):
        self.db.execute("UPDATE modules SET enabled = ? WHERE name = ? AND name != ?",
                        (1 if enabled else 0, name, CORE))
        self.log("module_enabled", module=name, enabled=bool(enabled))

    def modules(self, enabled_only=True):
        q = "SELECT name FROM modules" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY created, name"
        return [r[0] for r in self.db.execute(q)]

    def add_tokens(self, module, n):
        self.db.execute("UPDATE modules SET tokens = tokens + ? WHERE name = ?", (n, module))

    def tokens(self, module):
        return self.db.execute("SELECT tokens FROM modules WHERE name = ?", (module,)).fetchone()[0]

    # -- nodes / edges ---------------------------------------------------------
    def add_node(self, text, kind="WORD", note=None):
        # SELECT first: with AUTOINCREMENT, even an ignored INSERT OR IGNORE consumes an id,
        # which made ids sparse and inflated the capacity the word table is sized from.
        row = self.db.execute("SELECT id, note FROM nodes WHERE text = ? AND kind = ?", (text, kind)).fetchone()
        if row:
            if note and row[1] is None:
                self.db.execute("UPDATE nodes SET note = ? WHERE id = ?", (note, row[0]))
            return row[0]
        return self.db.execute("INSERT INTO nodes(text, kind, note) VALUES (?,?,?)", (text, kind, note)).lastrowid

    def node_id(self, text, kind="WORD"):
        row = self.db.execute("SELECT id FROM nodes WHERE text = ? AND kind = ?", (text, kind)).fetchone()
        return row[0] if row else None

    def node(self, node_id):
        row = self.db.execute("SELECT text, kind, note FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return {"text": row[0], "kind": row[1], "note": row[2]} if row else None

    def add_edge(self, src, dst, rel, *, source, module=CORE, weight=1.0, confidence=1.0,
                 status="accepted", run=None, src_kind="WORD", dst_kind="WORD"):
        s, d = self.add_node(src, src_kind), self.add_node(dst, dst_kind)
        if source == SOURCE_DICTIONARY:
            # ground truth: first write wins, later writes are ignored
            self.db.execute(
                "INSERT OR IGNORE INTO edges(src,dst,rel,module,source,weight,confidence,status,run) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (s, d, rel, module, source, weight, confidence, status, run))
        else:
            self.db.execute(
                "INSERT INTO edges(src,dst,rel,module,source,weight,confidence,status,run) "
                "VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(src,dst,rel,module,source) DO UPDATE SET "
                "weight=excluded.weight, confidence=excluded.confidence, status=excluded.status, "
                "run=excluded.run, evidence=evidence+1",
                (s, d, rel, module, source, weight, confidence, status, run))

    def adjust_edge(self, src, dst, rel, factor, *, module=CORE, src_kind="WORD", dst_kind="WORD", reason=""):
        """Scale an edge's effective strength (any source, dictionary included)
        without ever deleting it. Floor 0.05, ceiling 1.0."""
        s, d = self.node_id(src, src_kind), self.node_id(dst, dst_kind)
        if s is None or d is None:
            return 0
        n = self.db.execute(
            "UPDATE edges SET adjust = MIN(1.0, MAX(0.05, adjust * ?)) WHERE src=? AND dst=? AND rel=? AND module=?",
            (factor, s, d, rel, module)).rowcount
        if n:
            self.log("adjust_edge", src=src, dst=dst, rel=rel, module=module, factor=factor, reason=reason)
        return n

    def prune(self):
        """Housekeeping so growth does not become clutter: drop learned edges that
        are still only 'proposed', weak, and old. Dictionary and imported edges are
        never pruned. Returns how many were removed."""
        floor = self.policy("prune_confidence")
        age = int(self.policy("prune_after_runs"))
        latest = self.db.execute("SELECT COALESCE(MAX(id), 0) FROM runs").fetchone()[0]
        n = self.db.execute(
            "DELETE FROM edges WHERE source IN (?, ?) AND status = 'proposed' "
            "AND confidence * adjust < ? AND ? - COALESCE(run, 0) >= ?",
            (SOURCE_CARBIDE, SOURCE_TEXT, floor, latest, age)).rowcount
        if n:
            self.log("prune", removed=n, floor=floor, age_runs=age)
        return n

    def edges_of(self, text, kind="WORD", module=None):
        node = self.node_id(text, kind)
        if node is None:
            return []
        q = ("SELECT e.rel, n.text, n.kind, e.module, e.source, e.weight, e.confidence, e.adjust, e.status "
             "FROM edges e JOIN nodes n ON n.id = e.dst WHERE e.src = ?")
        args = [node]
        if module:
            q += " AND e.module = ?"
            args.append(module)
        keys = ("rel", "dst", "dst_kind", "module", "source", "weight", "confidence", "adjust", "status")
        return [dict(zip(keys, r)) for r in self.db.execute(q + " ORDER BY e.rel, e.weight DESC", args)]

    def bump_count(self, module, node, n):
        self.db.execute(
            "INSERT INTO term_counts(module, node, count) VALUES (?,?,?) "
            "ON CONFLICT(module, node) DO UPDATE SET count = count + excluded.count", (module, node, n))

    def count(self, module, node):
        row = self.db.execute("SELECT count FROM term_counts WHERE module = ? AND node = ?", (module, node)).fetchone()
        return row[0] if row else 0

    # -- discovered dimensions: traceable, and they keep their column forever ---
    def add_dimension(self, name, value_name, words, *, module, source, run=None,
                      status="proposed", confidence=0.2, detail=None):
        """Record a dimension Carbide discovered, with its provenance (which run,
        when, how confident) and its member words. A dimension gets a slot once
        and keeps it for good, so a trained model's column meanings never shift
        and a new dimension never changes the model's input width. A dimension
        already 'fixed' is not demoted back to 'proposed' by a later re-discovery."""
        row = self.db.execute("SELECT slot, status, confidence FROM dimensions WHERE name = ?", (name,)).fetchone()
        if row:
            slot = row[0]
            if row[1] == "fixed" and status == "proposed":
                status, confidence = "fixed", row[2]
            self.db.execute("UPDATE dimensions SET status=?, confidence=?, run=?, detail=? WHERE name=?",
                            (status, confidence, run, json.dumps(detail) if detail else None, name))
        else:
            slot = self.db.execute("SELECT COALESCE(MAX(slot), -1) + 1 FROM dimensions").fetchone()[0]
            if slot >= N_DIM_SLOTS:
                raise ValueError(f"all {N_DIM_SLOTS} dimension slots are in use; raise N_DIM_SLOTS "
                                 f"(this changes the model's input width, so plan a warm-start migration)")
            self.db.execute(
                "INSERT INTO dimensions(name, value_name, slot, module, status, confidence, run, created, detail) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (name, value_name, slot, module, status, confidence, run, time.time(),
                 json.dumps(detail) if detail else None))
            self.log("new_dimension", name=name, slot=slot, module=module, run=run, confidence=confidence)
        self.add_node(name, "DIM", note=value_name)
        for w in words:
            self.add_edge(w, name, REL_HAS_DIM, source=source, module=module, confidence=confidence,
                          run=run, dst_kind="DIM")
        return slot

    def fix_dimension(self, name, confidence=None):
        self.db.execute("UPDATE dimensions SET status = 'fixed', confidence = COALESCE(?, confidence) WHERE name = ?",
                        (confidence, name))
        self.log("fix_dimension", name=name, confidence=confidence)

    def dimensions(self, module=None):
        q = "SELECT name, value_name, slot, module, status, confidence, run, created FROM dimensions"
        args = []
        if module:
            q += " WHERE module = ?"
            args.append(module)
        keys = ("name", "value_name", "slot", "module", "status", "confidence", "run", "created")
        return [dict(zip(keys, r)) for r in self.db.execute(q + " ORDER BY slot", args)]

    def snapshot(self, dest):
        """Consistent copy of the whole graph, for rolling back a bad automatic step."""
        self.db.commit()
        out = sqlite3.connect(dest)
        with out:
            self.db.backup(out)
        out.close()

    def stats(self):
        q = self.db.execute
        return {
            "nodes": q("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "max_node_id": self.max_node_id(),
            "capacity": self.capacity(),
            "edges": q("SELECT COUNT(*) FROM edges").fetchone()[0],
            "edges_by_module_source_status": [
                {"module": m, "source": s, "status": st, "n": n} for m, s, st, n in
                q("SELECT module, source, status, COUNT(*) FROM edges GROUP BY 1,2,3 ORDER BY 1,2,3")],
            "edges_by_relation": dict(q("SELECT rel, COUNT(*) FROM edges GROUP BY rel ORDER BY 2 DESC").fetchall()),
            "runs": q("SELECT COUNT(*) FROM runs").fetchone()[0],
            "dimensions": self.db.execute("SELECT COUNT(*) FROM dimensions").fetchone()[0],
            "modules": self.modules(enabled_only=False),
            "policy": dict(q("SELECT name, value FROM policy").fetchall()),
        }
