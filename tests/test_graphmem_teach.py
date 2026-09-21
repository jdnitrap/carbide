"""Carbide teaches the graph: a gated probe on hidden states, proposals for unknown words, an audit of
the dictionary, and nothing at all when the measurement says Carbide has learned nothing."""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import discover_dimension as dd  # noqa: E402
from carbide_modules.config import config  # noqa: E402
from carbide_modules.graphmem import GraphStore, compile_for_vocab, store as gs, teach  # noqa: E402

CLASSES = ["NOUN", "VERB", "ADJECTIVE"]


def _world(separable=True, n_known=120, n_unknown=30, dim=16, seed=0):
    """A store where `n_known` words have a dictionary POS, plus vectors for known and unknown words."""
    g = torch.Generator().manual_seed(seed)
    centroids = torch.randn(3, dim, generator=g) * (3.0 if separable else 0.0)
    st = GraphStore(os.path.join(tempfile.mkdtemp(), "g.db"))
    vectors, truth = {}, {}
    with st.run("build") as r:
        for i in range(n_known + n_unknown):
            c = i % 3
            w = f"w{i}"
            vectors[w] = (centroids[c] + torch.randn(dim, generator=g), 10)
            truth[w] = CLASSES[c]
            st.add_node(w)
            if i < n_known:
                st.add_edge(w, CLASSES[c], gs.REL_HAS_POS, source=gs.SOURCE_DICTIONARY, run=r, dst_kind="POS")
    return st, vectors, truth


def test_a_probe_that_beats_the_baseline_proposes_edges_for_unknown_words():
    st, vectors, truth = _world()
    m = teach.probe_and_propose(st, vectors)
    assert m["passed"] and m["probe_accuracy"] > m["majority_baseline"] + 0.05, m
    assert m["proposed_edges"] == 30
    learned = st.db.execute(
        "SELECT n.text, d.text FROM edges e JOIN nodes n ON n.id=e.src JOIN nodes d ON d.id=e.dst "
        "WHERE e.source='carbide' AND e.module='learned'").fetchall()
    right = sum(truth[w] == pos for w, pos in learned)
    assert len(learned) == 30 and right >= 27, (right, len(learned))
    words = [f"w{i}" for i in range(150)]
    row = compile_for_vocab(st, words, 200).row_for("w120")          # an unknown word Carbide taught
    assert any(k.startswith("POS:") for k in row), "a confident learned edge should reach the table"


def test_a_probe_that_learned_nothing_adds_nothing_and_the_run_is_rejected():
    st, vectors, _ = _world(separable=False)
    m = teach.probe_and_propose(st, vectors)
    assert m["passed"] is False and m["proposed_edges"] == 0
    assert st.db.execute("SELECT COUNT(*) FROM edges WHERE source='carbide'").fetchone()[0] == 0
    assert st.db.execute("SELECT status FROM runs WHERE kind='teach'").fetchone()[0] == "rejected"


def test_too_few_labelled_words_is_refused_not_guessed():
    st, vectors, _ = _world(n_known=6, n_unknown=4)
    m = teach.probe_and_propose(st, vectors)
    assert m["passed"] is False and "too few" in m["reason"]


def test_a_dictionary_entry_the_probe_strongly_contradicts_is_scaled_down_never_deleted():
    st, vectors, truth = _world()
    victim = "w3"                                                    # truly a NOUN-cluster word ...
    st.db.execute("DELETE FROM edges WHERE src = ?", (st.node_id(victim),))
    with st.run("build") as r:                                       # ... but the dictionary says VERB
        st.add_edge(victim, "VERB", gs.REL_HAS_POS, source=gs.SOURCE_DICTIONARY, run=r, dst_kind="POS")
    m = teach.probe_and_propose(st, vectors)
    e = [x for x in st.edges_of(victim) if x["dst"] == "VERB"][0]
    assert m["passed"] and e["adjust"] < 1.0 and e["source"] == gs.SOURCE_DICTIONARY, (m, e)


def test_word_vectors_come_from_the_models_hidden_state_and_the_hook_is_removed():
    from carbide_modules import layers  # noqa: F401
    from carbide_modules.mdbe import Carbide
    torch.manual_seed(0)
    model = Carbide(d_model=16, n_layers=1, d_state=8)
    text = ("the cat sat on the mat. the dog ran to the cat. " * 60).encode()
    v = teach.collect_word_vectors(model, text, {"the", "cat", "zzz"}, seq_len=64)
    assert set(v) == {"the", "cat"} and v["the"][0].shape == (16,) and v["the"][1] > v["cat"][1] > 0
    assert len(model.head_norm._forward_hooks) == 0


def test_discovering_a_dimension_records_it_in_the_graph_with_provenance():
    tmp = tempfile.mkdtemp()
    db, jpath = os.path.join(tmp, "g.db"), os.path.join(tmp, "d.json")
    old = config.graph_db
    try:
        config.graph_db = db
        dd._persist("DISCOVERED_ANIMAL_LIKE", "ANIMAL_LIKE", ["cat", "dog"], jpath)
        assert not os.path.exists(db), "discovery must not create a graph that was never built"
        GraphStore(db).close()
        dd._persist("DISCOVERED_ANIMAL_LIKE", "ANIMAL_LIKE", ["cat", "dog"], jpath)
        dd._persist("DISCOVERED_ANIMAL_LIKE", "ANIMAL_LIKE", ["cat", "dog", "cow"], jpath, status="fixed", confidence=0.4)
        dd._persist("DISCOVERED_ANIMAL_LIKE", "ANIMAL_LIKE", ["cat"], jpath)     # re-proposed: must stay fixed
        with GraphStore(db) as st:
            d = st.dimensions()[0]
            assert d["name"] == "DISCOVERED_ANIMAL_LIKE" and d["slot"] == 0 and d["run"] is not None
            assert d["status"] == "fixed" and abs(d["confidence"] - 0.4) < 1e-9
            assert st.db.execute("SELECT COUNT(*) FROM runs WHERE kind='discover'").fetchone()[0] == 3
        assert json.load(open(jpath))["DISCOVERED_ANIMAL_LIKE"]["status"] == "fixed"
    finally:
        config.graph_db = old
