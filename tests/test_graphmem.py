"""Graph store: stable ids, protected dictionary edges, automatic growth/pruning/
retuning within bounds, gate-rejected runs, persistent dimension slots."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules.graphmem import store as m  # noqa: E402


def _store():
    return m.GraphStore(os.path.join(tempfile.mkdtemp(), "g.db"))


def test_ids_are_never_reused_after_a_delete():
    st = _store()
    st.add_node("dog")
    cat = st.add_node("cat")
    st.db.execute("DELETE FROM nodes WHERE id = ?", (cat,))
    assert st.add_node("bird") > cat


def test_dictionary_edges_cannot_be_overwritten_or_deleted():
    st = _store()
    with st.run("build") as r:
        st.add_edge("dog", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, weight=1.0, run=r, dst_kind="POS")
        st.add_edge("dog", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, weight=0.1, run=r, dst_kind="POS")
    assert st.db.execute("SELECT weight FROM edges").fetchone()[0] == 1.0
    st.adjust_edge("dog", "NOUN", m.REL_HAS_POS, 0.0001, dst_kind="POS")
    assert abs(st.db.execute("SELECT adjust FROM edges").fetchone()[0] - 0.05) < 1e-9  # scaled down, floor 0.05
    assert st.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1              # never deleted


def test_prune_drops_weak_old_learned_edges_but_never_dictionary_edges():
    st = _store()
    st.add_module("law")
    with st.run("build") as r1:
        st.add_edge("dog", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, run=r1, dst_kind="POS")
    with st.run("teach", "law") as r2:
        st.add_edge("tort", "law", m.REL_IN_DOMAIN, source=m.SOURCE_CARBIDE, module="law",
                    confidence=0.1, status="proposed", run=r2, dst_kind="DOMAIN")
    for _ in range(3):
        with st.run("teach", "law"):
            pass
    assert st.prune() == 1
    assert st.db.execute("SELECT COUNT(*) FROM edges WHERE source = 'dictionary'").fetchone()[0] == 1


def test_a_run_that_fails_its_gate_has_its_learned_edges_rejected():
    st = _store()
    st.add_module("law")
    with st.run("teach", "law") as r:
        st.add_edge("plaintiff", "law", m.REL_IN_DOMAIN, source=m.SOURCE_CARBIDE, module="law", run=r, dst_kind="DOMAIN")
    assert st.settle_run(r, False, {"general_loss": 2.9}) == 1
    assert st.db.execute("SELECT status FROM edges WHERE run = ?", (r,)).fetchone()[0] == "rejected"


def test_capacity_grows_in_blocks_with_headroom_and_never_shrinks():
    st = _store()
    cap0 = st.capacity()
    cap1 = st.ensure_capacity(cap0 + 1)
    assert cap1 > cap0 and cap1 % int(st.policy("capacity_block")) == 0
    assert st.ensure_capacity(5) == cap1
    assert st.events(kind="grow_capacity")


def test_dimensions_keep_their_slot_and_fixed_is_not_demoted():
    st = _store()
    a = st.add_dimension("DISCOVERED_NOUN_LIKE", "NOUN_LIKE", ["voyage"], module="discovered", source=m.SOURCE_CARBIDE)
    b = st.add_dimension("DISCOVERED_VERB_LIKE", "VERB_LIKE", ["run"], module="discovered", source=m.SOURCE_CARBIDE)
    st.fix_dimension("DISCOVERED_NOUN_LIKE", 0.35)
    again = st.add_dimension("DISCOVERED_NOUN_LIKE", "NOUN_LIKE", ["voyage", "ship"], module="discovered",
                             source=m.SOURCE_CARBIDE)
    row = [d for d in st.dimensions() if d["name"] == "DISCOVERED_NOUN_LIKE"][0]
    assert (a, b, again) == (0, 1, 0)
    assert row["status"] == "fixed" and abs(row["confidence"] - 0.35) < 1e-9


def test_dimension_slots_run_out_loudly_not_silently():
    st = _store()
    for i in range(m.N_DIM_SLOTS):
        st.add_dimension(f"D{i}", "V", [], module="discovered", source=m.SOURCE_CARBIDE)
    try:
        st.add_dimension("ONE_TOO_MANY", "V", [], module="discovered", source=m.SOURCE_CARBIDE)
    except ValueError:
        return
    raise AssertionError("running out of dimension slots must raise")


def test_autotune_is_bounded_and_logged():
    st = _store()
    assert len(st.autotune({"unk_rate": 0.2, "prune_rate": 0.9})) == 2
    for _ in range(30):
        st.autotune({"unk_rate": 0.2, "prune_rate": 0.9})
    assert st.policy("promote_min_count") >= 1 and st.policy("accept_confidence") <= 0.99
    assert st.events(kind="policy")


def test_everything_persists_across_a_restart():
    path = os.path.join(tempfile.mkdtemp(), "g.db")
    st = m.GraphStore(path)
    st.add_dimension("DISCOVERED_NOUN_LIKE", "NOUN_LIKE", ["voyage"], module="discovered", source=m.SOURCE_CARBIDE)
    cap = st.ensure_capacity(5000)
    st.snapshot(path + ".snap")
    st.close()
    again = m.GraphStore(path)
    assert len(again.dimensions()) == 1 and again.capacity() == cap
    assert os.path.getsize(path + ".snap") > 0


# ---------------------------------------------------------------- compile / ingest
import json  # noqa: E402

import torch  # noqa: E402

from carbide_modules.graphmem import compile as gc  # noqa: E402
from carbide_modules.graphmem import ingest as ing  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def _dict_store():
    st = _store()
    with st.run("build") as r:
        st.add_edge("dog", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, weight=1.0, run=r, dst_kind="POS")
        st.add_edge("dog", "noun.animal", m.REL_HAS_CLASS, source=m.SOURCE_DICTIONARY, run=r, dst_kind="CLASS")
        st.add_edge("run", "VERB", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, weight=0.6, run=r, dst_kind="POS")
        st.add_edge("run", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, weight=0.4, run=r, dst_kind="POS")
    return st


def test_compiled_table_lines_up_with_the_vocabulary():
    t = gc.compile_for_vocab(_dict_store(), ["dog", "run"], rows=8)
    assert t.table.shape == (8, gc.N_GRAPH_COLS) and len(t.column_names()) == gc.N_GRAPH_COLS
    assert float(t.table[0].abs().sum()) == 0.0, "row 0 (UNK) must stay empty"
    assert t.row_for("dog") == {"POS:NOUN": 1.0, "CLASS:noun.animal": 1.0}
    assert abs(t.row_for("run")["POS:VERB"] - 0.6) < 1e-6 and abs(t.row_for("run")["POS:NOUN"] - 0.4) < 1e-6


def test_learned_edges_need_confidence_and_a_passing_gate():
    st = _dict_store()
    with st.run("teach") as r:
        st.add_edge("dog", "VERB", m.REL_HAS_POS, source=m.SOURCE_CARBIDE, confidence=0.3, run=r, dst_kind="POS")
    assert "POS:VERB" not in gc.compile_for_vocab(st, ["dog"], 4).row_for("dog"), "below accept_confidence"
    with st.run("teach") as r2:
        st.add_edge("dog", "VERB", m.REL_HAS_POS, source=m.SOURCE_CARBIDE, confidence=0.9, run=r2, dst_kind="POS")
    assert "POS:VERB" in gc.compile_for_vocab(st, ["dog"], 4).row_for("dog")
    st.settle_run(r2, False)
    assert "POS:VERB" not in gc.compile_for_vocab(st, ["dog"], 4).row_for("dog"), "a rejected run must not reach Carbide"


def test_turning_a_module_off_removes_it_from_the_table():
    st = _dict_store()
    ing.dump_rows = None
    with st.run("ingest-triples", "law") as r:
        st.add_module("law")
        ing.ingest_triples(st, "law", [("tort", "IS_A", "wrong")], run=r)
    assert gc.compile_for_vocab(st, ["tort", "wrong"], 4).row_for("tort").get("DOMAIN_TERM") == 1.0
    st.set_module_enabled("law", False)
    assert "DOMAIN_TERM" not in gc.compile_for_vocab(st, ["tort", "wrong"], 4).row_for("tort")


def test_triples_already_in_the_core_are_not_duplicated():
    st = _dict_store()
    with st.run("build") as r:
        st.add_edge("dog", "canine", m.REL_IS_A, source=m.SOURCE_DICTIONARY, run=r)
    st.add_module("bio")
    stats = ing.ingest_triples(st, "bio", [("dog", "IS_A", "canine"), ("dog", "eats", "meat"), ("bad",), ("x", "y", "z", "nope")])
    assert stats == {"added": 1, "invalid": 2, "already_in_core": 1}, stats


def test_text_dump_finds_terms_specific_to_the_domain():
    st = _dict_store()
    the = st.add_node("the")
    st.add_tokens(m.CORE, 100_000)
    st.bump_count(m.CORE, the, 33_000)          # 'the' is common in general English
    text = "the plaintiff filed the tort claim against the defendant. " * 30
    stats = ing.ingest_text(st, "law", text)
    terms = {r[0] for r in st.db.execute(
        "SELECT n.text FROM edges e JOIN nodes n ON n.id = e.src WHERE e.rel = 'IN_DOMAIN' AND e.module = 'law'")}
    assert {"plaintiff", "tort", "defendant"} <= terms and "the" not in terms, terms
    assert stats["domain_terms"] >= 3


def test_unknown_words_retune_the_promotion_threshold_automatically():
    st = _dict_store()
    before = st.policy("promote_min_count")
    stats = ing.ingest_text(st, "fin", "amortization liquidity derivative " * 20)
    assert stats["unk_rate"] == 1.0 and st.policy("promote_min_count") == before - 1
    assert st.events(kind="policy")


def test_dump_reads_csv_jsonl_and_glossary_files():
    st = _dict_store()
    d = tempfile.mkdtemp()
    csv_path, jl, gl = (os.path.join(d, n) for n in ("t.csv", "t.jsonl", "g.txt"))
    open(csv_path, "w").write("subject,relation,object\naspirin,treats,pain\nibuprofen,treats,pain\n")
    open(jl, "w").write('{"s": "insulin", "r": "regulates", "o": "glucose", "c": 0.8}\n')
    open(gl, "w").write("hypertension: persistently high blood pressure\nanemia: too few red blood cells\n")
    assert ing.dump(st, "health", csv_path, "triples")["added"] == 2
    assert ing.dump(st, "health", jl, "triples")["added"] == 1
    assert ing.dump(st, "health", gl, "glossary")["added"] == 2
    assert st.node(st.node_id("anemia"))["note"] == "too few red blood cells"
    assert "health" in st.modules() and st.db.execute("SELECT COUNT(*) FROM runs WHERE module = 'health'").fetchone()[0] == 3


def test_discovered_dimensions_json_moves_into_the_graph_and_keeps_its_slots():
    path = os.path.join(REPO, "discovered_dimensions.json")
    n = len(json.load(open(path)))
    st = _store()
    assert ing.import_discovered_json(st, path)["dimensions"] == n
    first = {d["name"]: d["slot"] for d in st.dimensions()}
    ing.import_discovered_json(st, path)                       # importing again must not move anything
    assert {d["name"]: d["slot"] for d in st.dimensions()} == first and sorted(first.values()) == list(range(n))
    t = gc.compile_for_vocab(st, ["above"], 4)
    assert any(k.startswith("DIM:") for k in t.row_for("above")), "member words should light their dimension column"


def test_grow_embedding_keeps_every_old_row():
    e = torch.nn.Embedding(4, 3)
    g = gc.grow_embedding(e, 9)
    assert g.num_embeddings == 9 and torch.equal(g.weight[:4], e.weight)
    try:
        gc.grow_embedding(g, 2)
    except ValueError:
        return
    raise AssertionError("shrinking must be refused")


def test_word_table_follows_graph_capacity_automatically():
    st = _store()
    e = torch.nn.Embedding(8, 3)
    st.ensure_capacity(2000)
    grown, did = gc.sync_embedding(st, e)
    assert did and grown.num_embeddings == st.capacity() and torch.equal(grown.weight[:8], e.weight)
    assert gc.sync_embedding(st, grown)[1] is False


def test_pos_tags_match_the_models_own_pos_names():
    from carbide_modules import mdbe
    assert list(gc.POS_NAMES) == list(mdbe.POS_NAMES)


def test_node_ids_stay_dense_however_often_a_node_is_re_added():
    st = _store()
    for _ in range(500):
        st.add_node("dog")
        st.add_edge("dog", "NOUN", m.REL_HAS_POS, source=m.SOURCE_DICTIONARY, dst_kind="POS")
    assert st.max_node_id() == 2, st.max_node_id()      # dog + NOUN, nothing burned
    assert st.capacity() == int(st.policy("capacity_block"))
