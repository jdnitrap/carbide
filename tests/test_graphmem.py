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
