"""The real WordNet loader. Skips itself (passes) when nltk/WordNet are not installed:
    pip install nltk && python -m nltk.downloader wordnet"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules.graphmem import compile as gc  # noqa: E402
from carbide_modules.graphmem import store as m  # noqa: E402
from carbide_modules.graphmem import wordnet_loader as wl  # noqa: E402

WORDS = ["dog", "canine", "run", "quickly", "the", "big", "large", "zzqxv"]


def _wn():
    try:
        return wl._wordnet()
    except RuntimeError:
        return None


def test_wordnet_core_gives_pos_class_and_relations():
    wn = _wn()
    if wn is None:
        print("SKIP: WordNet not installed")
        return
    st = m.GraphStore(os.path.join(tempfile.mkdtemp(), "g.db"))
    with st.run("build") as r:
        stats = wl.load_wordnet_core(st, WORDS, run=r, wn=wn)
    edges = {(e["rel"], e["dst"]): e for e in st.edges_of("dog")}
    assert edges[(m.REL_HAS_POS, "NOUN")]["weight"] > 0.8
    assert (m.REL_HAS_CLASS, "noun.animal") in edges and (m.REL_IS_A, "canine") in edges
    run_pos = {e["dst"] for e in st.edges_of("run") if e["rel"] == m.REL_HAS_POS}
    assert {"VERB", "NOUN"} <= run_pos, "a polysemous word keeps every part of speech"
    assert {e["dst"]: e["weight"] for e in st.edges_of("quickly") if e["rel"] == m.REL_HAS_POS} == {"ADVERB": 1.0}
    assert st.edges_of("zzqxv") == [] and st.node_id("zzqxv"), "unknown words still get a node"
    assert any(e["dst"] == "large" and e["rel"] == m.REL_SYNONYM for e in st.edges_of("big"))
    assert stats["known"] == len(WORDS) - 2, stats          # WordNet knows neither "the" nor "zzqxv"
    assert st.edges_of("the") == [], "WordNet has no function words"
    assert all(e["source"] == m.SOURCE_DICTIONARY for e in st.edges_of("dog"))
    row = gc.compile_for_vocab(st, WORDS, 16).row_for("dog")
    assert row["POS:NOUN"] > 0.8 and row["CLASS:noun.animal"] > 0.5, row
    with st.run("build") as r3:
        assert wl.seed_function_words(st, run=r3) > 100
    t = gc.compile_for_vocab(st, WORDS + ["i"], 16)
    assert t.row_for("the") == {"POS:ARTICLE": 1.0}
    with st.run("build") as r4:
        wl.load_wordnet_core(st, ["i"], run=r4, wn=wn)
        wl.seed_function_words(st, run=r4)
    i_row = gc.compile_for_vocab(st, ["i"], 4).row_for("i")
    assert i_row["POS:PRONOUN"] > 0.85, i_row   # the pronoun reading beats WordNet's letter/iodine noun senses
    with st.run("build") as r2:
        assert wl.load_wordnet_core(st, WORDS, run=r2, wn=wn)["known"] == stats["known"]
    assert st.stats()["edges"] == st.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]  # idempotent: same edge count
