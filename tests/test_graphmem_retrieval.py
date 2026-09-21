"""Fact retrieval: the graph's knowledge as plain sentences a byte model can read."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules.graphmem import GraphStore, ingest, retrieval, store as gs  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _world():
    st = GraphStore(os.path.join(tempfile.mkdtemp(), "g.db"))
    with st.run("build") as r:
        st.add_node("dog", note="a domesticated carnivorous mammal")
        st.add_edge("dog", "canine", gs.REL_IS_A, source=gs.SOURCE_DICTIONARY, weight=1.0, run=r)
        st.add_edge("dog", "hound", gs.REL_SYNONYM, source=gs.SOURCE_DICTIONARY, weight=0.5, run=r)
        st.add_edge("dog", "NOUN", gs.REL_HAS_POS, source=gs.SOURCE_DICTIONARY, run=r, dst_kind="POS")
        st.add_node("the")
        st.add_tokens(gs.CORE, 100_000)
        st.bump_count(gs.CORE, st.node_id("the"), 6_000)
        st.bump_count(gs.CORE, st.node_id("dog"), 20)
    st.add_module("health")
    with st.run("ingest-triples", "health") as r:
        ingest.ingest_triples(st, "health", [("aspirin", "treats", "pain"), ("aspirin", "reduces", "fever")], run=r)
    return st


def test_facts_are_readable_sentences_best_first():
    st = _world()
    facts = retrieval.facts_for(st, "dog", max_facts=4)
    assert facts[0] == "dog: a domesticated carnivorous mammal."
    assert "dog is a kind of canine." in facts and "dog means about the same as hound." in facts
    assert not any("NOUN" in f for f in facts), "part-of-speech columns are not sentences"


def test_imported_domain_facts_use_their_own_relation_words_and_outrank_the_dictionary():
    st = _world()
    facts = retrieval.facts_for(st, "aspirin", max_facts=3)
    assert "aspirin treats pain." in facts and "aspirin reduces fever." in facts
    assert retrieval.facts_for(st, "nosuchword") == []


def test_the_informative_word_is_chosen_not_the_common_one():
    st = _world()
    facts = retrieval.retrieve(st, "the aspirin and the dog", max_words=1, per_word=2)
    assert facts and all(f.startswith("aspirin") for f in facts), facts      # a domain term beats a general word
    assert retrieval.retrieve(st, "the of and", max_words=2) == []


def test_a_disabled_module_stops_supplying_facts():
    st = _world()
    st.set_module_enabled("health", False)
    st.db.execute("UPDATE edges SET status = 'rejected' WHERE module = 'health'")   # what disabling means to a reader
    assert retrieval.facts_for(st, "aspirin") == []


def test_prefix_and_fact_corpus_format():
    st = _world()
    p = retrieval.prefix_for(st, "Tell me about aspirin.")
    assert p.startswith("Facts: aspirin") and p.endswith("\n")
    assert retrieval.prefix_for(st, "nothing known here at all") == ""
    out = os.path.join(tempfile.mkdtemp(), "c.txt")
    total, withf = retrieval.build_fact_corpus(st, "Aspirin helps. The sky is blue. The dog ran home.", out)
    text = open(out).read()
    assert (total, withf) == (3, 2), (total, withf)
    assert "aspirin treats pain." in text and "\nAspirin helps.\n" in text and "Facts: dog:" in text
    assert text.startswith("Facts: aspirin")


def test_shell_facts_command_and_generate_with_facts():
    import subprocess
    d = tempfile.mkdtemp()
    open(os.path.join(d, "h.tsv"), "w").write("subject\trelation\tobject\naspirin\ttreats\tpain\n")
    with open(os.path.join(REPO, "carbide_training_dataset.txt"), "rb") as f:
        open(os.path.join(d, "c.txt"), "wb").write(f.read(60_000))
    script = ["set model.d_model 16", "set model.n_layers 1", "set model.d_state 8", "set train.seq_len 32",
              "set train.batch_size 4", "train 10", "graph ingest health h.tsv triples", "graph facts aspirin",
              "gen +facts tell me about aspirin", "graph fact-corpus c.txt out.txt"]
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-m", "carbide_modules.shell", "--data", "c.txt"], cwd=d, env=env,
                         input="\n".join(script) + "\nquit\n", capture_output=True, text=True, timeout=600).stdout
    assert out.count("aspirin treats pain.") >= 2, out      # once from `graph facts`, once as the facts used by generate
    assert "wrote out.txt" in out
