"""Fact retrieval: turn what the graph knows into plain text a byte model can read.

The compiled table tells Carbide what TYPE a word is; it cannot make Carbide STATE a fact. The way a
text model uses stored knowledge is to have it in front of it as text. So: pick the informative
words of a prompt, look up the graph's facts about them (dictionary definitions and relations, plus
anything dumped into an enabled domain module), and verbalize them as short sentences:

    Facts: aspirin treats pain. pain: an unpleasant feeling. ...
    <prompt>

Two honest limits. Carbide only benefits if it is TRAINED on text in this format
(build_fact_corpus writes such a corpus from any text), and a small byte model will still paraphrase
poorly -- this gives it the facts, not the reasoning.
"""
import re

from .store import (CORE, REL_ANTONYM, REL_IN_DOMAIN, REL_IS_A, REL_PART_OF, REL_SYNONYM, REL_HAS_CLASS, REL_HAS_POS,
                    REL_HAS_DIM, SOURCE_DICTIONARY)

WORD_RE = re.compile(r"[a-z]+")
_SKIP = {REL_HAS_POS, REL_HAS_CLASS, REL_HAS_DIM, REL_IN_DOMAIN}     # descriptive columns, not sentences
_TEMPLATES = {
    REL_IS_A: "{s} is a kind of {o}.",
    REL_SYNONYM: "{s} means about the same as {o}.",
    REL_ANTONYM: "{s} is the opposite of {o}.",
    REL_PART_OF: "{s} is part of {o}.",
}


def _sentence(subject, rel, obj):
    if rel in _TEMPLATES:
        return _TEMPLATES[rel].format(s=subject, o=obj)
    return f"{subject} {rel.lower().replace('_', ' ')} {obj}."      # an imported relation: "aspirin treats pain."


def facts_for(store, word, *, max_facts=3):
    """Short sentences about one word, best first: its definition, then dictionary relations by
    weight, then imported domain facts by confidence."""
    node = store.node_id(word) or store.node_id(word, "TERM")
    if node is None:
        return []
    info = store.node(node) or {}
    facts = [f"{word}: {info['note'].rstrip('.')}."] if info.get("note") else []
    ranked = []
    for kind in ("WORD", "TERM"):
        for e in store.edges_of(word, kind):
            if e["rel"] in _SKIP or e["status"] != "accepted":
                continue
            trust = 1.0 if e["source"] == SOURCE_DICTIONARY else 2.0 if e["module"] != CORE else 0.5
            ranked.append((trust + e["weight"] * e["confidence"] * e["adjust"], e))
    for _, e in sorted(ranked, key=lambda t: -t[0]):
        sent = _sentence(word, e["rel"], e["dst"])
        if sent not in facts:
            facts.append(sent)
    return facts[:max_facts]


def _informativeness(store, word):
    """Domain terms first, then rarer words: the words worth spending a fact on."""
    node = store.node_id(word)
    if node is None:
        return None
    domain = store.db.execute("SELECT 1 FROM edges WHERE src = ? AND rel = ? AND module != ? LIMIT 1",
                              (node, REL_IN_DOMAIN, CORE)).fetchone() is not None
    return (1 if domain else 0, -store.count(CORE, node))


def retrieve(store, text, *, max_words=2, per_word=2):
    """Facts about the most informative known words in `text`."""
    seen, scored = set(), []
    for w in WORD_RE.findall(text.lower()):
        if w in seen or len(w) < 3:
            continue
        seen.add(w)
        score = _informativeness(store, w)
        if score is not None:
            scored.append((score, w))
    out = []
    for _, w in sorted(scored, reverse=True):
        facts = facts_for(store, w, max_facts=per_word)
        if facts:
            out.extend(facts)
        if len(out) >= max_words * per_word:
            break
    return out[:max_words * per_word]


def format_prefix(facts):
    return f"Facts: {' '.join(facts)}\n" if facts else ""


def prefix_for(store, prompt, **kw):
    return format_prefix(retrieve(store, prompt, **kw))


def build_fact_corpus(store, text, out_path, *, max_words=1, per_word=2):
    """A training corpus in the retrieval format: each sentence is preceded by the graph's facts
    about its most informative known word. Returns (sentences, sentences_with_facts)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    with_facts = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            prefix = prefix_for(store, s, max_words=max_words, per_word=per_word)
            with_facts += bool(prefix)
            f.write(f"{prefix}{s}\n\n")
    return len(sentences), with_facts
