"""Build the core (dictionary) module from WordNet, limited to the words Carbide
actually meets, so the graph stays small. nltk + its WordNet data are optional:
`pip install nltk` then `python -m nltk.downloader wordnet`."""
import hashlib
import re
from collections import Counter

from .store import (CORE, REL_ANTONYM, REL_HAS_CLASS, REL_HAS_POS, REL_IS_A, REL_PART_OF, REL_SYNONYM,
                    SOURCE_DICTIONARY)

WN_POS = {"n": "NOUN", "v": "VERB", "a": "ADJECTIVE", "s": "ADJECTIVE", "r": "ADVERB"}
MAX_SENSES = 3   # relations come from a word's first few (most frequent) senses


def corpus_words(path, vocab_size=5000, min_count=2, max_bytes=8_000_000):
    """(most frequent lowercase words, counts of those words, total tokens, sha1 of the bytes read)."""
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    tokens = re.findall(r"[a-z]+", raw.decode("ascii", errors="replace").lower())
    counts = Counter(tokens)
    words = [w for w, c in counts.most_common(vocab_size) if c >= min_count]
    return words, {w: counts[w] for w in words}, len(tokens), hashlib.sha1(raw).hexdigest()


def _wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        return wn
    except (ImportError, LookupError) as e:
        raise RuntimeError("WordNet is not available. Install it with: pip install nltk && "
                           "python -m nltk.downloader wordnet") from e


def seed_function_words(store, *, run=None):
    """WordNet has no entries for function words (the, of, and, she...) -- the closed
    classes that carry the grammar. Carbide's own rule lists (mdbe.py) already hold
    them, so seed them as dictionary edges, and scale down WordNet's rare open-class
    senses of the same words (e.g. 'i' the letter) so the function-word reading wins.
    Every scaling is logged."""
    from .. import mdbe
    closed = {"ARTICLE": mdbe._ARTICLES, "PRONOUN": mdbe._PRONOUNS, "PREPOSITION": mdbe._PREPOSITIONS,
              "CONJUNCTION": mdbe._CONJUNCTIONS, "AUXILIARY_VERB": mdbe._AUXILIARY_VERBS,
              "NUMERAL": mdbe._NUMERALS}
    n = 0
    for pos, words in closed.items():
        for w in sorted(words):
            store.add_edge(w, pos, REL_HAS_POS, source=SOURCE_DICTIONARY, module=CORE, weight=1.0, run=run,
                           dst_kind="POS")
            for open_pos in ("NOUN", "VERB", "ADJECTIVE", "ADVERB"):
                store.adjust_edge(w, open_pos, REL_HAS_POS, 0.1, dst_kind="POS", reason="closed-class word")
            n += 1
    return n


def load_wordnet_core(store, words, *, run=None, wn=None):
    """Add dictionary edges for `words` (all in module `core`, source `dictionary`):
    HAS_POS (weighted by sense frequency), HAS_CLASS (WordNet's 45 semantic classes),
    and IS_A / SYNONYM / ANTONYM / PART_OF between words in the vocabulary. Every
    word gets a node even if WordNet does not know it. Returns counts."""
    wn = wn or _wordnet()
    vocab = set(words)
    stats = {"words": len(words), "known": 0, "edges": 0}

    def edge(src, dst, rel, weight, src_kind="WORD", dst_kind="WORD"):
        store.add_edge(src, dst, rel, source=SOURCE_DICTIONARY, module=CORE, weight=weight, run=run,
                       src_kind=src_kind, dst_kind=dst_kind)
        stats["edges"] += 1

    for w in words:
        store.add_node(w)
    for w in words:
        synsets = wn.synsets(w)
        if not synsets:
            continue
        stats["known"] += 1
        pos_w, cls_w = Counter(), Counter()
        for s in synsets:
            base = wn.morphy(w, s.pos()) or w
            names = {w, base}
            cnt = 1 + sum(l.count() for l in s.lemmas() if l.name().lower() in names)
            pos_w[WN_POS[s.pos()]] += cnt
            cls_w[s.lexname()] += cnt
        total = sum(pos_w.values())
        for p, c in pos_w.items():
            edge(w, p, REL_HAS_POS, c / total, dst_kind="POS")
        total = sum(cls_w.values())
        for k, c in cls_w.items():
            edge(w, k, REL_HAS_CLASS, c / total, dst_kind="CLASS")
        store.add_node(w, note=synsets[0].definition())
        for rank, s in enumerate(synsets[:MAX_SENSES]):
            share = 1.0 / (1 + rank)
            for h in s.hypernyms():
                for n in h.lemma_names():
                    t = n.lower()
                    if t in vocab and t != w:
                        edge(w, t, REL_IS_A, share)
            for n in s.lemma_names():
                t = n.lower()
                if t in vocab and t != w:
                    edge(w, t, REL_SYNONYM, share)
            for part in s.part_holonyms() + s.substance_holonyms():
                for n in part.lemma_names():
                    t = n.lower()
                    if t in vocab and t != w:
                        edge(w, t, REL_PART_OF, share)
            for l in s.lemmas():
                if l.name().lower() == w:
                    for a in l.antonyms():
                        t = a.name().lower()
                        if t in vocab and t != w:
                            edge(w, t, REL_ANTONYM, share)
    return stats


def build_core(store, corpus_path, *, vocab_size=5000, min_count=2, extra_words=(), wn=None):
    """One automatic, recorded run: read the corpus, add its vocabulary, record
    general word frequencies (used later to spot domain-specific terms), and load
    the dictionary edges. Idempotent: re-running adds only what is new."""
    words, counts, total, sha = corpus_words(corpus_path, vocab_size, min_count)
    seen = set(words)
    for w in extra_words:  # e.g. Carbide's Layer-2 vocabulary, so every word it knows has a node
        if w not in seen:
            words.append(w)
            seen.add(w)
    with store.run("build-core", CORE, note=corpus_path, source_sha=sha) as run:
        stats = load_wordnet_core(store, words, run=run, wn=wn)
        stats["function_words"] = seed_function_words(store, run=run)
        for w, c in counts.items():
            store.bump_count(CORE, store.node_id(w), 0)  # ensure row exists
            store.db.execute("UPDATE term_counts SET count = ? WHERE module = ? AND node = ?",
                             (c, CORE, store.node_id(w)))
        store.db.execute("UPDATE modules SET tokens = ? WHERE name = ?", (total, CORE))
        store.ensure_capacity(store.max_node_id() + 1)
        stats.update(tokens=total, sha=sha[:10])
    return stats
