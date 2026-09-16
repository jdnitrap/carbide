"""Lets Carbide propose AND name genuinely NEW dimensions for itself
from real training data, fully automatically -- no human names or
confirms anything. The one non-negotiable rule: every discovered column
still gets a real, readable label, never a placeholder like dim_0 or
DIMENSION_1. Readability is enforced two ways: (1) names are built from
real signal (the known category a cluster's usage pattern most resembles
-- e.g. NOUN_LIKE), never a counter, and (2) a hard check refuses to
persist anything that still looks like a placeholder, as a safety net.

Real technique, not invented: the distributional hypothesis (words that
occur in similar contexts tend to share grammatical function) is a
standard, citable basis for unsupervised word clustering in
linguistics/NLP -- the same basis esgr's own discover_dimension.py cites
(that version proposes clusters and waits for a human to name them; this
one names them itself, adapted for Carbide's graph-free, byte-level
design and the explicit later direction to drop the human step).

"Context" is the POS tag of the neighboring word (via mdbe._pos_scan),
mined straight from the real training corpus -- both for the unknown
words being clustered, and for the reference: what (prev_pos, next_pos)
pattern do words ALREADY known to be a NOUN/VERB/ADJECTIVE/etc actually
have in this corpus? A new cluster is named after whichever known
category its own dominant pattern matches; if none match well enough,
it's named directly from the pattern itself (e.g. AFTER_ARTICLE_WORDS),
which is still a real, readable description of a real, checkable fact
-- never a meaningless index.
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
    """True if word already has a real hand-given part-of-speech category
    in mdbe.py -- these are NOT discovery candidates; the whole point is
    finding what's not covered yet."""
    return _pos_scan(word) is not None


def _context_signatures(words, include_word):
    """{word: Counter({(prev_pos, next_pos): n})} over every position i
    where include_word(words[i]) is True. prev_pos/next_pos come from
    mdbe._pos_scan on the neighboring word (None for an uncovered or
    missing neighbor -- a real, honest "unknown neighbor" state, not
    guessed)."""
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
    """Real pass over the actual training corpus, restricted to words
    with no hand-given POS category yet -- the real discovery
    candidates. Returns {word: Counter({(prev_pos, next_pos): n})}."""
    words = _read_corpus_words(corpus_path, sample_bytes)
    return _context_signatures(words, lambda w: not _is_covered(w))


def reference_category_signatures(corpus_path, sample_bytes=2_000_000):
    """The real, corpus-measured dominant (prev_pos, next_pos) pattern
    for words ALREADY hand-categorized as each known part of speech --
    "what does a typical NOUN actually sit next to in this text?" Used
    to name new clusters by resemblance. Returns {category_name:
    (prev_pos, next_pos)} for whichever categories have enough real
    data in this corpus to have a dominant pattern at all."""
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
    """Read-only. Mines real context signatures, then groups uncovered
    words by their DOMINANT (prev_pos, next_pos) signature -- words
    sharing the same most-common neighbor pattern are proposed as one
    cluster. Only clusters with >= min_cluster_size real distinct words,
    each seen >= min_freq times, are returned; small or noisy clusters
    are real evidence of nothing and are dropped rather than proposed.

    Returns a list of {signature, words, total_occurrences} dicts,
    largest cluster first. Touches no files -- purely reads the corpus."""
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
    """A readable fallback name built directly from the real (prev_pos,
    next_pos) pattern, for a cluster that doesn't resemble any known
    category closely enough -- still a real, checkable description
    (this cluster's words really do sit after an X and before a Y in
    the corpus), never an arbitrary index."""
    prev_pos, next_pos = signature
    prev_part = f"AFTER_{prev_pos}" if prev_pos else "AT_START"
    next_part = f"BEFORE_{next_pos}" if next_pos else "AT_END"
    return f"{prev_part}_{next_part}_WORDS"


def name_cluster(signature, reference_signatures):
    """Automatic, readable name for one proposed cluster: the known
    category whose own real corpus-measured signature matches this
    cluster's signature exactly, suffixed "_LIKE" (same pattern as
    esgr's own real discovered dimensions, ADJECTIVE_LIKE/NOUN_LIKE --
    here computed automatically instead of picked by a person). Falls
    back to a signature-derived label when no known category matches.
    Returns (dimension_name, value_name)."""
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


def _persist(dimension_name: str, value_name: str, words, path):
    _validate_real_name(dimension_name, "Dimension")
    _validate_real_name(value_name, "Value")
    discovered = {}
    if os.path.exists(path):
        with open(path) as f:
            discovered = json.load(f)
    key = dimension_name.strip().upper()
    discovered[key] = {
        "value_name": value_name.strip().upper(),
        "words": sorted(set(w.lower() for w in words)),
    }
    with open(path, "w") as f:
        json.dump(discovered, f, indent=2)
    return discovered[key]


def discover(corpus_path, sample_bytes=2_000_000, min_freq=50, min_cluster_size=10,
             path=DISCOVERED_PATH):
    """The full automatic pipeline: mine the real corpus, cluster
    uncovered words, name each cluster from real distributional
    resemblance to a known category (or, failing that, from its own
    real context pattern), and persist every result -- no human step.
    Two clusters that would auto-name identically get a real, readable
    disambiguator appended (their most frequent distinct word), never a
    bare counter. Returns the list of {dimension_name, value_name,
    words} dicts that were written."""
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
