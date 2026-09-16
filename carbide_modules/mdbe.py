"""MDBE embedding and the Selective-SSM model architecture."""
import csv
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CONSTRAINTS = 6


def mdbe_constraints(bytes_seq: torch.Tensor) -> torch.Tensor:
    f = bytes_seq.float()
    cols = []
    cols.append(((f >= 65) & (f <= 90)) | ((f >= 97) & (f <= 122)))
    cols.append((f >= 48) & (f <= 57))
    cols.append((f >= 65) & (f <= 90))
    cols.append(((f >= 33) & (f <= 47)) | ((f >= 58) & (f <= 64)) |
                ((f >= 91) & (f <= 96)) | ((f >= 123) & (f <= 126)))
    cols.append((f == 32) | (f == 9) | (f == 10) | (f == 13))
    lead = ((f >= 0) & (f < 0x80)) | ((f >= 0xC0) & (f <= 0xFF))
    cols.append(lead)
    return torch.stack(cols, dim=-1).float()


# ---------------------------------------------------------------------------
# Language-mechanics columns. Originally the 12 pipeline-stage mechanics
# named in the repo's own mdbe/language_mechanics_Column_key.csv (Channel
# .. Sociolinguistics) -- NOT the dialogue/speech-act columns (TURN,
# QUESTION_FORM, ROLE/TENSE/etc.) esgr's GRU head builds in head.py.
#
# 2026-09-15, at explicit user direction ("I need more stuff add to the
# columns"): Morphology and Syntax broken into their real sub-parts, and
# a full Grammatical group added (Tense/Number/Case, later expanded to
# Voice/Mood/Aspect/Person/Gender/Degree) plus a standalone Part of
# speech group -- none of these are in the original 12-mechanic CSV, all
# added the same way: one-hot-plus-a-real-"none"-bucket, identical shape
# to esgr's head.py WORD_DIMS. Of the original 12, only tokenization,
# pragmatics, and discourse were kept (real, context-dependent signals a
# bare per-byte embedding can't derive on its own); channel, phonetics,
# phonology, orthography, lexicon, semantics, and sociolinguistics were
# cut at the user's direction -- each was either a constant (channel/
# phonetics/phonology/semantics: zero discriminative value, redundant
# with a Linear layer's own bias), or redundant with columns that already
# exist elsewhere (orthography restates is_alpha/is_digit/is_punct/
# is_space; lexicon restates what Part of speech's closed classes already
# cover; sociolinguistics was a single-byte fact, the kind the learned
# embedding can already absorb on its own).
#
# The user's own reasoning for wanting even heuristic (not certain)
# columns like Syntax/Voice/Mood's IMPERATIVE: Carbide's *learned*
# embedding sits right next to these hand-given columns and trains on
# real data, so it can learn how much to trust an imperfect hint -- these
# don't have to be ground truth the way a loss target would. That is also
# why every group still gets a real "none" bucket rather than an implicit
# all-zero: "this doesn't apply" is answered by literally running the
# rule, so it is itself a checked fact, not a gap.
SCALAR_MECHANICS = ["tokenization", "pragmatics", "discourse", "negation"]

# One-hot-plus-"none" groups. Each (dim_name, value_names) pair gets
# len(value_names)+1 columns: bare value names (unprefixed) plus one
# "{dim_name}:NONE" bucket -- matches the naming convention worked out
# directly in the spreadsheet (values bare, only the none-bucket keeps
# its group prefix, since only that column would otherwise collide
# across every group sharing the word "none").
#
# Three bare-name collisions were found and resolved by renaming the
# MORE DERIVED/less-fundamental side (a spelling fact) rather than the
# grammatical-value side: Syntax's VERB (a clause ROLE) -> MAIN_VERB,
# keeping Part of speech's VERB (a lexical CATEGORY) bare; Morphology's
# PLURAL/COMPARATIVE/SUPERLATIVE (SUFFIX facts: "this word ends in -s/
# -er/-est") -> PLURAL_SUFFIX/COMPARATIVE_SUFFIX/SUPERLATIVE_SUFFIX,
# keeping Number's PLURAL and Degree's COMPARATIVE/SUPERLATIVE (the
# grammatical values themselves) bare.
MORPHOLOGY_NAMES = ["PLURAL_SUFFIX", "PAST_TENSE", "PRESENT_PARTICIPLE",
                     "COMPARATIVE_SUFFIX", "SUPERLATIVE_SUFFIX", "ADVERB_FORM",
                     "NOMINALIZATION"]
POS_NAMES = ["NOUN", "VERB", "ADJECTIVE", "ADVERB", "ARTICLE", "PRONOUN",
             "PREPOSITION", "CONJUNCTION", "AUXILIARY_VERB", "NUMERAL"]
# OBJECT (and, below, Voice/Mood's IMPERATIVE) is a real, documented
# approximation, not a certainty like everything else here: which word
# acts as an object depends on sentence structure, not word identity, so
# a hardcoded list can't give a hard fact the way POS or morphology can.
# This is the nearest-real-word heuristic (skipping one leading article)
# right after the first verb of each sentence -- deliberately causal
# (see _clause_scan below), so it stays exactly consistent between
# training and real streaming generation. A SUBJECT value was tried and
# removed: marking a word as subject required knowing a verb that comes
# LATER in the sentence, which is real information at full-batch training
# time but never available at real generation time -- exactly the kind
# of future-information leak this file otherwise refuses to introduce.
SYNTAX_NAMES = ["MAIN_VERB", "OBJECT"]
TENSE_NAMES = ["PAST", "PRESENT"]
NUMBER_NAMES = ["SINGULAR", "PLURAL"]
CASE_NAMES = ["SUBJECTIVE", "OBJECTIVE", "POSSESSIVE"]
VOICE_NAMES = ["ACTIVE", "PASSIVE"]
MOOD_NAMES = ["INDICATIVE", "IMPERATIVE", "INTERROGATIVE"]
ASPECT_NAMES = ["PROGRESSIVE", "PERFECT"]
PERSON_NAMES = ["FIRST", "SECOND", "THIRD"]
GENDER_NAMES = ["MASCULINE", "FEMININE", "NEUTER"]
DEGREE_NAMES = ["POSITIVE", "COMPARATIVE", "SUPERLATIVE"]

MECHANIC_DIMS = [
    ("MORPHOLOGY", MORPHOLOGY_NAMES),
    ("PART_OF_SPEECH", POS_NAMES),
    ("SYNTAX", SYNTAX_NAMES),
    ("TENSE", TENSE_NAMES),
    ("NUMBER", NUMBER_NAMES),
    ("CASE", CASE_NAMES),
    ("VOICE", VOICE_NAMES),
    ("MOOD", MOOD_NAMES),
    ("ASPECT", ASPECT_NAMES),
    ("PERSON", PERSON_NAMES),
    ("GENDER", GENDER_NAMES),
    ("DEGREE", DEGREE_NAMES),
]

# Dimensions Carbide proposed for itself from real corpus data (see
# discover_dimension.py) and a human then confirmed with a real name --
# discover_dimension.confirm_dimension() already refuses placeholder
# names (dim_0, DIMENSION_1, ...) at write time, so anything found here
# on disk is trusted to be real. Empty ({}, []) until someone actually
# confirms a discovery -- an honest, opt-in default, not a fabricated
# column. Loaded once at import time, same as every other MECHANIC_DIMS
# entry above; a newly confirmed dimension only takes effect on the next
# process start (model dimensions are fixed once TOTAL_CONSTRAINTS is
# computed below, same reason NUM_CONSTRAINTS itself can't change live).
DISCOVERED_DIMENSIONS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "discovered_dimensions.json"))


def _load_discovered_dimensions(path=DISCOVERED_DIMENSIONS_PATH):
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        data = json.load(f)
    dims = []
    word_lookup = {}
    for dim_name, entry in data.items():
        value_name = entry["value_name"]
        dims.append((dim_name, [value_name]))
        word_lookup[(dim_name, value_name)] = frozenset(entry["words"])
    return dims, word_lookup


DISCOVERED_MECHANIC_DIMS, _DISCOVERED_WORD_LOOKUP = _load_discovered_dimensions()
MECHANIC_DIMS = MECHANIC_DIMS + DISCOVERED_MECHANIC_DIMS


def _discovered_scan(word: str):
    """First matching confirmed-discovered dimension for the real word
    ending here, or None. Every (dimension, value) pair here passed
    discover_dimension.confirm_dimension()'s real-name check already --
    there is no placeholder-named column this can ever produce."""
    for (dim_name, value_name), words in _DISCOVERED_WORD_LOOKUP.items():
        if word in words:
            return dim_name, value_name
    return None

LANGUAGE_MECHANICS_NAMES = list(SCALAR_MECHANICS)
_seen_value_names = set(SCALAR_MECHANICS)
for _dim_name, _names in MECHANIC_DIMS:
    for _nm in _names:
        if _nm in _seen_value_names:
            raise ValueError(f"duplicate bare column name across mechanic groups: {_nm!r}")
        _seen_value_names.add(_nm)
        LANGUAGE_MECHANICS_NAMES.append(_nm)
    LANGUAGE_MECHANICS_NAMES.append(f"{_dim_name}:NONE")
NUM_LANGUAGE_MECHANICS = len(LANGUAGE_MECHANICS_NAMES)
TOTAL_CONSTRAINTS = NUM_CONSTRAINTS + NUM_LANGUAGE_MECHANICS
_COLUMN_INDEX = {name: i for i, name in enumerate(LANGUAGE_MECHANICS_NAMES)}
_MECHANIC_NONE_INDEX = {dim: _COLUMN_INDEX[f"{dim}:NONE"] for dim, _ in MECHANIC_DIMS}
_MECHANIC_VALUE_INDEX = {dim: {v: _COLUMN_INDEX[v] for v in names}
                          for dim, names in MECHANIC_DIMS}

# Real, hand-given word lists (same "explicitly hardcoded, human-
# interpretable" spirit as MDBE's own definition, and the same closed-
# class/pronoun words already sitting in
# mdbe/language_mechanics_workbook_Lists.csv) -- not exhaustive, but
# real words, not invented ones.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "into", "onto", "over", "under", "about", "after", "before",
    "and", "but", "or", "nor", "so", "yet", "not", "no",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
})
# PDTB-style discourse connectives -- same category esgr's grammar_extra.py
# DISCOURSE_MAP draws from, independently listed here so Carbide has no
# import dependency on esgr.
_DISCOURSE_WORDS = frozenset({
    "then", "after", "before", "when", "until", "once", "while", "if",
    "meanwhile", "because", "although", "though", "therefore", "however",
    "moreover", "furthermore", "thus", "consequently", "nevertheless",
    "otherwise",
})

# Closed classes: complete, exact word lists -- every member of the class
# really is in the list, so a match here is a hard fact.
_ARTICLES = frozenset({"the", "a", "an"})
_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
})
_PREPOSITIONS = frozenset({
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "into", "onto", "over", "under", "about", "after", "before", "through",
    "across", "without", "within", "among", "between", "against", "during", "upon",
})
_CONJUNCTIONS = frozenset({
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although",
    "though", "while", "when", "until", "since", "both",
})
_AUXILIARY_VERBS = frozenset({
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
})
_NUMERALS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "billions", "millions",
})
# Open classes: necessarily partial -- these vocabularies are unbounded,
# so a hardcoded list can only recognize the words it names ("none" for
# anything else is honest -- "no match in this list", not "definitely not
# a noun"). Seeded from carbide_training_dataset.txt's own most-frequent
# real content words, not invented, plus a handful of common generics.
_NOUNS = frozenset({
    "world", "life", "patterns", "stories", "knowledge", "power", "universe",
    "consciousness", "history", "revolution", "learning", "information",
    "ocean", "mind", "experience", "existence", "choices", "question",
    "beliefs", "truth", "meaning", "philosophy", "physics",
    "cat", "dog", "man", "woman", "time", "day", "hand", "child", "eye",
    "place", "fact", "case", "point", "government", "company", "number", "group",
})
_VERBS = frozenset({
    "becomes", "reveals", "live", "lives", "lead", "leads", "make", "makes",
    "matter", "matters", "run", "runs", "running", "ran", "sit", "sits",
    "sitting", "sat", "create", "creates", "created", "exist", "exists",
    "existed", "think", "thinks", "thought", "know", "knows", "knew",
})
_ADJECTIVES = frozenset({
    "human", "vast", "ordinary", "deep", "complex", "own", "classical",
    "same", "unexamined", "black", "good", "new", "first", "last", "long",
    "great", "little", "other", "old", "right", "big", "high", "different",
    "small", "large", "next", "early", "young", "important", "few", "public",
    "bad", "able",
})
_ADVERBS = frozenset({
    "not", "yet", "constantly", "only", "simultaneously", "also", "very",
    "often", "too", "usually", "really", "never", "always", "sometimes",
    "together", "likely", "simply", "far", "actually",
})
# Checked in this order: closed classes (complete, exact) before open
# classes (examples only); AUXILIARY_VERB before the open VERB list since
# an auxiliary IS a verb but the more specific closed-class tag wins.
_POS_LOOKUP_ORDER = [
    ("ARTICLE", _ARTICLES), ("PRONOUN", _PRONOUNS), ("PREPOSITION", _PREPOSITIONS),
    ("CONJUNCTION", _CONJUNCTIONS), ("AUXILIARY_VERB", _AUXILIARY_VERBS),
    ("NUMERAL", _NUMERALS),
    ("NOUN", _NOUNS), ("VERB", _VERBS), ("ADJECTIVE", _ADJECTIVES), ("ADVERB", _ADVERBS),
]

_IRREGULAR_PAST = frozenset({
    "was", "were", "had", "did", "went", "sat", "ran", "ate", "saw", "knew",
    "thought", "came", "took", "made", "said", "got", "gave", "found", "told",
    "became", "left", "felt", "brought", "began", "kept", "held", "wrote",
    "stood", "heard", "let", "meant", "set", "met", "paid", "sent", "built",
    "understood", "drew", "broke", "spoke", "rose", "chose", "arose", "led", "read",
})
# Real false-positive guards: common words that end in a suffix-shaped
# string without carrying that morphology. Not exhaustive (open-class
# vocabulary is unbounded), but real, common exceptions.
_PLURAL_STOPLIST = frozenset({
    "is", "was", "his", "this", "us", "yes", "gas", "bus", "plus", "minus",
    "focus", "always", "across", "less", "unless", "news",
})
_COMPARATIVE_STOPLIST = frozenset({
    "under", "over", "other", "her", "water", "after", "never", "whether",
    "number", "matter", "power", "order", "proper", "further", "together",
})
_NOMINALIZATION_SUFFIXES = ("tion", "ness", "ment", "ance", "ence", "ity")

_TENSE_PRESENT_AUX = frozenset({"is", "am", "are", "does", "do", "has", "have"})
_NUMBER_PLURAL_PRONOUNS = frozenset({"we", "you", "they", "us", "them", "our", "your", "their"})
_NUMBER_SINGULAR_PRONOUNS = frozenset({"i", "he", "she", "it", "me", "him", "her", "my", "his", "its"})
# Case is genuinely ambiguous for a few pronouns in English itself ("you"/
# "it" share one form for subjective and objective; "her" shares one for
# objective and possessive) -- not a gap in this list, a real fact about
# the language. Priority below picks one real tag to assign; it does not
# hide the ambiguity, it just can't emit two one-hot values at once.
_CASE_SUBJECTIVE = frozenset({"i", "he", "she", "we", "they", "you", "it"})
_CASE_OBJECTIVE = frozenset({"me", "him", "her", "us", "them", "you", "it"})
_CASE_POSSESSIVE = frozenset({"my", "your", "his", "her", "its", "our", "their"})

_NEGATION_WORDS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor",
})

# Voice/Aspect need to recognize a real "be"/"have" auxiliary immediately
# before the current word, and a real past-participle shape on the
# current word. Participles differ from simple past tense for some
# irregular verbs (eat/ate/EATEN, not eat/ate/ate) -- real, hand-listed,
# not exhaustive, same honest-partial-coverage limitation as every other
# open-class list here.
_BE_FORMS = frozenset({"is", "am", "are", "was", "were", "be", "been", "being"})
_HAVE_FORMS = frozenset({"has", "have", "had"})
_IRREGULAR_PARTICIPLES = frozenset({
    "been", "done", "gone", "seen", "given", "taken", "known", "written",
    "spoken", "broken", "chosen", "eaten", "shown", "grown", "thrown", "drawn",
    "made", "said", "found", "told", "brought", "built", "understood", "kept",
    "held", "stood", "heard", "let", "meant", "set", "met", "paid", "sent",
})

_PERSON_FIRST = frozenset({"i", "we", "me", "us", "my", "our", "myself", "ourselves"})
_PERSON_SECOND = frozenset({"you", "your", "yourself", "yourselves"})
_PERSON_THIRD = frozenset({
    "he", "she", "it", "they", "him", "her", "them", "his", "its", "their",
    "himself", "herself", "itself", "themselves",
})

_GENDER_MASCULINE = frozenset({"he", "him", "his", "himself"})
_GENDER_FEMININE = frozenset({"she", "her", "herself"})
_GENDER_NEUTER = frozenset({"it", "its", "itself"})

_DEGREE_COMPARATIVE_IRREGULAR = frozenset({"better", "worse", "more", "less"})
_DEGREE_SUPERLATIVE_IRREGULAR = frozenset({"best", "worst", "most", "least"})

# Longest hand-listed word anywhere above -- how far back a word can ever
# extend, so incremental.py's rolling byte-history buffer never truncates
# a real word mid-match.
_MAX_LOOKBACK = max(
    max(len(w) for w in _FUNCTION_WORDS | _DISCOURSE_WORDS),
    max(len(w) for cat in (_ARTICLES, _PRONOUNS, _PREPOSITIONS, _CONJUNCTIONS,
                            _AUXILIARY_VERBS, _NUMERALS, _NOUNS, _VERBS,
                            _ADJECTIVES, _ADVERBS) for w in cat),
    max(len(w) for w in _IRREGULAR_PAST),
    max(len(w) for w in _IRREGULAR_PARTICIPLES),
    max(len(w) for w in _NEGATION_WORDS),
    max(len(w) for w in _DEGREE_COMPARATIVE_IRREGULAR | _DEGREE_SUPERLATIVE_IRREGULAR),
)

# _clause_scan (SYNTAX/VOICE/ASPECT/MOOD) needs more history than any
# single word: it re-derives "has this sentence already had a verb" and
# "what real word came right before this one" by scanning back to the
# last real sentence boundary (or the start of the given text), every
# time it's called. In a full forward pass that's the whole sequence, so
# it's always exactly right. In incremental generation it's only ever as
# good as the rolling byte buffer incremental.py keeps -- a sentence
# longer than this bound would make the recomputation "forget" a verb or
# sentence-start that's fallen out of the window, the same class of
# bounded-context tradeoff SelectiveSSM's chunk_size and LocalByteConv's
# kernel_size already make elsewhere in this file, just sized for real
# sentence lengths instead of a few bytes. 256 covers the large majority
# of real sentences; genuinely longer ones can occasionally see SYNTAX/
# VOICE/ASPECT/MOOD disagree between training and live generation --
# documented, not hidden.
_CLAUSE_LOOKBACK = 256


def _decode_ascii_lower(byte_values) -> str:
    """Byte 0-255 -> single lowercase char, '\\0' standing in for anything
    outside printable ASCII (never matches a word or suffix, so it
    behaves as a real, inert boundary). Lowercasing here (not per-lookup)
    is what makes 'The' at a sentence start match the same word lists as
    'the' mid-sentence -- capitalization is real orthographic information
    (already captured separately by mdbe_constraints' is_upper), not a
    different word."""
    return "".join(chr(v).lower() if 0 <= v < 128 else "\0" for v in byte_values)


def _current_word(text_lower: str, t: int) -> str:
    """The real word ending at (and including) t: the maximal run of
    letters back to a genuine left boundary (start of string, or a
    non-letter immediately before). This is the ONLY substring a
    left-boundary match can ever resolve to at position t -- any shorter
    span inside the same run fails the left-boundary check by
    construction -- so, unlike the old multi-length scan, there is
    nothing to try but this one word. Causal only: never looks past t,
    exactly esgr's word_at_position() rule ("commits to its best reading
    of what's been seen so far")."""
    if not text_lower[t].isalpha():
        return ""
    start = t
    while start > 0 and text_lower[start - 1].isalpha():
        start -= 1
    return text_lower[start:t + 1]


def _lexical_scan(word: str):
    """Returns (is_function_word, is_discourse_word) for the real word
    ending here."""
    return word in _FUNCTION_WORDS, word in _DISCOURSE_WORDS


def _pos_scan(word: str):
    """First matching part-of-speech category for the real word ending
    here, or None. Closed classes are exact; open classes (NOUN/VERB/
    ADJECTIVE/ADVERB) only recognize the words hand-listed above."""
    for name, vocab in _POS_LOOKUP_ORDER:
        if word in vocab:
            return name
    return None


def _morphology_scan(word: str):
    """First matching morphological ending for the real word ending
    here, checked most-specific-suffix-first so e.g. 'consciousness'
    (NOMINALIZATION, -ness) is never also read as a plural (-s)."""
    if len(word) < 2:
        return None
    for suf in _NOMINALIZATION_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            return "NOMINALIZATION"
    if word.endswith("est") and len(word) > 3:
        return "SUPERLATIVE_SUFFIX"
    if word.endswith("er") and len(word) > 2 and word not in _COMPARATIVE_STOPLIST:
        return "COMPARATIVE_SUFFIX"
    if word in _IRREGULAR_PAST or (word.endswith("ed") and len(word) > 2):
        return "PAST_TENSE"
    if word.endswith("ing") and len(word) > 3:
        return "PRESENT_PARTICIPLE"
    if word.endswith("ly") and len(word) > 2:
        return "ADVERB_FORM"
    if word not in _PLURAL_STOPLIST and len(word) > 1 and (word.endswith("es") or word.endswith("s")):
        return "PLURAL_SUFFIX"
    return None


def _is_participle_shaped(word: str) -> bool:
    """A real (if partial) past-participle shape: an irregular participle,
    an irregular simple past (many English verbs share one form for both,
    e.g. 'made'/'made'), or a regular -ed ending."""
    return word in _IRREGULAR_PARTICIPLES or word in _IRREGULAR_PAST or \
        (word.endswith("ed") and len(word) > 2)


def _tense_scan(word: str, morphology):
    if morphology == "PAST_TENSE":
        return "PAST"
    if word in _TENSE_PRESENT_AUX:
        return "PRESENT"
    return None


def _number_scan(word: str, morphology):
    if word in _NUMBER_PLURAL_PRONOUNS:
        return "PLURAL"
    if word in _NUMBER_SINGULAR_PRONOUNS:
        return "SINGULAR"
    if morphology == "PLURAL_SUFFIX":
        return "PLURAL"
    return None


def _case_scan(word: str):
    if word in _CASE_POSSESSIVE:
        return "POSSESSIVE"
    if word in _CASE_SUBJECTIVE:
        return "SUBJECTIVE"
    if word in _CASE_OBJECTIVE:
        return "OBJECTIVE"
    return None


def _negation_scan(text_lower: str, t: int, word: str) -> bool:
    """A real negation marker ending here: either a hand-listed negation
    word, or the 't' of an "...n't" contraction (don't, isn't, won't...).
    The contraction check looks 2 bytes further back than `word` itself
    (past the apostrophe, to the 'n') -- still causal, still no lookahead
    past t, just reading a bit more of what already came before it."""
    if word in _NEGATION_WORDS:
        return True
    if word == "t" and t >= 2 and text_lower[t - 1] == "'" and text_lower[t - 2] == "n":
        return True
    return False


def _person_scan(word: str):
    if word in _PERSON_FIRST:
        return "FIRST"
    if word in _PERSON_SECOND:
        return "SECOND"
    if word in _PERSON_THIRD:
        return "THIRD"
    return None


def _gender_scan(word: str):
    if word in _GENDER_MASCULINE:
        return "MASCULINE"
    if word in _GENDER_FEMININE:
        return "FEMININE"
    if word in _GENDER_NEUTER:
        return "NEUTER"
    return None


def _degree_scan(word: str, morphology, pos_tag):
    """Grammatical degree, distinct from Morphology's suffix facts: this
    is the abstract value (POSITIVE/COMPARATIVE/SUPERLATIVE), covering
    both suffix-marked and irregular forms (better/best, more/most)."""
    if word in _DEGREE_SUPERLATIVE_IRREGULAR or morphology == "SUPERLATIVE_SUFFIX":
        return "SUPERLATIVE"
    if word in _DEGREE_COMPARATIVE_IRREGULAR or morphology == "COMPARATIVE_SUFFIX":
        return "COMPARATIVE"
    if pos_tag in ("ADJECTIVE", "ADVERB"):
        return "POSITIVE"
    return None


def _clause_scan(text_lower: str):
    """One causal left-to-right pass computing four real, sentence-local
    tag groups together: SYNTAX (MAIN_VERB/OBJECT), VOICE (ACTIVE/
    PASSIVE), ASPECT (PROGRESSIVE/PERFECT), MOOD (IMPERATIVE/INDICATIVE/
    INTERROGATIVE).

    Re-evaluated at EVERY letter of the CURRENT word, backward-only, the
    same rule _current_word already uses everywhere else in this module
    -- never "wait and see the whole word" (that requires a delimiter
    that, in real generation, hasn't happened yet and might never
    retroactively apply to a byte already fed through the SSM). Instead,
    every check here uses only (a) the word-so-far ending at the current
    position, and (b) the one real, ALREADY-COMPLETE word before it
    (`prev_word`, fixed the moment the previous real word finished, never
    reassigned mid-word). A `*_tagged_this_run` guard stops the SAME
    condition from refiring every time the current word grows by one
    more letter, without ever un-firing an earlier commitment -- an
    earlier, shorter-prefix guess ("run" inside "running") is left
    standing exactly as everywhere else in this file accepts.

    Because every decision only ever looks backward from the current
    byte, this produces byte-for-byte identical tags whether it is run
    over a whole sequence at once or one byte at a time -- see
    tests/test_incremental_decode.py -- PROVIDED the caller gives it
    enough real preceding text to find the current sentence's start and
    its most recent verb; incremental.py's rolling history buffer is
    sized generously (see _CLAUSE_LOOKBACK) for exactly this reason, not
    just for the longest single hand-listed word.

    SYNTAX's OBJECT, and MOOD's IMPERATIVE, are documented
    approximations, not certainties -- OBJECT is "the nearest real word
    after a verb", IMPERATIVE is "a verb as the first word of a
    sentence". (An earlier SUBJECT column was removed: marking a word as
    the subject required knowing a LATER verb, which is exactly the kind
    of future-information leak this whole file otherwise refuses to
    introduce -- real at full-batch time, but never available at real
    generation time, so keeping it would have taught the model to rely
    on a signal it can never actually have when generating live.)

    Returns (syntax_tags, voice_tags, aspect_tags, mood_tags), each
    {byte_index: tag_name}."""
    n = len(text_lower)
    syntax_tags, voice_tags, aspect_tags, mood_tags = {}, {}, {}, {}
    verb_seen_this_sentence = False
    pending_object = None  # None | "skip_article" | "tag_next"
    prev_word = None       # the last real, ALREADY-COMPLETE word (or None at a clause start)
    run_start = None       # start of the current in-progress word, or None between words
    sentence_word_count = 0
    object_done = verb_done = False  # guards: at most one commitment per run, never undone

    for t in range(n):
        ch = text_lower[t]
        if not ch.isalpha():
            if run_start is not None:
                prev_word = text_lower[run_start:t]
                run_start = None
            if ch in ".!?":
                if ch == "?":
                    mood_tags[t] = "INTERROGATIVE"
                verb_seen_this_sentence = False
                pending_object = None
                prev_word = None
                sentence_word_count = 0
            continue

        if run_start is None:
            run_start = t
            sentence_word_count += 1
            object_done = verb_done = False
        word = text_lower[run_start:t + 1]

        if not object_done and pending_object == "skip_article" and word in _ARTICLES:
            pending_object = "tag_next"
        elif not object_done and pending_object in ("skip_article", "tag_next"):
            syntax_tags[t] = "OBJECT"
            pending_object = None
            object_done = True

        if prev_word in _BE_FORMS and _is_participle_shaped(word):
            voice_tags[t] = "PASSIVE"
        elif word in _VERBS and prev_word not in _BE_FORMS:
            voice_tags[t] = "ACTIVE"

        if prev_word in _BE_FORMS and word.endswith("ing") and len(word) > 3:
            aspect_tags[t] = "PROGRESSIVE"
        elif prev_word in _HAVE_FORMS and _is_participle_shaped(word):
            aspect_tags[t] = "PERFECT"

        if not verb_done and not verb_seen_this_sentence and \
                (word in _AUXILIARY_VERBS or word in _VERBS):
            verb_seen_this_sentence = True
            verb_done = True
            syntax_tags[t] = "MAIN_VERB"
            mood_tags[t] = "IMPERATIVE" if sentence_word_count == 1 else "INDICATIVE"
            pending_object = "skip_article"

    return syntax_tags, voice_tags, aspect_tags, mood_tags


def language_mechanics_constraints(bytes_seq: torch.Tensor, history: torch.Tensor = None) -> torch.Tensor:
    """The language-mechanics tensor, shape (*bytes_seq.shape, NUM_LANGUAGE_MECHANICS).

    `history` (shape (B, H), optional) is real bytes that came before
    `bytes_seq` in the stream -- incremental_step() passes its rolling
    byte buffer here so a word/suffix/connective match that started
    before the current chunk still resolves correctly one byte at a
    time. A full forward pass over a whole sequence can omit it: the
    sequence already contains its own history, and the true start of a
    stream honestly has none to give."""
    B, T = bytes_seq.shape
    if history is None:
        history = bytes_seq.new_zeros(B, 0)
    H = history.shape[1]
    full = torch.cat([history, bytes_seq], dim=1)
    full_f = full.float()

    is_alpha = ((full_f >= 65) & (full_f <= 90)) | ((full_f >= 97) & (full_f <= 122))
    is_space = (full_f == 32) | (full_f == 9) | (full_f == 10) | (full_f == 13)
    prev_alpha = torch.cat([torch.zeros(B, 1, dtype=torch.bool), is_alpha[:, :-1]], dim=1)
    is_qmark_or_bang = (full == 63) | (full == 33)

    out_full = torch.zeros(B, H + T, NUM_LANGUAGE_MECHANICS)
    ci = _COLUMN_INDEX
    out_full[:, :, ci["tokenization"]] = (is_space | (is_alpha != prev_alpha)).float()
    out_full[:, :, ci["pragmatics"]] = is_qmark_or_bang.float()

    for dim, idx in _MECHANIC_NONE_INDEX.items():
        out_full[:, :, idx] = 1.0  # real "none" default everywhere, overwritten below

    for b in range(B):
        text = _decode_ascii_lower(full[b].tolist())
        syntax_tags, voice_tags, aspect_tags, mood_tags = _clause_scan(text)
        for pos in range(H, H + T):
            word = _current_word(text, pos)
            if word:
                _, is_disc = _lexical_scan(word)
                if is_disc:
                    out_full[b, pos, ci["discourse"]] = 1.0
                if _negation_scan(text, pos, word):
                    out_full[b, pos, ci["negation"]] = 1.0

                morph = _morphology_scan(word)
                if morph:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["MORPHOLOGY"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["MORPHOLOGY"][morph]] = 1.0

                pos_tag = _pos_scan(word)
                if pos_tag:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["PART_OF_SPEECH"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["PART_OF_SPEECH"][pos_tag]] = 1.0

                tense = _tense_scan(word, morph)
                if tense:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["TENSE"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["TENSE"][tense]] = 1.0

                number = _number_scan(word, morph)
                if number:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["NUMBER"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["NUMBER"][number]] = 1.0

                case = _case_scan(word)
                if case:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["CASE"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["CASE"][case]] = 1.0

                person = _person_scan(word)
                if person:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["PERSON"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["PERSON"][person]] = 1.0

                gender = _gender_scan(word)
                if gender:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["GENDER"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["GENDER"][gender]] = 1.0

                degree = _degree_scan(word, morph, pos_tag)
                if degree:
                    out_full[b, pos, _MECHANIC_NONE_INDEX["DEGREE"]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX["DEGREE"][degree]] = 1.0

                discovered = _discovered_scan(word)
                if discovered:
                    disc_dim, disc_value = discovered
                    out_full[b, pos, _MECHANIC_NONE_INDEX[disc_dim]] = 0.0
                    out_full[b, pos, _MECHANIC_VALUE_INDEX[disc_dim][disc_value]] = 1.0

            syn = syntax_tags.get(pos)
            if syn:
                out_full[b, pos, _MECHANIC_NONE_INDEX["SYNTAX"]] = 0.0
                out_full[b, pos, _MECHANIC_VALUE_INDEX["SYNTAX"][syn]] = 1.0

            voice = voice_tags.get(pos)
            if voice:
                out_full[b, pos, _MECHANIC_NONE_INDEX["VOICE"]] = 0.0
                out_full[b, pos, _MECHANIC_VALUE_INDEX["VOICE"][voice]] = 1.0

            aspect = aspect_tags.get(pos)
            if aspect:
                out_full[b, pos, _MECHANIC_NONE_INDEX["ASPECT"]] = 0.0
                out_full[b, pos, _MECHANIC_VALUE_INDEX["ASPECT"][aspect]] = 1.0

            mood = mood_tags.get(pos)
            if mood:
                out_full[b, pos, _MECHANIC_NONE_INDEX["MOOD"]] = 0.0
                out_full[b, pos, _MECHANIC_VALUE_INDEX["MOOD"][mood]] = 1.0

    return out_full[:, H:]


def all_constraints(bytes_seq: torch.Tensor, history: torch.Tensor = None) -> torch.Tensor:
    """The full live constraint tensor fed into MDBE/Block: the original
    6 byte-identity flags concatenated with the language-mechanics
    columns above."""
    return torch.cat([mdbe_constraints(bytes_seq),
                       language_mechanics_constraints(bytes_seq, history)], dim=-1)


BYTE_IDENTITY_COLUMN_NAMES = ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"]
# Single source of truth for the real, full column identity of
# all_constraints()'s output, in order -- not just a column COUNT.
# Loading a checkpoint by shape alone is a real trap: two different
# discovered-dimension sets can coincidentally total the same number of
# columns while meaning completely different things (found for real:
# an old checkpoint's discovered columns were PRONOUN_LIKE/
# AT_START_BEFORE_PREPOSITION_WORDS/etc from one corpus; the current
# schema's are PREPOSITION_LIKE/NOUN_LIKE/VERB_LIKE/etc from a
# different, replacement corpus -- both exactly 12 dimensions, so
# load_state_dict's shape check reported success while silently
# misapplying every discovered-column weight to the wrong real-world
# signal). training.save_checkpoint()/load_checkpoint() compare this
# full list, not just TOTAL_CONSTRAINTS, before ever trusting a shape
# match.
CONSTRAINT_COLUMN_NAMES = BYTE_IDENTITY_COLUMN_NAMES + LANGUAGE_MECHANICS_NAMES


def migrate_checkpoint_state_dict(old_state_dict, new_model):
    """Real, explicit weight migration for a checkpoint saved under an
    OLDER TOTAL_CONSTRAINTS (fewer discovered dimensions) than the model
    currently being built. Every weight the checkpoint actually has is
    kept EXACTLY as it was -- this never touches an already-learned
    value, it only decides what goes in the columns that didn't exist
    yet when the checkpoint was saved.

    New columns get 0.0, not a random draw. A zero weight contributes
    nothing to the forward pass until real training teaches it
    something -- the honest "not learned yet" state, the same
    discipline as every other feature in this file that would rather
    report 0 than fabricate a plausible-looking number. nn.Linear's own
    random init would silently inject a meaningless starting signal for
    a column that means something real (a named constraint) -- worse
    than admitting it hasn't been learned.

    Only mdbe.proj.weight and each Block's constraint_proj.weight can
    ever actually need this -- they're the only parameters whose shape
    depends on TOTAL_CONSTRAINTS (see MDBE.__init__ / Block.__init__).
    Everything else (the embedding table, every SSM parameter, the
    head) must match exactly; if it doesn't, that's a real d_model/
    n_layers/d_state change, not a dimension-count change, and this
    function refuses to guess at that -- it raises, same as a normal
    load would, rather than silently doing something wrong.

    Returns (migrated_state_dict, report) where report is a list of
    (key, old_width, new_columns_added) for every key actually migrated
    -- empty if every shape already matched (nothing to migrate)."""
    new_state = new_model.state_dict()
    migrated = {}
    report = []
    for key, new_tensor in new_state.items():
        if key not in old_state_dict:
            raise KeyError(f"Checkpoint is missing {key!r} entirely -- not a dimension-count "
                            "difference, refusing to guess.")
        old_tensor = old_state_dict[key]
        if old_tensor.shape == new_tensor.shape:
            migrated[key] = old_tensor
            continue

        is_constraint_input = key.endswith("mdbe.proj.weight") or key.endswith("constraint_proj.weight")
        same_rows = len(old_tensor.shape) == 2 and old_tensor.shape[0] == new_tensor.shape[0]
        grew_only_in_columns = same_rows and old_tensor.shape[1] < new_tensor.shape[1]
        if not (is_constraint_input and grew_only_in_columns):
            raise RuntimeError(
                f"Cannot migrate {key!r}: shape went from {tuple(old_tensor.shape)} to "
                f"{tuple(new_tensor.shape)}. A dimension-count change only ever adds columns "
                "to mdbe.proj/constraint_proj -- this looks like d_model/n_layers/d_state "
                "changed instead, which needs a real fresh model, not a migration."
            )

        merged = torch.zeros_like(new_tensor)
        old_width = old_tensor.shape[1]
        merged[:, :old_width] = old_tensor
        # merged[:, old_width:] stays 0.0 -- real, honest "not learned yet"
        migrated[key] = merged
        report.append((key, old_width, new_tensor.shape[1] - old_width))

    return migrated, report


def export_mdbe_table(model: "Carbide", filepath: str) -> None:
    """Writes the current learned embedding + constraint flags for all 256
    bytes to `filepath`. Shared by menu_save() (one-shot, on demand) and the
    periodic training-time snapshots (task1 refinement #4) so both produce
    the exact same format and can be diffed against each other directly."""
    rows = []
    for b in range(256):
        byte_tensor = torch.tensor([[b]])
        learned = model.mdbe.base(byte_tensor)[0, 0].tolist()
        live = all_constraints(byte_tensor)[0, 0].tolist()
        char_repr = repr(chr(b)) if 32 <= b < 127 else ""
        rows.append([b, char_repr] + [f"{v:.4f}" for v in learned]
                    + [f"{v:.0f}" for v in live])
    header = (["byte", "character"]
              + [f"cell_{i}" for i in range(model.mdbe.base.embedding_dim)]
              + ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"]
              + LANGUAGE_MECHANICS_NAMES)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(header)
        csv.writer(fh).writerows(rows)


class MDBE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.base = nn.Embedding(256, d_model)
        self.proj = nn.Linear(d_model + TOTAL_CONSTRAINTS, d_model)

    def forward(self, bytes_seq, live=True):
        learned = self.base(bytes_seq)
        cols = all_constraints(bytes_seq) if live \
               else torch.zeros(*bytes_seq.shape, TOTAL_CONSTRAINTS)
        return self.proj(torch.cat([learned, cols], dim=-1))


class SelectiveSSM(nn.Module):
    """Selective SSM with a chunked-parallel scan (task1 Part 1).

    The recurrence h_t = a_bar_t*h_{t-1} + b_bar_t is unchanged — this is
    still the same sequential state-space scan, just computed with fewer
    Python-level loop iterations. A naive fully-parallel formulation would
    use a cumulative-product/division trick, but a_bar = exp(delta*A) with
    A = -exp(A_log) (A_log ~ Uniform(1,16)) routinely underflows to exactly
    0 in float32, which would make a division-based scan emit NaN. Instead:
    split T into chunks of `chunk_size`; within a chunk, do a real
    sequential loop (safe, no division) but batched across ALL chunks at
    once (folded into the batch dim); carry state across chunks with a
    second, much shorter sequential loop (num_chunks steps). Net effect:
    ~T Python iterations become ~chunk_size + T/chunk_size (e.g. 128 -> ~24
    at the default chunk_size=16). See gradcheck-style verification in
    tests/test_scan.py comparing this directly against the plain loop.
    """

    def __init__(self, d_model, d_state=16, chunk_size=16):
        super().__init__()
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.A_log = nn.Parameter(torch.log(
            torch.exp(torch.empty(d_model, d_state).uniform_(1, 16))))
        self.W_delta = nn.Linear(d_model, d_model)
        self.W_B = nn.Linear(d_model, d_state)
        self.W_C = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        Bsz, T, d = x.shape
        delta = F.softplus(self.W_delta(x))                       # (B,T,d)
        A = -torch.exp(self.A_log)                                 # (d,ds)
        a_bar = torch.exp(delta.unsqueeze(-1) * A)                  # (B,T,d,ds)
        b_bar = delta.unsqueeze(-1) * self.W_B(x).unsqueeze(2)       # (B,T,d,ds)

        h = self._chunked_scan(a_bar, b_bar)                        # (B,T,d,ds)

        Cx = self.W_C(x)                                            # (B,T,ds) — position-wise, same as per-t
        y = (Cx.unsqueeze(2) * h).sum(-1)                            # (B,T,d)
        return y + self.D * x

    def _chunked_scan(self, a_bar, b_bar):
        Bsz, T, d, ds = a_bar.shape
        C = min(self.chunk_size, T)
        num_chunks = math.ceil(T / C)
        Tp = num_chunks * C
        pad = Tp - T
        if pad > 0:
            a_bar = torch.cat([a_bar, a_bar.new_ones(Bsz, pad, d, ds)], dim=1)
            b_bar = torch.cat([b_bar, b_bar.new_zeros(Bsz, pad, d, ds)], dim=1)

        a_c = a_bar.reshape(Bsz * num_chunks, C, d, ds)
        b_c = b_bar.reshape(Bsz * num_chunks, C, d, ds)

        # Within-chunk recurrence: real sequential loop, but batched across
        # every chunk simultaneously — this is the piece that used to be a
        # T-iteration loop and is now only a C-iteration one.
        h_local = a_c.new_zeros(Bsz * num_chunks, d, ds)
        h_locals = []
        for t in range(C):
            h_local = a_c[:, t] * h_local + b_c[:, t]
            h_locals.append(h_local)
        h_local = torch.stack(h_locals, dim=1)                       # (B*nc,C,d,ds)
        P = torch.cumprod(a_c, dim=1)                                # (B*nc,C,d,ds) — pure product, safe even if it hits 0

        h_local = h_local.reshape(Bsz, num_chunks, C, d, ds)
        P = P.reshape(Bsz, num_chunks, C, d, ds)

        # Carry state across chunks — sequential, but only num_chunks steps.
        carry = a_bar.new_zeros(Bsz, d, ds)
        carries = [carry]
        for i in range(num_chunks - 1):
            carry = P[:, i, -1] * carries[-1] + h_local[:, i, -1]
            carries.append(carry)
        carry_stack = torch.stack(carries, dim=1)                    # (B,nc,d,ds)

        h_full = P * carry_stack.unsqueeze(2) + h_local              # (B,nc,C,d,ds)
        return h_full.reshape(Bsz, Tp, d, ds)[:, :T]


class Block(nn.Module):
    """task1 refinement #1: the constraint flags are re-injected here via
    a small additive projection, not just consumed once by MDBE.proj at the
    input. MDBE.proj immediately blends learned+defined columns together, so
    without this the flags' traceability doesn't survive past layer 1 — the
    whole point of hand-coding them was to keep a legible "is this byte a
    digit/space/etc." signal available, and that only holds if later layers
    can still see it directly, not just a linearly-mixed trace of it. Now
    carries all 18 (6 byte-identity + 12 language-mechanics), not just 6."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.constraint_proj = nn.Linear(TOTAL_CONSTRAINTS, d_model, bias=False)

    def forward(self, x, constraint_cols):
        h = self.norm(x) + self.constraint_proj(constraint_cols)
        return x + self.ssm(h)


class LocalByteConv(nn.Module):
    """task1 refinement #2: a short CAUSAL depthwise 1D conv over a few
    neighboring bytes, right after MDBE and before the SSM stack, so the
    network gets a head start on merging bytes into word-like chunks
    instead of having to learn that unassisted in its first 1-2 SSM layers
    — the same job a BPE tokenizer would otherwise do for free.

    Causal is not optional: this is an autoregressive next-byte predictor,
    so padding is LEFT-only (kernel only ever sees the current position and
    the `kernel_size-1` positions before it). A non-causal conv would leak
    future bytes into predicting the current one — inflated training loss,
    broken generation, since generate() only ever has past bytes to give it.
    Depthwise (groups=d_model): local temporal mixing per channel, cheap;
    channel mixing already happens elsewhere via the Linear layers.
    """

    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)

    def forward(self, x):
        xt = x.transpose(1, 2)                            # (B,d_model,T)
        xt = F.pad(xt, (self.kernel_size - 1, 0))          # left-pad only — causal
        return self.conv(xt).transpose(1, 2)               # (B,T,d_model)


MODES = ("full", "no_constraints", "plain_embedding")


class Carbide(nn.Module):
    def __init__(self, d_model=64, n_layers=2, d_state=16):
        super().__init__()
        self.mdbe = MDBE(d_model)
        self.local_conv = LocalByteConv(d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_state=d_state) for _ in range(n_layers)])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)

    def forward(self, bytes_seq, mode="full"):
        live = (mode == "full")
        if mode == "plain_embedding":
            x = self.mdbe.base(bytes_seq)
        else:
            x = self.mdbe(bytes_seq, live=live)

        x = x + self.local_conv(x)

        constraint_cols = all_constraints(bytes_seq) if live \
            else torch.zeros(*bytes_seq.shape, TOTAL_CONSTRAINTS, device=bytes_seq.device)
        for block in self.blocks:
            x = block(x, constraint_cols)

        return self.head(self.head_norm(x))
