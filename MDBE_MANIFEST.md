# MDBE Column Manifest

**Start here if you do not write programs:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
How to dump the notebooks: [TRACEABILITY.md](TRACEABILITY.md).
This file is the technical notebook: which columns exist and what they mean.

**Official layout: grammar BESIDE the six hex flags, never inside them.**

The row ID is the real byte (`0x00`–`0xFF`). The six flags decode that
hex. Language-mechanics columns are extra named dimensions on the same
row-for-this-position. An experiment that projected grammar into the
six axes ("inside") is not the default: it can shave a little loss but
it destroys the meaning of `is_digit`.

## The three notebooks (implemented)

| Layer | Code | Rows | Named columns | Learned cells |
|---|---|---|---|---|
| 1 letter book | `mdbe_constraints()` + `LayerStack.byte` | 256 hex IDs | 6 hard flags | `nn.Embedding(256, d_model)` |
| 2 word book | `spans_from_bytes` + grammar block + `LayerStack.word` | top-N words + UNK | language-mechanics names | `nn.Embedding(WORD_VOCAB_SIZE, d_model)` |
| 3 sentence book | `causal_pack` + `sentence_columns` | packed slot per open sentence | DECLARATIVE, INTERROGATIVE, IMPERATIVE, NEGATED, HAS_OBJECT, MULTI_CLAUSE | pack projection + `l3_proj` |

All three stay named and dumpable via `strips()` / `named_values()`.
Values are mixed for the guessing engine; names stay on Copy A.

Entry point: `from carbide_modules.layers import Carbide, LayerStack, strips`.

## Where these columns live

Each byte (0-255) is represented, just before the mixer, as named strips
plus learned rows:

| columns | count | source | learned? |
|---|---|---|---|
| `base(byte)` / `LayerStack.byte` | `d_model` | `nn.Embedding(256, d_model)` | yes — trained, no defined meaning per-dimension until `trace_weights.py` labels it |
| constraint flags | 6 | `mdbe_constraints()` | no — deterministic function of the byte, never touched by the optimizer |
| language mechanics | `NUM_LANGUAGE_MECHANICS` | `language_mechanics_constraints()` | rule scores by default; **beside** the six flags |
| word row | `d_model` | `LayerStack.word` | yes — trained per word id |
| sentence dims | 6 | `sentence_columns()` | rule scores from the current sentence pack |
| pack slot | `d_model` | running mean of Layer-2 vectors in the open sentence | yes — the pack is built from learned word rows |

`MDBE.proj` / `LayerStack.mix` then project the concatenated values
back down to `d_model`. That is Copy B. **Important:** this projection
is a dense matmul — there is no surviving "column 3 is always is_digit"
in the vector the SSM sees. The traceable point is the *input* to the
projection (`strips()`), not its output.

Re-injecting the full beside strip at every SSM block
(`Block.constraint_proj`, width = 6 + grammar + 6 sentence dims)
exists precisely because of this: without it, the named values are
visible to the network once and get blended away by every layer after
that.

## The six hand-defined columns

Computed directly from the raw byte value `b` (0-255), matching
`mdbe_constraints()` exactly:

| # | name | true when |
|---|---|---|
| 0 | `is_alpha` | `b` in `A-Z` (65-90) or `a-z` (97-122) |
| 1 | `is_digit` | `b` in `0-9` (48-57) |
| 2 | `is_upper` | `b` in `A-Z` (65-90) |
| 3 | `is_punct` | `b` in `!-/` (33-47), `:-@` (58-64), `` [-` `` (91-96), or `{-~` (123-126) |
| 4 | `is_space` | `b` is space (32), tab (9), newline (10), or carriage return (13) |
| 5 | `utf8_lead` | `b` is a UTF-8 lead byte: `< 0x80` (single-byte ASCII) or `>= 0xC0` (start of a 2/3/4-byte sequence) — false for continuation bytes (`0x80`-`0xBF`) |

Each is a 0/1 float, always computed live and correct (never learned, never
wrong) — that's the entire point of hand-defining them rather than hoping
the model discovers "0x30-0x39 are digits" from data on its own.

## Grammar dimensions (soft scores, not 0/1)

These are the original MDBE idea extended past ASCII: hand-coded structure
about the *word / clause*, concatenated **beside** the 6 facts.

They are **not** facts. A closed-class hit (`the` → ARTICLE) is almost
certain; a suffix guess and a clause role are not; a discovered `*_LIKE`
cluster is a blob from this corpus. So the writer stores a confidence
instead of 1.0, and that group's `:NONE` column gets `1 - confidence`.

| kind | typical confidence |
|---|---|
| 6 byte flags | 1.0 (still hard bits) |
| closed-class POS / pronoun grammar | 0.95 / 0.90 |
| morphology suffix | 0.70 (0.85 if irregular-list hit) |
| open-class POS lists | 0.45 |
| syntax / voice / mood / aspect guesses | 0.40 |
| discovered `*_LIKE` clusters | 0.20 |

The word embedding on the same row is a separate learned cell. After
training you can read both: the rule hint and the model’s opinion.

## Sentence dimensions (Layer 3)

| name | rough meaning |
|---|---|
| DECLARATIVE | leftover mass after question/command hints |
| INTERROGATIVE | question / interrogative grammar in this sentence so far |
| IMPERATIVE | command hint |
| NEGATED | negation mechanic seen |
| HAS_OBJECT | object role seen |
| MULTI_CLAUSE | crude length flag for a long open sentence |

These are packed from Layer-2 values with a causal running mean so a
sentence is never stored as one giant unique title.

## Checking whether they're actually helping

`mdbe_table.csv` (Save Outputs) and the files in `mdbe_snapshots/`
both export the per-byte state. Word and sentence dumps come from
`export_word_table` / `export_sentence_table` in `layers.py`.
`weight_trace.csv` labels otherwise-opaque hidden units by the named
column they weight most.

Ablation modes:

- `full` / `l1_l2_l3` — hex + word book + sentence pack
- `l1_l2` — hex + word book, no sentence pack
- `flags_only` — 6 facts, grammar and L3 zeroed
- `no_constraints` — learned byte row only
- `plain_embedding` — raw embedding, projection skipped

There is **no** `inside_6` default. Grammar is not projected into the six
hex-decode axes.

## What the 2026-09-11 flag ablation showed

3 seeds, 1500 steps/variant, d_model=256 / n_layers=4 / d_state=32, real corpus:

| mode | seed=42 | seed=123 | seed=7 |
|---|---|---|---|
| `full` | 0.4227 | 0.4552 | 0.4614 |
| `no_constraints` | 0.4279 | 0.4654 | 0.4888 |
| `plain_embedding` | 0.4518 | 0.5516 | 0.4948 |

Every seed: `full` < `no_constraints` < `plain_embedding`.
Do not trust a one-seed ablation.

## What the Layer 1/2/3 ablation showed (2026-09-18)

Smaller model (d_model=128, 2 layers, seq_len=128, 500 steps, 3 seeds),
mean of last 50 steps:

| mode | seed=42 | seed=123 | seed=7 | mean |
|---|---|---|---|---|
| beside (byte + flags + grammar on bytes) | 2.1361 | 2.1705 | 2.0948 | 2.1338 |
| l1_l2 (hex + word-row ingest) | 2.0136 | 2.0720 | 2.0418 | **2.0425** |
| l1_l2_l3 (+ sentence pack) | 2.0313 | 2.0636 | 2.0447 | 2.0465 |

> **Superseded (2026-09-21):** the table above was measured with a Layer 2 lookahead leak (now fixed).
> Corrected held-out means: beside 1.7316, l1_l2 1.7526, l1_l2_l3 1.7529, l1 1.7721.
> See the correction in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Layer 1 + Layer 2 is the clear win on this budget. Layer 3 is a near-tie.
It stays in the default stack because the labeled sentence book is part
of the design, not because the current pack already beats Layer 2 on loss.

A later 800-step training run on `carbide_training_dataset.txt`
(8,021,709 bytes, d_model=128, 2 layers) moved loss from 2.883 → 1.593.
Health check only.
