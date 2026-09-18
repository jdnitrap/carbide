# MDBE Column Manifest

**Start here if you do not write programs:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
This file is the technical notebook: which columns exist and what they mean.

**Official layout: grammar BESIDE the six hex flags, never inside them.**

The row ID is the real byte (`0x00`–`0xFF`). The six flags decode that
hex. Language-mechanics columns are extra named dimensions on the same
row-for-this-position. An experiment that projected grammar into the
six axes ("inside") is not the default: it can shave a little loss but
it destroys the meaning of `is_digit`.

The intended three notebooks:

- Layer 1 — letter book (this file’s six flags)
- Layer 2 — word book (grammar columns, later a real word row)
- Layer 3 — sentence book (sentence facts on a packed slot)

All three stay named and dumpable. Values may be mixed for the guessing
engine; names stay on the strip.


task1 refinement #5. Documents exactly which of MDBE's per-byte columns are
learned vs. hand-defined, and what each hand-defined one encodes — this used
to only be readable by reading `mdbe_constraints()` in `carbide_modules/mdbe.py`
directly.

## Where these columns live

Each byte (0-255) is represented, just before `MDBE.proj`, as the
concatenation of two blocks:

| columns | count | source | learned? |
|---|---|---|---|
| `base(byte)` | `d_model` (256 by default) | `nn.Embedding(256, d_model)` | yes — trained by gradient descent, no defined meaning per-dimension |
| constraint flags | 6 | `mdbe_constraints()` | no — deterministic function of the byte value, recomputed every forward pass, never touched by the optimizer |
| language mechanics | `NUM_LANGUAGE_MECHANICS` | `language_mechanics_constraints()` | no — rules on the current word/clause; **beside** the six flags, never mixed into them |

`MDBE.proj` (a `Linear(d_model + TOTAL_CONSTRAINTS, d_model)`) then
projects the learned row + **all** constraint columns (6 facts + grammar)
back down to `d_model`. That is the single combined embedding. **Important:**
this projection is a dense matmul — there is no surviving "column 3 is
always is_digit" in the `d_model`-wide vector that comes out of
`MDBE.forward`. The traceable point is the *input* to `.proj`, not its output.

task1 refinement #1 (re-injecting the constraint flags at every SSM block
via `Block.constraint_proj`) exists precisely because of this: without it,
the six raw flags are visible to the network exactly once, at the very
first projection, and get progressively blended away by every layer after
that. Re-injection uses the **full beside strip** (6 + grammar).

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

## Grammar dimensions (soft, not 0/1)

These are the original MDBE idea extended past ASCII: hand-coded structure
about the *word / clause*, concatenated with the 6 facts and the learned
row, then projected into one vector.

They are **not** facts. A closed-class hit (`the` → ARTICLE) is almost
certain; a suffix guess and a clause role are not; a discovered `*_LIKE`
cluster is a blob from this corpus. So the writer now stores a confidence
instead of 1.0, and that group's `:NONE` column gets `1 - confidence`.

| kind | typical confidence |
|---|---|
| 6 byte flags | 1.0 (still hard bits) |
| closed-class POS / pronoun grammar | 0.95 / 0.90 |
| morphology suffix | 0.70 (0.85 if irregular-list hit) |
| open-class POS lists | 0.45 |
| syntax / voice / mood / aspect guesses | 0.40 |
| discovered `*_LIKE` clusters | 0.20 |

Checkpoint shapes are unchanged (`TOTAL_CONSTRAINTS` is the same). Only
the numbers written into the grammar columns changed.

Ablation modes now include `flags_only`: live 6 facts, grammar block
zeroed. That is the test of "did growing MDBE past the 6 flags help?"
The 2026-09-11 numbers below only compare the original 6 flags against
no flags. They do not cover this grammar stack.

## Checking whether they're actually helping

`mdbe_table.csv` (Save Outputs) and the files in `mdbe_snapshots/`
(refinement #4 — periodic snapshots during training, not just a one-shot
export) both export the full per-byte state: the `d_model` learned
embedding cells plus these 6 flags, side by side, for all 256 bytes.
Comparing snapshots across training steps is how to check whether
same-class bytes (e.g. all digit bytes) actually drift closer together in
the learned embedding space over training, rather than assuming they do.

The ablation study (`Run ablation study` in the menu) trains four
variants — `full` (6 facts + soft grammar **beside**), `flags_only` (6 facts,
grammar zeroed), `no_constraints` (all constraint columns zeroed),
`plain_embedding` (projection skipped, raw embedding only).
`flags_only` vs `full` is the grammar-expansion test; `flags_only` vs
`no_constraints` retests the original 6 facts.

There is **no** `inside_6` default. Grammar is not projected into the six
hex-decode axes.

## What the ablation actually showed (2026-09-11, 3 seeds, 1500 steps/variant each, d_model=256/n_layers=4/d_state=32, real 5MB corpus)

The first run (seed=42) alone looked equivocal — `full` and `no_constraints`
were within ~1% of each other, which read as "the live constraint values
aren't clearly helping." Two more independent seeds settled it:

| mode | seed=42 | seed=123 | seed=7 |
|---|---|---|---|
| `full` | 0.4227 | 0.4552 | 0.4614 |
| `no_constraints` | 0.4279 | 0.4654 | 0.4888 |
| `plain_embedding` | 0.4518 | 0.5516 | 0.4948 |

**Every single seed shows the same ordering: `full` < `no_constraints` <
`plain_embedding`.** `full` beat `no_constraints` in 3/3 seeds, and the gap
between them *grew* in the two additional seeds (0.010 and 0.027) rather
than shrinking toward noise — the opposite of what you'd expect if the
first result were a fluke. This is a real, repeatable effect, not
measurement noise: both the hand-coded constraint flags AND the extra
projection layer they ride in on contribute — `full` > `no_constraints` >
`plain_embedding`, consistently. Three seeds is still not exhaustive, but
it's a genuinely confirmed direction, not a coin flip.

**Lesson in the process, not just the result:** the single-seed run alone
was actively misleading — it understated a real, consistent effect. Don't
trust a one-seed ablation for a design decision; three independent seeds
turned "looks equivocal" into "confirmed, and the effect is larger than
the first run suggested."
