# MDBE Column Manifest

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

`MDBE.proj` (a `Linear(d_model+6, d_model)`) then projects this
`(d_model+6)`-wide concatenation back down to `d_model`. **Important:** this
projection is a dense matmul — it mixes the 6 defined columns together with
the learned ones. There is no surviving "column 3 is always is_digit" in the
`d_model`-wide representation that comes out of `MDBE.forward`. The
traceable point is the *input* to `.proj`, not its output.

task1 refinement #1 (re-injecting the constraint flags at every SSM block
via `Block.constraint_proj`) exists precisely because of this: without it,
the six raw flags are visible to the network exactly once, at the very
first projection, and get progressively blended away by every layer after
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

## Checking whether they're actually helping

`mdbe_table.csv` (Save Outputs) and the files in `mdbe_snapshots/`
(refinement #4 — periodic snapshots during training, not just a one-shot
export) both export the full per-byte state: the `d_model` learned
embedding cells plus these 6 flags, side by side, for all 256 bytes.
Comparing snapshots across training steps is how to check whether
same-class bytes (e.g. all digit bytes) actually drift closer together in
the learned embedding space over training, rather than assuming they do.

The ablation study (`Run ablation study` in the menu, refinement #3) trains
three variants — `full` (both learned embedding and constraint flags),
`no_constraints` (flags zeroed, everything else identical), `plain_embedding`
(flags AND the projection skipped entirely, raw embedding only) — for real
evidence on whether the constraint columns help loss/convergence, rather
than assuming they do.

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
