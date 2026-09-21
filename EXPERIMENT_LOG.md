# Experiment / Verification Log

**In everyday language:** this page is the lab diary. It is a list of
bugs we found, checks we ran, and what we decided to keep or set aside.
You do not need it to understand Carbide. For that, start at
[HOW_IT_WORKS.md](HOW_IT_WORKS.md). To open the notebooks, see
[TRACEABILITY.md](TRACEABILITY.md).

See also `MDBE_MANIFEST.md` for column lists and ablation tables.

## History (2026-09-11, from prior session records)

**Directory cleanup:** `carbide/` used to hold three parallel
implementations plus an undocumented TUI. Consolidated to one program;
old variants moved to `_archive/` (not deleted).

**Module split:** single-file `carbide.py` split into the
`carbide_modules/` package for editability. Required fixing a real
cross-module bug (bare `global` reassignment only works within one
file; cross-module writes now go through the owning module explicitly).
Verified via smoke test matching the original's exact param count
(52,608 at the old default scale).

**Concurrency bug found, then fixed:** `menu_hyperparams()` /
`menu_checkpoints()` / `menu_load_dataset()` could mutate shared
`model`/`opt`/`data` globals from the main-menu thread while
`background_training_thread` concurrently used them.
`train_state.training_lock` only guarded `train_state`'s own fields,
not these globals. Fixed by gating the dangerous menu paths behind
`train_state.training_active` instead of adding more locks. Verified
with a script mocking `input()` asserting model/opt/data identity is
unchanged while training is active.

**A real race in background training:** display loops could check
`training_active` before the background thread set it, misreporting
"Training complete at step 0" for training that was still running
mid-run. Fixed.

**Other bugs fixed (verified against the real CLI):** asymmetric
checkpoint suffix handling, missing numeric input validation on menus,
stale model/opt after hyperparameter shape change (now nulled to force
reinit), unsafe `torch.load` (now `weights_only=True`), checkpoint LR
not restored on load, hardcoded 128-byte context window in `generate()`
(now uses `config.seq_len`), orphaned non-daemon background thread on
exit.

**`d_state` forwarding bug (found during `new_carbide_architect` work,
fixed here too):** `Block.__init__` never accepted/forwarded `d_state`
to `SelectiveSSM` — every `d_state=32` setting silently ran at the
hardcoded default 16. This means the "522,112 params" figure recorded
in early task1 work was measured under the bug; real d_state=32 default
is 571,392 params. Relative comparisons (ablation ordering, 3-seed
confirmation) are unaffected since the bug applied uniformly.

**CPU scan benchmark (motivated the chunked-scan fix):** the original
sequential-loop scan scaled 52,608 → 2,261,952 params (43x) but dropped
throughput 6,114 → 108 tok/sec (~57x slower) — Python dispatch overhead
compounds across layers × timesteps, not FLOPs. After the chunked
parallel scan: throughput scales ~linearly with params (43x params →
~44x time), a 2.8x–3.6x speedup depending on config.

**C++ rewrite — built, benchmarked, deliberately set aside.** Full
hand-written reverse-mode autodiff tensor engine, zero external
dependencies (not a translation of the Python version). Every primitive
gradient-checked; full CLI concurrency-guard and checkpoint round-trip
verified via scripted runs. Benchmark: Python/PyTorch ~5x faster
(39.7 vs 7.7 steps/sec) — PyTorch's backend is BLAS-optimized/fused/
multi-threaded, the hand-rolled engine isn't. One perf pass (same-shape
fast paths in add/mul, hoisting a per-timestep reshape) narrowed the
gap to ~1.9x slower. A C++ parallel-scan attempt was tried, verified
correct, but was actually slower (rebuilding full arrays via concat
each round costs more than the sequential loop saves) — reverted.
Decision: focus stays on `carbide_modules/` (Python).

## 2026-09-12 — Incremental caching prototype, then integrated for real

Built and correctness-verified in the sibling `carbide_speculative/`
repo first (`incremental_ssm.py`, matches full forward pass to
~1e-6/1e-7). Measured **~63.6x speedup** for the SSM's own plain
decoding (0.48ms cached vs 30.56ms uncached per byte) — bigger and
simpler than the speculative-decoding project that had originally
motivated looking at decode cost at all. Subsequently integrated for
real into this repo as `carbide_modules/incremental.py` +
`generation.py`.

## 2026-09-13 — Repo split-out, fresh test re-run

Split `carbide/` out as its own standalone repo. Re-ran the existing
test suite directly (no `pytest` installed on this machine, ran each
test script directly instead) to confirm nothing broke in the split:

- `tests/test_scan.py` — **PASS**, all T/chunk-size/dtype combinations,
  including float32 at training precision (max param diff ~3e-4,
  expected float32 noise level)
- `tests/test_incremental_decode.py` — **PASS**, all context lengths
  tested (L=1 to L=128), max abs diff ~7e-7 to ~9e-7; `generate()`
  output confirmed bit-identical to the old full-recompute
  implementation across 4 seeds
- `tests/test_sft.py` — **PASS**, all 3 checks (masked loss matches
  response-only reference; scrambling prompt-position targets changes
  nothing; loss decreases over repeated steps)

**Correction to a stale claim found in prior session notes:** those
notes stated incremental caching was still just a prototype living only
in `carbide_speculative/`, with `generation.py` "currently untouched."
That is no longer true as of this session — `incremental.py` is real,
integrated, wired into `generation.py`'s `generate()` function, and
covered by its own test. If picking up `carbide_speculative/` again,
its own recorded caveat applies: its measured 1.9x speculative-decoding
speedup was calculated against the *old, uncached* baseline and is now
known-stale — see that repo's README for the honest current status.

## 2026-09-18 — Three notebooks, beside layout, training health check

Wired Layer 1 (hex + 6 flags), Layer 2 (word table + grammar beside
the flags), Layer 3 (causal sentence pack + 6 sentence dims) in
`carbide_modules/layers.py`. Named strips stay dumpable. The SSM sees
only mixed numbers. Incremental decode carries `pack_state` so a
one-byte step matches a full forward.

3-seed layer ablation (d_model=128, 2 layers, seq_len=128, 500 steps):

- beside mean last-50 = 2.1338
- l1_l2 mean last-50 = **2.0425**
- l1_l2_l3 mean last-50 = 2.0465

Layer 1 + Layer 2 is the loss win. Layer 3 is a near-tie on this
budget and stays because the labeled sentence book is part of the
design.

800-step training on `carbide_training_dataset.txt` (8,021,709 bytes,
same small stack): first-50 loss 2.883 → last-50 1.593. Health check,
not a claim of language understanding.

Also locked: soft grammar scores (not 0/1) on Layer 2/3; hard 0/1 only
on the six hex flags; auto-discovered columns park beside existing
ones and can be marked fixed after training
(`discover_dimension.py`).

## 2026-09-21 — Learned cells, “watch it think,” docs pass

Docs were still calling Layer 2 and Layer 3 “the intended next
notebooks.” They are implemented. README, HOW_IT_WORKS,
MDBE_MANIFEST, and a new TRACEABILITY page now say so in everyday
language.

Learned cells vs rule scores: a learned number on a named row and
named column is the model’s opinion. That is more useful for “open
the brain” than a hand-written 0.95. It is still not a thought-movie
of the SSM. `trace_weights.py` labels hidden units by the named
column they listen to most. `weight_trace.csv` and
`embedding_dimension_names.json` are the dumps.

Prior-art check (deep dive, not a claim of a legal search):
byte-level selective SSMs exist (MambaByte). Hierarchical byte models
exist (SpaceByte, H-Net). Named linguistic features and probing
exist. We did not find another public stack that keeps (1) the real
hex as the row ID, (2) six hex-decode flags, (3) grammar *beside*
those flags, (4) a word table ingested from those hex spans, (5) a
fixed sentence slot with named sentence mechanics, and (6) a dumpable
named strip that is never what the SSM is fed. Pieces are common.
The whole recipe is Carbide’s as far as this check could see.

## 2026-09-21 — Correction: the "Layer 1 + Layer 2 is the loss win" result was a lookahead leak

**What was wrong.** `spans_from_bytes` in `layers.py` gave every letter of a
word the id of the *whole* word. So at byte `t` of "the", Layer 2 already
knew the word was "the" — the model was handed the answer it was being
trained to predict. In modes `l1_l2` / `l1_l2_l3` / `full`, changing only
*later* bytes changed the logits at *earlier* positions by about 1.0–1.5
(modes `l1`, `flags_only`, `no_constraints` changed by 0). The 2026-09-18
ablation below (beside 2.1338 / l1_l2 2.0425 / l1_l2_l3 2.0465) and the
800-step health check were measured with that leak, so they do not support
"Layer 1 + Layer 2 is the loss win". Two related faults: generation only has
the partly spelled word, so training and generation disagreed; and the
incremental decoder differed from a full forward pass by 0.2–0.9 on the
layered model (`test_incremental_decode.py` builds the old `mdbe.Carbide`, so
it never exercised the layer stack).

**Fix.** A word is now known at the delimiter byte that completes it and is
carried forward; letters of a word still being spelled see the previous
completed word. Word/sentence ids and Layer 3's running maxima are computed by
one streaming routine used by both the full forward pass and `incremental_step`.
`tests/test_causality.py` checks prefix causality in every mode, and
incremental == full past the 255-byte window; all of it fails on the old code.

**Re-measured** with the same setup as the old ablation (d_model=128, 2 layers,
seq_len=128, 500 steps, seeds 0/1/2; batch 16, lr 3e-3, d_state 16; identical
batches and initialisation per seed for every arm; `ablate_layers.py`). Mean
of the last-50-step training loss, with the per-seed spread:

| arm | code | params | last-50 train loss (mean, sd) | held-out loss (mean) |
|---|---|---|---|---|
| beside (old `mdbe.Carbide`, mode full) | fixed | 163,136 | 1.7695 (0.007) | 1.7316 |
| l1 (flags only) | fixed | 471,744 | 1.8114 (0.008) | 1.7721 |
| l1_l2 | fixed | 471,744 | 1.7845 (0.004) | 1.7526 |
| l1_l2_l3 | fixed | 471,744 | 1.7858 (0.002) | 1.7529 |
| l1_l2 | original (leaky) | 471,744 | 1.3167 (0.011) | 1.3085 |
| l1_l2_l3 | original (leaky) | 471,744 | 1.3312 (0.002) | 1.3184 |

Held-out = last 500 KB of the corpus, never trained on. The leaky models were
also scored with *causal* word ids (what generation sees): l1_l2 **3.693**,
l1_l2_l3 **3.540** held-out — far worse than the ~1.75 of models trained on the
fixed code. Nearly all of the old apparent gain was the leak (≈0.47 of train loss).

**What the corrected numbers say, at this budget.** Layer 2 gives a small,
consistent gain over flags only (l1 → l1_l2: 0.027 train / 0.020 held-out,
every seed). Layer 3 adds nothing measurable (l1_l2 vs l1_l2_l3 within noise).
The old `beside` layout is best on both measures, with about a third of the
parameters — so "Layer 1 + Layer 2 is the loss win" is not supported, and the
layered stack does not beat `beside` here. Caveats: one training budget
(500 steps, small model), one corpus, 3 seeds; the layered arms carry a
2048-row word embedding that may need more steps; `l1` carries the same unused
word rows.

**Also found.** `load_dataset(filepath)` never built the word vocab (only the
default-file route did) and `word_vocab()` cached an empty vocab forever if
called first — both left every word UNK. `export_word_table` read grammar at
each word's *first* letter, so every word showed `PART_OF_SPEECH:NONE`.
`tests/test_soft_constraints.py` needs `CONF_CLOSED_POS` and
`all_constraints(flags_only=)`, which were never committed to `mdbe.py`
(commit a2f87a7 added only the test). `cli.py` / `training.py` still import
`Carbide` from `mdbe.py`, so the menu's training does not use the layer stack.
