# Carbide

Carbide is a small program that reads text the way a person does it in stages: **letter → word → sentence**. It writes each stage in a notebook with names you can open and check.

You do not need to be a programmer to understand the idea. Start here:

- Everyday tour: [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
- How to open the notebooks and watch a guess: [TRACEABILITY.md](TRACEABILITY.md)
- Named columns in full: [MDBE_MANIFEST.md](MDBE_MANIFEST.md)
- Lab diary: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

## What it is, in one screen

| Stage | Notebook | What it writes | What you can open |
|---|---|---|---|
| Layer 1 | Letter book | The real letter ID (the same hex the computer already uses), plus six yes/no facts | `mdbe_table.csv` |
| Layer 2 | Word book | The word those letters just spelled, plus grammar columns (article, noun, verb, …) | word table export |
| Layer 3 | Sentence book | A short running summary of the words so far, plus sentence facts (statement, question, command, negated, …) | sentence table export |

A guessing engine then tries to predict the **next letter**. The engine is a *selective state-space model* (same family as “Mamba”). That is a technical name for a reader with a working memory that updates as each letter arrives. **It is not the notebooks.** The notebooks stay readable.

## Design rules (the team’s contract)

1. The letter’s ID is the real letter the computer already uses — not a made-up locker number.
2. Letter facts and grammar facts stay in **separate columns**. We never hide “verb” inside “is this a digit?” Grammar sits **beside** the six hex facts, never inside them.
3. All three layers can be printed with names. If you cannot dump them, the design has slipped.
4. Values from the notebooks are mixed for the engine. Names stay on a second copy of the page so a person can still read them.
5. This copy of the program runs on an ordinary home CPU. It does not need a data-center graphics card.

## Soft scores and learned cells

Two different numbers live in the word and sentence books:

- A **rule score** is a hint we wrote by hand (“`the` looks like an article, about 0.95”).
- A **learned cell** is the model’s own number for that same row and column after training.

Both stay tied to a named row and a named column, so both are still traceable. The rule score is the starter hint. The learned cell is the model’s opinion. Opening the model means reading those named cells, not decoding a pile of mystery bits.

The six Layer-1 facts stay hard yes/no. They decode the hex. They are not guesses.

## What this is not

It is not ChatGPT. It is not a finished product that answers questions. It is a research program for studying a *labeled* way to read text, small enough to run and inspect at home.

Nobody else appears to have shipped this exact stack: real hex rows + named grammar beside those rows + a word table + a sentence slot + a dumpable strip that never goes into the SSM as names. Pieces of it exist (byte-level Mamba, grammar features, interpretability tools). The whole recipe is Carbide’s.

## For people who will run the code

From this folder:

```
python3 -m carbide_modules
```

That opens a menu: train, generate, save tables, compare versions.

- Training books: `carbide_training_dataset.txt`
- Letter table export: `mdbe_table.csv`
- Combined labeled table: `full_table.csv`
- Weight-listen table: `weight_trace.csv`

Default size (menu model): width 256, 4 layers. Smaller test runs are fine while you check the notebooks.

## Project layout

- `carbide_modules/` — the program
  - `layers.py` — Layer 1 / 2 / 3 stack and the named strip
  - `mdbe.py` — six hex flags, grammar columns, selective SSM
  - `trace_weights.py` — names for otherwise-opaque learned cells
  - `discover_dimension.py` — proposed extra columns, fixed after training
- `tests/` — checks that the reader and the notebooks still agree with themselves
- `_archive/` and `carbide_module_cplusplus/` — older or side work, not the active program

## Status

Layer 1, Layer 2, and Layer 3 are live on this branch. Grammar sits beside the six hex flags. Training on the included books does reduce error (a short 800-step run on the 8 MB book dropped from about 2.88 to about 1.59). That is a health check, not a claim that the program understands language.

Adding the word book (Layer 1 + Layer 2) helped the guessing game more than the current sentence pack. Layer 3 is still kept because the labeled sentence facts are part of the design, not because they already win on loss.
