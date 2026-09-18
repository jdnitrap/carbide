# Carbide

Carbide is a small program that reads text the way a person does it in stages: **letter → word → sentence**. It writes each stage in a notebook with names you can open and check.

You do not need to be a programmer to understand the idea. The short tour is [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## What it is, in one screen

| Stage | Notebook | What it writes |
|---|---|---|
| Layer 1 | Letter book | The real letter ID, plus six yes/no facts (letter? number? capital? punctuation? space? start of a multi-byte character?) |
| Layer 2 | Word book | The word those letters just spelled, plus grammar columns (article, noun, verb, …) |
| Layer 3 | Sentence book | A short summary of the words so far, plus sentence facts (statement, question, command, negated, …) |

A guessing engine then tries to predict the **next letter**. The engine is a *selective state-space model* (same family as “Mamba”). That is a technical name for a reader with a working memory that updates as each letter arrives. It is **not** the notebooks. The notebooks stay readable.

## Design rules (the team’s contract)

1. The letter’s ID is the real letter the computer already uses — not a made-up locker number.
2. Letter facts and grammar facts stay in **separate columns**. We never hide “verb” inside “is this a digit?”
3. All three layers can be printed with names. If you cannot dump them, the design has slipped.
4. This copy of the program runs on an ordinary home CPU. It does not need a data-center graphics card.

## What this is not

It is not ChatGPT. It is not a finished product that answers questions. It is a research program for studying a *labeled* way to read text, small enough to run and inspect at home.

## For people who will run the code

From this folder:

```
python3 -m carbide_modules
```

That opens a menu: train, generate, save tables, compare versions.

- Training books: `carbide_training_dataset.txt`
- Letter table export: `mdbe_table.csv`
- Column-by-column technical list: [MDBE_MANIFEST.md](MDBE_MANIFEST.md)
- History of experiments: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

Default size (menu model): width 256, 4 layers. Smaller test runs are fine while you check the notebooks.

## Project layout

- `carbide_modules/` — the program
- `tests/` — checks that the reader and the notebooks still agree with themselves
- `HOW_IT_WORKS.md` — this idea in everyday language
- `MDBE_MANIFEST.md` — the named columns in full
- `_archive/` and `carbide_module_cplusplus/` — older or side work, not the active program

## Status

The letter book and the “facts sit beside grammar” rule are the documented baseline on this branch. The word book and sentence book are the intended next notebooks in the same design. Training on the included books does reduce error. That is a health check, not a claim that the program understands language.
