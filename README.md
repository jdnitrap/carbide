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

Grammar columns hold the rule's confidence (`the` is an article: 0.95; a suffix guess: 0.70; a
discovered cluster: 0.20) and each group's `:NONE` column holds one minus that. This costs a little
loss at the current scale (see the newest entry in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)), so it is a
switch: `set model.soft_grammar off`. A checkpoint remembers which mode it was trained in.

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

## The shell and the TUI

Besides the menu there is a command shell, and a terminal UI built on it (same layout as the
esgm-gru project): a status panel with an editable settings panel beside it, a log, and a command line.

```
python3 -m carbide_modules.shell --data carbide_training_dataset.txt     # plain commands
python3 -m carbide_modules.tui   --data carbide_training_dataset.txt     # needs: pip install textual
```

`help` lists the commands: `train`, `generate`, `preset`, `save`/`load`, `set`, and `graph ...`.
`set` shows and changes every knob (model size, learning rate, sampling controls, graph thresholds);
each one says where it is saved. A trained model, its settings and its graph carry over a restart.
In the TUI: F2 opens the settings panel (Enter edits a value or cycles a choice), F5 trains 100
steps, F8 stops training, F9 saves.

## Creative text

`generate` and the menu REPL take more than temperature and top-k: top-p, a repetition penalty, a
repeated-phrase block, a seed, and three presets (`calm`, `balanced`, `wild`). With everything at its
default the sampling is identical to before. Expect plausible words and style from a model this small,
not coherent long stories.

## Graph memory

A knowledge graph that sits beside Carbide as long-term memory (`carbide_modules/graphmem/`, SQLite,
built and grown automatically -- there are no commands to hand-edit it):

```
pip install nltk && python -m nltk.downloader wordnet         # once, for the dictionary
python -m carbide_modules.graphmem build-core --corpus carbide_training_dataset.txt
python -m carbide_modules.graphmem ingest --module law --file law.tsv --format triples   # dump domain data in
python -m carbide_modules.graphmem import-dimensions          # keep discovered dimensions across restarts
```

- **Dictionary:** WordNet for the words in your corpus (part of speech, semantic class, is-a, synonyms,
  antonyms, part-of) plus Carbide's own function-word lists, which WordNet lacks.
- **Modules:** law, health, finance, math... are separate modules you dump data into (`triples`,
  `glossary`, or raw `text`, which finds the terms specific to that domain). A module can be turned off.
- **Growth without retraining:** node ids only grow and are never reused, dictionary edges can never be
  overwritten, capacity grows in blocks, thresholds retune themselves inside fixed bounds, and every
  automatic change is written to an audit log. Discovered dimensions keep a permanent column slot, so a
  new one never changes the model's input width.
- **Carbide reads it** through a compiled table, in the `graph` model kind (`set model.kind graph`).
  `graph teach` lets Carbide propose grammar for words the dictionary lacks; it is applied only if a
  probe on Carbide's hidden states beats the majority-class baseline on words it was not trained on.
- Whether the graph-fed model actually beats the plain one is measured, not assumed: see the newest
  entry in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Growing over time

- **Depth growth (`set train.auto_grow on`, off by default).** When loss plateaus, `train` adds one block
  that starts as an exact no-op, trains briefly, and keeps it only if held-out loss (a reserved tail that
  training never sees) improves; otherwise it rolls back exactly. Every attempt is in `growth_log.jsonl`.
- **Facts in front of the prompt.** `graph facts <words>` and `generate +facts <prompt>` put the graph's
  definitions and domain facts, as plain sentences, before the prompt; `graph fact-corpus in.txt out.txt`
  writes a training text in that format. A first test at small scale (1,200 steps, 96-wide) showed Carbide did
  **not** learn to use the facts (correct facts scored the same as another sentence's); see the newest entry in
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The compiled graph features, by contrast, do help a little.
- **A local model as teacher.** `teacher <model> <genre> <count> <out.txt> --license-ok` has an Ollama model
  write text, filters it with the graph (ASCII, not repetitive, mostly known words), and records every
  attempt in a manifest. It refuses to run until you say you have checked the model's licence. Generation is
  slow on a CPU (about 8 seconds per 60-word passage with phi3).

## Project layout

- `carbide_modules/` — the program
  - `layers.py` — Layer 1 / 2 / 3 stack and the named strip
  - `mdbe.py` — six hex flags, grammar columns, selective SSM
  - `trace_weights.py` — names for otherwise-opaque learned cells
  - `discover_dimension.py` — proposed extra columns, fixed after training
  - `graphmem/` — the graph memory: `store.py` (rules it enforces), `wordnet_loader.py`, `ingest.py`, `compile.py`, `teach.py`
  - `shell.py`, `tui.py`, `settings.py` — command shell, terminal UI, and the one registry of knobs
  - `generation.py` — sampling controls and presets
  - `growth.py`, `teacher.py` — gated depth growth; a local model as a teacher
- `tests/` — checks that the reader and the notebooks still agree with themselves
- `_archive/` and `carbide_module_cplusplus/` — older or side work, not the active program

## Status

Layer 1, Layer 2, and Layer 3 are live on this branch. Grammar sits beside the six hex flags. Training on the included books does reduce error (a short 800-step run on the 8 MB book dropped from about 2.88 to about 1.59). That is a health check, not a claim that the program understands language.

Adding the word book (Layer 1 + Layer 2) helped the guessing game more than the current sentence pack. Layer 3 is still kept because the labeled sentence facts are part of the design, not because they already win on loss.

**Correction (2026-09-21):** the size of that Layer 2 gain, and the loss numbers above, were measured while Layer 2 could see a word's identity before the word was finished (a lookahead leak, now fixed). With the leak removed, Layer 2 gives only a small gain (about 0.02 held-out loss), Layer 3 adds nothing measurable, and the plain `beside` layout is best. See the 2026-09-21 correction in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
