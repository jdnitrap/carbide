# Opening the notebooks (plain language)

This page answers the question: **can I watch Carbide think?**

Short answer: you can watch the three notebooks, and you can see which named columns a hidden unit listens to. You cannot watch a private movie of every number inside the guessing engine. Nobody’s model really gives you that movie. Carbide’s promise is smaller and real: every fact we claim to care about has a name, a row, and a number you can dump.

## What “traceable” means here

Pick one letter in a sentence, say the `t` in *the*.

You can print:

| layer | name | number |
|---|---|---|
| L1 | is_alpha | 1 |
| L1 | is_digit | 0 |
| L2 | word | the |
| L2 | ARTICLE | 0.95 |
| L2 | VERB | 0.02 |
| L3 | DECLARATIVE | 0.80 |

That table is Copy A — the named strip. It is produced by `strips()` in `carbide_modules/layers.py`. Training does not erase it.

## Soft score vs learned cell

People use “weight” to mean two different things. Carbide keeps them apart.

| | Soft score (rule) | Learned cell (model opinion) |
|---|---|---|
| Who wrote it? | A person, as a hint | Training, by guessing the next letter |
| Example | `the` → ARTICLE 0.95 | the word-row for *the* after 800 steps |
| Can it be 0 or 1? | Layer 1 facts are 0/1. Grammar is usually a strength. | Any real number the trainer moved |
| Still named? | Yes — column ARTICLE, row *the* | Yes — same row, same column, different number |
| Good for | Starting the model with language mechanics | Seeing what the model actually used |

A learned cell on a named square is *more* useful for “open the brain” than a soft score, because it is the model’s vote, not our hint. It is still not a thought-movie. It is a labeled opinion.

Layer 1 stays 0/1 on purpose. Those six columns decode the hex. If they became soft, “is this a digit?” would stop being a fact.

## What you can open today

| File | What it shows |
|---|---|
| `mdbe_table.csv` / `mdbe_table.xlsx` | All 256 letter rows + six facts + grammar columns |
| `full_table.csv` | Same, plus every learned embedding dimension given a best-effort name |
| `weight_trace.csv` | For each hidden unit: which named column it listens to most |
| `embedding_dimension_names.json` | Those best-effort names, with the real correlation next to them |
| word table export | Layer 2 rows as words, not bytes |
| sentence table export | Layer 3 facts pinned to whole sentences |
| `discovered_dimensions.json` | Extra columns the model proposed, still beside the original ones |

The menu item **Save Outputs** writes several of these.

## What you cannot open

After Copy A is mixed into Copy B, a single slot in the guessing engine is no longer “the ARTICLE column.” It is soup.

You also cannot replay “why it said *q* next” as a clean chain of English reasons. What you *can* do:

1. Dump the named strip at that letter.
2. Ask which named columns the next hidden units weighted hardest (`weight_trace.csv`).
3. Turn one named column off and see whether the guess changes (ablation).

That is the honest toolkit. Researchers call pieces of it “probing” and “activation patching.” Carbide just keeps the names on the page so those tools have something real to point at.

## How this is different from a normal language model

A normal tokenizer gives a locker number to a chunk of text. That locker number is not the letter, and the dimensions behind it are not named.

Carbide’s letter ID *is* the letter. The extra dimensions have names a person chose. The word book uses those letters as an index into a word row. The sentence book uses the word rows as an index into a short sentence slot. Names stay on Copy A the whole way.

## Watching a training run

A short run on the included 8 MB book (800 steps, small model) moved the guessing error from about 2.88 down to about 1.59. That means the notebooks plus the engine got better at the next-letter game. It does not mean the notebooks are “right English.” Open the word table after training and read the cells yourself.
