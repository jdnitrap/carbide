# How Carbide works (plain language)

This note is for anyone. You do not need to know how a computer works.

Carbide is a program that reads books one letter at a time and tries to guess the next letter. While it does that, it keeps three labeled notebooks so a person can see *what it thought it saw* — not only a pile of mystery numbers.

## The simple picture

Think of reading as three steps you already do without noticing:

1. **See the mark.** That shape is a `t`, not a `7`, not a space.
2. **Group marks into a word.** `t` + `h` + `e` is the word *the*.
3. **Hear the sentence.** *The cat sat.* is a statement. *Did it run?* is a question.

Carbide does the same three steps on purpose, and it **writes them down with names**.

## Layer 1 — the letter book

Every letter the computer already knows has a real ID. The letter `t` is the same ID the machine has used for decades (`0x74`). We do not invent a new locker number and hide the letter.

On that row we also write six plain facts:

- Is it a letter?
- Is it a number?
- Is it a capital?
- Is it punctuation?
- Is it a space?
- Is it the start of a multi-byte character?

Those six answers are yes or no. They decode the mark. They are not “maybe a verb.”

## Layer 2 — the word book

Layer 1 strings marks into a word. That word is the title of a row in the **word book**.

On the *the* row we write grammar in named columns: article, noun, verb, and so on. Some answers are almost sure (`the` is an article). Some are guesses (a word that only *looks like* a noun in this book). Guesses are written as a strength, not a fake “100% yes.”

You can open this book and read:

> word: *the* — ARTICLE: high — VERB: low

That is the point. A person can check it.

## Layer 3 — the sentence book

Layer 2’s words, in order, are a sentence. Layer 3 does not store the whole sentence as one giant title (almost every sentence in a book is unique, so that book would never fill in).

It keeps a **fixed-size summary** of the words so far, and writes sentence facts next to it:

- statement / question / command
- is it negated? (*not*, *never*)
- does it look like it has an object?

You can still dump this layer with names. You can also pin those tags onto the words you are already looking at.

## What the “learning engine” is

After the three notebooks are filled, a separate engine guesses the next letter. In this project that engine is a **selective state-space model** (same family as Mamba). You can think of it as a reader with a short working memory that updates as each letter arrives. It is not the notebooks. The notebooks stay outside so they can still be printed.

The engine sees the *numbers* from the notebooks. The *names* stay on the page for you.

## What we refuse to do

- We do not mash “this is a verb” into the same six boxes that mean “this is a digit.” Grammar sits **beside** the letter facts, never inside them.
- We do not replace letters with mystery token IDs the way some large language models do. The letter ID stays the real letter.
- We do not throw the notebooks away after mixing. If you cannot print Layer 1, 2, and 3 with names, the design has failed.

## How it learns

We show it real books (the files in this repo). After each stretch of text it guesses the next letter. When it is wrong, we nudge the learned parts. The six letter facts stay facts. Grammar and sentence columns can start as rules and get “fixed” later: keep what the model actually uses, quiet what it ignores.

A short training run on the included books did learn: the error went down. That does not mean the program “understands” English. It means the labeled books plus the engine got better at the guessing game.

## If you only remember one paragraph

Carbide reads letters, builds words, then builds sentences — and it keeps a named notebook at each step. The guessing engine trains on those notebooks. You can open the notebooks. That is the whole idea.
