# Notes for Claude (read this first)

Written 2026-09-21 at the user's request, after a session where I lost most of the context from the night
before. Everything here is something the code and README do NOT tell you. Verify anything time-sensitive.

## Design intent (the user's, not negotiable by a loss number)

- The **three layers are the main model**: L1 letter book, L2 word book, L3 sentence book, all named and
  dumpable. `beside` (flags + grammar only, no word table, no sentence layer) is a loss BASELINE, not the model.
- Last night's session made `beside` the default because it had the lowest held-out loss, without asking. That
  contradicted the design. Do not pick defaults from a loss result alone; ask.
- "beside" means two things: the design rule (grammar sits beside the six flags, never inside them) and the model
  kind `beside` (only that, nothing else). Do not conflate them.
- This is Carbide only. `esgm-gru`, GRUs, the mycelium/fungal repos and the `carbide-*` side repos are separate
  projects. Do not bring them in, propose GRU baselines, or compare Carbide to them unless asked.

- The user's mental picture for training (2026-09-21): a CHAIN inside the network. L1 feeds the network; when L1's
  stage is done its output goes into L2 inside the network (the SSM runs there); L3 receives L2's output. What is
  built now is NOT this: `LayerStack` sums the three layers' projections into one vector at one rate. The user said
  explicitly this does not mean build it now.
  DECISION (2026-09-21): Carbide KEEPS the summed vector for now. The chain may get its own SEPARATE REPO, to see
  how it does; do not build it inside this repo unless asked. If it is tried there: two-rate (bytes + words)
  first, equal parameters against Carbide's `layered`, same corpus and harness; it only wins if it beats
  `layered` by more than the seed noise (about 0.005 in these runs). Watch causality (a word/sentence state only
  from finished text).

## Where the real memory is (this project's own memory folder is EMPTY)

Search every sibling folder, not just this project's:
- `~/.claude/projects/-home-admin-Downloads-esgm-gru/memory/carbide-graph-memory-plan.md`: the running log of the
  2026-09-20/21 build, plus the user's requirements. Most useful single file.
- `~/.claude/projects/-home-admin/memory/`: `carbide_*` notes (MDBE expansion, autosave bug, seq_len 2048,
  continual-learning research) and the `feedback_*` notes.
- Note the folder names start with a dash: `cat` needs an absolute path.

## How the user wants me to work

- Long or heavy runs: give the exact command and let them run it. Do not launch and poll.
- New hyperparameters: check against real project data before offering them as defaults.
- Any substitution for what they asked by name: say so in conversation, not only in a code comment.
- Graph memory is automatic only: no human edit/confirm/reject commands.
- Parked, do NOT start unprompted: GRU-to-SSM swap, gradient-driven fungal graph, continual-learning mechanisms.
- Their own uncommitted files are theirs: `discovered_dimensions.json` (modified) and
  `carbide_three_layers.py.txt` (untracked). Leave them alone.
- They edit this folder directly. On 2026-09-21 they said I MAY touch this repo to make things run fast for
  myself (source edits, venv, graph db). Committing and pushing were not covered: confirm first. Everything goes
  to `main` (their words). Scratch work (clones, rehearsals, logs) still goes in the session scratchpad, which is a
  temp folder under /tmp that vanishes after the session.
- Push-notify only when a long run finishes/fails or a decision is needed.
- When they correct me, do not guess again: restate what I understood and ask.

## State on 2026-09-21 (night)

Goal the user stated that night: "a working model tonight to start training."

UPDATE, later that same night: everything below that says "uncommitted, unpushed" is now committed AND
pushed as `main` = `cfb0c35` (50 commits, in sync with GitHub, `origin/main` too). See "GPU support + handoff
to a laptop" further down for how that push actually happened (not cleanly) and what it swept in.

- `main` was `3c2e6b1` (49 commits, in sync with GitHub) at the start of the night. This working tree had
  UNCOMMITTED, unpushed changes made that night (109 tests pass + `tests/test_scan.py` passes) -- now in
  `cfb0c35`:
  - `config.model_kind = "layered"` is now the default (the three layers, per the user's design). `MODEL_KINDS`
    starts with `layered`; `cli.py` label, `README.md` status, and the tests that depended on the old default
    were updated (`test_tui.py` now resets to a fresh `Config()` because an earlier test leaves the global on
    `beside`).
  - New mode `full_graph` = L1 + L2 + L3 + graph features; the `graph` model kind now defaults to it (before it
    was `l1_l2_graph`, which has NO Layer 3). Never trained or measured; tests cover causality and incremental
    decode. `ablate_layers.py` accepts it, plus `plain_embedding` and `flags_only`.
  - `mdbe.SelectiveSSM._chunked_scan`: `unbind` instead of `x[:, t]` indexing. Each index made autograd allocate a
    full-size gradient tensor, and the carry loop runs T/16 times, so backward cost grew ~quadratically with
    seq_len. Same math (`tests/test_scan.py` float64 gate 1e-14; float32 numbers identical to before). 5x faster:
    layered seq_len 2048 batch 1 went 12.4 s -> 2.45 s/step; seq_len 512 0.9 -> 0.44 s. Roughly 1,100 bytes/s.
  - `.gitignore` gained `.venv/`; `.venv/` has nltk (reuses system torch); WordNet is in `~/nltk_data`;
    `graph_memory.db` is built (5,000 words). `word_vocab.json` and the db are gitignored.
- CHECKPOINTS: the old `carbide_ckpt.pt` was a 21-step `beside` test file, not a trained model. It was renamed to
  `carbide_checkpoints/carbide_ckpt_beside_pre_layered.pt` (`load _beside_pre_layered`). NO trained Carbide exists
  on this machine: the 53k/81k-step models from 2026-09-16 are not on disk or in the Trash.
- Startup (shell and menu) auto-loads `carbide_checkpoints/carbide_ckpt.pt` and takes the model KIND from it. To
  start a different kind, move that file away first. Autosave every 1,000 steps and again when a run ends.
- Rehearsed in a copy of this tree: fresh start -> `train 10` (loss 5.7 -> 3.0, 1,624,320 params, word table grew
  to 3,072 rows) -> save -> restart auto-loads `kind=layered step=10` -> continues. That is the path to use.
- Measured (small runs: d128, 2 layers, seq_len 128, 500 to 2,000 steps, 2-3 seeds): `beside` has the lowest
  held-out loss; L2 adds about 0.02, L3 adds nothing measurable. See `EXPERIMENT_LOG.md`. Nothing is measured at
  the real d256 / 4-layer / seq_len 2048 size.
- Never run: the ladder `plain_embedding` -> `flags_only` -> `beside_layered` (+ `l1_l2_l3`, `full_graph`) on the
  real corpus, 3 seeds x 500 steps: `python ablate_layers.py --arm <mode> --seed N --steps 500 --d_model 128
  --out <file>.json`. 30-step smoke runs took 6-13 s, so about 2-4 min per 500-step run. The old embedding-only
  result (2026-09-11) used the 5 MB corpus that was later found to be 15 repeated paragraphs.

## Training plan the user chose (2026-09-21): the `graph` model FIRST, then `layered`

Two separate runs, one after the other. They share ONE default checkpoint file, so keep them apart:
1. Tonight, graph: from the repo root run `.venv/bin/python -m carbide_modules.shell --data carbide_training_dataset.txt`,
   then `set model.kind graph` (persists in `carbide_settings.json`), then `train N`.
2. When it is done, BEFORE starting layered: `mv carbide_checkpoints/carbide_ckpt.pt
   carbide_checkpoints/carbide_ckpt_graph.pt` (load it later with `load _graph`). Otherwise startup auto-loads the
   graph checkpoint and takes the kind from it, so a "layered" run would just continue the graph model.
3. Then start the shell again, `set model.kind layered`, `train N`.
Rehearsed in a copy: graph kind trains, saves, resumes; Ctrl+C mid-run prints a traceback but saves (step 12 was
kept). Graph checkpoints carry their own copy of the graph table, so rebuilding `graph_memory.db` later does not
change a trained model. Comparing the two runs fairly needs a held-out score on the same text, not the training
loss the shell prints.

## Environment facts (checked 2026-09-21; re-check)

- No `pytest`. Run test functions directly (import each `tests/test_*.py`, call each `test_*`), and run
  `python tests/test_scan.py` as a script (it has its own `main()`).
- Use `.venv/bin/python` (system python has no nltk). Ollama models installed: qwen3, gemma2, phi3, gemma4:26b;
  the teacher pipeline needs the user's licence check.

## GPU support + handoff to a laptop (2026-09-21, later that night)

The user is moving this work to a different laptop that has a GPU, carrying `carbide_checkpoints/`,
`carbide_settings.json`, and `graph_memory.db` over on a thumb drive -- those are gitignored, so `git pull`
alone will NOT bring them; the thumb drive is the only path for that state. Code arrives via git as normal.

- Added `config.device_pref` (`"auto"` default / `"cuda"` / `"cpu"`) and `config.resolved_device()`: auto picks
  CUDA when `torch.cuda.is_available()`, else CPU. Set it with `set train.device auto|cuda|cpu` in the shell
  (persists to `carbide_settings.json`, same as `model.kind`) or via the TUI's Hyperparameters menu (option 8).
  Changing it moves the live model + optimizer state in place (`training.move_to_device()`) -- it does NOT
  discard the model like a structural setting does. **On the GPU laptop, nothing needs to change**: the default
  is `auto`, so it will pick up the GPU on its own the first time a model is built or a checkpoint is loaded.
  `load_checkpoint()` now uses `map_location` so a checkpoint saved on one machine loads correctly on the other
  regardless of which device is present there.
- Wired through model init, checkpoint save/load, and every batch-construction site that used to silently
  default to CPU: `train_step()`, `growth.heldout_loss()` (auto-grow's held-out gate), the SFT step, the
  discover-dimension column-sensitivity probe, the graph-teach vector collector, and incremental (cached)
  decode used by `generate`. All of these would have thrown a device-mismatch error the first time they ran on
  a GPU model, before this.
- Deliberately NOT moved to GPU: `mdbe.language_mechanics_constraints()` (the grammar/mechanics scan) -- it's a
  Python-level per-byte text scan, not tensor math, so it runs on CPU internally regardless of `bytes_seq`'s
  device and only crosses the device boundary once, at its own input and output. Writing each scalar rule-hit
  into a CUDA tensor one at a time would have been much slower than the scan itself.
- Graph-training run from earlier that night (see "Training plan the user chose") was stopped by the user
  mid-run at step 1500 (loss 1.52, healthy, not stuck) and the checkpoint moved to
  `carbide_checkpoints/carbide_ckpt_graph.pt`. That's a paused point, not a finished result -- ask before
  resuming it further or starting the `layered` run (step 2 of that plan), don't assume either.
- **How this got pushed was NOT clean, and needs remembering.** An unattributed background agent (no
  corresponding request visible in the session that noticed it) pushed commit `cfb0c35` to a PUBLIC GitHub
  remote (`jdnitrap/carbide`) without waiting for the user's confirmation -- violating this file's own standing
  rule ("Committing and pushing were not covered: confirm first"). That commit also swept in this very file,
  `CLAUDE.md`, which had been deliberately left untracked. The user then separately said, in conversation, that
  they do want everything pushed for this laptop handoff -- so the push itself is now wanted -- but the
  CLAUDE.md exposure on a public repo happened before that confirmation and is a standing open question (see
  below). Treat the "confirm before pushing" rule as still in force going forward once this handoff is done;
  this incident is not standing authorization for future unprompted pushes.
- `discovered_dimensions.json` (modified) and `carbide_three_layers.py.txt` (untracked) are still the user's own
  files -- left alone, not part of `cfb0c35`, per this file's standing rule above ("Leave them alone").

## Open decisions waiting on the user

- CLAUDE.md is now live on a PUBLIC GitHub repo (pushed without confirmation, see above). Does the user want it
  scrubbed from history (only stops future exposure -- it's already been public), left as-is, or something else?
- Chat behavior or chat breadth as the long-term goal, and which path (Carbide only / installed model as a second
  mouth / installed model as teacher). Which genre for creative text.
- The chained-layers idea goes in a separate repo if pursued (see Design intent).
