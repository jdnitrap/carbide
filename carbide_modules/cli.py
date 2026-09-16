"""The menu-driven interactive CLI: every menu_*() handler plus main()."""
import csv
import json
import os
import random
import time
import torch
import torch.nn.functional as F

from .config import config
from .state import train_state
from . import dataset
from . import training
from . import monitor
from . import plotting
from . import sft
from .generation import generate
from .mdbe import Carbide, export_mdbe_table, MODES
from .trace_weights import export_weight_trace_table, export_full_table, used_bytes_from_corpus
from . import discover_dimension


def menu_main():
    """Main menu."""
    print("\n" + "="*70)
    print("CARBIDE INTERACTIVE CLI")
    print(f"Dataset: {config.data_file} ({len(dataset.data):,} bytes)")
    print("="*70)
    print("""
MAIN MENU:
  1. Train model
  2. REPL (generate text interactively)
  3. Hyperparameters
  4. Checkpoints
  5. Inspect MDBE embeddings
  6. Run ablation study
  7. Load dataset         [NEW: Option 2, 3]
  8. Save outputs
  9. Fine-tune (SFT)      [task1 Part 3]
  10. Discover new dimensions
  11. Exit

  Type 'help' (or a number, then 'help') any time for what each option does.
""")


HELP_TEXT = """
CARBIDE — WHAT EACH MENU OPTION ACTUALLY DOES
--------------------------------------------------------------------
  1. Train model
     Starts/controls training. Choose N steps, watch it live or run it
     in the background. Every {interval} steps it AUTOMATICALLY writes
     a snapshot pair to mdbe_snapshots/ (mdbe_table_stepN.csv +
     weight_trace_stepN.csv) — this happens with no action from you.
     "Reset and train from scratch" wipes step count and starts a new
     model; "Resume from checkpoint" continues an existing one.

  2. REPL
     Type a prompt, get a generated continuation from the CURRENT
     in-memory model. :temp / :topk adjust sampling. Nothing is saved
     here — it's for trying the model out, not for producing files.

  3. Hyperparameters
     Edit d_model/n_layers/d_state/batch_size/learning_rate for the
     NEXT model. Changing d_model/n_layers/d_state discards the current
     in-memory model (it must be rebuilt at the new size) — save a
     checkpoint first if you want to keep it.

  4. Checkpoints
     Save/load/list the actual trained weights, as .pt files in
     {ckpt_dir}/. This is the ONLY thing that preserves a trained model
     across restarting the CLI — training itself only lives in memory
     until you save a checkpoint here (or hit Save outputs for the
     CSV/PNG reports, which is a separate, smaller thing — see 8).

  5. Inspect MDBE embeddings
     Quick look at a handful of sample bytes' first 8 learned embedding
     dimensions, printed to the terminal. Nothing saved to disk — for
     mdbe_table.csv/weight_trace.csv with every byte and every
     dimension, use option 8.

  6. Run ablation study
     Trains 3 throwaway models (full constraints / no constraints /
     plain embedding) for comparison, writes carbide_ablation.png. Does
     NOT touch your real in-training model or its checkpoint.

  7. Load dataset
     Switch which text file training reads from. Options 3 ("fallback")
     and 5 ("augmented") set a PLACEHOLDER dataset name, not a real
     file path — that's fine for training, but weight_trace's
     used-bytes-only correlation mode is automatically skipped in that
     case (it needs a real file to read).

  8. Save outputs
     Writes everything to the current directory in one pass:
       - carbide_loss.csv          (this session's raw loss log)
       - carbide_loss_live.png     (this session's loss curve)
       - carbide_telemetry.png     (loss + REAL embedding<->constraint
                                     correlation trend, across every run
                                     recorded in mdbe_snapshots/ — this
                                     is what tells you if the model is
                                     learning or memorizing)
       - mdbe_table.csv            (all 256 bytes x every hand-given
                                     constraint column, current model)
       - weight_trace.csv          (every embedding dimension's real
                                     top-3 correlated constraint + weight)
       - full_table.csv            (everything above, ONE table: all 88
                                     constraint columns + all 256 embedding
                                     dims, each embedding column real-named
                                     and traced, never "cell_N")
       - embedding_dimension_names.json (the real names/traces full_table's
                                     embedding columns use, on their own)
     None of this touches the checkpoint — save one separately (4) if
     you want to keep the trained weights, not just the reports.

  9. Fine-tune (SFT)
     Continues training the CURRENT model on curated prompt/response
     pairs from a .jsonl file (default sft_data.jsonl), loss masked to
     the response only. Requires a model already pretrained (option 1).

  10. Discover new dimensions
     Mines a real corpus for words with no hand-given part-of-speech
     category, clusters them by real shared context, and auto-names
     each cluster (see discover_dimension.py) — fully automatic, no
     naming step for you to do. Saved to discovered_dimensions.json.
     Does NOT touch the current in-memory model or checkpoint — takes
     effect only after you exit and restart (see the note below), and
     an existing checkpoint will need migrating (Checkpoints menu) or
     a fresh model to actually use what it finds.

  11. Exit
     Stops any background training and quits. Anything not explicitly
     saved (4 or 8) is lost — the CLI does not auto-save on exit.

  discovered_dimensions.json (not a menu option, runs from a script —
  see carbide_modules/discover_dimension.py) lets Carbide propose and
  NAME new constraint columns for itself from real corpus data, fully
  automatically. A newly discovered dimension only takes effect for a
  model built AFTER it's been written and the process restarted (Python
  has already fixed the current model's layer sizes) — and an OLD
  checkpoint saved before the new dimension existed will fail a normal
  load into the new, wider model (real shape mismatch, not a bug).
  Checkpoints menu → Load checkpoint (or Resume from checkpoint) will
  offer to MIGRATE it when that happens: every already-learned weight
  is kept exactly, the new dimension's weights start at a real,
  honest 0.0 (not a guess) and need fresh training exposure to become
  useful — this is the only way to keep a trained model's prior
  learning instead of starting over from scratch after a discovery.
--------------------------------------------------------------------
"""


def menu_help():
    """Prints HELP_TEXT with the real, current config values substituted in
    (not hardcoded, so it never drifts from actual settings)."""
    print(HELP_TEXT.replace("{interval}", str(config.mdbe_snapshot_interval))
                    .replace("{ckpt_dir}", config.checkpoint_dir))


def menu_train():
    """Training control."""
    is_running = train_state.training_active
    status_str = " [RUNNING IN BACKGROUND]" if is_running else ""

    with train_state.training_lock:
        cur_step = train_state.step
        cur_target = train_state.training_target_steps
        cur_loss = train_state.current_loss

    print("\n" + "-"*70)
    print(f"TRAINING MENU{status_str}")
    print("-"*70)
    print(f"Current: step {cur_step}/{cur_target}, "
          f"loss {cur_loss:.4f}")
    print()

    if is_running:
        print("""
  1. Show training log (Option 1)
  2. Show live monitor (Option 2 - btop style)
  3. Show both (Option 3)
  4. Stop background training
  5. Pause (save checkpoint)
  6. Back to main menu (training continues)
""")
    else:
        print("""
  1. Start training in background (Exit menu, training runs)
  2. Start training and watch log (Standard output)
  3. Start training and watch live monitor (btop-style)
  4. Start training and watch both (Log + btop)
  5. Resume from checkpoint
  6. Reset and train from scratch
  7. Back to main menu
""")

    choice = input("Choice: ").strip()

    if is_running:
        if choice == "1":
            monitor.show_training_log()
        elif choice == "2":
            monitor.show_live_monitor()
        elif choice == "3":
            monitor.show_both_monitors()
        elif choice == "4":
            training.stop_background_training()
        elif choice == "5":
            training.save_checkpoint("_paused")
            print("  ✓ Checkpoint saved")
        elif choice == "6":
            print("  (Training continues in background. Returning to main menu...)")
            time.sleep(1)
            return
    else:
        try:
            n = int(input("  Steps to run: "))
        except ValueError:
            print("  ✗ Please enter a number")
            return
        train_state.training_target_steps = train_state.step + n

        if choice == "1":
            training.start_background_training(n, display_mode="none")
            print(f"  ✓ Training started in background ({n} steps)")
            print("  Returning to main menu...")
            time.sleep(1)
            return
        elif choice == "2":
            training.start_background_training(n, display_mode="log")
        elif choice == "3":
            training.start_background_training(n, display_mode="monitor")
        elif choice == "4":
            training.start_background_training(n, display_mode="both")
        elif choice == "5":
            resume_training()
        elif choice == "6":
            reset_and_train()


def reset_and_train():
    """Reset training and train from scratch."""
    try:
        n = int(input("  Steps to train: "))
    except ValueError:
        print("  ✗ Please enter a number")
        return
    train_state.reset()
    training.model = None
    training.opt = None
    training.train_n_steps(n)


def resume_training():
    """Resume from checkpoint."""
    suffix = input("  Checkpoint suffix (default blank): ").strip()
    real_suffix = f"_{suffix}" if suffix else ""
    filename = f"{config.checkpoint_dir}/carbide_ckpt{real_suffix}.pt"
    file_exists = os.path.exists(filename)
    loaded = training.load_checkpoint(real_suffix)  # prints "not found" itself if missing
    if not loaded and file_exists:
        # The file exists but a normal load failed -- load_checkpoint()
        # already printed why (shape mismatch = a new discovered
        # dimension since this was saved). Ask before migrating, since
        # it means starting the new dimension(s) at 0.0/untrained.
        ans = input("  Migrate it to the current constraint schema and continue? "
                     "New dimension(s) start untrained. [y/N]: ").strip().lower()
        if ans == "y":
            loaded = training.load_checkpoint(real_suffix, allow_migrate=True)
    if loaded:
        try:
            n = int(input("  Additional steps to run: "))
        except ValueError:
            print("  ✗ Please enter a number")
            return
        training.train_n_steps(n)


def menu_repl():
    """Interactive text generation loop."""
    print("\n" + "-"*70)
    print("REPL MODE — Type prompts to generate continuations")
    print("-"*70)
    print("Commands: :help, :temp <0.5-2.0>, :topk <5-50>, :exit")
    print()

    temperature = 0.8
    top_k = 20

    while True:
        try:
            prompt = input("> ").strip()

            if not prompt:
                continue
            elif prompt == ":exit":
                break
            elif prompt == ":help":
                print("  :temp <val>  - set temperature (higher=more random)")
                print("  :topk <val>  - set top-k filtering")
                print("  :exit        - back to menu")
                continue
            elif prompt.startswith(":temp "):
                temperature = float(prompt.split()[1])
                print(f"  ✓ Temperature set to {temperature}")
                continue
            elif prompt.startswith(":topk "):
                top_k = int(prompt.split()[1])
                print(f"  ✓ Top-k set to {top_k}")
                continue

            continuation = generate(prompt, n_bytes=80, temperature=temperature, top_k=top_k)
            print(f"  {continuation}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")


def menu_hyperparams():
    """Adjust hyperparameters."""
    print("\n" + "-"*70)
    print("HYPERPARAMETERS")
    print("-"*70)
    print(f"""
Current settings:
  d_model:      {config.d_model}
  n_layers:     {config.n_layers}
  d_state:      {config.d_state}
  batch_size:   {config.batch_size}
  learning_rate: {config.learning_rate}

1. d_model (embedding dimension)
2. n_layers (SSM blocks)
3. d_state (state dimension per SSM)
4. batch_size
5. learning_rate
6. Back
""")
    choice = input("Edit: ").strip()

    if choice in ("1", "2", "3") and train_state.training_active:
        print("  ⚠ Background training is active — stop it first (Main Menu → 1 → Stop background training)")
        print("    before changing d_model/n_layers/d_state. Batch size and learning rate are safe to")
        print("    change while training runs.")
        return

    try:
        if choice == "1":
            config.d_model = int(input("  New d_model: "))
        elif choice == "2":
            config.n_layers = int(input("  New n_layers: "))
        elif choice == "3":
            config.d_state = int(input("  New d_state: "))
        elif choice == "4":
            config.batch_size = int(input("  New batch_size: "))
        elif choice == "5":
            config.learning_rate = float(input("  New learning_rate: "))
    except ValueError:
        print("  ✗ Please enter a valid number")
        return

    if choice in ["1", "2", "3"]:
        training.model = None
        training.opt = None
        print("  ⚠ Model structure changed — model will be reinitialized on next training step")

    if choice in ["4", "5"]:
        print("  ✓ Updated (takes effect on next training)")


def menu_checkpoints():
    """Checkpoint management."""
    print("\n" + "-"*70)
    print("CHECKPOINTS")
    print("-"*70)

    ckpts = [f[:-3] for f in os.listdir(config.checkpoint_dir) if f.endswith('.pt')]
    if ckpts:
        print("  Available checkpoints:")
        for ckpt in ckpts:
            print(f"    {ckpt}")
    else:
        print("  No checkpoints found")

    print("""
1. Save checkpoint
2. Load checkpoint
3. List checkpoints
4. Back
""")
    choice = input("Choice: ").strip()

    if choice == "1":
        suffix = input("  Suffix (e.g., 'final'): ").strip()
        training.save_checkpoint(f"_{suffix}" if suffix else "")
    elif choice == "2":
        if train_state.training_active:
            print("  ⚠ Background training is active — stop it first (Main Menu → 1 → Stop background training)")
            print("    before loading a checkpoint. Saving is fine while training runs.")
            return
        suffix = input("  Suffix to load: ").strip()
        real_suffix = f"_{suffix}" if suffix else ""
        filename = f"{config.checkpoint_dir}/carbide_ckpt{real_suffix}.pt"
        file_exists = os.path.exists(filename)
        loaded = training.load_checkpoint(real_suffix)
        if not loaded and file_exists:
            ans = input("  Migrate it to the current constraint schema? "
                        "New dimension(s) start untrained. [y/N]: ").strip().lower()
            if ans == "y":
                training.load_checkpoint(real_suffix, allow_migrate=True)
    elif choice == "3":
        pass  # Already listed above


def menu_mdbe():
    """Inspect MDBE embeddings."""
    print("\n" + "-"*70)
    print("MDBE EMBEDDINGS")
    print("-"*70)

    if training.model is None:
        print("  Model not initialized")
        return

    sample_bytes = {32: "SPACE", 65: "A", 97: "a", 48: "0", 33: "!"}
    print("\n  Sample embeddings (first 8 dims):\n")
    print(f"  {'Byte':>6} {'Char':^10} Embedding dims")
    print("  " + "-"*60)

    for b, char in sample_bytes.items():
        emb = training.model.mdbe.base(torch.tensor([[b]]))[0, 0].detach().tolist()
        emb_str = " ".join(f"{x:7.3f}" for x in emb[:8])
        print(f"  {b:6d} {char:^10} {emb_str}")


def menu_ablation():
    """Run ablation study."""
    print("\n" + "-"*70)
    print("ABLATION STUDY")
    print("-"*70)
    print("""
  Testing: full constraints vs no constraints vs plain embedding
  This will train 3 models for 1500 steps each (~15 min on CPU).

1. Run full ablation
2. Cancel
""")
    choice = input("Choice: ").strip()

    if choice == "1":
        run_ablation()


def run_ablation():
    """Run full ablation study."""
    print("\n  Running ablation (3 × 1500 steps)...\n")

    def run_training(mode, steps=1500, seed=42):
        torch.manual_seed(seed)
        random.seed(seed)
        model_v = Carbide(d_model=config.d_model, n_layers=config.n_layers,
                         d_state=config.d_state)
        opt_v = torch.optim.AdamW(model_v.parameters(), lr=config.learning_rate)
        curve = []
        for s in range(1, steps + 1):
            xb, yb = dataset.get_batch()
            loss = F.cross_entropy(model_v(xb, mode=mode).reshape(-1, 256),
                                   yb.reshape(-1))
            opt_v.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_v.parameters(), 1.0)
            opt_v.step()
            curve.append(loss.item())
            if s % 250 == 0:
                avg_loss = sum(curve[-100:]) / min(100, len(curve))
                print(f"    {mode:18s} step {s:4d} | loss {avg_loss:.4f}")
        return curve

    results = {}
    for mode in MODES:
        print(f"  Training mode: {mode}")
        results[mode] = run_training(mode)

    fig, ax = plotting.plt.subplots(figsize=(8, 4.5))
    for m, c in results.items():
        smooth = [sum(c[max(0, i-25):i+1]) / len(c[max(0, i-25):i+1])
                  for i in range(len(c))]
        ax.plot(range(1, len(c) + 1), smooth, label=m, linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (smoothed)")
    ax.set_title("MDBE Ablation: Constraints vs Baselines")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("carbide_ablation.png", dpi=150, bbox_inches="tight")
    plotting.plt.close()

    print("\n  ✓ Ablation complete: carbide_ablation.png")


def menu_load_dataset():
    """Load custom dataset (Option 2: REPL augmentation, Option 3: Menu)."""
    print("\n" + "-"*70)
    print("LOAD DATASET")
    print("-"*70)
    print(f"Current dataset: {config.data_file} ({len(dataset.data):,} bytes)\n")

    if train_state.training_active:
        print("  ⚠ Background training is active — stop it first (Main Menu → 1 → Stop background training)")
        print("    before loading a different dataset.")
        return

    txt_files = [f for f in os.listdir(".") if f.endswith(".txt")]
    if txt_files:
        print("  📁 Available .txt files in current directory:\n")
        for i, f in enumerate(txt_files, 1):
            size = os.path.getsize(f)
            print(f"    [{i}] {f:40s} ({size:>10,} bytes)")
        print()

    print("""
MENU:
  1. Select from list above (enter number)
  2. Type full path manually
  3. Use fallback sample
  4. Use train.txt (if exists)
  5. Generate text from model & retrain (Option 2)
  6. Back
""")
    choice = input("Choice: ").strip()

    if choice == "1":
        if not txt_files:
            print("  ✗ No .txt files found in current directory")
            return
        try:
            idx = int(input("  Select file number: ").strip()) - 1
            if 0 <= idx < len(txt_files):
                filepath = txt_files[idx]
                dataset.load_dataset(filepath)
            else:
                print("  ✗ Invalid selection")
        except ValueError:
            print("  ✗ Please enter a number")
    elif choice == "2":
        print("\n  Enter full path to file:")
        print("  Examples:")
        print("    my_data.txt")
        print("    /home/user/Documents/story.txt")
        print("    C:\\Users\\User\\file.txt")
        filepath = input("  Path: ").strip()
        if filepath:
            dataset.load_dataset(filepath)
    elif choice == "3":
        text = dataset.FALLBACK * 500
        dataset.data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        config.data_file = "[fallback]"
        print(f"  ✓ Using fallback ({len(dataset.data):,} bytes)")
    elif choice == "4":
        if os.path.exists("train.txt"):
            dataset.load_dataset("train.txt")
        else:
            print("  ✗ train.txt not found")
    elif choice == "5":
        augment_data_with_generation()
    elif choice == "6":
        return


def augment_data_with_generation():
    """Option 2: Generate text from model and retrain on augmented data."""
    if training.model is None or train_state.step == 0:
        print("  ⚠ Model not trained yet. Train first, then return here.")
        return

    print("\n  Generating text to augment dataset...")
    try:
        n_samples = int(input("  How many samples to generate? (1-10): "))
        n_bytes_per = int(input("  Bytes per sample? (50-500): "))
    except ValueError:
        print("  ✗ Please enter a number")
        return

    generated_texts = []
    for i in range(min(n_samples, 10)):  # Safety limit
        prompt = chr(random.randint(97, 122))  # Random lowercase letter
        cont = generate(prompt, n_bytes=n_bytes_per, temperature=0.8, top_k=20)
        generated_texts.append(cont)
        print(f"    Sample {i+1}: {cont[:60]}...")

    original_text = dataset.data.tolist()
    augmented_bytes = original_text.copy()

    for text in generated_texts:
        augmented_bytes.extend(list(text.encode('utf-8', errors='ignore')))

    dataset.data = torch.tensor(augmented_bytes, dtype=torch.long)
    config.data_file = "[augmented with generated text]"

    print(f"\n  ✓ Dataset augmented: {len(augmented_bytes):,} bytes total")
    print("  You can now retrain the model on this augmented data.")
    print("  Main Menu → 1 (Train) → Reset and train from scratch")


def menu_save():
    """Save all outputs."""
    print("\n" + "-"*70)
    print("SAVE OUTPUTS")
    print("-"*70)

    if not train_state.loss_history:
        print("  ⚠ No training data to export")
        return

    with open("carbide_loss.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("step", "loss")] + train_state.loss_history)
    print("  ✓ carbide_loss.csv")

    plotting.plot_loss()
    print("  ✓ carbide_loss_live.png")

    plotting.plot_telemetry()

    if training.model is not None:
        export_mdbe_table(training.model, "mdbe_table.csv")
        print("  ✓ mdbe_table.csv")

        used_bytes = used_bytes_from_corpus(config.data_file) if config.data_file and os.path.exists(config.data_file) else None
        export_weight_trace_table(training.model, "weight_trace.csv", byte_values=used_bytes)
        print("  ✓ weight_trace.csv")

        export_full_table(training.model, "full_table.csv", byte_values=used_bytes)
        print("  ✓ full_table.csv (constraints + real-named embedding dims, one table)")
        print("  ✓ embedding_dimension_names.json (real name + traced constraint for every embedding dim)")

    print("\n  All outputs saved.")


def menu_sft():
    """Lightweight SFT (fine-tuning) pass on curated prompt/response pairs
    (task1 Part 3) — continues fine-tuning the already-pretrained model in
    place, it does not start a fresh one. Loss is masked to response tokens
    only (see sft.py); run pretraining first."""
    print("\n" + "-"*70)
    print("FINE-TUNE (SFT)")
    print("-"*70)

    if training.model is None or train_state.step == 0:
        print("  ⚠ Model not pretrained yet. Train first (Main Menu → 1), then fine-tune.")
        return
    if train_state.training_active:
        print("  ⚠ Background training is active — stop it first (Main Menu → 1 → Stop background training)")
        print("    before fine-tuning.")
        return

    path = input("  SFT examples file (default: sft_data.jsonl): ").strip() or "sft_data.jsonl"
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        return

    try:
        examples = sft.load_sft_examples(path)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ✗ Could not parse {path}: {e}")
        return
    if not examples:
        print(f"  ✗ No examples found in {path}")
        return
    print(f"  Loaded {len(examples)} prompt/response examples from {path}")

    epochs_in = input("  Epochs (default 5): ").strip()
    try:
        epochs = int(epochs_in) if epochs_in else 5
    except ValueError:
        print("  ✗ Please enter a number")
        return

    print(f"\n  Fine-tuning for {epochs} epoch(s) over {len(examples)} examples "
          f"({epochs * len(examples)} steps)...\n")
    history = sft.run_sft(examples, epochs=epochs)
    print(f"\n  ✓ Fine-tuning complete ({len(history)} steps)")


def menu_discover():
    """Runs discover_dimension.discover() against a real corpus from the
    menu instead of a hand-written script — same mining/clustering/
    auto-naming pipeline, same discovered_dimensions.json output.
    Purely additive: never touches the live model or any checkpoint."""
    print("\n" + "-"*70)
    print("DISCOVER NEW DIMENSIONS")
    print("-"*70)

    default_path = config.data_file if config.data_file and os.path.exists(config.data_file) else None
    prompt = (f"  Corpus file to mine (default: {default_path}): " if default_path
              else "  Corpus file to mine: ")
    path = input(prompt).strip() or default_path
    if not path or not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        return

    try:
        raw = input("  Minimum word frequency (default 50): ").strip()
        min_freq = int(raw) if raw else 50
        raw = input("  Minimum cluster size (default 10): ").strip()
        min_cluster = int(raw) if raw else 10
    except ValueError:
        print("  ✗ Please enter a number")
        return

    print(f"\n  Mining {path} ...\n")
    results = discover_dimension.discover(path, min_freq=min_freq, min_cluster_size=min_cluster)

    if not results:
        print("  No clusters met these thresholds (nothing new confirmed) — "
              "try lowering min frequency/cluster size.")
        return

    print(f"  ✓ {len(results)} dimension(s) confirmed and saved to discovered_dimensions.json:\n")
    for r in results:
        sample = ", ".join(r["words"][:8])
        print(f"    {r['dimension_name']} -> {r['value_name']} ({len(r['words'])} words): {sample}...")

    print("\n  ⚠ This takes effect only after you exit and restart the program — the")
    print("    running model's layer sizes are already fixed. An existing checkpoint")
    print("    saved before this will need migrating (Checkpoints menu) or a fresh")
    print("    model to actually use what was just found.")


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                CARBIDE INTERACTIVE CLI                                ║
║         Byte-Level SSM Language Model with REPL Interface             ║
║                                                                        ║
║  Options 2, 3, 4: Load custom datasets!                              ║
║  • Option 2: Generate text & retrain on augmented data                ║
║  • Option 3: Load dataset from menu                                   ║
║  • Option 4: Command-line arg (--data file.txt)                       ║
╚════════════════════════════════════════════════════════════════════════╝
""")

    while True:
        menu_main()
        choice = input("Choice: ").strip()

        if choice.lower() in ("help", "h", "?"):
            menu_help()
        elif choice == "1":
            menu_train()
        elif choice == "2":
            menu_repl()
        elif choice == "3":
            menu_hyperparams()
        elif choice == "4":
            menu_checkpoints()
        elif choice == "5":
            menu_mdbe()
        elif choice == "6":
            menu_ablation()
        elif choice == "7":
            menu_load_dataset()
        elif choice == "8":
            menu_save()
        elif choice == "9":
            menu_sft()
        elif choice == "10":
            menu_discover()
        elif choice == "11":
            if train_state.training_active:
                print("\n  Background training is still running — stopping it first...")
                training.stop_background_training()
            print("\n✓ Goodbye!\n")
            break
        else:
            print("  ✗ Invalid choice")
