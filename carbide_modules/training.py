"""Model init, one training step, checkpoints, and the background thread.

`model` and `opt` live only here. Other modules that need the current
model must go through `training.model` / `training.opt` (attribute
access on this module) rather than importing the names directly —
importing the name would freeze a stale None/reference across the
reinit that happens in init_model()/load_checkpoint()."""
import math
import os
import threading
import torch
import torch.nn.functional as F

from .config import config
from .state import train_state
from . import dataset
from . import monitor
from .mdbe import Carbide, export_mdbe_table, migrate_checkpoint_state_dict, CONSTRAINT_COLUMN_NAMES
from .trace_weights import export_weight_trace_table, used_bytes_from_corpus

model = None
opt = None


def _maybe_snapshot_mdbe(step):
    """task1 refinement #4: periodic MDBE-table snapshot during training,
    not just the one-shot export in Save Outputs — lets you check whether
    same-class bytes actually drift closer together in embedding space over
    training rather than assuming it. Also snapshots the weight-trace table
    (trace_weights.py) alongside it, same interval, same reasoning: lets
    you check whether hidden dimensions actually converge onto a
    consistent, meaningful constraint label over training, rather than
    just asserting it once at the end."""
    if step % config.mdbe_snapshot_interval == 0:
        os.makedirs(config.mdbe_snapshot_dir, exist_ok=True)  # absent on a fresh clone
        path = f"{config.mdbe_snapshot_dir}/mdbe_table_step{step}.csv"
        export_mdbe_table(model, path)
        trace_path = f"{config.mdbe_snapshot_dir}/weight_trace_step{step}.csv"
        used_bytes = used_bytes_from_corpus(config.data_file) if config.data_file and os.path.exists(config.data_file) else None
        export_weight_trace_table(model, trace_path, byte_values=used_bytes)


def _maybe_autosave_checkpoint(step):
    """Periodic checkpoint save during training, to the default (blank-
    suffix) filename -- the same one "Resume from checkpoint" loads by
    default. Without this, a run's actual weights only ever get written to
    disk when someone explicitly saves from the Checkpoints menu; everything
    trained since the last manual save lives only in RAM and is lost the
    moment the process exits."""
    if step % config.checkpoint_autosave_interval == 0:
        save_checkpoint()


MODEL_KINDS = ("beside", "layered", "graph")


def _build_model(kind=None, with_table=True):
    """The one place a model object is created, so init and checkpoint-load agree."""
    kind = kind or config.model_kind
    if kind == "beside":
        return Carbide(d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state)
    if kind == "layered":
        from .layers import Carbide as LayeredCarbide
        return LayeredCarbide(d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state)
    if kind == "graph":
        from .graphmem import N_GRAPH_COLS
        from .layers import Carbide as LayeredCarbide
        m = LayeredCarbide(d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state,
                           graph_cols=N_GRAPH_COLS, default_mode="l1_l2_graph")
        if with_table:
            m.set_graph_table(compile_graph_table())
        return m
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def _kind_of(m):
    if not hasattr(m, "layers"):
        return "beside"
    return "graph" if m.layers.graph_cols else "layered"


def compile_graph_table():
    """Compile the graph memory (config.graph_db) into the table the graph model reads,
    aligned to the current Layer-2 vocabulary."""
    from .graphmem import GraphStore, compile_for_vocab
    from .layers import WORD_VOCAB_SIZE, word_vocab
    if not os.path.exists(config.graph_db):
        raise RuntimeError(f"no graph database at {config.graph_db!r}. Build it first:\n"
                           f"  python -m carbide_modules.graphmem build-core --corpus <your corpus>")
    words = list(word_vocab()[0])
    if not words:
        raise RuntimeError("the Layer-2 word vocabulary is empty; load a dataset first")
    with GraphStore(config.graph_db) as store:
        return compile_for_vocab(store, words, WORD_VOCAB_SIZE).table


def refresh_graph_table():
    """Re-read the graph into the live graph model (e.g. after new knowledge was dumped in)."""
    if model is None or _kind_of(model) != "graph":
        raise RuntimeError("the current model is not a graph model")
    model.set_graph_table(compile_graph_table())


def init_model():
    global model, opt
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {n_params:,} parameters")


def train_step():
    """Run one training step, return loss."""
    if model is None:
        init_model()

    xb, yb = dataset.get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def save_checkpoint(suffix=""):
    """Save model, optimizer, and training state."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)  # absent on a fresh clone
    filename = f"{config.checkpoint_dir}/carbide_ckpt{suffix}.pt"
    kind = _kind_of(model)
    extra = {}
    if kind in ("layered", "graph"):
        from .layers import word_vocab
        # the exact vocabulary this model's word rows were trained against
        extra['word_vocab'] = list(word_vocab()[0])
    from . import mdbe
    torch.save({
        'model_kind': kind,
        'soft_grammar': mdbe.SOFT_GRAMMAR,   # the input values this model was trained on
        **extra,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'train_state': train_state.to_dict(),
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_state': config.d_state,
            'learning_rate': config.learning_rate,
        },
        # The real, full, ordered column identity this checkpoint's
        # constraint-facing weights were trained against -- NOT just a
        # count. Two different discovered-dimension sets can total the
        # same column count while meaning something completely
        # different (see mdbe.CONSTRAINT_COLUMN_NAMES's docstring for
        # the real case that motivated this); load_checkpoint() compares
        # this list, not TOTAL_CONSTRAINTS, before trusting a shape
        # match.
        'constraint_names': CONSTRAINT_COLUMN_NAMES,
    }, filename)
    print(f"  ✓ Checkpoint saved: {filename}")


def load_checkpoint(suffix="", allow_migrate=False):
    """Load model, optimizer, and training state.

    allow_migrate=True lets a checkpoint saved under an OLDER, narrower
    TOTAL_CONSTRAINTS (fewer discovered dimensions) load into today's
    wider model instead of failing outright -- see
    mdbe.migrate_checkpoint_state_dict for exactly what that does and
    doesn't preserve. The optimizer state is deliberately NOT migrated
    in that case (Adam's momentum buffers are shape-locked to the old
    parameter sizes same as the weights, and reconciling those too
    isn't worth the complexity for what amounts to a few steps of
    re-warming momentum) -- a fresh AdamW is created instead, same as
    any new model gets. This only ever matters after a real dimension-
    count change; a normal checkpoint load is unaffected either way."""
    global model, opt
    filename = f"{config.checkpoint_dir}/carbide_ckpt{suffix}.pt"
    if not os.path.exists(filename):
        print(f"  ✗ Checkpoint not found: {filename}")
        return False

    ckpt = torch.load(filename, weights_only=True)
    config.d_model = ckpt['config']['d_model']
    config.n_layers = ckpt['config']['n_layers']
    config.d_state = ckpt['config']['d_state']
    config.learning_rate = ckpt['config'].get('learning_rate', config.learning_rate)

    kind = ckpt.get('model_kind', 'beside')  # checkpoints from before model kinds were beside models
    config.model_kind = kind
    from . import mdbe
    # a checkpoint with no record predates soft scores: it was trained on hard 0/1 grammar values
    mdbe.SOFT_GRAMMAR = bool(ckpt.get('soft_grammar', False))
    model = _build_model(kind, with_table=False)   # the table itself is in the checkpoint's buffers
    if kind in ("layered", "graph") and 'word_vocab' in ckpt:
        from .layers import install_word_vocab
        note = install_word_vocab(ckpt['word_vocab'])
        if note:
            print(f"  ⚠ {note}")

    # Real column IDENTITY check, not just a count/shape check. Two
    # different discovered-dimension sets can total the same number of
    # columns while meaning something completely different (a real trap
    # this project hit: an old checkpoint's discovered columns were
    # PRONOUN_LIKE/AT_START_BEFORE_PREPOSITION_WORDS/etc from one
    # corpus; the replacement corpus's discovered columns were
    # PREPOSITION_LIKE/NOUN_LIKE/VERB_LIKE/etc -- both exactly 12
    # dimensions, so a shape-only check would report success while
    # silently misapplying every discovered-column weight to the wrong
    # real-world signal). Refuse BEFORE even attempting a load whenever
    # that's the case, regardless of whether shapes happen to line up.
    old_names = ckpt.get('constraint_names')
    current_names = CONSTRAINT_COLUMN_NAMES
    safe_prefix_migration = False
    if old_names is not None and old_names != current_names:
        safe_prefix_migration = (len(old_names) < len(current_names) and
                                  current_names[:len(old_names)] == old_names)
        if not safe_prefix_migration:
            print("  ✗ This checkpoint's real constraint columns don't match the current "
                  f"schema (saved with {len(old_names)} named columns, now "
                  f"{len(current_names)}), and the saved columns are NOT a clean prefix of "
                  "today's -- some column's IDENTITY changed (e.g. a discovered dimension "
                  "was replaced by a different one, not just added on top), not merely the "
                  "count.")
            print("    Migrating would silently apply a weight learned for one real "
                  "constraint onto a different one -- refusing even with allow_migrate=True.")
            print("    Train a fresh model instead.")
            model = None
            return False

    migrated_this_load = False
    try:
        model.load_state_dict(ckpt['model_state'])
    except RuntimeError as e:
        if old_names is None:
            print("  ⚠ This checkpoint predates column-identity tracking -- only the raw "
                  "tensor shape could be checked here, which can't rule out a same-count-"
                  "but-different-meaning schema change. Proceed with that in mind.")
        if not allow_migrate:
            print("  ✗ Checkpoint doesn't match the current constraint schema:")
            print(f"    {e}")
            print("    Call load_checkpoint(suffix, allow_migrate=True) to migrate it: "
                  "old weights kept exactly, new dimension(s) start at 0.0 (real "
                  "'not learned yet', not a guess) and need fresh training exposure.")
            model = None
            return False
        migrated, report = migrate_checkpoint_state_dict(ckpt['model_state'], model)
        model.load_state_dict(migrated)
        migrated_this_load = True
        print("  ⚠ Migrated checkpoint to the current, wider constraint schema:")
        for key, old_width, new_cols in report:
            print(f"    {key}: kept {old_width} learned column(s), added {new_cols} new "
                  f"column(s) at 0.0 (not yet trained)")

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if not migrated_this_load:
        opt.load_state_dict(ckpt['opt_state'])
    else:
        print("    optimizer momentum reset (shape-locked to the old width same as the "
              "weights) -- a fresh AdamW will re-warm over the next several steps")

    train_state.step = ckpt['train_state']['step']
    train_state.loss_history = ckpt['train_state']['loss_history']

    print(f"  ✓ Checkpoint loaded: {filename} (step {train_state.step})")
    return True


def background_training_thread(n, display_mode, started_event=None):
    """Background thread that runs training."""
    if model is None:
        init_model()
        _maybe_snapshot_mdbe(train_state.step)  # baseline, before any steps this run

    train_state.training_active = True
    if started_event is not None:
        started_event.set()
    start_step = train_state.step

    try:
        for step in range(start_step + 1, start_step + n + 1):
            if not train_state.training_active:
                break

            loss = train_step()

            with train_state.training_lock:
                train_state.loss_history.append((step, loss))
                train_state.step = step
                train_state.current_loss = loss

            _maybe_snapshot_mdbe(step)
            _maybe_autosave_checkpoint(step)

    except Exception as e:
        print(f"  ✗ Training error: {e}")
    finally:
        train_state.training_active = False
        # Always save on the way out -- normal completion, an early Stop, or
        # an exception mid-step -- so progress since the last periodic
        # autosave (which only lands every checkpoint_autosave_interval
        # steps) is never silently dropped when the process exits.
        if model is not None:
            save_checkpoint()


def start_background_training(n, display_mode="none"):
    """Start training in background thread."""
    started = threading.Event()
    train_state.training_thread = threading.Thread(
        target=background_training_thread,
        args=(n, display_mode, started),
        daemon=False
    )
    train_state.training_thread.start()
    # Wait for the thread to actually set training_active=True before the
    # display loops below check it — otherwise there's a race where they can
    # see training_active still False and exit instantly ("step 0"), even
    # though training is genuinely running moments later in the background.
    started.wait(timeout=10)

    if display_mode == "log":
        monitor.show_training_log()
    elif display_mode == "monitor":
        monitor.show_live_monitor()
    elif display_mode == "both":
        monitor.show_both_monitors()


def stop_background_training():
    """Stop background training."""
    train_state.training_active = False
    print("  ✓ Stopping background training...")
    if train_state.training_thread:
        train_state.training_thread.join(timeout=5)
    print(f"  ✓ Stopped at step {train_state.step}")


def set_learning_rate(lr):
    """Change the learning rate everywhere it matters. config.learning_rate alone does nothing
    once the optimizer exists -- AdamW keeps the rate it was built with."""
    config.learning_rate = lr
    if opt is not None:
        for group in opt.param_groups:
            group["lr"] = lr


def train_n_steps(n, should_stop=None):
    """Run n training steps. `should_stop()` (optional) is polled between steps so a caller
    (the shell/TUI) can stop a long run cleanly; the checkpoint is still saved on the way out."""
    if model is None:
        init_model()
        _maybe_snapshot_mdbe(train_state.step)  # baseline, before any steps this run

    start_step = train_state.step
    try:
        for step in range(start_step + 1, start_step + n + 1):
            if should_stop is not None and should_stop():
                print(f"  ■ Stopped early at step {train_state.step}")
                break
            loss = train_step()
            train_state.loss_history.append((step, loss))
            train_state.step = step
            train_state.current_loss = loss
            _maybe_snapshot_mdbe(step)
            _maybe_autosave_checkpoint(step)

            if step % max(1, n // 10) == 0 or step == start_step + 1:
                ppl = math.exp(loss) if loss < 10 else float('inf')
                print(f"  step {step:5d} | loss {loss:.4f} | perplexity {ppl:.1f}")
    finally:
        # Always save on the way out (normal completion or an exception
        # mid-step) -- see _maybe_autosave_checkpoint's docstring for why
        # this can't be left to periodic autosave alone.
        if model is not None:
            save_checkpoint()

    print(f"  ✓ Trained {train_state.step - start_step} steps (total: {train_state.step})")
