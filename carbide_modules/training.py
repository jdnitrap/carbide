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
from .mdbe import Carbide, export_mdbe_table

model = None
opt = None


def _maybe_snapshot_mdbe(step):
    """task1 refinement #4: periodic MDBE-table snapshot during training,
    not just the one-shot export in Save Outputs — lets you check whether
    same-class bytes actually drift closer together in embedding space over
    training rather than assuming it."""
    if step % config.mdbe_snapshot_interval == 0:
        path = f"{config.mdbe_snapshot_dir}/mdbe_table_step{step}.csv"
        export_mdbe_table(model, path)


def init_model():
    global model, opt
    model = Carbide(d_model=config.d_model, n_layers=config.n_layers,
                    d_state=config.d_state)
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
    filename = f"{config.checkpoint_dir}/carbide_ckpt{suffix}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'train_state': train_state.to_dict(),
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_state': config.d_state,
            'learning_rate': config.learning_rate,
        }
    }, filename)
    print(f"  ✓ Checkpoint saved: {filename}")


def load_checkpoint(suffix=""):
    """Load model, optimizer, and training state."""
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

    model = Carbide(d_model=config.d_model,
                    n_layers=config.n_layers,
                    d_state=config.d_state)
    model.load_state_dict(ckpt['model_state'])
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    opt.load_state_dict(ckpt['opt_state'])

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

    except Exception as e:
        print(f"  ✗ Training error: {e}")
    finally:
        train_state.training_active = False


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


def train_n_steps(n):
    """Run n training steps."""
    if model is None:
        init_model()
        _maybe_snapshot_mdbe(train_state.step)  # baseline, before any steps this run

    start_step = train_state.step
    for step in range(start_step + 1, start_step + n + 1):
        loss = train_step()
        train_state.loss_history.append((step, loss))
        train_state.step = step
        train_state.current_loss = loss
        _maybe_snapshot_mdbe(step)

        if step % max(1, n // 10) == 0 or step == start_step + 1:
            ppl = math.exp(loss) if loss < 10 else float('inf')
            print(f"  step {step:5d} | loss {loss:.4f} | perplexity {ppl:.1f}")

    print(f"  ✓ Trained {n} steps (total: {train_state.step})")
