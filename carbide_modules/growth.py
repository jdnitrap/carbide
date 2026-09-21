"""Automatic model growth: add depth when training plateaus, keep it only if it measurably helps.

Growing is done so it cannot hurt at the moment it happens: a new block is a copy of the last one
with its SSM output zeroed (W_C and the D skip term), so it is an exact identity -- the model's
outputs are unchanged -- and then it trains like any other block. That is depth growth only;
widening d_model is not attempted (it changes every layer's shape and cannot be made exactly
function-preserving through the LayerNorms without more machinery).

The gate is a measurement: held-out loss (the reserved tail of the data, never trained on) before
vs after a short probation of training. If the grown model is not better by `keep_margin` it is
rolled back exactly -- weights, optimizer, layer count and step counter -- and the attempt is
logged either way (growth_log.jsonl), so every automatic change to the model is traceable.
Off by default (`set train.auto_grow on`): it changes the model's size and therefore its speed.
"""
import copy
import json
import time

import torch
import torch.nn.functional as F

from . import dataset, training
from .config import config
from .state import train_state

LOG_PATH = "growth_log.jsonl"
HOLDOUT = 0.05


def grow_depth(model, n=1):
    """Append n identity-initialised blocks. Returns the new blocks."""
    new = []
    for _ in range(n):
        block = copy.deepcopy(model.blocks[-1])
        with torch.no_grad():
            block.ssm.W_C.weight.zero_()
            block.ssm.W_C.bias.zero_()
            block.ssm.D.zero_()
        model.blocks.append(block)
        new.append(block)
    return new


@torch.no_grad()
def heldout_loss(model, n_batches=8, seed=1234):
    g = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(n_batches):
            x, y = dataset.get_holdout_batch(g)
            losses.append(F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def plateaued(losses, window=100, min_gain=0.01):
    """True when the mean loss over the last `window` steps improved by less than `min_gain` nats
    over the window before it."""
    if len(losses) < 2 * window:
        return False
    prev, last = losses[-2 * window:-window], losses[-window:]
    return sum(prev) / window - sum(last) / window < min_gain


def _extend_optimizer():
    """A fresh AdamW over all parameters that keeps the old parameters' momentum. (Adding a second
    parameter group instead would make the saved optimizer state impossible to reload.)"""
    old = training.opt
    fresh = torch.optim.AdamW(training.model.parameters(), lr=config.learning_rate)
    for p, state in old.state.items():
        fresh.state[p] = state
    training.opt = fresh


def _snapshot():
    return {"model": copy.deepcopy(training.model), "opt": copy.deepcopy(training.opt.state_dict()),
            "n_layers": config.n_layers, "step": train_state.step,
            "history": list(train_state.loss_history)}


def _restore(snap):
    training.model = snap["model"]
    training.opt = torch.optim.AdamW(training.model.parameters(), lr=config.learning_rate)
    training.opt.load_state_dict(snap["opt"])
    config.n_layers = snap["n_layers"]
    train_state.step = snap["step"]
    train_state.loss_history = snap["history"]


def try_growth(probation=100, keep_margin=0.005, should_stop=None, log_path=LOG_PATH):
    """One gated growth attempt. Returns the record that was logged."""
    dataset.set_holdout(HOLDOUT)
    before = heldout_loss(training.model)
    snap = _snapshot()
    layers_before = len(training.model.blocks)
    grow_depth(training.model)
    _extend_optimizer()
    config.n_layers = len(training.model.blocks)
    training.train_n_steps(probation, should_stop=should_stop)
    after = heldout_loss(training.model)
    kept = after < before - keep_margin
    if not kept:
        _restore(snap)
    rec = {"time": time.time(), "step": train_state.step, "layers_before": layers_before,
           "layers_after": len(training.model.blocks), "heldout_before": round(before, 5),
           "heldout_after": round(after, 5), "kept": kept, "keep_margin": keep_margin}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {'✓ grew' if kept else '↩ rolled back'}: {layers_before} -> {len(training.model.blocks) if kept else layers_before} "
          f"layers (held-out {before:.4f} -> {after:.4f})")
    return rec


def train_with_growth(n, should_stop=None, window=100, probation=100, max_layers=8, min_gain=0.01,
                      keep_margin=0.005, log_path=LOG_PATH):
    """Train n steps; whenever loss plateaus (and there is room), attempt one gated growth."""
    dataset.set_holdout(HOLDOUT)
    if training.model is None:
        training.init_model()
    done, since_growth = 0, 0
    while done < n and not (should_stop and should_stop()):
        chunk = min(window, n - done)
        before_step = train_state.step
        training.train_n_steps(chunk, should_stop=should_stop)
        ran = train_state.step - before_step
        done, since_growth = done + ran, since_growth + ran
        if ran < chunk:
            break
        losses = [loss for _, loss in train_state.loss_history]
        if (len(training.model.blocks) < max_layers and since_growth >= 2 * window
                and plateaued(losses, window, min_gain) and n - done >= probation):
            rec = try_growth(probation, keep_margin, should_stop, log_path)
            done += probation if rec["kept"] else 0
            since_growth = 0
