"""Loss/ablation plotting. The sole importer of matplotlib.pyplot — other
modules that need it (e.g. cli.run_ablation) go through `plotting.plt`."""
import csv
import glob
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import config
from .state import train_state


def plot_loss():
    """Update loss curve in real-time."""
    if not train_state.loss_history:
        return

    steps, losses = zip(*train_state.loss_history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses, linewidth=0.8, alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Carbide Training Loss (step {train_state.step})")
    ax.grid(alpha=0.3)
    fig.savefig("carbide_loss_live.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_ablation():
    """Plot ablation comparison (requires running ablation first)."""
    if not os.path.exists("carbide_ablation.png"):
        print("  ⚠ Run ablation study first")
        return
    print("  ✓ Ablation plot ready at carbide_ablation.png")


def _read_weight_trace_top1_mean(path):
    """Real mean |top-1 correlation| across all embedding dims in one
    weight_trace snapshot CSV -- same statistic used to diagnose real
    learning (correlation rising with loss falling) vs memorization
    (correlation falling while loss falls) earlier this session."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return statistics.mean(abs(float(r["embedding_corr_1"])) for r in rows)


def _split_into_runs(records_by_mtime):
    """Splits (step, value, mtime) records into separate runs wherever
    step decreases when walked in SAVE ORDER (mtime) -- a fresh run
    resets train_state.step to 0, so a drop in the order snapshots were
    actually written is real evidence a new run started. Sorting by
    step first (the obvious-looking approach) is wrong: two runs' step
    ranges can overlap (e.g. a fresh 1,000-step run and an old 58,000-
    step run both cover step 0-1000), which silently merges them into
    one fake continuous run. Each returned run is then sorted by step
    internally, for plotting."""
    ordered = sorted(records_by_mtime, key=lambda r: r[2])
    runs, current = [], []
    prev_step = -1
    for step, value, mtime in ordered:
        if step < prev_step and current:
            runs.append(current)
            current = []
        current.append((step, value))
        prev_step = step
    if current:
        runs.append(current)
    return [sorted(run, key=lambda r: r[0]) for run in runs]


def plot_telemetry():
    """Real training telemetry across every run recorded in
    mdbe_snapshots/ (Carbide's own automatic every-200-step snapshots,
    see training._maybe_snapshot_mdbe) -- not just the current run's
    raw loss, but whether the model is learning or memorizing: loss
    should fall AND embedding<->constraint correlation should rise
    together for real learning; loss falling while correlation falls is
    the memorization signature this exact check caught earlier this
    session (see mdbe_table.xlsx's Read me / weight_trace.xlsx).

    Reads every mdbe_snapshots/weight_trace_step*.csv on disk (real,
    dense, 200-step-interval data -- no manual sampling), splits them
    into separate runs wherever the step count resets, and plots mean
    top-1 |correlation| per run alongside the current run's loss curve
    (from train_state.loss_history, if any). Writes
    carbide_telemetry.png. Safe to call with zero or one snapshot --
    prints what it found instead of crashing on sparse data."""
    files = glob.glob(os.path.join(config.mdbe_snapshot_dir, "weight_trace_step*.csv"))
    if not files:
        print("  ⚠ No mdbe_snapshots/weight_trace_step*.csv found yet -- train at least "
              f"{config.mdbe_snapshot_interval} steps first (snapshots are automatic).")
        return

    records = []
    for f in files:
        step = int(os.path.basename(f).split("step")[-1].replace(".csv", ""))
        records.append((step, _read_weight_trace_top1_mean(f), os.path.getmtime(f)))
    runs = _split_into_runs(records)

    fig, (ax_loss, ax_corr) = plt.subplots(1, 2, figsize=(12, 4.5))

    if train_state.loss_history:
        steps, losses = zip(*train_state.loss_history)
        ax_loss.plot(steps, losses, linewidth=0.6, alpha=0.7, color="#2a78d6")
        ax_loss.set_yscale("log")
        ax_loss.axhline(5.545, linestyle="--", color="gray", linewidth=0.8,
                         label="random guess (ln 256)")
        ax_loss.legend(fontsize=8)
    else:
        ax_loss.text(0.5, 0.5, "no loss_history this session\n(load a checkpoint or keep training)",
                     ha="center", va="center", transform=ax_loss.transAxes, fontsize=9, color="gray")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("cross-entropy loss (log scale)")
    ax_loss.set_title("Loss — current session")
    ax_loss.grid(alpha=0.3)

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
    for i, run in enumerate(runs):
        steps, vals = zip(*run)
        ax_corr.plot(steps, vals, marker="o", markersize=2.5, linewidth=1.2,
                     color=colors[i % len(colors)], label=f"run {i+1} (step {steps[0]}-{steps[-1]})")
    ax_corr.set_xlabel("training step")
    ax_corr.set_ylabel("mean top-1 |embedding correlation|")
    ax_corr.set_title("Embedding ↔ constraint alignment — every run")
    ax_corr.legend(fontsize=8)
    ax_corr.grid(alpha=0.3)

    fig.suptitle("Carbide training telemetry: loss down AND correlation up = real learning; "
                  "loss down, correlation down = memorization", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig("carbide_telemetry.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ carbide_telemetry.png ({len(runs)} run(s), {len(records)} snapshots)")
    for i, run in enumerate(runs):
        first, last = run[0], run[-1]
        trend = "rising (healthy)" if last[1] > first[1] else "falling (check for memorization)"
        print(f"    run {i+1}: step {first[0]}->{last[0]}, mean |r| {first[1]:.3f}->{last[1]:.3f} [{trend}]")
