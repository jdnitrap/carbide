"""Loss/ablation plotting. The sole importer of matplotlib.pyplot — other
modules that need it (e.g. cli.run_ablation) go through `plotting.plt`."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
