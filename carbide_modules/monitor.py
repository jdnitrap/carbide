"""Live training displays: the ASCII loss graph, the btop-style status
screen, and the three watch loops menu_train() drops into."""
import math
import os
import time

from .config import config
from .state import train_state
from . import dataset


def draw_ascii_graph(losses, max_width=60, max_height=8):
    """Draw ASCII graph of losses (like btop)."""
    if not losses:
        return "No data yet"

    recent_losses = losses[-max_width:] if len(losses) > max_width else losses
    if not recent_losses:
        return "No data"

    min_loss = min(recent_losses)
    max_loss = max(recent_losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    scaled = []
    for loss in recent_losses:
        scaled_val = int((loss - min_loss) / loss_range * (max_height - 1))
        scaled.append(max(0, min(max_height - 1, scaled_val)))

    lines = []
    for height in range(max_height - 1, -1, -1):
        line = "│ "
        for val in scaled:
            if val == height:
                line += "█"
            elif val > height:
                line += "│"
            else:
                line += " "
        line += " │"
        lines.append(line)

    lines.append("└" + "─" * (len(scaled) + 1) + "┘")
    return "\n".join(lines)


def print_live_status(total_steps):
    """Print live training status (btop-style)."""
    elapsed = train_state.get_elapsed()
    speed = train_state.get_speed()

    if train_state.step > 0 and speed > 0:
        eta_sec = (total_steps - train_state.step) / speed
        eta_min = int(eta_sec / 60)
        eta_sec = int(eta_sec % 60)
    else:
        eta_min = 0
        eta_sec = 0

    ppl = math.exp(train_state.current_loss) if train_state.current_loss < 10 else float('inf')

    os.system('clear' if os.name == 'posix' else 'cls')

    print("\n" + "="*70)
    print("CARBIDE LIVE MONITOR")
    print("="*70)
    print(f"Dataset: {config.data_file} ({len(dataset.data):,} bytes)")
    print()

    progress = (train_state.step / total_steps * 50) if total_steps > 0 else 0
    bar = "█" * int(progress) + "░" * (50 - int(progress))
    print(f"Progress: [{bar}] {train_state.step}/{total_steps}")
    print()

    print(f"Loss:       {train_state.current_loss:8.4f}")
    print(f"Perplexity: {ppl:8.1f}")
    bytes_per_sec = speed * config.batch_size * config.seq_len
    print(f"Speed:      {speed:8.2f} steps/sec  ({bytes_per_sec:,.0f} bytes/sec)")
    print(f"Elapsed:    {int(elapsed//60):3d}:{int(elapsed%60):02d}")
    print(f"ETA:        {eta_min:3d}:{eta_sec:02d}")
    print()

    if train_state.loss_history:
        print("Loss Curve (last 60 steps):")
        print(draw_ascii_graph([l for _, l in train_state.loss_history], max_width=60, max_height=7))
    else:
        print("Loss Curve: (waiting for first step)")
    print()

    print("="*70)


def show_training_log():
    """Show training log (Option 1) - standard output."""
    last_printed_step = 0

    print("\n" + "="*70)
    print("TRAINING LOG (Option 1)")
    print("="*70)
    print("Press Ctrl+C to return to menu (training continues in background)\n")

    try:
        while train_state.training_active:
            with train_state.training_lock:
                current_step = train_state.step
                current_loss = train_state.current_loss

            if current_step > last_printed_step:
                if current_step % 10 == 0 or current_step == 1:
                    ppl = math.exp(current_loss) if current_loss < 10 else float('inf')
                    print(f"step {current_step:5d} | loss {current_loss:.4f} | perplexity {ppl:.1f}")
                last_printed_step = current_step

            time.sleep(0.1)

        print(f"\n✓ Training complete at step {train_state.step}")
    except KeyboardInterrupt:
        print("\n✓ Returned to menu (training continues in background)")


def show_live_monitor():
    """Show live monitor (Option 2) - btop style."""
    print("\nStarting live monitor...")
    print("Press Ctrl+C to return to menu (training continues in background)\n")
    time.sleep(0.5)

    try:
        while train_state.training_active:
            with train_state.training_lock:
                if train_state.loss_history:
                    print_live_status(train_state.training_target_steps)
            time.sleep(1)

        print(f"\n✓ Training complete at step {train_state.step}")
    except KeyboardInterrupt:
        print("\n✓ Returned to menu (training continues in background)")


def show_both_monitors():
    """Show both log and live monitor (Option 3)."""
    last_printed_step = 0

    print("\n" + "="*70)
    print("TRAINING LOG + LIVE MONITOR (Option 3)")
    print("="*70)
    print("Press Ctrl+C to return to menu (training continues in background)\n")

    try:
        update_counter = 0
        while train_state.training_active:
            with train_state.training_lock:
                current_step = train_state.step
                current_loss = train_state.current_loss

            if current_step > last_printed_step and current_step % 10 == 0:
                ppl = math.exp(current_loss) if current_loss < 10 else float('inf')
                print(f"step {current_step:5d} | loss {current_loss:.4f} | perplexity {ppl:.1f}")
                last_printed_step = current_step

            if update_counter % 20 == 0 and train_state.loss_history:
                print_live_status(train_state.training_target_steps)

            update_counter += 1
            time.sleep(0.1)

        print(f"\n✓ Training complete at step {train_state.step}")
    except KeyboardInterrupt:
        print("\n✓ Returned to menu (training continues in background)")
