"""
================================================================================
FILE: carbide_interactive.py
VERSION: 2.0 (Interactive CLI with Data Loading)
DESCRIPTION: Full-featured interactive CLI + REPL + checkpoints + data options
================================================================================

Carbide Interactive CLI — Byte-Level SSM Language Model with REPL Interface

Features:
  • Start/pause/resume training with real-time loss
  • REPL-style text generation (type → generate continuations live)
  • Hyperparameter tuning (learning rate, batch size, d_state)
  • Save/load checkpoints mid-training
  • Real-time loss plotting
  • Ablation mode selector
  • MDBE embedding inspection
  • Load custom datasets (Option 2, 3, 4)

Usage:
  python3 carbide_interactive.py [--data <path>]
  python3 carbide_interactive.py                    # uses train.txt or fallback
  python3 carbide_interactive.py --data my_text.txt # loads my_text.txt
"""

import math, os, random, csv, json, pickle, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import threading

# ==============================================================================
# CONFIG & STATE
# ==============================================================================

class Config:
    def __init__(self):
        self.d_model = 64
        self.n_layers = 2
        self.d_state = 16
        self.batch_size = 1
        self.seq_len = 128
        self.learning_rate = 3e-3
        self.steps_total = 3000
        self.checkpoint_dir = "carbide_checkpoints"
        self.data_file = None  # Track current dataset
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

config = Config()

# Parse command-line arguments (Option 4)
parser = argparse.ArgumentParser(description="Carbide Interactive CLI")
parser.add_argument("--data", type=str, default=None, help="Path to custom dataset file")
args = parser.parse_args()

class TrainingState:
    def __init__(self):
        self.step = 0
        self.loss_history = []
        self.paused = False
        self.current_loss = 0.0
        self.start_time = time.time()
        self.training_thread = None
        self.training_active = False
        self.training_target_steps = 0
        self.training_lock = threading.Lock()
        
    def get_elapsed(self):
        return time.time() - self.start_time
    
    def get_speed(self):
        elapsed = self.get_elapsed()
        if elapsed > 0 and self.step > 0:
            return self.step / elapsed
        return 0
        
    def to_dict(self):
        return {
            'step': self.step,
            'loss_history': self.loss_history,
            'config': {
                'd_model': config.d_model,
                'n_layers': config.n_layers,
                'd_state': config.d_state,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
            }
        }

train_state = TrainingState()

# ==============================================================================
# DATA LOADING
# ==============================================================================

FALLBACK = ("the little cat saw the sun. the sun was big and warm. "
            "the cat ran to the hill. the hill was green. ")

def load_dataset(filepath=None):
    """Load dataset from file or use default (Option 2, 3, 4)."""
    global data
    
    if filepath:
        # Option 4: Command-line argument OR Option 3: Menu selection
        filepath = filepath.strip()  # Remove whitespace
        
        if not os.path.exists(filepath):
            print(f"  ✗ File not found: {filepath}")
            print(f"  Current directory: {os.getcwd()}")
            print(f"  Available .txt files: {', '.join([f for f in os.listdir('.') if f.endswith('.txt')]) or '(none)'}")
            return False
        
        if not os.path.isfile(filepath):
            print(f"  ✗ Not a file: {filepath}")
            return False
        
        try:
            text = open(filepath, encoding="utf-8", errors="ignore").read()
            data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
            config.data_file = filepath
            print(f"  ✓ Loaded {filepath}")
            print(f"  ✓ {len(text):,} chars, {len(data):,} bytes")
            return True
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return False
    else:
        # Option 1: Check for train.txt, fallback to default
        if os.path.exists("train.txt"):
            text = open("train.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "train.txt"
            print(f"✓ loaded train.txt ({len(text):,} chars)")
        else:
            text = FALLBACK * 500
            config.data_file = "[fallback]"
            print("✓ using built-in fallback sample")
        
        # Convert text to bytes
        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        print(f"✓ {len(data):,} bytes total\n")
        return True

# Initialize data (check command-line arg first)
if args.data:
    load_dataset(args.data)
else:
    load_dataset()

def get_batch(batch=None, seq_len=None):
    if batch is None:
        batch = config.batch_size
    if seq_len is None:
        seq_len = config.seq_len
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,))
    xs = torch.stack([data[i : i + seq_len] for i in ix])
    ys = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return xs, ys

# ==============================================================================
# MDBE + SSM (same as carbide.py)
# ==============================================================================

NUM_CONSTRAINTS = 6

def mdbe_constraints(bytes_seq: torch.Tensor) -> torch.Tensor:
    f = bytes_seq.float()
    cols = []
    cols.append(((f >= 65) & (f <= 90)) | ((f >= 97) & (f <= 122)))
    cols.append((f >= 48) & (f <= 57))
    cols.append((f >= 65) & (f <= 90))
    cols.append(((f >= 33) & (f <= 47)) | ((f >= 58) & (f <= 64)) |
                ((f >= 91) & (f <= 96)) | ((f >= 123) & (f <= 126)))
    cols.append((f == 32) | (f == 9) | (f == 10) | (f == 13))
    lead = ((f >= 0) & (f < 0x80)) | ((f >= 0xC0) & (f <= 0xFF))
    cols.append(lead)
    return torch.stack(cols, dim=-1).float()

class MDBE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.base = nn.Embedding(256, d_model)
        self.proj = nn.Linear(d_model + NUM_CONSTRAINTS, d_model)

    def forward(self, bytes_seq, live=True):
        learned = self.base(bytes_seq)
        cols = mdbe_constraints(bytes_seq) if live \
               else torch.zeros(*bytes_seq.shape, NUM_CONSTRAINTS)
        return self.proj(torch.cat([learned, cols], dim=-1))

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(
            torch.exp(torch.empty(d_model, d_state).uniform_(1, 16))))
        self.W_delta = nn.Linear(d_model, d_model)
        self.W_B = nn.Linear(d_model, d_state)
        self.W_C = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        Bsz, T, d = x.shape
        delta = F.softplus(self.W_delta(x))
        A = -torch.exp(self.A_log)
        a_bar = torch.exp(delta.unsqueeze(-1) * A)
        b_bar = delta.unsqueeze(-1) * self.W_B(x).unsqueeze(2)
        h = torch.zeros(Bsz, d, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            h = a_bar[:, t] * h + b_bar[:, t]
            y_t = (self.W_C(x[:, t]).unsqueeze(1) * h).sum(-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1) + self.D * x

class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model)

    def forward(self, x):
        return x + self.ssm(self.norm(x))

MODES = ("full", "no_constraints", "plain_embedding")

class Carbide(nn.Module):
    def __init__(self, d_model=64, n_layers=2, d_state=16):
        super().__init__()
        self.mdbe = MDBE(d_model)
        self.blocks = nn.Sequential(*[Block(d_model) for _ in range(n_layers)])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)

    def forward(self, bytes_seq, mode="full"):
        if mode == "plain_embedding":
            x = self.mdbe.base(bytes_seq)
        else:
            x = self.mdbe(bytes_seq, live=(mode == "full"))
        x = self.blocks(x)
        return self.head(self.head_norm(x))

# ==============================================================================
# LIVE MONITOR FUNCTIONS
# ==============================================================================

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
    
    # Scale losses to height
    scaled = []
    for loss in recent_losses:
        scaled_val = int((loss - min_loss) / loss_range * (max_height - 1))
        scaled.append(max(0, min(max_height - 1, scaled_val)))
    
    # Draw graph
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
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n" + "="*70)
    print("CARBIDE LIVE MONITOR")
    print("="*70)
    print(f"Dataset: {config.data_file} ({len(data):,} bytes)")
    print()
    
    # Progress bar
    progress = (train_state.step / total_steps * 50) if total_steps > 0 else 0
    bar = "█" * int(progress) + "░" * (50 - int(progress))
    print(f"Progress: [{bar}] {train_state.step}/{total_steps}")
    print()
    
    # Stats
    print(f"Loss:       {train_state.current_loss:8.4f}")
    print(f"Perplexity: {ppl:8.1f}")
    print(f"Speed:      {speed:8.2f} steps/sec")
    print(f"Elapsed:    {int(elapsed//60):3d}:{int(elapsed%60):02d}")
    print(f"ETA:        {eta_min:3d}:{eta_sec:02d}")
    print()
    
    # Loss graph
    if train_state.loss_history:
        print("Loss Curve (last 60 steps):")
        print(draw_ascii_graph([l for _, l in train_state.loss_history], max_width=60, max_height=7))
    else:
        print("Loss Curve: (waiting for first step)")
    print()
    
    print("="*70)

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

model = None
opt = None

def init_model():
    global model, opt
    model = Carbide(d_model=config.d_model, n_layers=config.n_layers, 
                    d_state=config.d_state)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {n_params:,} parameters")

def train_step():
    """Run one training step, return loss."""
    global model, opt
    if model is None:
        init_model()
    
    xb, yb = get_batch()
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

# ==============================================================================
# GENERATION
# ==============================================================================

@torch.no_grad()
def generate(prompt="the ", n_bytes=100, temperature=0.8, top_k=20):
    """Generate bytes from prompt."""
    if model is None:
        return "[Model not trained yet]"
    
    model.eval()
    out = bytearray(prompt.encode('utf-8') if isinstance(prompt, str) else prompt)
    
    for _ in range(n_bytes):
        ctx = torch.tensor(list(out[-config.seq_len:])).unsqueeze(0)
        logits = model(ctx)[0, -1] / temperature
        kth = logits.topk(top_k).values[-1]
        probs = F.softmax(logits.masked_fill(logits < kth, float("-inf")), -1)
        out.append(int(torch.multinomial(probs, 1)))
    
    return out.decode("utf-8", errors="ignore")

# ==============================================================================
# PLOTTING
# ==============================================================================

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

# ==============================================================================
# INTERACTIVE CLI
# ==============================================================================

def menu_main():
    """Main menu."""
    print("\n" + "="*70)
    print("CARBIDE INTERACTIVE CLI")
    print(f"Dataset: {config.data_file} ({len(data):,} bytes)")
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
  9. Exit
""")

def menu_train():
    """Training control."""
    global train_state
    
    # Check if training is running in background
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
        # Background training is active
        if choice == "1":
            show_training_log()
        elif choice == "2":
            show_live_monitor()
        elif choice == "3":
            show_both_monitors()
        elif choice == "4":
            stop_background_training()
        elif choice == "5":
            save_checkpoint("_paused")
            print("  ✓ Checkpoint saved")
        elif choice == "6":
            print("  (Training continues in background. Returning to main menu...)")
            time.sleep(1)
            return
    else:
        # Start new training
        try:
            n = int(input("  Steps to run: "))
        except ValueError:
            print("  ✗ Please enter a number")
            return
        train_state.training_target_steps = train_state.step + n
        
        if choice == "1":
            start_background_training(n, display_mode="none")
            print(f"  ✓ Training started in background ({n} steps)")
            print("  Returning to main menu...")
            time.sleep(1)
            return
        elif choice == "2":
            start_background_training(n, display_mode="log")
        elif choice == "3":
            start_background_training(n, display_mode="monitor")
        elif choice == "4":
            start_background_training(n, display_mode="both")
        elif choice == "5":
            resume_training()
        elif choice == "6":
            reset_and_train()

def background_training_thread(n, display_mode):
    """Background thread that runs training."""
    global model, opt, train_state
    
    if model is None:
        init_model()
    
    train_state.training_active = True
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
    
    except Exception as e:
        print(f"  ✗ Training error: {e}")
    finally:
        train_state.training_active = False

def start_background_training(n, display_mode="none"):
    """Start training in background thread."""
    global train_state
    
    train_state.training_thread = threading.Thread(
        target=background_training_thread,
        args=(n, display_mode),
        daemon=False
    )
    train_state.training_thread.start()
    
    if display_mode == "log":
        show_training_log()
    elif display_mode == "monitor":
        show_live_monitor()
    elif display_mode == "both":
        show_both_monitors()

def stop_background_training():
    """Stop background training."""
    global train_state
    train_state.training_active = False
    print("  ✓ Stopping background training...")
    if train_state.training_thread:
        train_state.training_thread.join(timeout=5)
    print(f"  ✓ Stopped at step {train_state.step}")

def train_n_steps(n):
    """Run n training steps."""
    global model, opt, train_state
    if model is None:
        init_model()
    
    start_step = train_state.step
    for step in range(start_step + 1, start_step + n + 1):
        loss = train_step()
        train_state.loss_history.append((step, loss))
        train_state.step = step
        train_state.current_loss = loss
        
        if step % max(1, n // 10) == 0 or step == start_step + 1:
            ppl = math.exp(loss) if loss < 10 else float('inf')
            print(f"  step {step:5d} | loss {loss:.4f} | perplexity {ppl:.1f}")
    
    print(f"  ✓ Trained {n} steps (total: {train_state.step})")

def show_training_log():
    """Show training log (Option 1) - standard output."""
    global train_state
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
    global train_state
    
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
    global train_state
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
            
            # Show log every 10 steps
            if current_step > last_printed_step and current_step % 10 == 0:
                ppl = math.exp(current_loss) if current_loss < 10 else float('inf')
                print(f"step {current_step:5d} | loss {current_loss:.4f} | perplexity {ppl:.1f}")
                last_printed_step = current_step
            
            # Show live monitor every 20 steps
            if update_counter % 20 == 0 and train_state.loss_history:
                print_live_status(train_state.training_target_steps)
            
            update_counter += 1
            time.sleep(0.1)
        
        print(f"\n✓ Training complete at step {train_state.step}")
    except KeyboardInterrupt:
        print("\n✓ Returned to menu (training continues in background)")

def reset_and_train():
    """Reset training and train from scratch."""
    global model, opt, train_state
    try:
        n = int(input("  Steps to train: "))
    except ValueError:
        print("  ✗ Please enter a number")
        return
    train_state = TrainingState()
    model = None
    opt = None
    train_n_steps(n)

def resume_training():
    """Resume from checkpoint."""
    suffix = input("  Checkpoint suffix (default blank): ").strip()
    if load_checkpoint(f"_{suffix}" if suffix else ""):
        try:
            n = int(input("  Additional steps to run: "))
        except ValueError:
            print("  ✗ Please enter a number")
            return
        train_n_steps(n)

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
            
            # Generate continuation
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
    global model, opt
    choice = input("Edit: ").strip()

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
        model = None
        opt = None
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
        save_checkpoint(f"_{suffix}" if suffix else "")
    elif choice == "2":
        suffix = input("  Suffix to load: ").strip()
        load_checkpoint(f"_{suffix}" if suffix else "")
    elif choice == "3":
        pass  # Already listed above

def menu_mdbe():
    """Inspect MDBE embeddings."""
    print("\n" + "-"*70)
    print("MDBE EMBEDDINGS")
    print("-"*70)
    
    if model is None:
        print("  Model not initialized")
        return
    
    sample_bytes = {32: "SPACE", 65: "A", 97: "a", 48: "0", 33: "!"}
    print("\n  Sample embeddings (first 8 dims):\n")
    print(f"  {'Byte':>6} {'Char':^10} Embedding dims")
    print("  " + "-"*60)
    
    for b, char in sample_bytes.items():
        emb = model.mdbe.base(torch.tensor([[b]]))[0, 0].detach().tolist()
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
            xb, yb = get_batch()
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
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
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
    plt.close()
    
    print("\n  ✓ Ablation complete: carbide_ablation.png")

def menu_load_dataset():
    """Load custom dataset (Option 2: REPL augmentation, Option 3: Menu)."""
    global data
    print("\n" + "-"*70)
    print("LOAD DATASET")
    print("-"*70)
    print(f"Current dataset: {config.data_file} ({len(data):,} bytes)\n")
    
    # List available .txt files in current directory
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
                load_dataset(filepath)
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
            load_dataset(filepath)
    elif choice == "3":
        text = FALLBACK * 500
        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        config.data_file = "[fallback]"
        print(f"  ✓ Using fallback ({len(data):,} bytes)")
    elif choice == "4":
        if os.path.exists("train.txt"):
            load_dataset("train.txt")
        else:
            print("  ✗ train.txt not found")
    elif choice == "5":
        augment_data_with_generation()
    elif choice == "6":
        return

def augment_data_with_generation():
    """Option 2: Generate text from model and retrain on augmented data."""
    global data
    
    if model is None or train_state.step == 0:
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
    
    # Combine original + generated
    original_text = data.tolist()
    augmented_bytes = original_text.copy()
    
    for text in generated_texts:
        augmented_bytes.extend(list(text.encode('utf-8', errors='ignore')))
    
    # Update global data
    data = torch.tensor(augmented_bytes, dtype=torch.long)
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
    
    # Save loss CSV
    with open("carbide_loss.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("step", "loss")] + train_state.loss_history)
    print("  ✓ carbide_loss.csv")
    
    # Save loss PNG
    plot_loss()
    print("  ✓ carbide_loss_live.png")
    
    # Save MDBE table
    if model is not None:
        rows = []
        for b in range(256):
            byte_tensor = torch.tensor([[b]])
            learned = model.mdbe.base(byte_tensor)[0, 0].tolist()
            live = mdbe_constraints(byte_tensor)[0, 0].tolist()
            char_repr = repr(chr(b)) if 32 <= b < 127 else ""
            rows.append([b, char_repr] + [f"{v:.4f}" for v in learned]
                        + [f"{v:.0f}" for v in live])
        header = (["byte", "character"]
                  + [f"cell_{i}" for i in range(model.mdbe.base.embedding_dim)]
                  + ["is_alpha", "is_digit", "is_upper", "is_punct", "is_space", "utf8_lead"])
        with open("mdbe_table.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(header)
            csv.writer(fh).writerows(rows)
        print("  ✓ mdbe_table.csv")
    
    print("\n  All outputs saved.")

# ==============================================================================
# MAIN LOOP
# ==============================================================================

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
        
        if choice == "1":
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
            if train_state.training_active:
                print("\n  Background training is still running — stopping it first...")
                stop_background_training()
            print("\n✓ Goodbye!\n")
            break
        else:
            print("  ✗ Invalid choice")

if __name__ == "__main__":
    main()
