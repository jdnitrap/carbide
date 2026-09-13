"""
================================================================================
FILE: carbide_v3.0_tui.py
VERSION: 3.0 (Full btop-style Terminal UI)
DESCRIPTION: Complete interactive TUI like btop/htop with panels, shortcuts, real-time updates
================================================================================

Carbide TUI v3.0 — Full btop-style Terminal User Interface

Features:
  • btop-style layout with multiple panels
  • Real-time loss graph (main panel)
  • Statistics panel (loss, speed, ETA, etc)
  • Dataset info panel
  • Keyboard shortcuts (like btop)
  • Background training thread-safe
  • Interactive controls without exiting
  • Live updates every 100ms

Keyboard Shortcuts (like btop):
  [S] Start training         [P] Pause training        [R] Resume training
  [L] Toggle loss log        [M] Toggle live monitor   [C] Save checkpoint
  [O] Save outputs          [H] Help                  [Q] Quit

Usage:
  python3 carbide_v3.0_tui.py [--data file.txt] [--steps 5000]
"""

import math, os, random, csv, json, pickle, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import threading

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

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
        self.steps_total = 5000
        self.checkpoint_dir = "carbide_checkpoints"
        self.data_file = None
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

config = Config()

parser = argparse.ArgumentParser(description="Carbide TUI v3.0")
parser.add_argument("--data", type=str, default=None, help="Path to custom dataset")
parser.add_argument("--steps", type=int, default=5000, help="Number of training steps")
args = parser.parse_args()

class TrainingState:
    def __init__(self):
        self.step = 0
        self.loss_history = deque(maxlen=200)  # Keep last 200 losses
        self.paused = False
        self.current_loss = 0.0
        self.start_time = time.time()
        self.training_thread = None
        self.training_active = False
        self.training_target_steps = 0
        self.training_lock = threading.Lock()
        self.show_log = True
        self.show_monitor = True
        
    def get_elapsed(self):
        return time.time() - self.start_time
    
    def get_speed(self):
        elapsed = self.get_elapsed()
        if elapsed > 0 and self.step > 0:
            return self.step / elapsed
        return 0

train_state = TrainingState()

# ==============================================================================
# DATA LOADING
# ==============================================================================

FALLBACK = ("the little cat saw the sun. the sun was big and warm. "
            "the cat ran to the hill. the hill was green. ")

def load_dataset(filepath=None):
    global data
    
    if filepath:
        filepath = filepath.strip()
        if not os.path.exists(filepath):
            return False
        
        try:
            text = open(filepath, encoding="utf-8", errors="ignore").read()
            data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
            config.data_file = filepath
            return True
        except:
            return False
    else:
        if os.path.exists("train.txt"):
            text = open("train.txt", encoding="utf-8", errors="ignore").read()
            config.data_file = "train.txt"
        else:
            text = FALLBACK * 500
            config.data_file = "[fallback]"
        
        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        return True

if args.data:
    load_dataset(args.data)
else:
    load_dataset()

config.steps_total = args.steps

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
# MODEL (SIMPLIFIED)
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
        cols = mdbe_constraints(bytes_seq) if live else torch.zeros(*bytes_seq.shape, NUM_CONSTRAINTS)
        return self.proj(torch.cat([learned, cols], dim=-1))

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.exp(torch.empty(d_model, d_state).uniform_(1, 16))))
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

model = None
opt = None

def init_model():
    global model, opt
    model = Carbide(d_model=config.d_model, n_layers=config.n_layers, d_state=config.d_state)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def train_step():
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

# ==============================================================================
# TRAINING BACKGROUND THREAD
# ==============================================================================

def background_training_thread(n):
    """Background training loop."""
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
                train_state.loss_history.append(loss)
                train_state.step = step
                train_state.current_loss = loss
    finally:
        train_state.training_active = False

# ==============================================================================
# TUI RENDERING
# ==============================================================================

def draw_ascii_graph(losses, width=60, height=8):
    """Draw ASCII loss graph."""
    if not losses:
        return ["[No data yet]"]
    
    losses_list = list(losses)[-width:]
    if not losses_list:
        return ["[No data yet]"]
    
    min_loss = min(losses_list)
    max_loss = max(losses_list)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
    
    scaled = []
    for loss in losses_list:
        scaled_val = int((loss - min_loss) / loss_range * (height - 1))
        scaled.append(max(0, min(height - 1, scaled_val)))
    
    lines = []
    for h in range(height - 1, -1, -1):
        line = "│ "
        for val in scaled:
            line += "█" if val == h else ("│" if val > h else " ")
        line += " │"
        lines.append(line)
    
    lines.append("└" + "─" * (len(scaled) + 1) + "┘")
    return lines

def show_startup_menu():
    """Show startup menu before training begins."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    moth_logo = [
        "    /\\__/\\",
        "   ( o.o )",
        "    > ^ <",
        "   /|   |\\",
        "  (_|   |_)",
    ]
    
    print("╔" + "═"*78 + "╗")
    for i, line in enumerate(moth_logo):
        if i == 0:
            print("║ " + line + " │ CARBIDE TUI v3.0 - Neural Network Trainer".ljust(76 - len(line)) + "║")
        elif i == 1:
            print("║ " + line + " │ Byte-Level SSM with MDBE Embeddings".ljust(76 - len(line)) + "║")
        else:
            print("║ " + line.ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    print("║ " + f"Dataset: {config.data_file}".ljust(76) + "║")
    print("║ " + f"Size: {len(data):,} bytes".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("║ STARTUP OPTIONS:".ljust(78) + "║")
    print("║ " + " "*76 + "║")
    print("║   1. Start training immediately (auto TUI updates)".ljust(78) + "║")
    print("║   2. Load custom dataset first".ljust(78) + "║")
    print("║   3. Configure training (steps, learning rate, etc)".ljust(78) + "║")
    print("║   4. View model info".ljust(78) + "║")
    print("║   5. Resume from checkpoint".ljust(78) + "║")
    print("║ " + " "*76 + "║")
    print("╚" + "═"*78 + "╝")
    
    choice = input("\nChoice (1-5): ").strip()
    return choice

def show_dataset_menu():
    """Show dataset loading menu."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("╔" + "═"*78 + "╗")
    print("║ LOAD DATASET".ljust(79) + "║")
    print("╠" + "═"*78 + "╣")
    print("║ " + f"Current: {config.data_file} ({len(data):,} bytes)".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    
    txt_files = [f for f in os.listdir(".") if f.endswith(".txt")]
    if txt_files:
        print("║ Available .txt files:".ljust(79) + "║")
        for i, f in enumerate(txt_files, 1):
            size = os.path.getsize(f)
            print("║ " + f"[{i}] {f} ({size:,} bytes)".ljust(76) + "║")
        print("║ " + " "*76 + "║")
    
    print("║   1. Select from list above (enter number)".ljust(78) + "║")
    print("║   2. Type custom path".ljust(78) + "║")
    print("║   3. Use train.txt (if exists)".ljust(78) + "║")
    print("║   4. Back to startup menu".ljust(78) + "║")
    print("║ " + " "*76 + "║")
    print("╚" + "═"*78 + "╝")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        if txt_files:
            try:
                idx = int(input("Select file number: ")) - 1
                if 0 <= idx < len(txt_files):
                    load_dataset(txt_files[idx])
                    print(f"✓ Loaded {txt_files[idx]}")
                    time.sleep(1)
            except:
                print("✗ Invalid selection")
                time.sleep(1)
    elif choice == "2":
        path = input("File path: ").strip()
        if load_dataset(path):
            print(f"✓ Loaded {path}")
        else:
            print("✗ Failed to load file")
        time.sleep(1)
    elif choice == "3":
        if os.path.exists("train.txt"):
            load_dataset("train.txt")
            print("✓ Loaded train.txt")
        else:
            print("✗ train.txt not found")
        time.sleep(1)
    
    return show_startup_menu()

def show_config_menu():
    """Show configuration menu."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("╔" + "═"*78 + "╗")
    print("║ TRAINING CONFIGURATION".ljust(79) + "║")
    print("╠" + "═"*78 + "╣")
    print("║ " + f"Steps to run:     {config.steps_total}".ljust(76) + "║")
    print("║ " + f"Learning rate:    {config.learning_rate}".ljust(76) + "║")
    print("║ " + f"Model dims:       d_model={config.d_model}, n_layers={config.n_layers}, d_state={config.d_state}".ljust(76) + "║")
    print("║ " + f"Batch size:       {config.batch_size}".ljust(76) + "║")
    print("║ " + f"Sequence len:     {config.seq_len}".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("║   1. Change steps to run".ljust(78) + "║")
    print("║   2. Change learning rate".ljust(78) + "║")
    print("║   3. Change d_model".ljust(78) + "║")
    print("║   4. Change n_layers".ljust(78) + "║")
    print("║   5. Change d_state".ljust(78) + "║")
    print("║   6. Back to startup menu".ljust(78) + "║")
    print("║ " + " "*76 + "║")
    print("╚" + "═"*78 + "╝")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        steps = int(input("Steps to run: "))
        config.steps_total = steps
        print(f"✓ Set to {steps} steps")
        time.sleep(1)
    elif choice == "2":
        lr = float(input("Learning rate: "))
        config.learning_rate = lr
        print(f"✓ Set to {lr}")
        time.sleep(1)
    elif choice == "3":
        dm = int(input("d_model: "))
        config.d_model = dm
        print(f"✓ Set to {dm}")
        time.sleep(1)
    elif choice == "4":
        nl = int(input("n_layers: "))
        config.n_layers = nl
        print(f"✓ Set to {nl}")
        time.sleep(1)
    elif choice == "5":
        ds = int(input("d_state: "))
        config.d_state = ds
        print(f"✓ Set to {ds}")
        time.sleep(1)
    
    return show_startup_menu()

def show_model_info():
    """Show model information."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("╔" + "═"*78 + "╗")
    print("║ MODEL INFORMATION".ljust(79) + "║")
    print("╠" + "═"*78 + "╣")
    print("║ " + "Architecture: Selective State Space Model (SSM)".ljust(76) + "║")
    print("║ " + f"d_model (dimensions): {config.d_model}".ljust(76) + "║")
    print("║ " + f"n_layers (blocks): {config.n_layers}".ljust(76) + "║")
    print("║ " + f"d_state (state size): {config.d_state}".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("║ INPUT: Byte-level (0-255) UTF-8 text".ljust(79) + "║")
    print("║ MDBE: Mechanically Defined Byte Embeddings".ljust(79) + "║")
    print("║   • 64-dim learned embedding per byte".ljust(79) + "║")
    print("║   • 6 rule-based constraints (alpha, digit, upper, punct, space, utf8)".ljust(79) + "║")
    print("║ " + " "*76 + "║")
    print("║ OUTPUT: 256-way softmax (next byte prediction)".ljust(79) + "║")
    print("║ " + " "*76 + "║")
    print("║ TRAINING:".ljust(79) + "║")
    print("║ " + f"Optimizer: AdamW (lr={config.learning_rate})".ljust(76) + "║")
    print("║ " + f"Gradient clip: 1.0".ljust(76) + "║")
    print("║ " + f"Loss: Cross-entropy on next-byte prediction".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("╚" + "═"*78 + "╝")
    
    input("\nPress Enter to return...")
    return show_startup_menu()
    """Render full TUI screen."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    elapsed = train_state.get_elapsed()
    speed = train_state.get_speed()
    
    if train_state.step > 0 and speed > 0:
        eta_sec = (config.steps_total - train_state.step) / speed
        eta_min, eta_sec = int(eta_sec // 60), int(eta_sec % 60)
    else:
        eta_min, eta_sec = 0, 0
    
    ppl = math.exp(train_state.current_loss) if train_state.current_loss < 10 else 0
    
    # Moth Logo
    moth_logo = [
        "    /\\__/\\",
        "   ( o.o )",
        "    > ^ <",
        "   /|   |\\",
        "  (_|   |_)",
    ]
    
    # Header with logo
    print("╔" + "═"*78 + "╗")
    for i, line in enumerate(moth_logo):
        if i == 0:
            print("║ " + line + " │ CARBIDE TUI v3.0 - btop-style Neural Network Trainer".ljust(76 - len(line)) + "║")
        elif i == 1:
            print("║ " + line + " │ Byte-Level SSM Language Model with MDBE Embeddings".ljust(76 - len(line)) + "║")
        else:
            print("║ " + line.ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Dataset panel
    print("║ DATASET" + " "*71 + "║")
    print("║ " + config.data_file.ljust(76) + "║")
    print("║ " + f"Size: {len(data):,} bytes".ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Stats panel
    progress = (train_state.step / config.steps_total * 50) if config.steps_total > 0 else 0
    bar = "█" * int(progress) + "░" * (50 - int(progress))
    print("║ TRAINING" + " "*69 + "║")
    print("║ " + f"Progress: [{bar}] {train_state.step}/{config.steps_total}".ljust(76) + "║")
    print("║ " + f"Loss: {train_state.current_loss:8.4f} │ Perplexity: {ppl:8.1f}".ljust(76) + "║")
    print("║ " + f"Speed: {speed:8.2f} steps/sec │ Elapsed: {int(elapsed//60):3d}:{int(elapsed%60):02d} │ ETA: {eta_min:3d}:{eta_sec:02d}".ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Loss graph panel
    print("║ LOSS CURVE" + " "*67 + "║")
    for line in draw_ascii_graph(train_state.loss_history, width=70, height=6):
        print("║ " + line.ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Status and help panel
    status = "RUNNING" if train_state.training_active else "IDLE"
    paused = " (PAUSED)" if train_state.paused else ""
    print("║ STATUS" + " "*71 + "║")
    print("║ " + f"State: {status}{paused}".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("║ KEYBOARD SHORTCUTS:" + " "*57 + "║")
    print("║ " + "[S]tart │ [P]ause │ [R]esume │ [C]heckpoint │ [O]utputs │ [H]elp │ [Q]uit".ljust(76) + "║")
    print("╚" + "═"*78 + "╝")

def handle_input():
    """Handle keyboard input (non-blocking)."""
    if os.name == 'posix':
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            char = sys.stdin.read(1).lower()
            return char
    return None

# ==============================================================================
# TUI RENDERING
# ==============================================================================

def draw_ascii_graph(losses, width=60, height=8):
    """Draw ASCII loss graph."""
    if not losses:
        return ["[No data yet]"]
    
    losses_list = list(losses)[-width:]
    if not losses_list:
        return ["[No data yet]"]
    
    min_loss = min(losses_list)
    max_loss = max(losses_list)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
    
    scaled = []
    for loss in losses_list:
        scaled_val = int((loss - min_loss) / loss_range * (height - 1))
        scaled.append(max(0, min(height - 1, scaled_val)))
    
    lines = []
    for h in range(height - 1, -1, -1):
        line = "│ "
        for val in scaled:
            line += "█" if val == h else ("│" if val > h else " ")
        line += " │"
        lines.append(line)
    
    lines.append("└" + "─" * (len(scaled) + 1) + "┘")
    return lines

def render_tui():
    """Render full TUI screen."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    elapsed = train_state.get_elapsed()
    speed = train_state.get_speed()
    
    if train_state.step > 0 and speed > 0:
        eta_sec = (config.steps_total - train_state.step) / speed
        eta_min, eta_sec = int(eta_sec // 60), int(eta_sec % 60)
    else:
        eta_min, eta_sec = 0, 0
    
    ppl = math.exp(train_state.current_loss) if train_state.current_loss < 10 else 0
    
    # Moth Logo
    moth_logo = [
        "    /\\__/\\",
        "   ( o.o )",
        "    > ^ <",
        "   /|   |\\",
        "  (_|   |_)",
    ]
    
    # Header with logo
    print("╔" + "═"*78 + "╗")
    for i, line in enumerate(moth_logo):
        if i == 0:
            print("║ " + line + " │ CARBIDE TUI v3.0 - btop-style Neural Network Trainer".ljust(76 - len(line)) + "║")
        elif i == 1:
            print("║ " + line + " │ Byte-Level SSM Language Model with MDBE Embeddings".ljust(76 - len(line)) + "║")
        else:
            print("║ " + line.ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Dataset panel
    print("║ DATASET" + " "*71 + "║")
    print("║ " + config.data_file.ljust(76) + "║")
    print("║ " + f"Size: {len(data):,} bytes".ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Stats panel
    progress = (train_state.step / config.steps_total * 50) if config.steps_total > 0 else 0
    bar = "█" * int(progress) + "░" * (50 - int(progress))
    print("║ TRAINING" + " "*69 + "║")
    print("║ " + f"Progress: [{bar}] {train_state.step}/{config.steps_total}".ljust(76) + "║")
    print("║ " + f"Loss: {train_state.current_loss:8.4f} │ Perplexity: {ppl:8.1f}".ljust(76) + "║")
    print("║ " + f"Speed: {speed:8.2f} steps/sec │ Elapsed: {int(elapsed//60):3d}:{int(elapsed%60):02d} │ ETA: {eta_min:3d}:{eta_sec:02d}".ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Loss graph panel
    print("║ LOSS CURVE" + " "*67 + "║")
    for line in draw_ascii_graph(train_state.loss_history, width=70, height=6):
        print("║ " + line.ljust(76) + "║")
    print("╠" + "═"*78 + "╣")
    
    # Status and help panel
    status = "RUNNING" if train_state.training_active else "IDLE"
    paused = " (PAUSED)" if train_state.paused else ""
    print("║ STATUS" + " "*71 + "║")
    print("║ " + f"State: {status}{paused}".ljust(76) + "║")
    print("║ " + " "*76 + "║")
    print("║ KEYBOARD SHORTCUTS:" + " "*57 + "║")
    print("║ " + "[S]tart │ [P]ause │ [R]esume │ [C]heckpoint │ [O]utputs │ [H]elp │ [Q]uit".ljust(76) + "║")
    print("╚" + "═"*78 + "╝")

def handle_input():
    """Handle keyboard input (non-blocking)."""
    if os.name == 'posix':
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            char = sys.stdin.read(1).lower()
            return char
    return None

# ==============================================================================
# MAIN TUI LOOP
# ==============================================================================

def run_tui():
    """Main TUI loop."""
    global train_state
    
    # Show startup menu first
    while True:
        choice = show_startup_menu()
        
        if choice == "1":
            break  # Start training
        elif choice == "2":
            show_dataset_menu()
        elif choice == "3":
            show_config_menu()
        elif choice == "4":
            show_model_info()
        elif choice == "5":
            # Resume from checkpoint
            checkpoint_files = [f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pt')]
            if checkpoint_files:
                print(f"✓ Found {len(checkpoint_files)} checkpoints")
                for f in checkpoint_files:
                    print(f"  - {f}")
            else:
                print("✗ No checkpoints found")
            time.sleep(2)
    
    # Start training
    config.steps_total = args.steps
    train_state.training_target_steps = args.steps
    train_state.training_thread = threading.Thread(
        target=background_training_thread,
        args=(args.steps,),
        daemon=False
    )
    train_state.training_thread.start()
    
    try:
        while True:
            render_tui()
            
            char = handle_input()
            if char:
                if char == 'q':
                    print("\nQuitting Carbide TUI...")
                    train_state.training_active = False
                    break
                elif char == 's':
                    if not train_state.training_active:
                        train_state.training_active = True
                        train_state.training_thread = threading.Thread(
                            target=background_training_thread,
                            args=(args.steps - train_state.step,),
                            daemon=False
                        )
                        train_state.training_thread.start()
                elif char == 'p':
                    train_state.paused = True
                    train_state.training_active = False
                elif char == 'r':
                    train_state.paused = False
                    if not train_state.training_active:
                        train_state.training_active = True
                        train_state.training_thread = threading.Thread(
                            target=background_training_thread,
                            args=(args.steps - train_state.step,),
                            daemon=False
                        )
                        train_state.training_thread.start()
                elif char == 'c':
                    if model:
                        torch.save({
                            'model_state': model.state_dict(),
                            'opt_state': opt.state_dict(),
                            'step': train_state.step,
                        }, f"{config.checkpoint_dir}/carbide_ckpt_tui.pt")
                        print("✓ Checkpoint saved")
                        time.sleep(1)
                elif char == 'o':
                    if train_state.loss_history:
                        losses = list(train_state.loss_history)
                        with open("carbide_loss_tui.csv", "w", newline="") as f:
                            csv.writer(f).writerows([("step", "loss")])
                            for i, loss in enumerate(losses, train_state.step - len(losses)):
                                csv.writer(f).writerow([i, loss])
                        print("✓ Outputs saved")
                        time.sleep(1)
                elif char == 'h':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print("""
CARBIDE TUI v3.0 - KEYBOARD SHORTCUTS

[S] Start training     - Start or restart training
[P] Pause training     - Pause the training loop
[R] Resume training    - Resume from pause
[C] Checkpoint         - Save model checkpoint
[O] Outputs            - Save loss curve to CSV
[H] Help               - Show this help screen
[Q] Quit               - Exit Carbide TUI

Press any key to continue...""")
                    input()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        train_state.training_active = False
    finally:
        if train_state.training_thread:
            train_state.training_thread.join(timeout=5)
        print(f"Final step: {train_state.step}")
        print(f"Final loss: {train_state.current_loss:.4f}")

if __name__ == "__main__":
    run_tui()
