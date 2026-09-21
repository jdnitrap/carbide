"""Layer ablation for the three-layer MDBE stack. Same batches, seeds and init for every arm.

  python3 ablate_layers.py --arm {beside,l1,l1_l2,l1_l2_l3,l1_l2_graph,l1_graph} --seed 0 --steps 500 --d_model 128
  (run from the repo root; --repo points it at another checkout, e.g. an older commit)

`beside` = the old mdbe.Carbide in mode "full" (six flags + grammar beside them, no word/sentence
layers). The other arms use layers.Carbide with that mode. Works on any tree that has
carbide_modules/layers.py, so the same script measures the original and the fixed code.
Training data excludes the last HELD_OUT bytes of the corpus, which are used only for the
end-of-run held-out loss.
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="."); ap.add_argument("--arm", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--d_model", type=int, default=64); ap.add_argument("--n_layers", type=int, default=2)
ap.add_argument("--d_state", type=int, default=16); ap.add_argument("--seq_len", type=int, default=128)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--threads", type=int, default=3); ap.add_argument("--vocab", default=None, help="word_vocab.json to use (default: the repo's own; built from the corpus if missing)")
ap.add_argument("--out", default=None); ap.add_argument("--save", default=None)
ap.add_argument("--load", default=None, help="skip training; evaluate this saved layers.Carbide state_dict")
ap.add_argument("--tag", default="")
ap.add_argument("--graphdb", default="graph_memory.db", help="graph memory used by the graph arms (l1_l2_graph, l1_graph)")
a = ap.parse_args()

sys.path.insert(0, a.repo); os.chdir(a.repo)
torch.set_num_threads(a.threads)
from carbide_modules import layers as L
from carbide_modules.mdbe import Carbide as OldCarbide
if a.vocab:
    L.WORD_VOCAB_PATH = a.vocab
L._VOCAB_WORDS = None; L._WORD_TO_ID = None
if not os.path.exists(L.WORD_VOCAB_PATH):
    L.build_word_vocab("carbide_training_dataset.txt")
    L._VOCAB_WORDS = None; L._WORD_TO_ID = None

HELD_OUT = 500_000
data = torch.tensor(list(open("carbide_training_dataset.txt", "rb").read()), dtype=torch.long)
train_end = len(data) - HELD_OUT

def make_model():
    torch.manual_seed(a.seed)
    if a.arm in L.GRAPH_MODES:
        from carbide_modules.graphmem import GraphStore, N_GRAPH_COLS, compile_for_vocab
        m = L.Carbide(d_model=a.d_model, n_layers=a.n_layers, d_state=a.d_state,
                      graph_cols=N_GRAPH_COLS, default_mode=a.arm)
        with GraphStore(a.graphdb) as store:
            m.set_graph_table(compile_for_vocab(store, L.word_vocab()[0], L.WORD_VOCAB_SIZE).table)
        return m
    return (OldCarbide if a.arm == "beside" else L.Carbide)(d_model=a.d_model, n_layers=a.n_layers, d_state=a.d_state)
mode = "full" if a.arm == "beside" else a.arm

def batches(lo, hi, g, n):
    for _ in range(n):
        ix = torch.randint(lo, hi - a.seq_len - 1, (a.batch,), generator=g)
        yield (torch.stack([data[i:i + a.seq_len] for i in ix]), torch.stack([data[i + 1:i + a.seq_len + 1] for i in ix]))

@torch.no_grad()
def held_out(model):
    model.eval(); g = torch.Generator().manual_seed(999)
    ls = [F.cross_entropy(model(x, mode=mode).reshape(-1, 256), y.reshape(-1)).item()
          for x, y in batches(train_end, len(data), g, 20)]
    model.train(); return sum(ls) / len(ls)

res = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "d_model": a.d_model, "n_layers": a.n_layers,
       "seq_len": a.seq_len, "batch": a.batch, "lr": a.lr, "repo": os.path.basename(a.repo), "tag": a.tag}
model = make_model()
res["params"] = sum(p.numel() for p in model.parameters())
if a.load:
    model.load_state_dict(torch.load(a.load, weights_only=True)); res["held_out"] = held_out(model)
    res["loaded"] = a.load
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(1000 + a.seed)
    losses, t0 = [], time.time()
    for step, (x, y) in enumerate(batches(0, train_end, g, a.steps)):
        loss = F.cross_entropy(model(x, mode=mode).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step % 25 == 0: print(f"[{a.arm} s{a.seed} {res['repo']}] step {step} loss {losses[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    res.update(first50=sum(losses[:50]) / 50, last50=sum(losses[-50:]) / 50, seconds=time.time() - t0,
               held_out=held_out(model))
    if a.save and a.arm != "beside": torch.save(model.state_dict(), a.save)
if a.out:
    json.dump(res, open(a.out, "w"))
print("RESULT", json.dumps(res), flush=True)
