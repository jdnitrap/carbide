"""Does Carbide learn to USE graph facts placed in front of a sentence?

  python3 experiment_facts.py --db graph_memory.db --steps 1200

Two identical small models are trained on the same text: `plain` on the raw sentences, `facts` on the
retrieval format ("Facts: ...\\n<sentence>"). Both are then scored on held-out sentences, on the bytes of
the SENTENCE only:

  plain  / no facts            the baseline
  facts  / no facts            the same model with the prefix removed
  facts  / correct facts       the prefix retrieved for that very sentence
  facts  / shuffled facts      the prefix retrieved for a DIFFERENT sentence

"Uses the facts" means correct < shuffled by more than noise (shuffled controls for the model merely being
used to seeing a Facts line). "The facts help" means correct < plain. Held-out sentences come from the end
of the corpus, never trained on. Facts are the dictionary's own definitions, which usually restate words in
the sentence, so a small win here is copying evidence, not reasoning.
"""
import argparse, json, os, random, re, sys, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="."); ap.add_argument("--db", default="graph_memory.db")
ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--d_model", type=int, default=96)
ap.add_argument("--n_layers", type=int, default=2); ap.add_argument("--seq_len", type=int, default=384); ap.add_argument("--prefix_max", type=int, default=160)
ap.add_argument("--batch", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--train_bytes", type=int, default=1_200_000); ap.add_argument("--eval_sentences", type=int, default=300)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--arm", choices=["plain", "facts", "both"], default="both"); ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, a.repo); os.chdir(a.repo); torch.set_num_threads(a.threads)
from carbide_modules.graphmem import GraphStore, retrieval
from carbide_modules.mdbe import Carbide

raw = open("carbide_training_dataset.txt", "rb").read().decode("ascii", errors="ignore")
train_text, held_text = raw[:a.train_bytes], raw[-600_000:]
def clip(prefix):
    """Cap a facts prefix so prefix + sentence always fits one training window (cut at a word, keep the newline)."""
    if len(prefix) <= a.prefix_max:
        return prefix
    return prefix[:a.prefix_max].rsplit(" ", 1)[0].rstrip(" .,;:") + ".\n"


split = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if 40 <= len(s.strip()) <= 200]
store = GraphStore(a.db)
t0 = time.time()
fact_train = []
for s in split(train_text):
    fact_train.append(clip(retrieval.prefix_for(store, s, max_words=1, per_word=2)) + s + "\n\n")
plain_train = [s + "\n\n" for s in split(train_text)]
with_facts = sum(1 for f, p in zip(fact_train, plain_train) if len(f) > len(p))
print(f"fact corpus: {len(fact_train)} sentences, {with_facts} with facts ({time.time()-t0:.0f}s)", flush=True)

rng = random.Random(a.seed)
held = [s for s in split(held_text) if retrieval.prefix_for(store, s, max_words=1, per_word=2)]
rng.shuffle(held); held = held[:a.eval_sentences]
prefixes = [clip(retrieval.prefix_for(store, s, max_words=1, per_word=2)) for s in held]
shuffled = prefixes[1:] + prefixes[:1]
store.close()
print(f"held-out sentences with facts: {len(held)}", flush=True)

def encode(parts): return torch.tensor(list("".join(parts).encode("ascii", errors="ignore")), dtype=torch.long)

def train(sentences, tag):
    data = encode(sentences); torch.manual_seed(a.seed)
    m = Carbide(d_model=a.d_model, n_layers=a.n_layers, d_state=16)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr); g = torch.Generator().manual_seed(1000 + a.seed)
    for step in range(a.steps):
        ix = torch.randint(0, len(data) - a.seq_len - 1, (a.batch,), generator=g)
        x = torch.stack([data[i:i + a.seq_len] for i in ix]); y = torch.stack([data[i + 1:i + a.seq_len + 1] for i in ix])
        loss = F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 100 == 0: print(f"[{tag}] step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return m.eval()

@torch.no_grad()
def sentence_loss(model, prefix_list, held_list=None):
    """Mean cross-entropy over the bytes of each held-out SENTENCE (the prefix is context only)."""
    held_list = held if held_list is None else held_list
    losses = []
    for s, p in zip(held, prefix_list):
        ctx, body = encode([p]), encode([s + "\n\n"])
        seq = torch.cat([ctx, body])[:a.seq_len + 1]
        x, y = seq[:-1].unsqueeze(0), seq[1:]
        start = max(len(ctx) - 1, 0)                                # first position whose target is a sentence byte
        if start >= len(y):                                         # nothing of the sentence fits: skip, never NaN
            continue
        logits = model(x)[0]
        losses.append(F.cross_entropy(logits[start:], y[start:]).item())
    if len(losses) < 0.95 * len(held_list):
        raise RuntimeError(f"only {len(losses)}/{len(held_list)} held-out sentences could be scored")
    return sum(losses) / len(losses)

res = {"seed": a.seed, "steps": a.steps, "d_model": a.d_model, "seq_len": a.seq_len, "n_held": len(held),
       "fact_sentences_with_facts": with_facts}
none = [""] * len(held)
if a.arm in ("plain", "both"):
    plain = train(plain_train, "plain"); res["plain/no_facts"] = sentence_loss(plain, none)
if a.arm in ("facts", "both"):
    fm = train(fact_train, "facts")
    res["facts/no_facts"] = sentence_loss(fm, none)
    res["facts/correct_facts"] = sentence_loss(fm, prefixes)
    res["facts/shuffled_facts"] = sentence_loss(fm, shuffled)
print("RESULT", json.dumps(res), flush=True)
if a.out: json.dump(res, open(a.out, "w"))
