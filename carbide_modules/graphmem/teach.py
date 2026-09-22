"""Carbide teaches the graph -- automatically, and only what a measurement supports.

Carbide is a byte model; what it can honestly contribute is *evidence about how words behave in
context*. So: read Carbide's hidden state at the byte right after each occurrence of a word (the
model has just seen the whole word), average per word, and train a small linear probe to predict
the dictionary's part of speech from it. If the probe beats the majority-class baseline on
words it did NOT train on (out-of-fold), Carbide demonstrably knows something about word grammar,
and then:
  * words the dictionary has no part of speech for get PROPOSED edges (source `carbide`, with the
    probe's probability as confidence -- they only reach Carbide's table above accept_confidence);
  * dictionary entries the probe strongly and repeatedly disagrees with are scaled down (never
    deleted).
If the probe does not beat the baseline the run is recorded as REJECTED and nothing is added.
No human is in the loop; the gate is the measurement.

Not claimed: this cannot invent meaning or definitions, and a barely-trained model will simply fail
the gate -- which is the correct outcome.
"""
import re
from collections import Counter

import torch

from .compile import POS_NAMES
from .store import REL_HAS_POS, SOURCE_CARBIDE, SOURCE_DICTIONARY

LEARNED = "learned"
WORD_RE = re.compile(r"[a-z]+")


@torch.no_grad()
def collect_word_vectors(model, corpus_bytes, words, *, seq_len=128, max_bytes=400_000, batch=16):
    """{word: (mean hidden vector at the byte after the word, occurrences)} for the given words."""
    words = set(words)
    grabbed = {}
    hook = model.head_norm.register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", o.detach()))
    sums, counts = {}, Counter()
    model.eval()
    device = next(model.parameters()).device
    try:
        data = corpus_bytes[:max_bytes]
        windows = [data[i:i + seq_len] for i in range(0, len(data) - seq_len, seq_len)]
        for lo in range(0, len(windows), batch):
            chunk = windows[lo:lo + batch]
            x = torch.tensor([list(w) for w in chunk], dtype=torch.long, device=device)
            model(x)
            h = grabbed["h"]
            for b, w in enumerate(chunk):
                text = bytes(w).decode("ascii", errors="replace").lower()
                for m in WORD_RE.finditer(text):
                    if m.group() in words and m.end() < len(text):   # the delimiter after the word exists
                        # moved to CPU here: probe_and_propose's fit is a tiny CPU-only linear
                        # probe (not part of the hot training loop) and never needs GPU tensors
                        v = h[b, m.end()].detach().cpu()
                        sums[m.group()] = sums.get(m.group(), 0) + v
                        counts[m.group()] += 1
    finally:
        hook.remove()
    return {w: (sums[w] / counts[w], counts[w]) for w in sums}


def _fit(X, y, n_classes, epochs=300, lr=0.05, wd=1e-3):
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(X), y).backward()
        opt.step()
    return lin


def _dictionary_pos(store, word):
    """Dictionary part-of-speech label: the strongest dictionary HAS_POS edge, or None."""
    best = None
    for e in store.edges_of(word):
        if e["rel"] == REL_HAS_POS and e["source"] == SOURCE_DICTIONARY and e["dst"] in POS_NAMES:
            eff = e["weight"] * e["confidence"] * e["adjust"]
            if best is None or eff > best[1]:
                best = (e["dst"], eff)
    return best[0] if best else None


def probe_and_propose(store, vectors, *, min_count=5, gate_margin=0.05, folds=5, disagree_prob=0.9):
    """Run one teaching pass over `vectors` ({word: (vector, count)}). Returns the run's metrics."""
    usable = {w: v for w, (v, c) in vectors.items() if c >= min_count}
    labeled = {w: _dictionary_pos(store, w) for w in usable}
    known = [w for w, lab in labeled.items() if lab]
    unknown = [w for w, lab in labeled.items() if not lab]
    metrics = {"words_with_vectors": len(usable), "labeled": len(known), "unlabeled": len(unknown)}
    with store.run("teach", LEARNED) as run:
        store.add_module(LEARNED, kind="learned")
        if len(known) < folds * 4 or len({labeled[w] for w in known}) < 2:
            metrics.update(passed=False, reason="too few dictionary-labeled words to test a probe")
            store.settle_run(run, False, metrics)
            return metrics
        X = torch.stack([usable[w] for w in known])
        X = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
        y = torch.tensor([POS_NAMES.index(labeled[w]) for w in known])
        order = torch.randperm(len(known), generator=torch.Generator().manual_seed(0))
        oof_pred, oof_prob = torch.zeros_like(y), torch.zeros(len(known))
        for f in range(folds):
            test = order[f::folds]
            train = torch.tensor([i for i in order.tolist() if i not in set(test.tolist())])
            probs = _fit(X[train], y[train], len(POS_NAMES))(X[test]).softmax(-1)
            oof_prob[test], oof_pred[test] = probs.max(-1)
        acc = float((oof_pred == y).float().mean())
        baseline = float(torch.bincount(y).max()) / len(y)
        passed = acc >= baseline + gate_margin
        metrics.update(probe_accuracy=round(acc, 4), majority_baseline=round(baseline, 4), passed=passed)
        proposed = audited = 0
        if passed:
            everything = _fit(X, y, len(POS_NAMES))
            mu, sd = torch.stack([usable[w] for w in known]).mean(0), torch.stack([usable[w] for w in known]).std(0).clamp(min=1e-6)
            floor = store.policy("accept_confidence")
            for w in unknown:
                p, i = everything(((usable[w] - mu) / sd).unsqueeze(0)).softmax(-1)[0].max(0)
                store.add_edge(w, POS_NAMES[int(i)], REL_HAS_POS, source=SOURCE_CARBIDE, module=LEARNED,
                               weight=float(p), confidence=float(p), run=run,
                               status="accepted" if float(p) >= floor else "proposed", dst_kind="POS")
                proposed += 1
            for j, w in enumerate(known):
                if float(oof_prob[j]) >= disagree_prob and int(oof_pred[j]) != int(y[j]):
                    audited += store.adjust_edge(w, labeled[w], REL_HAS_POS, 0.9, dst_kind="POS",
                                                 reason=f"probe strongly predicts {POS_NAMES[int(oof_pred[j])]}")
        metrics.update(proposed_edges=proposed, dictionary_edges_scaled_down=audited)
        store.settle_run(run, passed, metrics)
    pruned = store.prune()
    total = store.db.execute("SELECT COUNT(*) FROM edges WHERE source = ?", (SOURCE_CARBIDE,)).fetchone()[0] + pruned
    metrics["policy_changes"] = [f"{c[0]}->{c[1]}" for c in store.autotune({"prune_rate": pruned / max(total, 1)})]
    metrics["pruned"] = pruned
    return metrics
