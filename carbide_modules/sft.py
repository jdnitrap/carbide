"""task1 Part 3 — a lightweight SFT (supervised fine-tuning) stage on top
of the already-pretrained model. Trains on curated prompt->response pairs
with the loss masked to the response tokens only: the model is never
trained to "predict" the prompt, only to produce a good response given
one — standard SFT practice, and the reason this isn't just more
pretraining on the same data. Deliberately small, per task1: a lightweight
pass on a small curated set, not a full RLHF/DPO pipeline.

Continues fine-tuning training.model/training.opt in place — it does not
start a fresh model. Run pretraining first.
"""
import json
import math
import random
import torch
import torch.nn.functional as F

from . import training


def load_sft_examples(filepath):
    """Reads a JSONL file of {"prompt": ..., "response": ...} objects,
    returns [(prompt_bytes, response_bytes), ...] as raw UTF-8 byte lists
    (ints 0-255 — same representation dataset.py uses)."""
    examples = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = list(obj["prompt"].encode("utf-8"))
            r = list(obj["response"].encode("utf-8"))
            examples.append((p, r))
    return examples


def sft_step(prompt_bytes, response_bytes):
    """One masked-loss SFT step on a single (prompt, response) example."""
    if training.model is None:
        training.init_model()

    full = prompt_bytes + response_bytes
    if len(full) < 2:
        return None  # nothing to predict

    device = next(training.model.parameters()).device
    xb = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    yb = torch.tensor([full[1:]], dtype=torch.long, device=device)

    # mask[i] corresponds to target position i+1 in `full`: 1.0 if that
    # position is inside the response, 0.0 if it's still the prompt — only
    # response-token predictions contribute to the loss.
    prompt_len = len(prompt_bytes)
    mask = torch.zeros(len(full) - 1, device=device)
    mask[max(0, prompt_len - 1):] = 1.0

    logits = training.model(xb)                          # (1, T, 256)
    logp = F.log_softmax(logits.reshape(-1, 256), dim=-1)
    target = yb.reshape(-1)
    nll = -logp[torch.arange(len(target), device=device), target]        # (T,)

    denom = mask.sum().clamp(min=1.0)
    loss = (nll * mask).sum() / denom

    training.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(training.model.parameters(), 1.0)
    training.opt.step()
    return loss.item()


def run_sft(examples, epochs=5, shuffle=True):
    """Runs `epochs` passes over `examples`, printing per-epoch progress.
    Returns the per-step (step, loss) history."""
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        if shuffle:
            random.shuffle(order)
        epoch_losses = []
        for idx in order:
            p, r = examples[idx]
            loss = sft_step(p, r)
            if loss is None:
                continue
            step += 1
            epoch_losses.append(loss)
            history.append((step, loss))
        avg = sum(epoch_losses) / max(1, len(epoch_losses))
        ppl = math.exp(avg) if avg < 10 else float("inf")
        print(f"  epoch {epoch:3d}/{epochs} | avg loss {avg:.4f} | perplexity {ppl:.1f}")
    return history
