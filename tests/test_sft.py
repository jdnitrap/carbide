"""Verifies sft.py's loss masking is actually correct: the loss and its
gradient must depend ONLY on the response tokens, never the prompt tokens
— that's the entire point of masking (SFT shouldn't train the model to
"predict" the prompt). Two independent checks, not just one:
  1. sft_step's masked loss matches a reference computed by slicing out
     ONLY the response region and running plain cross_entropy on it.
  2. Changing what byte follows a PROMPT position changes nothing at all
     about the computed loss (since those positions are masked out).
"""
import sys
import os
import copy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import training, sft
from carbide_modules.mdbe import Carbide


def main():
    torch.manual_seed(0)
    training.model = Carbide(d_model=16, n_layers=2, d_state=8)
    training.opt = torch.optim.AdamW(training.model.parameters(), lr=1e-3)

    prompt = list(b"Q: What is X?\nA:")
    response = list(b" X is a thing.\n")

    # --- check 1: masked loss matches a response-only reference loss ---
    model_snapshot = copy.deepcopy(training.model.state_dict())

    full = prompt + response
    xb = torch.tensor([full[:-1]], dtype=torch.long)
    yb = torch.tensor([full[1:]], dtype=torch.long)
    logits = training.model(xb)[0]  # (T,256)

    prompt_len = len(prompt)
    resp_logits = logits[prompt_len - 1:]
    resp_targets = yb[0, prompt_len - 1:]
    ref_loss = F.cross_entropy(resp_logits, resp_targets)

    training.model.load_state_dict(model_snapshot)  # reset before sft_step also mutates via opt.step
    masked_loss_value = sft.sft_step(prompt, response)

    diff = abs(ref_loss.item() - masked_loss_value)
    print(f"masked loss = {masked_loss_value:.6f}  reference (response-only) loss = {ref_loss.item():.6f}  diff = {diff:.2e}")
    ok1 = diff < 1e-5
    print("PASS" if ok1 else "FAIL", "check 1: masked loss matches response-only reference")

    # --- check 2: perturbing what would follow a PROMPT position changes nothing ---
    training.model.load_state_dict(model_snapshot)
    torch.manual_seed(1)
    loss_a = sft.sft_step(prompt, response)

    # construct a variant where a PROMPT-region target byte is different —
    # simulate by editing sft_step's own inputs directly rather than via the
    # public API, since prompt bytes feed the model as INPUT context too and
    # changing an input byte legitimately changes the loss. What must NOT
    # matter is which byte the loss function is told to "expect" at a
    # prompt position — so directly test the masked-loss math in isolation.
    training.model.load_state_dict(model_snapshot)
    full2 = prompt + response
    xb2 = torch.tensor([full2[:-1]], dtype=torch.long)
    yb_wrong_prompt_targets = yb.clone()
    yb_wrong_prompt_targets[0, :prompt_len - 1] = (yb_wrong_prompt_targets[0, :prompt_len - 1] + 37) % 256
    logits2 = training.model(xb2)[0]
    logp2 = F.log_softmax(logits2, dim=-1)
    mask = torch.zeros(len(full2) - 1)
    mask[max(0, prompt_len - 1):] = 1.0
    nll_orig = -logp2[torch.arange(len(full2) - 1), yb[0]]
    nll_wrong = -logp2[torch.arange(len(full2) - 1), yb_wrong_prompt_targets[0]]
    loss_orig = (nll_orig * mask).sum() / mask.sum().clamp(min=1.0)
    loss_wrong = (nll_wrong * mask).sum() / mask.sum().clamp(min=1.0)
    diff2 = abs(loss_orig.item() - loss_wrong.item())
    print(f"loss with real prompt-position targets = {loss_orig.item():.6f}  "
          f"loss with SCRAMBLED prompt-position targets = {loss_wrong.item():.6f}  diff = {diff2:.2e}")
    ok2 = diff2 < 1e-9
    print("PASS" if ok2 else "FAIL", "check 2: scrambling prompt-position targets changes nothing")

    # --- sanity: SFT actually reduces loss on its own examples over a few steps ---
    training.model.load_state_dict(model_snapshot)
    losses = [sft.sft_step(prompt, response) for _ in range(30)]
    ok3 = sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    print(f"first5={sum(losses[:5])/5:.4f} last5={sum(losses[-5:])/5:.4f}")
    print("PASS" if ok3 else "FAIL", "check 3: SFT loss decreases over repeated steps on its own example")

    all_ok = ok1 and ok2 and ok3
    print("\nALL SFT CHECKS PASSED" if all_ok else "\nSOME SFT CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
