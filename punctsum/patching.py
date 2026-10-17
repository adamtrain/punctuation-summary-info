"""Activation patching (BRIEF.md section 4.6): where is the property information *used*?

For a pair (clean member, corrupted member; identical token positions), run the corrupted passage with the residual
stream at one (layer, position) replaced by the clean run's residual at the same place, and measure the readout's
logit difference LD = logit(clean answer) - logit(corrupted answer) at the final position. Recovery for a pair is
(LD_patched - LD_corrupted) / (LD_clean - LD_corrupted): 0 = the patch carries nothing the readout uses, 1 = it
restores the clean behavior completely. Many patches for the same pair run as one batch: each batch item gets one
(layer, positions) patch through per-layer hooks. A patch can cover several positions at once (a span), which detects
information spread thinly over many tokens that no single-position patch would show.
"""

import torch

from punctsum.model import resid_hook_names


@torch.no_grad()
def patch_pair(model, clean: list[int], corrupted: list[int], cells: list[tuple[int, int | list[int]]],
               answers: tuple[int, int], max_tokens: int = 4096) -> tuple[float, float, list[float]]:
    """cells: (layer, position or list of positions) patches; layer indexes resid_hook_names (0 = embeddings,
    l = output of block l-1).
    Returns (LD_clean, LD_corrupted, [LD_patched per cell])."""
    names = resid_hook_names(model)
    device = model.cfg.device
    good, bad = answers
    _, cache = model.run_with_cache(torch.tensor([clean], device=device), names_filter=lambda n: n in names)
    clean_resid = [cache[n][0] for n in names]  # each [T, d]
    ld = lambda logits: (logits[:, -1, good] - logits[:, -1, bad]).float().cpu()  # noqa: E731
    ld_clean = float(ld(model(torch.tensor([clean], device=device)))[0])
    ld_corr = float(ld(model(torch.tensor([corrupted], device=device)))[0])

    out: list[float] = []
    step = max(1, max_tokens // len(corrupted))
    for s in range(0, len(cells), step):
        batch = cells[s : s + step]
        toks = torch.tensor([corrupted] * len(batch), device=device)
        hooks = []
        for li, name in enumerate(names):
            items = [(b, pos) for b, (l, pos) in enumerate(batch) if l == li]
            if not items:
                continue

            def hook(resid, hook, items=items, li=li):
                for b, pos in items:
                    resid[b, pos] = clean_resid[li][pos]  # pos is an int or a list of positions
                return resid

            hooks.append((name, hook))
        out.extend(ld(model.run_with_hooks(toks, fwd_hooks=hooks)).tolist())
    return ld_clean, ld_corr, out
