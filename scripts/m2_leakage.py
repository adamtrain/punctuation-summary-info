"""Milestone 2: filler-only leakage check (negative control) and city-token probe (positive control).

The pair design already makes filler leakage impossible: both members of a pair share identical filler,
and fillers are drawn independently of the label. This script therefore checks the *probe pipeline*:
- Negative control: probe the label from passages made only of the filler sentences (no property
  sentence), one randomly chosen member per pair. Accuracy must stay inside the chance band; that band
  also says how far from 50% test accuracy wanders by luck at this test-set size.
- Positive control: probe the label at the city's last token in P, on held-out templates *and cities*.
  It should be high in at least some layers; layer 0 (embeddings) shows how much is readable from the
  city's tokens alone (spelling), before any context.
- Shuffled-label baseline for the positive control.
Uses the k=4 pairs (300 train / 100 test). Writes outputs/m2/leakage_report.txt and leakage.json.
"""

import json
import random

import numpy as np

from punctsum import PROJECT_ROOT
from punctsum.conditions import Passage, render
from punctsum.dataset import load_jsonl, passage_for, render_condition
from punctsum.materials import FILLERS, FIXED_PROPER, LABELS, READOUT
from punctsum.model import load, resid_at, tokenize_pieces, tokenize_with_readout
from punctsum.probes import chance_band, fit_eval
from punctsum.roles import label_tokens

K = 4
SEED = 0
OUT = PROJECT_ROOT / "outputs" / "m2"


def main():
    model = load("gpt2")
    pairs = [p for p in load_jsonl(OUT / "pairs.jsonl") if p["k"] == K]
    rng = random.Random(SEED)
    lines = [f"Pairs with k={K}: {sum(p['split'] == 'train' for p in pairs)} train / {sum(p['split'] == 'test' for p in pairs)} test\n"]
    report = {}

    # Negative control: filler-only text, one member (random label) per pair, last token.
    seqs, pos, y, split = [], [], [], []
    for p in pairs:
        label = rng.choice(LABELS)
        filler = Passage([FILLERS[j] for j in p["fillers"]], proper_nouns=set(FIXED_PROPER))
        _, toks, _ = tokenize_pieces(model, render(filler, "P"))
        seqs.append(toks[0].tolist())
        pos.append([toks.shape[1] - 1])
        y.append(LABELS.index(label))
        split.append(p["split"])
    X = resid_at(model, seqs, pos)[:, 0]  # [n, layers, d]
    y, split = np.array(y), np.array(split)
    tr, te = split == "train", split == "test"
    lo, hi = chance_band(int(te.sum()))
    accs = [fit_eval(X[tr, l], y[tr], X[te, l], y[te]) for l in range(X.shape[1])]
    report["negative_filler_only"] = accs
    lines.append(f"Negative control (filler-only, last token), test n={te.sum()}, chance band [{lo:.2f}, {hi:.2f}]:")
    lines.append("  " + " ".join(f"L{l}:{a:.2f}" for l, a in enumerate(accs)))
    lines.append(f"  outside band: {[l for l, a in enumerate(accs) if not lo <= a <= hi] or 'none'}")

    # Positive control: P, both members, city's last token and the final readout position.
    seqs, pos, y, split = [], [], [], []
    for p in pairs:
        for label in LABELS:
            pieces = render_condition(p, label, "P", model)
            _, toks, pot, n_readout = tokenize_with_readout(model, pieces, READOUT)
            rows = label_tokens(passage_for(p, label), pieces, pot, n_readout)
            city_last = max(t for t, r in enumerate(rows) if r["kind"] == "property")
            seqs.append(toks[0].tolist())
            pos.append([city_last, len(rows) - 1])
            y.append(LABELS.index(label))
            split.append(p["split"])
    X = resid_at(model, seqs, pos)
    y, split = np.array(y), np.array(split)
    tr, te = split == "train", split == "test"
    lo, hi = chance_band(int(te.sum()))
    y_shuf = np.random.default_rng(SEED).permutation(y[tr])
    for name, j in (("city_last_token", 0), ("readout_final", 1)):
        accs = [fit_eval(X[tr, j, l], y[tr], X[te, j, l], y[te]) for l in range(X.shape[2])]
        shuf = [fit_eval(X[tr, j, l], y_shuf, X[te, j, l], y[te]) for l in range(X.shape[2])]
        report[f"positive_{name}"] = accs
        report[f"shuffled_{name}"] = shuf
        lines.append(f"\nPositive control at {name} (P), test n={te.sum()} (held-out templates and cities), chance band [{lo:.2f}, {hi:.2f}]:")
        lines.append("  real labels:     " + " ".join(f"L{l}:{a:.2f}" for l, a in enumerate(accs)))
        lines.append("  shuffled labels: " + " ".join(f"L{l}:{a:.2f}" for l, a in enumerate(shuf)))

    (OUT / "leakage_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "leakage.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
