"""Milestone 2: calibrate control C's substitution rate r so its unusualness is 0.5x, 1x and 1.5x U's.

Unusualness = increase in word-token NLL over P. Uses training pairs only (40 per filler length, 200
pairs, 400 passages per condition), sweeps a grid of r, and linearly interpolates between grid points.
Prints the chosen rates; copy them into punctsum/dataset.py C_STRENGTHS.
Writes outputs/m2/calibration.txt and outputs/m2/calibration.json.
"""

import json
import statistics

import numpy as np

from punctsum import PROJECT_ROOT
from punctsum.behavior import passage_label_pairs, score
from punctsum.dataset import FILLER_LENGTHS, load_jsonl
from punctsum.model import load

PER_K = 40
GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26]
TARGETS = {"C_lo": 0.5, "C_mid": 1.0, "C_hi": 1.5}
OUT = PROJECT_ROOT / "outputs" / "m2"


def main():
    model = load("gpt2")
    pairs = load_jsonl(OUT / "pairs.jsonl")
    calib = [p for k in FILLER_LENGTHS for p in [q for q in pairs if q["k"] == k and q["split"] == "train"][:PER_K]]
    conds = ["P", "U"] + [f"C@{r}" for r in GRID]
    items = passage_label_pairs(calib, conds)
    results = score(model, items)

    by = {}
    for (pair, _, cond), r in zip(items, results):
        by.setdefault(cond, []).append(r)
        by.setdefault((cond, pair["k"]), []).append(r)
    mean = lambda key: statistics.mean(r["word_nll"] for r in by[key])  # noqa: E731
    base, target = mean("P"), mean("U") - mean("P")
    deltas = np.array([mean(f"C@{r}") - base for r in GRID])

    lines = [f"Calibration on {len(calib)} training pairs ({len(items)} renderings). Word-NLL increase over P:"]
    lines.append(f"  U: {target:+.3f}  (per k: " + ", ".join(f"k={k} {mean(('U', k)) - mean(('P', k)):+.3f}" for k in FILLER_LENGTHS) + ")")
    for r, d in zip(GRID, deltas):
        per_k = ", ".join(f"{mean((f'C@{r}', k)) - mean(('P', k)):+.3f}" for k in FILLER_LENGTHS)
        lines.append(f"  C r={r:<5} {d:+.3f}  = {d / target:4.2f}x U   (per k: {per_k})")
    assert np.all(np.diff(deltas) > 0), "unusualness should rise with r; check the grid"
    chosen = {name: round(float(np.interp(mult * target, deltas, GRID)), 3) for name, mult in TARGETS.items()}
    lines.append("\nChosen rates (linear interpolation): " + ", ".join(f"{n}={r} ({TARGETS[n]}x U)" for n, r in chosen.items()))
    (OUT / "calibration.txt").write_text("\n".join(lines) + "\n")
    (OUT / "calibration.json").write_text(json.dumps({
        "grid": GRID, "deltas": deltas.tolist(), "u_delta": target, "chosen": chosen, "pairs": len(calib),
    }, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
