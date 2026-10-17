"""Milestone 2 review: choose a separator-free continuation readout (BRIEF.md risk 6.4).

The question readout ("\n\nQuestion: Which country was this in?\nAnswer:") hands the unpunctuated condition a
newline separator right after the passage. A continuation readout carries on in words straight into the answer,
so no condition gets separators from the readout. Each candidate is appended identically in every condition.
Screened on 100 training pairs (20 per filler length) in P and U; reports the pair-level mean logit diff
(average of the two members, which cancels any label-independent bias) and pair accuracy.
Writes outputs/m2/readout_screen.txt.
"""

import statistics

from punctsum import PROJECT_ROOT
from punctsum.behavior import passage_label_pairs, score
from punctsum.dataset import FILLER_LENGTHS, load_jsonl
from punctsum.materials import READOUT
from punctsum.model import load

PER_K = 20
CANDIDATES = [
    READOUT,
    " and all of this happened in the country of",
    " and all of this was in the country of",
    " which all took place in the country of",
    " all of it happened in the country of",
    " and the country where all of this happened is",
]
OUT = PROJECT_ROOT / "outputs" / "m2"


def main():
    model = load("gpt2")
    pairs = load_jsonl(OUT / "pairs.jsonl")
    sub = [p for k in FILLER_LENGTHS for p in [q for q in pairs if q["k"] == k and q["split"] == "train"][:PER_K]]
    items = passage_label_pairs(sub, ["P", "U"])
    lines = [f"{len(sub)} training pairs; pair-level mean logit diff (true - other) and pair accuracy\n"]
    for ro in CANDIDATES:
        res = score(model, items, readout=ro)
        pair_vals: dict[tuple[str, str], list[float]] = {}
        for (p, _, c), r in zip(items, res):
            pair_vals.setdefault((c, p["pair_id"]), []).append(r["diff"])
        cells = []
        for c in ("P", "U"):
            vals = [statistics.mean(v) for (cc, _), v in pair_vals.items() if cc == c]
            cells.append(f"{c}: {statistics.mean(vals):+.2f} (acc {sum(v > 0 for v in vals) / len(vals):.2f})")
        lines.append(f"{ro!r:52} " + "   ".join(cells))
    (OUT / "readout_screen.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
