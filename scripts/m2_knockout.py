"""Milestone 2 review: attention knockout x readout x condition (BRIEF.md section 4.7, risks 6.1 and 6.4).

Does the label reach the readout only by looking straight back at the city, or through later positions
(summary points)? Readout tokens are blocked from attending to parts of the passage at every layer:
  none        baseline
  s0words     the property sentence's words (city included); its punctuation stays visible
  s0all       the whole property sentence
  punctonly   every passage word; only BOS and punctuation stay visible (punctuated conditions only)
Run with both readouts: the question readout and the separator-free continuation readout.
100 pairs per filler length (500 pairs, 1,000 renderings per cell). Metric: pair-level logit diff (mean of
the two members, which cancels label-independent bias), with 95% bootstrap CIs over pairs.
Writes outputs/m2/knockout.jsonl, knockout_report.txt, knockout_summary.json.
"""

import json
import time

from punctsum import PROJECT_ROOT
from punctsum.behavior import pair_bootstrap, passage_label_pairs, score
from punctsum.dataset import FILLER_LENGTHS, load_jsonl, save_jsonl
from punctsum.materials import READOUT, READOUT_CONT
from punctsum.model import load, peak_rss_gb

PER_K = 100
CONDS = ["P", "U", "Ucap", "M", "K", "C_mid"]
PUNCTUATED = ["P", "M", "K", "C_mid"]
READOUTS = {"question": READOUT, "continuation": READOUT_CONT}
OUT = PROJECT_ROOT / "outputs" / "m2"


def main():
    t0 = time.time()
    model = load("gpt2")
    pairs = load_jsonl(OUT / "pairs.jsonl")
    sub = [p for k in FILLER_LENGTHS for p in [q for q in pairs if q["k"] == k][:PER_K]]
    k_of = {p["pair_id"]: p["k"] for p in sub}
    rows = []
    for ro_name, ro in READOUTS.items():
        for ko in ["none", "s0words", "s0all", "punctonly"]:
            conds = PUNCTUATED if ko == "punctonly" else CONDS
            items = passage_label_pairs(sub, conds)
            for (p, lab, c), r in zip(items, score(model, items, readout=ro, knockout=ko)):
                rows.append({"readout": ro_name, "knockout": ko, "cond": c, "pair_id": p["pair_id"], "k": p["k"],
                             "label": lab, "diff": r["diff"]})
            print(f"done {ro_name}/{ko} at {time.time() - t0:.0f}s", flush=True)
    save_jsonl(rows, OUT / "knockout.jsonl")

    pv: dict[tuple, list[float]] = {}
    for r in rows:
        pv.setdefault((r["readout"], r["knockout"], r["cond"], r["pair_id"]), []).append(r["diff"])
    pair_mean = {key: sum(v) / len(v) for key, v in pv.items()}

    summary, lines = {}, [f"{len(sub)} pairs; pair-level logit diff [95% CI]; acc = share of pairs where the true country wins.\n"]
    for ro_name in READOUTS:
        for ks, kname in [((k,), f"k={k}") for k in FILLER_LENGTHS] + [(tuple(FILLER_LENGTHS), "all")]:
            lines.append(f"== readout={ro_name}, {kname} ==")
            lines.append(f"  {'cond':6} " + "".join(f"{ko:>27}" for ko in ["none", "s0words", "s0all", "punctonly"]))
            for c in CONDS:
                cells = []
                for ko in ["none", "s0words", "s0all", "punctonly"]:
                    vals = {pid: v for (r_, k_, c_, pid), v in pair_mean.items()
                            if r_ == ro_name and k_ == ko and c_ == c and k_of[pid] in ks}
                    if not vals:
                        cells.append(f"{'—':>27}")
                        continue
                    m, lo, hi = pair_bootstrap(vals)
                    acc = sum(v > 0 for v in vals.values()) / len(vals)
                    summary[f"{ro_name}|{kname}|{c}|{ko}"] = {"diff": [m, lo, hi], "acc": acc}
                    cells.append(f"{m:+6.2f} [{lo:+.2f},{hi:+.2f}] {acc:.2f}".rjust(27))
                lines.append(f"  {c:6} " + "".join(cells))
            lines.append("")
    lines.append(f"Runtime {time.time() - t0:.0f} s; peak RSS {peak_rss_gb():.2f} GB")
    (OUT / "knockout_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "knockout_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
