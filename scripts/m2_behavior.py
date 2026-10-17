"""Milestone 2: behavioral baseline on the full dataset, with bootstrap confidence intervals.

For every pair, label and condition (2,000 pairs x 2 labels x 8 conditions = 32,000 renderings): the
readout logit diff (true minus other country) and the word-NLL unusualness. Confidence intervals resample
pairs (2,000 bootstrap draws, 95% percentile intervals); differences from P are paired by pair.
Writes outputs/m2/behavior.jsonl (one row per rendering), outputs/m2/behavior_report.txt and
outputs/m2/behavior_summary.json.
"""

import json
import time

from punctsum import PROJECT_ROOT
from punctsum.behavior import pair_bootstrap, passage_label_pairs, score
from punctsum.dataset import CONDITIONS, FILLER_LENGTHS, load_jsonl, save_jsonl
from punctsum.model import load, peak_rss_gb

OUT = PROJECT_ROOT / "outputs" / "m2"


def main():
    t0 = time.time()
    model = load("gpt2")
    pairs = load_jsonl(OUT / "pairs.jsonl")
    items = passage_label_pairs(pairs, CONDITIONS)
    results = score(model, items)
    rows = [
        {"pair_id": p["pair_id"], "k": p["k"], "split": p["split"], "label": lab, "cond": c, **r}
        for (p, lab, c), r in zip(items, results)
    ]
    save_jsonl(rows, OUT / "behavior.jsonl")

    # Pair-level values: the mean of the pair's two members.
    pv: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        pv.setdefault((r["cond"], r["pair_id"]), {"diff": [], "word_nll": [], "correct": []})
        cell = pv[(r["cond"], r["pair_id"])]
        cell["diff"].append(r["diff"])
        cell["word_nll"].append(r["word_nll"])
        cell["correct"].append(float(r["diff"] > 0))
    k_of = {p["pair_id"]: p["k"] for p in pairs}

    def values(cond, metric, ks):
        return {pid: sum(v[metric]) / len(v[metric]) for (c, pid), v in pv.items() if c == cond and k_of[pid] in ks}

    summary = {}
    lines = [f"Behavior on {len(pairs)} pairs x 2 labels x {len(CONDITIONS)} conditions = {len(rows)} renderings.",
             "Cells: mean logit diff [95% CI]; Δ = difference from P, paired by pair [95% CI]; acc = share correct;",
             "×U = word-NLL increase over P as a multiple of U's (same k). 400 pairs per k; 2,000 for 'all'.\n"]
    for ks, name in [((k,), f"k={k}") for k in FILLER_LENGTHS] + [(tuple(FILLER_LENGTHS), "all")]:
        base = values("P", "diff", ks)
        nll_p = values("P", "word_nll", ks)
        u_delta = sum(values("U", "word_nll", ks)[pid] - nll_p[pid] for pid in nll_p) / len(nll_p)
        lines.append(f"== {name} ==")
        lines.append(f"  {'cond':6} {'logit diff [95% CI]':>24} {'Δ vs P [95% CI]':>26} {'acc':>5} {'×U':>5}")
        for cond in CONDITIONS:
            d = values(cond, "diff", ks)
            m, lo, hi = pair_bootstrap(d)
            dm, dlo, dhi = pair_bootstrap({pid: d[pid] - base[pid] for pid in d})
            acc = sum(values(cond, "correct", ks).values()) / len(d)
            nll = values(cond, "word_nll", ks)
            xu = (sum(nll[pid] - nll_p[pid] for pid in nll) / len(nll)) / u_delta
            summary[f"{name}|{cond}"] = {"diff": [m, lo, hi], "delta": [dm, dlo, dhi], "acc": acc, "xU": xu}
            delta = "" if cond == "P" else f"{dm:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
            lines.append(f"  {cond:6} {m:+8.2f} [{lo:+.2f}, {hi:+.2f}]   {delta:>24} {acc:5.3f} {xu:5.2f}")
        lines.append("")
    lines.append(f"Runtime {time.time() - t0:.0f} s; peak RSS {peak_rss_gb():.2f} GB")
    (OUT / "behavior_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "behavior_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
