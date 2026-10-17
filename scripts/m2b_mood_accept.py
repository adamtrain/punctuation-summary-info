"""Milestone 2b: acceptance test for the mood property (pre-registered criteria in BRIEF.md section 11.2).

For each variant (mixed, consistent): 100 pairs per filler length (500 pairs), conditions P, U, U-cap, M, K, C_mid,
continuation readout, knockouts none / s0words / s0all / punctonly (punctonly only for punctuated conditions).
Reports pair-level logit diff (great - terrible for the true label, mean of the pair's two members) with 95% bootstrap
CIs, pair accuracy, member-level AUC, and the share of P's margin retained under the s0words knockout, then the
verdict on criteria C1 (pair acc >= 0.90 and AUC >= 0.80) and C2 (s0words retains >= 25%).
Inputs: outputs/m2b/adj_screen.jsonl and filler_screen.jsonl (from scripts/m2b_mood_screen.py).
Writes outputs/m2b/pairs_{variant}.jsonl, accept.jsonl, accept_report.txt, accept_summary.json.
"""

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from punctsum import PROJECT_ROOT
from punctsum.behavior import pair_bootstrap, passage_label_pairs, score
from punctsum.dataset import FILLER_LENGTHS, load_jsonl, save_jsonl
from punctsum.model import load, peak_rss_gb
from punctsum.mood import make_mood_pairs, neutral_fillers

SEED = 0
PER_K = 100
CONDS = ["P", "U", "Ucap", "M", "K", "C_mid"]
PUNCTUATED = ["P", "M", "K", "C_mid"]
KNOCKOUTS = ["none", "s0words", "s0all", "punctonly"]
OUT = PROJECT_ROOT / "outputs" / "m2b"


def ratio_ci(num: dict[str, float], den: dict[str, float], n_boot: int = 2000, seed: int = 0):
    """Ratio of means over pairs, with a 95% bootstrap CI resampling pairs."""
    ids = sorted(num)
    a, b = np.array([num[i] for i in ids]), np.array([den[i] for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boots = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    t0 = time.time()
    model = load("gpt2")
    adj = load_jsonl(OUT / "adj_screen.jsonl")
    fillers = neutral_fillers(load_jsonl(OUT / "filler_screen.jsonl"))
    rows = []
    for variant in ("mixed", "consistent"):
        pairs = make_mood_pairs(adj, fillers, PER_K, SEED, variant)
        save_jsonl(pairs, OUT / f"pairs_{variant}.jsonl")
        for ko in KNOCKOUTS:
            items = passage_label_pairs(pairs, PUNCTUATED if ko == "punctonly" else CONDS)
            for (p, lab, c), r in zip(items, score(model, items, readout="continuation", knockout=ko)):
                rows.append({"variant": variant, "knockout": ko, "cond": c, "pair_id": p["pair_id"], "k": p["k"],
                             "split": p["split"], "label": lab, "diff": r["diff"]})
            print(f"done {variant}/{ko} at {time.time() - t0:.0f}s", flush=True)
    save_jsonl(rows, OUT / "accept.jsonl")

    pv: dict[tuple, list[float]] = {}
    member: dict[tuple, list[tuple[int, float]]] = {}
    for r in rows:
        pv.setdefault((r["variant"], r["knockout"], r["cond"], r["pair_id"], r["k"]), []).append(r["diff"])
        s = r["diff"] if r["label"] == "good" else -r["diff"]  # great - terrible, label-independent score
        member.setdefault((r["variant"], r["knockout"], r["cond"], r["k"]), []).append((int(r["label"] == "good"), s))

    def pair_vals(variant, ko, cond, ks):
        return {pid: sum(v) / len(v) for (v_, k_, c_, pid, kk), v in pv.items()
                if v_ == variant and k_ == ko and c_ == cond and kk in ks}

    def auc(variant, ko, cond, ks):
        m = [x for kk in ks for x in member.get((variant, ko, cond, kk), [])]
        return float(roc_auc_score([y for y, _ in m], [s for _, s in m])) if m else float("nan")

    summary, lines = {}, ["Mood acceptance test, continuation readout. Cells: pair-level logit diff [95% CI], pair acc, member AUC.\n"]
    for variant in ("mixed", "consistent"):
        for ks, kname in [((k,), f"k={k}") for k in FILLER_LENGTHS] + [(tuple(FILLER_LENGTHS), "all")]:
            lines.append(f"== {variant}, {kname} ==")
            lines.append(f"  {'cond':6} " + "".join(f"{ko:>33}" for ko in KNOCKOUTS))
            for c in CONDS:
                cells = []
                for ko in KNOCKOUTS:
                    vals = pair_vals(variant, ko, c, ks)
                    if not vals:
                        cells.append(f"{'—':>33}")
                        continue
                    m, lo, hi = pair_bootstrap(vals)
                    acc = sum(v > 0 for v in vals.values()) / len(vals)
                    a = auc(variant, ko, c, ks)
                    summary[f"{variant}|{kname}|{c}|{ko}"] = {"diff": [m, lo, hi], "pair_acc": acc, "auc": a}
                    cells.append(f"{m:+6.2f} [{lo:+.2f},{hi:+.2f}] {acc:.2f} {a:.2f}".rjust(33))
                lines.append(f"  {c:6} " + "".join(cells))
            ret = {c: ratio_ci(pair_vals(variant, "s0words", c, ks), pair_vals(variant, "none", c, ks)) for c in CONDS}
            for c, (r, lo, hi) in ret.items():
                summary[f"{variant}|{kname}|{c}|retained_s0words"] = [r, lo, hi]
            lines.append("  s0words retention (share of no-knockout margin): "
                         + ", ".join(f"{c} {r:.2f} [{lo:.2f},{hi:.2f}]" for c, (r, lo, hi) in ret.items()))
            lines.append("")

    lines.append("== Pre-registered acceptance (P, continuation readout, pooled over k) ==")
    for variant in ("mixed", "consistent"):
        s = summary[f"{variant}|all|P|none"]
        r = summary[f"{variant}|all|P|retained_s0words"]
        c1 = s["pair_acc"] >= 0.90 and s["auc"] >= 0.80
        c2 = r[0] >= 0.25
        lines.append(f"  {variant:10}: pair acc {s['pair_acc']:.3f}, AUC {s['auc']:.3f} -> C1 {'PASS' if c1 else 'FAIL'}; "
                     f"s0words retains {r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}] -> C2 {'PASS' if c2 else 'FAIL'}")
        summary[f"{variant}|verdict"] = {"C1": c1, "C2": c2}
    lines.append(f"\nRuntime {time.time() - t0:.0f} s; peak RSS {peak_rss_gb():.2f} GB")
    (OUT / "accept_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "accept_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
