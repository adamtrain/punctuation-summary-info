"""Post-hoc analyses for the write-up (BRIEF.md Section 10.17), computed from each model's saved outputs; nothing is
rerun on a model. Writes outputs/models/posthoc.md (the tables used in the brief) and posthoc.json.

  python scripts/m6_posthoc.py

1. Long-distance mood margins (no knockout) for every condition, and each condition's change from P with a
   pair-bootstrap CI (from long_knockout.jsonl).
2. Mood patching per model: the best single position and the whole post-property passage, P vs U vs U-cap, at the
   best layer (from patching_summary.json), plus the two words that follow s0's commas (from the patching CSVs).
3. The punctuation-only knockout (the readout sees only BOS and punctuation) for P and M, by model, as a margin and as
   a fraction of the no-knockout margin (from knockout_summary.json).
"""

import csv
import json

from punctsum import PROJECT_ROOT
from punctsum.behavior import pair_bootstrap

OUT = PROJECT_ROOT / "outputs" / "models"
MODELS = ["gpt2", "gpt-neo-125m", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "llama-3.1-8b"]
LONG_CONDS = ["P", "U", "Ucap", "M", "K", "C_mid"]
KS = (1, 4, 16)


def long_mood(model):
    """{(level, cond): {pair_id: margin}} for mood with no knockout; a pair's margin is the mean over its two labels."""
    acc: dict[tuple, dict[str, list[float]]] = {}
    for line in (OUT / model / "long_knockout.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["group"].startswith("mood") and r["knockout"] == "none":
            acc.setdefault((r["k"], r["cond"]), {}).setdefault(r["pair_id"], []).append(r["diff"])
    return {key: {p: sum(v) / len(v) for p, v in pairs.items()} for key, pairs in acc.items()}


def post_comma(model, k, cond):
    """Best-layer recovery of the words right after s0's first and second comma (or where they were)."""
    best = {}
    for r in csv.DictReader((OUT / model / "patching" / f"patching_mood_k{k}_{cond}.csv").open()):
        if r["row"] in ("s0.post_comma1", "s0.post_comma2"):
            best[r["row"]] = max(best.get(r["row"], 0.0), float(r["recovery"]))
    return best


def main():
    md, js = [], {"long_mood": {}, "patching": {}, "punctonly": {}}

    md += ["## 1. Long-distance mood margins (no knockout)", "",
           "Margin per condition; Δ = condition − P per pair, mean [95% pair-bootstrap CI]. 100 pairs per level.", ""]
    md += ["| model | level | P | U | U-cap | M | K | C_mid | U − P | M − P | K − P | C_mid − P |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        data = long_mood(m)
        for level in ("L900", "L1500"):
            if (level, "P") not in data:
                continue
            p = data[(level, "P")]
            means = {c: sum(data[(level, c)].values()) / len(data[(level, c)]) for c in LONG_CONDS if (level, c) in data}
            deltas = {}
            for c in ("U", "M", "K", "C_mid"):
                d = data[(level, c)]
                deltas[c] = pair_bootstrap({i: d[i] - p[i] for i in p if i in d})
            js["long_mood"][f"{m}|{level}"] = {"margin": means, "delta_vs_P": deltas}
            md.append(f"| {m} | {level} | " + " | ".join(f"{means[c]:.2f}" for c in LONG_CONDS) + " | "
                      + " | ".join(f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for v in deltas.values()) + " |")

    md += ["", "## 2. Mood patching: best single position and whole post-property passage (best layer)", "",
           "Recovery (fraction of the clean − corrupted gap). best = best single position after the property words "
           "(row name in brackets); passage = every passage position after the property words, patched together.", ""]
    md += ["| model | k | P best | U best | U-cap best | P passage | U passage | U-cap passage |",
           "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        ps = json.loads((OUT / m / "patching_summary.json").read_text())
        for k in KS:
            row = {c: ps[f"mood_k{k}_{c}"] for c in ("P", "U", "Ucap")}
            js["patching"][f"{m}|k={k}"] = row
            md.append(f"| {m} | {k} | " + " | ".join(
                f"{row[c]['best_position'][0]:.3f} ({row[c]['best_position'][3]})" for c in ("P", "U", "Ucap"))
                + " | " + " | ".join(f"{row[c]['passage_after'][0]:.3f}" for c in ("P", "U", "Ucap")) + " |")

    md += ["", "### Words after s0's commas (best-layer recovery, k = 4)", "",
           "| model | post_comma1 P | post_comma1 U | post_comma2 P | post_comma2 U |", "|---|---|---|---|---|"]
    for m in MODELS:
        p, u = post_comma(m, 4, "P"), post_comma(m, 4, "U")
        js["patching"][f"{m}|post_comma|k=4"] = {"P": p, "U": u}
        md.append(f"| {m} | {p.get('s0.post_comma1', 0):.3f} | {u.get('s0.post_comma1', 0):.3f} | "
                  f"{p.get('s0.post_comma2', 0):.3f} | {u.get('s0.post_comma2', 0):.3f} |")

    md += ["", "## 3. Punctuation-only knockout, mood (pooled over k = 1, 4, 16)", "",
           "Margin when the readout can see only BOS and punctuation, and that as a fraction of the condition's own "
           "no-knockout margin.", "", "| model | P | M | P fraction | M fraction |", "|---|---|---|---|---|"]
    for m in MODELS:
        ko = json.loads((OUT / m / "knockout_summary.json").read_text())
        row = {c: {"punctonly": ko[f"mood|all|{c}"]["punctonly"], "none": ko[f"mood|all|{c}"]["none"]} for c in ("P", "M")}
        js["punctonly"][m] = row
        po = {c: row[c]["punctonly"][0] for c in row}
        md.append(f"| {m} | {po['P']:.2f} | {po['M']:.2f} | {po['P'] / row['P']['none'][0]:.2f} | "
                  f"{po['M'] / row['M']['none'][0]:.2f} |")

    (OUT / "posthoc.md").write_text("\n".join(md) + "\n")
    (OUT / "posthoc.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
