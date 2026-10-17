"""Milestone 2b: screen the mood property's ingredients on GPT-2 small.

1. Adjectives: valence of each candidate (" great" - " terrible" logit diff after "The {noun} was {adj}. Overall,
   it was", averaged over 5 nouns), relative to neutral adjectives. Kept if single-token and inside the strength
   band (punctsum.mood.ADJ_BAND) in the right direction.
2. Fillers: valence of each filler sentence alone; the 64 closest to the median are kept for mood passages,
   so filler doesn't push the readout toward good or bad.
3. Readouts: candidate question and continuation readouts on 100 consistent-valence and 100 mixed pairs
   (20 per filler length), conditions P and U; pair-level logit diff and pair accuracy.
Writes outputs/m2b/screen_report.txt, adj_screen.jsonl, filler_screen.jsonl.
"""

import statistics

from punctsum import PROJECT_ROOT
from punctsum.behavior import passage_label_pairs, score
from punctsum.dataset import save_jsonl
from punctsum.model import load
from punctsum.mood import make_mood_pairs, neutral_fillers, screen_adjectives, screen_fillers

SEED = 0
OUT = PROJECT_ROOT / "outputs" / "m2b"
READOUT_CANDIDATES = [
    "\n\nQuestion: Overall, was it good or bad?\nAnswer:",
    "\n\nQuestion: Was the experience good or bad?\nAnswer:",
    " and overall it was",
    " and all in all it was",
    " and overall the whole experience was",
    " so overall it was",
]


def main():
    model = load("gpt2")
    lines = []
    adj = screen_adjectives(model)
    save_jsonl(adj, OUT / "adj_screen.jsonl")
    lines.append(f"== 1. Adjectives: great-terrible logit diff relative to neutral adjectives (neutral mean {adj[0]['ref']:+.2f}) ==")
    for lab in ("good", "bad"):
        rows = sorted((r for r in adj if r["label"] == lab), key=lambda r: -abs(r["rel"]))
        kept = [r for r in rows if r["keep"]]
        mean_kept = statistics.mean(abs(r["rel"]) for r in kept) if kept else float("nan")
        lines.append(f"{lab}: kept {len(kept)}/{len(rows)}, mean |rel| of kept {mean_kept:.2f}")
        lines.append("  kept:    " + ", ".join(f"{r['adj']}({r['rel']:+.1f})" for r in kept))
        lines.append("  dropped: " + ", ".join(f"{r['adj']}({r['rel']:+.1f}{'' if r['single_token'] else ',multi-tok'})" for r in rows if not r["keep"]))

    fil = screen_fillers(model)
    save_jsonl(fil, OUT / "filler_screen.jsonl")
    keep = neutral_fillers(fil)
    diffs = sorted(r["diff"] for r in fil)
    lines.append(f"\n== 2. Fillers: valence range {diffs[0]:+.2f} .. {diffs[-1]:+.2f}, median {diffs[len(diffs) // 2]:+.2f}; kept {len(keep)} closest to median ==")
    dropped = sorted((r for r in fil if r["filler"] not in keep), key=lambda r: r["diff"])
    lines.append("  dropped: " + " | ".join(f"{r['text']} ({r['diff']:+.2f})" for r in dropped))

    lines.append("\n== 3. Readouts (answers great/terrible): pair-level logit diff (mean of members' true-minus-other diffs), pair and member accuracy ==")
    for variant in ("consistent", "mixed"):
        pairs = make_mood_pairs(adj, keep, 20, SEED, variant)
        pairs = [p for p in pairs if p["split"] == "train"]
        items = passage_label_pairs(pairs, ["P", "U"])
        lines.append(f"-- {variant} ({len(pairs)} train pairs) --")
        for ro in READOUT_CANDIDATES:
            res = score(model, items, readout=ro)
            pv: dict[tuple[str, str], list[float]] = {}
            member_correct = {"P": [], "U": []}
            for (p, _, c), r in zip(items, res):
                pv.setdefault((c, p["pair_id"]), []).append(r["diff"])
                member_correct[c].append(r["diff"] > 0)
            cells = []
            for c in ("P", "U"):
                vals = [statistics.mean(v) for (cc, _), v in pv.items() if cc == c]
                cells.append(f"{c}: {statistics.mean(vals):+.2f} pair-acc {sum(v > 0 for v in vals) / len(vals):.2f} "
                             f"member-acc {sum(member_correct[c]) / len(member_correct[c]):.2f}")
            lines.append(f"  {ro!r:58} " + " | ".join(cells))
    (OUT / "screen_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
