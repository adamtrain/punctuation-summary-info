"""Milestone 2: screen cities, generate France/Japan pairs, and check rendering and role labels.

  python scripts/m2_generate.py            # small sample (8 pairs per filler length) + examples report
  python scripts/m2_generate.py --n 400    # full dataset (written to outputs/m2/pairs.jsonl)

Every rendering of every pair in every condition is tokenized and labelled, which checks that tokens map
cleanly to words, the readout doesn't change passage tokenization, and both members of a pair have
identical token counts and roles (required for patching).
"""

import argparse
import collections
import statistics

import torch

from punctsum import PROJECT_ROOT
from punctsum.conditions import to_text
from punctsum.dataset import (
    CONDITIONS,
    city_buckets,
    FILLER_LENGTHS,
    make_pairs,
    passage_for,
    render_condition,
    save_jsonl,
    screen_cities,
    split_indices,
)
from punctsum.materials import FILLERS, LABELS, READOUT, S0_TEMPLATES
from punctsum.model import load, peak_rss_gb, tokenize_with_readout
from punctsum.roles import label_tokens

SEED = 0
OUT = PROJECT_ROOT / "outputs" / "m2"


def labelled(model, pair, label, cond):
    pieces = render_condition(pair, label, cond, model)
    text, toks, piece_of_token, n_readout = tokenize_with_readout(model, pieces, READOUT)
    rows = label_tokens(passage_for(pair, label), pieces, piece_of_token, n_readout)
    return text, toks[0].tolist(), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="pairs per filler length")
    args = ap.parse_args()
    sample = args.n <= 20
    model = load("gpt2")
    lines: list[str] = []
    log = lines.append

    # 1. City screen.
    screen = screen_cities(model)
    save_jsonl(screen, OUT / "city_screen.jsonl")
    log("== 1. City screen (knowledge prompt logit diff, threshold 2.0) ==")
    for label in LABELS:
        rows = sorted((r for r in screen if r["label"] == label), key=lambda r: -r["diff"])
        kept = [f"{r['city']}({r['ntok']}t,{r['diff']:+.1f})" for r in rows if r["keep"]]
        dropped = [f"{r['city']}({r['diff']:+.1f})" for r in rows if not r["keep"]]
        log(f"{label}: kept {len(kept)}/{len(rows)}\n  kept:    {', '.join(kept)}\n  dropped: {', '.join(dropped)}")
    for split, bs in city_buckets(screen, SEED + 2).items():
        log(f"{split} cities by token count: " + "; ".join(
            f"{n} tok: " + " | ".join(f"{lab} {', '.join(cs)}" for lab, cs in b.items()) for n, b in sorted(bs.items())
        ))

    # 2. Template split.
    s0_split, f_split = split_indices(S0_TEMPLATES, SEED), split_indices(FILLERS, SEED + 1)
    log("\n== 2. Template split (stratified by clause count) ==")
    log(f"s0 templates: {sum(v == 'train' for v in s0_split.values())} train / {sum(v == 'test' for v in s0_split.values())} test")
    log(f"fillers:      {sum(v == 'train' for v in f_split.values())} train / {sum(v == 'test' for v in f_split.values())} test")

    # 3. Generate pairs and check every rendering.
    pairs = make_pairs(screen, args.n, SEED)
    save_jsonl(pairs, OUT / ("pairs_sample.jsonl" if sample else "pairs.jsonl"))
    lengths = collections.defaultdict(list)
    diffs = collections.defaultdict(list)  # sample mode: logit diff (true label - other) at the readout
    answer = {lab: model.to_single_token(" " + lab) for lab in LABELS}
    n_checked = 0
    for pair in pairs:
        for cond in CONDITIONS:
            (_, ta, ra), (_, tb, rb) = rendered = [labelled(model, pair, lab, cond) for lab in LABELS]
            if len(ta) != len(tb) or [r["role"] for r in ra] != [r["role"] for r in rb]:
                raise AssertionError(f"pair members differ in {cond}: {pair['pair_id']}")
            lengths[(pair["k"], cond)].append(len(ta))
            n_checked += 2
            if sample:
                for lab, (_, toks, _) in zip(LABELS, rendered):
                    other = next(x for x in LABELS if x != lab)
                    with torch.no_grad():
                        logits = model(torch.tensor([toks], device=model.cfg.device))[0, -1]
                    diffs[(pair["k"], cond)].append((logits[answer[lab]] - logits[answer[other]]).item())
    log(f"\n== 3. Generated {len(pairs)} pairs ({len(pairs) * 2} passages); checked {n_checked} renderings ==")
    log("All renderings tokenize cleanly; pair members match in token count and roles in every condition.")
    log("Split: " + ", ".join(f"{s}={sum(p['split'] == s for p in pairs)} pairs" for s in ("train", "test")))
    log("City token counts used: " + str(dict(collections.Counter(p["city_ntok"] for p in pairs))))
    log("\nMean tokens (passage + 13 readout + BOS) by filler length:")
    log(f"  {'k':>3} " + " ".join(f"{c:>6}" for c in CONDITIONS))
    for k in FILLER_LENGTHS:
        log(f"  {k:>3} " + " ".join(f"{statistics.mean(lengths[(k, c)]):6.1f}" for c in CONDITIONS))
    if sample:
        log("\nBehavior: mean logit diff (true country - other) at the readout, 16 passages per cell:")
        log(f"  {'k':>3} " + " ".join(f"{c:>6}" for c in CONDITIONS))
        for k in FILLER_LENGTHS:
            log(f"  {k:>3} " + " ".join(f"{statistics.mean(diffs[(k, c)]):+6.2f}" for c in CONDITIONS))
        log("  share correct (diff > 0) by k, P only: " + ", ".join(
            f"k={k}: {sum(d > 0 for d in diffs[(k, 'P')]) / len(diffs[(k, 'P')]):.2f}" for k in FILLER_LENGTHS))

    if sample:
        # 4. One pair in every condition.
        pair = next(p for p in pairs if p["k"] == 2)
        log(f"\n== 4. Example pair {pair['pair_id']} (split={pair['split']}, cities {pair['cities']}) ==")
        for cond in CONDITIONS:
            for label in LABELS:
                text = to_text(render_condition(pair, label, cond, model))[0]
                log(f"  [{cond:5}|{label[:2]}] {text}")

        # 5. Role tables.
        for cond in ["P", "U", "M", "K"]:
            text, toks, rows = labelled(model, pair, LABELS[0], cond)
            log(f"\n== 5. Role table: {cond}, {LABELS[0]} ==")
            log(f"  {'t':>3} {'token':16} {'role':24} {'wf':2} {'aft':3} flags")
            for t, (tok, r) in enumerate(zip(toks, rows)):
                log(f"  {t:>3} {model.to_string(tok)!r:16} {r['role']:24} {'y' if r['word_final'] else '-':2} "
                    f"{'y' if r['after_city'] else '-':3} {','.join(r['flags'])}")

        # 6. Short-length and long-length examples in P.
        log("\n== 6. P examples at each filler length ==")
        for k in FILLER_LENGTHS:
            pair = next(p for p in pairs if p["k"] == k)
            log(f"  k={k:<2} {to_text(render_condition(pair, LABELS[1], 'P', model))[0]}")

    log(f"\nPeak RSS: {peak_rss_gb():.2f} GB")
    (OUT / ("sample_report.txt" if sample else "report.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
