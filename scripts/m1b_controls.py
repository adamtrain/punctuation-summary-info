"""Milestone 1b: calibrate control C (word substitution), and pilot conditions M (misplaced
punctuation) and K (restart capitals) on GPT-2 small.

"Unusualness" is measured as mean next-token NLL over *word* tokens only. Removing punctuation removes
easy-to-predict tokens, which inflates an all-token average without making the words more surprising.
Writes outputs/m1b/report.txt and outputs/m1b/results.json.
"""

import json
import random
import statistics

import torch

from punctsum import PROJECT_ROOT
from punctsum.conditions import Substitution, render
from punctsum.model import load, peak_rss_gb, single_token_word, tokenize_pieces
from punctsum.pilot import NEUTRAL_POOL, READOUT, STATES, all_passages

SEED = 0
DRAWS = 5  # random draws per passage for the stochastic conditions (C, M, K)
SUB_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
RESTARTS = [1, 2, 3]
OUT = PROJECT_ROOT / "outputs" / "m1b"


@torch.no_grad()
def measure(model, pieces, state: str) -> dict:
    text, toks, piece_of_token = tokenize_pieces(model, pieces)
    assert torch.equal(toks, model.to_tokens(text)), f"alignment tokens differ from to_tokens: {text!r}"
    logprobs = model(toks)[0, :-1].log_softmax(-1)
    nll = -logprobs.gather(-1, toks[0, 1:, None])[:, 0]
    is_word = torch.tensor([pieces[i].kind == "word" for i in piece_of_token[1:]])
    other = [s for s in STATES if s != state][0]
    final = model(model.to_tokens(text + READOUT))[0, -1]
    diff = final[model.to_single_token(" " + state)] - final[model.to_single_token(" " + other)]
    return {
        "word_nll": nll[is_word].mean().item(),
        "all_nll": nll.mean().item(),
        "total_nll": nll.sum().item(),
        "tokens": toks.shape[1],
        "diff": diff.item(),
    }


def summarize(rows: list[dict]) -> dict:
    out = {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
    out["acc"] = sum(r["diff"] > 0 for r in rows) / len(rows)
    out["n"] = len(rows)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = load("gpt2")
    pool = [w for w in NEUTRAL_POOL if single_token_word(model, w)]
    keeps = lambda w: single_token_word(model, w)  # noqa: E731
    passages = all_passages()
    lines: list[str] = []
    log = lines.append
    log(f"Substitution pool: {len(pool)}/{len(NEUTRAL_POOL)} words are single tokens in GPT-2")

    # Condition specs: (label, condition, kwargs builder, stochastic?)
    specs = [("P", "P", {}, False), ("U", "U", {}, False), ("Ucap", "Ucap", {}, False), ("M", "M", {}, True)]
    specs += [(f"C{p:.1f}", "C", {"sub": Substitution(p, pool, keeps)}, True) for p in SUB_PROBS]
    specs += [(f"K{n}", "K", {"restarts": n}, True) for n in RESTARTS]

    results = {}
    examples = {}
    for label, cond, kw, stochastic in specs:
        rows = []
        for pi, (p, city, state) in enumerate(passages):
            for d in range(DRAWS if stochastic else 1):
                rng = random.Random(f"{SEED}-{label}-{pi}-{d}")
                pieces = render(p, cond, rng=rng, **kw)
                rows.append(measure(model, pieces, state))
                if pi == 0 and d == 0:
                    examples[label] = pieces
        results[label] = summarize(rows)

    base = results["P"]["word_nll"]
    target = results["U"]["word_nll"] - base
    log(f"\n== Unusualness and behavior (48 passages; x{DRAWS} random draws for M, C, K) ==")
    log(f"{'cond':6} {'word NLL':>8} {'Δ vs P':>7} {'×U Δ':>6} {'all NLL':>8} {'total':>7} {'tokens':>6} {'logit diff':>10} {'acc':>5}")
    for label, r in results.items():
        delta = r["word_nll"] - base
        log(
            f"{label:6} {r['word_nll']:8.3f} {delta:+7.3f} {delta / target:6.2f} {r['all_nll']:8.3f} "
            f"{r['total_nll']:7.1f} {r['tokens']:6.1f} {r['diff']:+10.2f} {r['acc']:5.2f}"
        )

    # Pick C strengths whose word-NLL increase is closest to 0.5x, 1x and 1.5x U's increase.
    chosen = {}
    for mult in (0.5, 1.0, 1.5):
        label = min(
            (lab for lab in results if lab.startswith("C")),
            key=lambda lab: abs((results[lab]["word_nll"] - base) - mult * target),
        )
        chosen[mult] = label
    log(f"\nU's word-NLL increase over P: {target:+.3f} nats/token")
    log("Closest C strengths: " + ", ".join(f"{m}x U -> {lab}" for m, lab in chosen.items()))

    log("\n== Example renderings (template 1, Denver) ==")
    from punctsum.conditions import to_text

    for label in ["P", "U", "Ucap", "M", *sorted(set(chosen.values())), "K1", "K2", "K3"]:
        log(f"  {label:5} {to_text(examples[label])[0]}")

    log(f"\nPeak RSS: {peak_rss_gb():.2f} GB")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(
        json.dumps({"seed": SEED, "draws": DRAWS, "pool": pool, "results": results, "chosen": chosen}, indent=1)
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
