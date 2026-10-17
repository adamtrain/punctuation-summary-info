"""Milestone 1: tokenization, answer tokens, per-condition loss, and a behavioral check on GPT-2 small.

Writes a human-readable report to outputs/m1/report.txt and raw numbers to outputs/m1/results.json.
"""

import json
import random
import statistics
from pathlib import Path

import torch

from punctsum import PROJECT_ROOT
from punctsum.conditions import Passage, render_P, render_U, render_Ucap
from punctsum.model import load, peak_rss_gb
from punctsum.pilot import CITIES, READOUT, STATES, TEMPLATES, make_passage

SEED = 0
OUT = PROJECT_ROOT / "outputs" / "m1"

KNOWLEDGE_PROMPT = "{CITY} is a city in the state of"

PROBE_STRINGS = [
    ",", " ,", ".", " .", ". The", ", then", "\n", "\n\n", " and", " then",
    " Colorado", " Texas", "Colorado", "Texas",
]


# Casing-based control candidates, kept here only to reproduce the milestone 1 report; both were
# rejected (BRIEF.md decision log), so they are not part of punctsum.conditions.
def _join(sentences, comma: str, period: str) -> str:
    return " ".join((comma + " ").join(" ".join(clause) for clause in sent) + period for sent in sentences)


def _map_words(p: Passage, fn):
    return [
        [[fn(w, ci == 0 and wi == 0, w in p.proper_nouns) for wi, w in enumerate(clause)] for ci, clause in enumerate(sent)]
        for sent in p.sentences
    ]


def render_C_wordcase(p: Passage, rng: random.Random, prob: float) -> str:
    s = _map_words(
        p,
        lambda w, initial, proper: w.capitalize() if not (initial or proper) and rng.random() < prob else w,
    )
    return _join(s, ",", ".")


def render_C_charcase(p: Passage, rng: random.Random, prob: float) -> str:
    def flip(w: str, initial: bool, proper: bool) -> str:
        if proper:
            return w
        return "".join(c.swapcase() if rng.random() < prob else c for c in w)

    return _join(_map_words(p, flip), ",", ".")


def render_all(p: Passage, rng: random.Random) -> dict[str, str]:
    return {
        "P": render_P(p),
        "U": render_U(p),
        "Ucap": render_Ucap(p),
        "C_wordcase30": render_C_wordcase(p, rng, 0.3),
        "C_wordcase60": render_C_wordcase(p, rng, 0.6),
        "C_charcase10": render_C_charcase(p, rng, 0.1),
        "C_charcase25": render_C_charcase(p, rng, 0.25),
    }


@torch.no_grad()
def passage_nll(model, text: str) -> tuple[float, float, int]:
    """Mean and total next-token NLL over the passage tokens (excluding BOS as a target)."""
    toks = model.to_tokens(text)
    logits = model(toks)
    logprobs = logits[0, :-1].log_softmax(-1)
    nll = -logprobs.gather(-1, toks[0, 1:, None])[:, 0]
    return nll.mean().item(), nll.sum().item(), nll.numel()


@torch.no_grad()
def answer_logit_diff(model, text: str, correct: str, wrong: str) -> float:
    logits = model(model.to_tokens(text))[0, -1]
    c, w = model.to_single_token(" " + correct), model.to_single_token(" " + wrong)
    return (logits[c] - logits[w]).item()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    model = load("gpt2")
    lines: list[str] = []
    log = lines.append
    results: dict = {"seed": SEED, "model": "gpt2"}

    # 1. Tokenization of individual strings.
    log("== 1. Tokenization of probe strings ==")
    for s in PROBE_STRINGS:
        log(f"{s!r:>14} -> {model.to_str_tokens(s, prepend_bos=False)}")
    log("\nCity names (leading space / no space / lowercase):")
    for state, cities in CITIES.items():
        for c in cities:
            log(
                f"  {state:9} {c:10} ' {c}'={model.to_str_tokens(' ' + c, prepend_bos=False)}"
                f"  '{c}'={model.to_str_tokens(c, prepend_bos=False)}"
                f"  ' {c.lower()}'={model.to_str_tokens(' ' + c.lower(), prepend_bos=False)}"
            )

    # 2. One passage in every condition, token by token.
    log("\n== 2. Example passage in each condition ==")
    p = make_passage(TEMPLATES[0], "Boulder")
    for name, text in render_all(p, random.Random(SEED)).items():
        toks = model.to_str_tokens(text)
        log(f"\n[{name}] ({len(toks)} tokens incl. BOS)\n  text: {text}")
        log("  tokens: " + " | ".join(f"{i}:{t!r}" for i, t in enumerate(toks)))
    log("\nReadout suffix tokens: " + " | ".join(repr(t) for t in model.to_str_tokens(READOUT, prepend_bos=False)))

    # 3. Per-condition loss (how unusual is each condition to the model?).
    log("\n== 3. Passage NLL per condition (48 passages: 3 templates x 16 cities) ==")
    per_cond: dict[str, list[tuple[float, float, int]]] = {}
    for tmpl in TEMPLATES:
        for cities in CITIES.values():
            for c in cities:
                for name, text in render_all(make_passage(tmpl, c), rng).items():
                    per_cond.setdefault(name, []).append(passage_nll(model, text))
    results["nll"] = {}
    log(f"  {'condition':14} {'mean/token':>10} {'total/passage':>13} {'tokens':>7}")
    for name, vals in per_cond.items():
        mean_tok = statistics.mean(v[0] for v in vals)
        total = statistics.mean(v[1] for v in vals)
        ntok = statistics.mean(v[2] for v in vals)
        results["nll"][name] = {"mean_per_token": mean_tok, "total_per_passage": total, "tokens": ntok}
        log(f"  {name:14} {mean_tok:10.3f} {total:13.1f} {ntok:7.1f}")

    # 4. Behavioral check: can GPT-2 small map city -> state at all?
    log("\n== 4. Behavioral check: logit diff (correct state - other state) ==")
    log("Knowledge prompt alone: " + repr(KNOWLEDGE_PROMPT))
    results["knowledge"] = {}
    for state, cities in CITIES.items():
        other = [s for s in STATES if s != state][0]
        for c in cities:
            d = answer_logit_diff(model, KNOWLEDGE_PROMPT.replace("{CITY}", c), state, other)
            results["knowledge"][c] = d
            log(f"  {c:10} ({state:8}) diff={d:+.2f}")
    log("\nPassage + readout, by condition (mean diff, accuracy = share with diff > 0):")
    results["behavior"] = {}
    for name in ["P", "U", "Ucap", "C_wordcase30", "C_charcase10"]:
        diffs = []
        for tmpl in TEMPLATES:
            for state, cities in CITIES.items():
                other = [s for s in STATES if s != state][0]
                for c in cities:
                    text = render_all(make_passage(tmpl, c), random.Random(SEED))[name] + READOUT
                    diffs.append(answer_logit_diff(model, text, state, other))
        acc = sum(d > 0 for d in diffs) / len(diffs)
        results["behavior"][name] = {"mean_diff": statistics.mean(diffs), "acc": acc, "diffs": diffs}
        log(f"  {name:14} mean diff={statistics.mean(diffs):+.2f}  acc={acc:.2f}  (n={len(diffs)})")

    # Top predictions for one example, to see what GPT-2 wants to say after the readout.
    text = render_P(make_passage(TEMPLATES[0], "Boulder")) + READOUT
    top = model(model.to_tokens(text))[0, -1].topk(10)
    log("\nTop-10 next tokens after readout (P, Boulder): "
        + ", ".join(f"{model.to_string(i)!r}" for i in top.indices))

    results["peak_rss_gb"] = peak_rss_gb()
    log(f"\nPeak RSS: {results['peak_rss_gb']:.2f} GB")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
