"""Milestone 2 dataset: minimal pairs with a filler-length variable, for one or more properties.

The original property is country (France/Japan, defined here). Other properties (e.g. mood, punctsum/mood.py)
register a Property in PROPERTIES; each pair records its property name ("country" if absent).

A *pair* is one s0 template plus k filler sentences, instantiated twice: once with a French city and
once with a Japanese city of the same token count. The test split uses held-out templates, fillers and
cities, so a probe can only do well on it by reading the label, not by memorizing a template or city. Pairs give label balance by construction, cancel any
template or filler association with a label, and are exactly the clean/corrupted inputs that activation
patching needs. Pairs store structure only; condition texts are rendered on demand, deterministically,
from a seed derived from the pair id, so both members of a pair get identical perturbations.
"""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from punctsum.conditions import Passage, Piece, Substitution, render
from punctsum.materials import (
    CITY,
    CITY_CANDIDATES,
    FILLERS,
    FIXED_PROPER,
    KNOWLEDGE_PROMPT,
    NEUTRAL_POOL,
    S0_TEMPLATES,
    LABELS,
    READOUT,
    READOUT_CONT,
)
from punctsum.model import single_token_word

FILLER_LENGTHS = [1, 2, 4, 8, 16]
TEST_FRACTION = 0.25
KNOWLEDGE_THRESHOLD = 2.0  # min logit diff (correct label - other) on the knowledge prompt
K_WORDS_PER_RESTART = 10  # condition K: one spurious capital per ~10 words
# Substitution rates giving 0.5x, 1x, 1.5x U's word-NLL increase over P on GPT-2 small
# (scripts/m2_calibrate_c.py, outputs/m2/calibration.txt). Recalibrate for other models.
C_STRENGTHS = {"C_lo": 0.026, "C_mid": 0.057, "C_hi": 0.079}
CONDITIONS = ["P", "U", "Ucap", "M", "K", *C_STRENGTHS]
# Conditions for the expensive internal analyses (probes, patching): C_lo dropped (decision log 2026-10-17).
INTERNAL_CONDITIONS = ["P", "U", "Ucap", "M", "K", "C_mid", "C_hi"]


def split_indices(items: list, seed: int) -> dict[int, str]:
    """Assign each template to train/test, stratified by clause count, so test holds unseen templates."""
    rng = random.Random(seed)
    out = {}
    for n_clauses in sorted({len(t) for t in items}):
        idx = [i for i, t in enumerate(items) if len(t) == n_clauses]
        rng.shuffle(idx)
        n_test = round(len(idx) * TEST_FRACTION)
        out.update({i: ("test" if j < n_test else "train") for j, i in enumerate(idx)})
    return out


@torch.no_grad()
def screen_cities(model) -> list[dict]:
    """Token count and knowledge margin for every candidate city."""
    rows = []
    for label, cities in CITY_CANDIDATES.items():
        other = next(lab for lab in LABELS if lab != label)
        good, bad = model.to_single_token(" " + label), model.to_single_token(" " + other)
        for city in cities:
            logits = model(model.to_tokens(KNOWLEDGE_PROMPT.replace(CITY, city)))[0, -1]
            diff = (logits[good] - logits[bad]).item()
            ntok = len(model.tokenizer(" " + city, add_special_tokens=False)["input_ids"])
            rows.append({"city": city, "label": label, "ntok": ntok, "diff": diff, "keep": diff >= KNOWLEDGE_THRESHOLD})
    return rows


def city_buckets(screen: list[dict], seed: int) -> dict[str, dict[int, dict[str, list[str]]]]:
    """Kept cities as {split: {token count: {label: [cities]}}}. Within each (token count, label) group a
    TEST_FRACTION share (at least one, if the group has two or more) is held out for the test split.
    Only token counts with cities for both labels in a split are returned for that split."""
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[str]] = {}
    for r in screen:
        if r["keep"]:
            groups.setdefault((r["ntok"], r["label"]), []).append(r["city"])
    out: dict[str, dict[int, dict[str, list[str]]]] = {"train": {}, "test": {}}
    for (ntok, label), cities in sorted(groups.items()):
        cities = sorted(cities)
        rng.shuffle(cities)
        n_test = max(1, round(len(cities) * TEST_FRACTION)) if len(cities) >= 2 else 0
        for split, part in (("test", cities[:n_test]), ("train", cities[n_test:])):
            out[split].setdefault(ntok, {lab: [] for lab in LABELS})[label] = part
    return {split: {n: b for n, b in bs.items() if all(b.values())} for split, bs in out.items()}


def make_pairs(screen: list[dict], n_per_length: int, seed: int) -> list[dict]:
    """n_per_length pairs for each filler length; a TEST_FRACTION share uses only test templates."""
    rng = random.Random(seed)
    s0_split, filler_split = split_indices(S0_TEMPLATES, seed), split_indices(FILLERS, seed + 1)
    buckets = city_buckets(screen, seed + 2)
    pairs = []
    for k in FILLER_LENGTHS:
        n_test = round(n_per_length * TEST_FRACTION)
        for i in range(n_per_length):
            split = "test" if i < n_test else "train"
            s0_pool = [j for j, s in s0_split.items() if s == split]
            f_pool = [j for j, s in filler_split.items() if s == split]
            usable = buckets[split]
            ntok = rng.choices(list(usable), [min(len(c) for c in b.values()) for b in usable.values()])[0]
            pairs.append({
                "pair_id": f"k{k}-{i:05d}",
                "k": k,
                "split": split,
                "s0": rng.choice(s0_pool),
                "fillers": rng.sample(f_pool, k),
                "city_ntok": ntok,
                "cities": {lab: rng.choice(usable[ntok][lab]) for lab in LABELS},
            })
    return pairs


def passage_for(pair: dict, label: str) -> Passage:
    city = pair["cities"][label]
    s0 = [[city if w == CITY else w for w in clause] for clause in S0_TEMPLATES[pair["s0"]]]
    sentences = [s0] + [FILLERS[j] for j in pair["fillers"]]
    return Passage(sentences, proper_nouns=FIXED_PROPER | {city}, property_words={city})


@dataclass(frozen=True)
class Property:
    name: str
    labels: list[str]
    readouts: dict[str, str]  # "question" and "continuation" (separator-free, primary for internals)
    passage_for: Callable[[dict, str], Passage]
    answers: dict[str, str] | None = None  # label -> answer word (token " " + word); default: the label itself

    def answer(self, label: str) -> str:
        return " " + (self.answers[label] if self.answers else label)


COUNTRY = Property("country", LABELS, {"question": READOUT, "continuation": READOUT_CONT}, passage_for)
PROPERTIES: dict[str, Property] = {"country": COUNTRY}


def property_of(pair: dict) -> Property:
    return PROPERTIES[pair.get("property", "country")]


_pools: dict[int, list[str]] = {}


def render_condition(pair: dict, label: str, cond: str, model) -> list[Piece]:
    p = property_of(pair).passage_for(pair, label)
    rng = random.Random(f"{pair['pair_id']}-{cond}")  # same perturbation for both members of the pair
    if cond in ("P", "U", "Ucap", "M"):
        return render(p, cond, rng=rng)
    if cond == "K":
        n_words = sum(len(c) for s in p.sentences for c in s)
        return render(p, "K", rng=rng, restarts=max(1, round(n_words / K_WORDS_PER_RESTART)))
    if cond in C_STRENGTHS or cond.startswith("C@"):  # "C@0.08" = control C at rate 0.08 (calibration)
        r = C_STRENGTHS[cond] if cond in C_STRENGTHS else float(cond[2:])
        if id(model) not in _pools:
            _pools[id(model)] = [w for w in NEUTRAL_POOL if single_token_word(model, w)]
        sub = Substitution(r, _pools[id(model)], lambda w: single_token_word(model, w))
        return render(p, "C", rng=rng, sub=sub)
    raise ValueError(f"unknown condition {cond!r}")


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
