"""Milestone 6: shared items and pairs for the Pythia ladder (BRIEF.md decision log, "option 2").

From the per-model screens in outputs/models/pythia-*/screen/: cities kept by the frozen knowledge screen in every size;
adjectives with the right direction and strength >= 1.2 (relative to neutral words) in every size (no ceiling);
the 64 fillers closest to each model's median valence, averaged over sizes. All Pythia sizes share one tokenizer, so token
counts (and pairing) come from any one size's screen. Writes outputs/models/pythia-ladder/: items.json,
pairs_country.jsonl (400 per filler length), pairs_mood.jsonl (consistent, 400 per filler length) and
pairs_mood_mixed.jsonl (100 per filler length, for the acceptance test).
"""

import json

import numpy as np

from punctsum import PROJECT_ROOT
from punctsum.dataset import load_jsonl, make_pairs, save_jsonl
from punctsum.mood import N_NEUTRAL_FILLERS, make_mood_pairs

SIZES = ["pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"]
SEED = 0
ADJ_FLOOR = 1.2
N_TEST_ADJ = 3
OUT = PROJECT_ROOT / "outputs" / "models" / "pythia-ladder"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scr = {s: PROJECT_ROOT / "outputs" / "models" / s / "screen" for s in SIZES}
    cities = {s: {r["city"]: r for r in load_jsonl(scr[s] / "city_screen.jsonl")} for s in SIZES}
    adjs = {s: {r["adj"]: r for r in load_jsonl(scr[s] / "adj_screen.jsonl")} for s in SIZES}
    fills = {s: load_jsonl(scr[s] / "filler_screen.jsonl") for s in SIZES}

    ref = SIZES[0]
    city_screen = [{**r, "keep": all(cities[s][c]["keep"] for s in SIZES)} for c, r in cities[ref].items()]
    sign = {"good": 1, "bad": -1}
    adj_screen = [{**r, "keep": r["single_token"] and all(sign[r["label"]] * adjs[s][a]["rel"] >= ADJ_FLOOR for s in SIZES)}
                  for a, r in adjs[ref].items()]
    dist = np.zeros(len(fills[ref]))
    for s in SIZES:
        d = np.array([r["diff"] for r in fills[s]])
        dist += np.abs(d - np.median(d)) / len(SIZES)
    fillers = sorted(int(fills[ref][i]["filler"]) for i in np.argsort(dist)[:N_NEUTRAL_FILLERS])

    items = {
        "cities": {lab: sorted(r["city"] for r in city_screen if r["keep"] and r["label"] == lab) for lab in ("France", "Japan")},
        "adjectives": {lab: sorted(r["adj"] for r in adj_screen if r["keep"] and r["label"] == lab) for lab in ("good", "bad")},
        "fillers": fillers,
        "adj_floor": ADJ_FLOOR,
        "n_test_adj": N_TEST_ADJ,
    }
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    save_jsonl(make_pairs(city_screen, 400, SEED), OUT / "pairs_country.jsonl")
    save_jsonl(make_mood_pairs(adj_screen, fillers, 400, SEED, "consistent", N_TEST_ADJ), OUT / "pairs_mood.jsonl")
    save_jsonl(make_mood_pairs(adj_screen, fillers, 100, SEED, "mixed", N_TEST_ADJ), OUT / "pairs_mood_mixed.jsonl")
    print({k: ({kk: len(vv) for kk, vv in v.items()} if isinstance(v, dict) else (len(v) if isinstance(v, list) else v))
           for k, v in items.items()})


if __name__ == "__main__":
    main()
