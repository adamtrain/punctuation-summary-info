"""Long-distance pairs (BRIEF.md decision log): property sentence + a natural-text PG19 excerpt + readout.

  python scripts/m6_long_pairs.py excerpts            # 100 excerpts per level from 100 different books
  python scripts/m6_long_pairs.py pairs <items_set>   # e.g. pythia-ladder, gpt2-items

Levels: L900 (~850 tokens of excerpt, so property sentence + excerpt + readout fits GPT-2's 1,024-token window in every
condition) and L1500 (~1,450 tokens; for Pythia's 2,048 window and Llama). Token counts use the GPT-2 tokenizer; the
driver checks each model's window. The same excerpts are used for every model. Long pairs reuse the first 100 k=1 pairs
of an item set (same property sentences, cities and adjectives) and replace the filler with excerpt i.
"""

import json
import random
import sys

from transformers import AutoTokenizer

from punctsum import PROJECT_ROOT
from punctsum.dataset import load_jsonl, save_jsonl
from punctsum.longtext import BOOKS, excerpt

LEVELS = {"L900": 850, "L1500": 1450}
OUT = PROJECT_ROOT / "outputs" / "models"
EXCERPTS = OUT / "long_excerpts.jsonl"


def build_excerpts() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    count = lambda t: len(tok(t)["input_ids"])  # noqa: E731
    books = sorted(BOOKS.glob("*.txt"))
    rows = []
    for level, n in LEVELS.items():
        rng = random.Random(f"pg19-{level}")
        for i, b in enumerate(books):
            for attempt in range(5):  # a new random start if the book runs out before the excerpt is full
                try:
                    e = excerpt(b, n, count, rng)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"no excerpt of {n} tokens found in book {b.stem}")
            rows.append({"level": level, "index": i, "attempts": attempt + 1, **e})
    save_jsonl(rows, EXCERPTS)
    for level in LEVELS:
        lv = [r for r in rows if r["level"] == level]
        kept = sum(r["kept_count"] for r in lv)
        skipped = sum(r["skipped"] for r in lv)
        print(f"{level}: {len(lv)} excerpts, tokens {min(r['tokens'] for r in lv)}-{max(r['tokens'] for r in lv)}, "
              f"sentences kept {kept}, skipped {skipped} ({skipped / (kept + skipped):.0%})")


def build_pairs(items_set: str) -> None:
    d = OUT / items_set
    ex = load_jsonl(EXCERPTS)
    for kind in ("country", "mood"):
        base = [p for p in load_jsonl(d / f"pairs_{kind}.jsonl") if p["k"] == 1][:100]
        for level in LEVELS:
            lv = sorted((e for e in ex if e["level"] == level), key=lambda e: e["index"])
            pairs = [{**p, "pair_id": f"{p['pair_id']}-{level}", "k": level, "long_filler": e["sentences"],
                      "long_proper": e["proper"], "long_source": f"pg19-test/{e['book']}#p{e['start_paragraph']}"}
                     for p, e in zip(base, lv)]
            save_jsonl(pairs, d / f"pairs_{kind}_{level}.jsonl")
            print(f"{items_set}: {kind} {level}: {len(pairs)} pairs")


if __name__ == "__main__":
    if sys.argv[1] == "excerpts":
        build_excerpts()
    else:
        build_pairs(sys.argv[2])
