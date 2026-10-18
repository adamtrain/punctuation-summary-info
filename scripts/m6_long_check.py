"""Tokenizer-only check of the long-distance items (no model weights; CPU): every pair, label and condition must
tokenize without a token straddling two pieces, and appending each readout must not change the passage's tokens.

  python scripts/m6_long_check.py <items_set> <model> [<model> ...]
"""

import sys

from punctsum import PROJECT_ROOT  # noqa: F401  (sets HF_HOME before transformers is imported)
import torch
from transformers import AutoTokenizer

from punctsum.behavior import passage_label_pairs
from punctsum.dataset import CONDITIONS, load_jsonl, property_of, render_condition
from punctsum.model import tokenize_with_readout
from m6_models import MODELS
from m6_scale import revision
from m6_long_pairs import LEVELS, OUT


class TokenizerOnly:
    """The two things the rendering and tokenization code needs from a model."""

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[name], revision=revision(name))

    def to_tokens(self, text, prepend_bos=True):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor([[self.tokenizer.bos_token_id] * prepend_bos + ids])


def main(items_set, names):
    pairs = [p for kind in ("country", "mood") for lv in LEVELS
             for p in load_jsonl(OUT / items_set / f"pairs_{kind}_{lv}.jsonl")]
    for name in names:
        model, bad, n, longest = TokenizerOnly(name), [], 0, {}
        for p, lab, c in passage_label_pairs(pairs, CONDITIONS):
            pieces = render_condition(p, lab, c, model)
            for ro in property_of(p).readouts.values():
                try:
                    _, toks, _, _ = tokenize_with_readout(model, pieces, ro)
                    longest[p["k"]] = max(longest.get(p["k"], 0), toks.shape[1])
                except ValueError as e:
                    bad.append((p["pair_id"], lab, c, str(e)[:120]))
                n += 1
        print(f"{items_set} / {name}: {n} renderings, {len(bad)} failures, longest {longest}", flush=True)
        for b in bad[:5]:
            print("   ", b)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
