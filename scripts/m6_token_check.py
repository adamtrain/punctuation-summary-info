"""Tokenizer-only check of an item set (no model weights; CPU). For every pair file in the set (standard and
long-distance), every label, condition and readout:
  1. the rendering tokenizes without a token straddling two pieces (else roles would be ambiguous);
  2. compare the ids we feed the model (passage and readout tokenized separately, then joined) with tokenizing the whole
     text at once. They must be identical for GPT-2, GPT-Neo and Pythia (the frozen pipeline's tokens are unchanged); for
     Llama 3 they differ where a final period fuses with the question readout (".\\n\\n"), which is the point of joining.

  python scripts/m6_token_check.py <items_set> <model> [<model> ...]          # all pair files
  python scripts/m6_token_check.py <items_set> <model> ... --long             # long-distance pair files only
"""

import sys
from collections import Counter

from punctsum import PROJECT_ROOT  # noqa: F401  (sets HF_HOME before transformers is imported)
import torch
from transformers import AutoTokenizer

from punctsum.behavior import passage_label_pairs
from punctsum.dataset import CONDITIONS, load_jsonl, property_of, render_condition
from punctsum.model import joint_tokens, tokenize_with_readout
from m6_models import MODELS
from m6_scale import revision
from m6_long_pairs import OUT


class TokenizerOnly:
    """The two things the rendering and tokenization code needs from a model."""

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[name], revision=revision(name))

    def to_tokens(self, text, prepend_bos=True):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor([[self.tokenizer.bos_token_id] * prepend_bos + ids])


def main(items_set, names, long_only=False):
    files = sorted((OUT / items_set).glob("pairs_*_L*.jsonl" if long_only else "pairs_*.jsonl"))
    pairs = [p for f in files for p in load_jsonl(f)]
    for name in names:
        model, bad, n, longest, differs = TokenizerOnly(name), [], 0, {}, Counter()
        for p, lab, c in passage_label_pairs(pairs, CONDITIONS):
            pieces = render_condition(p, lab, c, model)
            for ro_name, ro in property_of(p).readouts.items():
                n += 1
                try:
                    _, toks, _, _ = tokenize_with_readout(model, pieces, ro)
                except ValueError as e:
                    bad.append((p["pair_id"], lab, c, str(e)[:120]))
                    continue
                longest[p["k"]] = max(longest.get(p["k"], 0), toks.shape[1])
                if not torch.equal(toks, joint_tokens(model, pieces, ro)):
                    differs[(ro_name, c)] += 1
        print(f"{items_set} / {name}: {len(files)} files, {n} renderings, {len(bad)} failures, "
              f"{sum(differs.values())} where joined != whole-text tokens {dict(differs)}; longest {longest}", flush=True)
        for b in bad[:5]:
            print("   ", b)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--long"]
    main(args[0], args[1:], long_only="--long" in sys.argv)
