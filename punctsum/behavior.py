"""Behavioral scoring: how unusual each rendering is to the model, and how well it answers the readout.

One forward pass per rendering (passage + readout) gives both:
  word_nll   mean next-token NLL over the passage's *word* tokens (the unusualness measure; punctuation
             tokens are excluded because they are easy to predict and would bias the average)
  all_nll    mean NLL over all passage tokens; total_nll is the sum
  diff       logit(true label) - logit(other label) at the final readout position
"""

import numpy as np
import torch

from punctsum.dataset import property_of, render_condition
from punctsum.model import length_batches, tokenize_with_readout
from punctsum.roles import label_tokens

PUNCT_KINDS = {"comma", "period", "misplaced_comma", "misplaced_period"}

# Attention knockouts (BRIEF.md section 4.7). Each maps a token's role row to True if readout queries
# may NOT attend to it. Queries are always the readout tokens; BOS and readout tokens are never blocked.
KNOCKOUTS = {
    "none": lambda r: False,
    "s0words": lambda r: r["segment"] == "s0" and r["kind"] not in PUNCT_KINDS,
    "s0all": lambda r: r["segment"] == "s0",
    "punctonly": lambda r: r["segment"] not in ("bos", "readout") and r["kind"] not in PUNCT_KINDS,
}


def score(model, items: list[tuple[dict, str, str]], max_tokens: int = 4096, readout: str = "question",
          knockout: str = "none") -> list[dict]:
    """items: (pair, label, condition). `readout` is a readout name of the pair's property ("question" or
    "continuation") or a literal readout string. Returns one result dict per item, in order. With a knockout,
    readout tokens cannot attend to the blocked passage tokens in any layer or head."""
    seqs, word_masks, n_passage, answers, blocked = [], [], [], [], []
    for pair, label, cond in items:
        prop = property_of(pair)
        readout_text = prop.readouts.get(readout, readout)
        pieces = render_condition(pair, label, cond, model)
        _, toks, piece_of_token, n_readout = tokenize_with_readout(model, pieces, readout_text)
        if knockout != "none":
            rows = label_tokens(prop.passage_for(pair, label), pieces, piece_of_token, n_readout)
            is_q = torch.tensor([r["segment"] == "readout" for r in rows])
            is_k = torch.tensor([KNOCKOUTS[knockout](r) for r in rows])
            blocked.append(is_q[:, None] & is_k[None, :])
        seqs.append(toks[0].tolist())
        n_passage.append(len(piece_of_token))
        # Target t (predicting token t from t-1) counts as a word token if token t belongs to a word piece.
        word_masks.append(np.array([pi >= 0 and pieces[pi].kind == "word" for pi in piece_of_token[1:]]))
        other = next(lab for lab in prop.labels if lab != label)
        answers.append((model.to_single_token(prop.answer(label)), model.to_single_token(prop.answer(other))))

    def fn(logits, idx):
        res = []
        logprobs = logits.log_softmax(-1)
        for b, i in enumerate(idx):
            n = n_passage[i]
            tgt = torch.tensor(seqs[i][1:n])
            nll = -logprobs[b, : n - 1].gather(-1, tgt[:, None])[:, 0].numpy()
            good, bad = answers[i]
            res.append({
                "word_nll": float(nll[word_masks[i]].mean()),
                "all_nll": float(nll.mean()),
                "total_nll": float(nll.sum()),
                "tokens": len(seqs[i]),
                "diff": float(logits[b, -1, good] - logits[b, -1, bad]),
            })
        return res

    out = [None] * len(seqs)
    names = [f"blocks.{l}.attn.hook_attn_scores" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        for idx in length_batches(seqs, max_tokens):
            toks = torch.tensor([seqs[i] for i in idx])
            if knockout == "none":
                logits = model(toks)
            else:
                mask = torch.stack([blocked[i] for i in idx])[:, None]  # [B, 1, Q, K], broadcast over heads

                def hook(scores, hook, mask=mask):
                    return scores.masked_fill(mask, float("-inf"))

                logits = model.run_with_hooks(toks, fwd_hooks=[(n, hook) for n in names])
            for i, r in zip(idx, fn(logits, idx)):
                out[i] = r
    return out


def pair_bootstrap(values: dict[str, float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling *pairs* (the independent unit; the two members of a pair
    share template, fillers and perturbation). values maps pair_id -> that pair's value."""
    v = np.array(list(values.values()))
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def passage_label_pairs(pairs: list[dict], conds: list[str]) -> list[tuple[dict, str, str]]:
    return [(p, lab, c) for p in pairs for c in conds for lab in property_of(p).labels]

