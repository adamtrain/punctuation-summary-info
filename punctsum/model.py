"""Model loading. All models go through TransformerLens's TransformerBridge in compatibility mode,
which gives HookedTransformer-style hook names (blocks.{L}.hook_resid_post) and numerics."""

import os
import resource
import sys

import punctsum  # noqa: F401  (sets HF_HOME before transformer_lens is imported)
import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from punctsum.conditions import Piece, to_text


def default_device() -> str:
    """PUNCTSUM_DEVICE if set, else the best available: CUDA (cloud GPU), MPS (Apple GPU), CPU.
    MPS logits match CPU to ~3e-5 on GPT-2 (BRIEF.md section 10.9)."""
    if os.environ.get("PUNCTSUM_DEVICE"):
        return os.environ["PUNCTSUM_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name: str = "gpt2", device: str | None = None, dtype: torch.dtype = torch.float32) -> TransformerBridge:
    model = TransformerBridge.boot_transformers(name, device=device or default_device(), dtype=dtype)
    model.enable_compatibility_mode(disable_warnings=True)
    model.eval()
    return model


def tokenize_pieces(model, pieces: list[Piece]) -> tuple[str, torch.Tensor, list[int]]:
    """Tokenize a rendered condition and map every token back to the Piece it came from.

    Returns (text, tokens of shape [1, seq] with BOS prepended, piece index per token, -1 for BOS).
    Raises if a token straddles two pieces, which would make role labels ambiguous.
    """
    text, spans = to_text(pieces)
    enc = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = [model.tokenizer.bos_token_id] + enc["input_ids"]
    piece_of_token = [-1]
    for s, e in enc["offset_mapping"]:
        while s < e and text[s] == " ":  # GPT-2-style tokens carry their leading space
            s += 1
        idx = next(i for i, (a, b) in enumerate(spans) if a <= s < b)
        if e > spans[idx][1]:
            raise ValueError(f"token {text[s:e]!r} straddles pieces in {text!r}")
        piece_of_token.append(idx)
    return text, torch.tensor([ids]), piece_of_token


def tokenize_with_readout(model, pieces: list[Piece], readout: str) -> tuple[str, torch.Tensor, list[int], int]:
    """Like tokenize_pieces, with the readout suffix appended. Checks that appending the readout doesn't
    change how the passage tokenizes. Returns (full text, tokens, piece index per passage token,
    number of readout tokens)."""
    text, toks, piece_of_token = tokenize_pieces(model, pieces)
    full = model.to_tokens(text + readout).cpu()
    n = toks.shape[1]
    if not torch.equal(full[:, :n], toks):
        raise ValueError(f"readout changes passage tokenization: {text!r}")
    return text + readout, full, piece_of_token, full.shape[1] - n


def single_token_word(model, word: str) -> bool:
    """True if ' word' (mid-sentence form, with leading space) is one token."""
    return len(model.tokenizer(" " + word, add_special_tokens=False)["input_ids"]) == 1


def resid_hook_names(model) -> list[str]:
    """Residual stream at every depth: the embeddings (block 0's input), then each block's output.
    Index i of the result is "layer i" in our plots (0 = embeddings, n_layers = final block)."""
    return ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]


def length_batches(seqs: list[list[int]], max_tokens: int = 4096):
    """Group sequence indices into batches of equal-length sequences (so no padding is needed),
    each at most max_tokens tokens in total. Keeps peak memory bounded: GPT-2 logits for 4096 tokens
    are about 0.8 GB."""
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append(i)
    for n, idx in sorted(by_len.items()):
        step = max(1, max_tokens // n)
        for s in range(0, len(idx), step):
            yield idx[s : s + step]


@torch.no_grad()
def forward_batched(model, seqs: list[list[int]], fn, max_tokens: int = 4096) -> list:
    """Run the model over many token sequences; fn(logits [B, T, V], batch indices) returns one result per
    sequence. Results come back in input order."""
    out = [None] * len(seqs)
    for idx in length_batches(seqs, max_tokens):
        logits = model(torch.tensor([seqs[i] for i in idx], device=model.cfg.device))
        for i, r in zip(idx, fn(logits, idx)):
            out[i] = r
    return out


@torch.no_grad()
def resid_at(model, seqs: list[list[int]], positions: list[list[int]], max_tokens: int = 4096) -> np.ndarray:
    """Residual stream vectors at the given positions of each sequence, at every depth.
    Returns float32 [n_seqs, n_positions, n_layers + 1, d_model]; positions[i] lists positions for seqs[i]
    (all lists must have the same length)."""
    names = resid_hook_names(model)
    n_pos = len(positions[0])
    out = np.zeros((len(seqs), n_pos, len(names), model.cfg.d_model), dtype=np.float32)
    for idx in length_batches(seqs, max_tokens):
        toks = torch.tensor([seqs[i] for i in idx], device=model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        stacked = torch.stack([cache[n] for n in names], dim=2)  # [B, T, L+1, d]
        for b, i in enumerate(idx):
            out[i] = stacked[b, positions[i]].float().cpu().numpy()
    return out


def peak_rss_gb() -> float:
    # ru_maxrss is in bytes on macOS but kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9
