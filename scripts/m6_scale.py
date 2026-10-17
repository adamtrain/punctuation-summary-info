"""Milestone 6: run the frozen protocol (git tag method-freeze-v1; BRIEF.md decision log) on a scaling model.

  python scripts/m6_scale.py verify pythia-160m [pythia-410m ...]   # TransformerLens vs plain Hugging Face, memory
  python scripts/m6_scale.py screen pythia-160m [...]               # city, adjective and filler screens

Models are named as in scripts/m6_models.py and loaded at the exact revision recorded in
outputs/m6/models_manifest.json, with Hugging Face in offline mode (only the downloaded weights can be used).
Everything is computed in fp32. Outputs go to outputs/models/<model>/.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never fetch: use exactly the pinned, downloaded weights

import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_models import MODELS  # noqa: E402

from punctsum import PROJECT_ROOT  # noqa: E402
from punctsum.dataset import load_jsonl, property_of, render_condition, save_jsonl, screen_cities  # noqa: E402
from punctsum.materials import LABELS  # noqa: E402
from punctsum.model import load, peak_rss_gb, tokenize_with_readout  # noqa: E402
from punctsum.mood import MOOD_ANSWERS, screen_adjectives, screen_fillers  # noqa: E402

MANIFEST = PROJECT_ROOT / "outputs" / "m6" / "models_manifest.json"


def out_dir(name: str) -> Path:
    d = PROJECT_ROOT / "outputs" / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def revision(name: str) -> str:
    return json.loads(MANIFEST.read_text())[name]["revision"]


def load_model(name: str):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model = load(MODELS[name], revision=revision(name), stage_on_cpu=True)  # same path for every scaling model
    return model, {"load_s": round(time.time() - t, 1), "load_peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "weights_gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


def free() -> None:
    """Release GPU memory. Callers must `del` their own references first; deleting inside a helper only drops the
    helper's reference (the bug that made the first 12B verification run out of memory)."""
    gc.collect()
    torch.cuda.empty_cache()


def sample_texts(model_for_render, n_pairs: int = 8) -> list[tuple[str, str, str]]:
    """(full text with continuation readout, true answer, other answer) for a few P passages of both properties."""
    out = []
    for path in (PROJECT_ROOT / "outputs" / "m2" / "pairs.jsonl", PROJECT_ROOT / "outputs" / "m3" / "pairs_mood.jsonl"):
        for p in load_jsonl(path)[:n_pairs]:
            prop = property_of(p)
            for lab in prop.labels:
                other = next(x for x in prop.labels if x != lab)
                pieces = render_condition(p, lab, "P", model_for_render)
                text, *_ = tokenize_with_readout(model_for_render, pieces, prop.readouts["continuation"])
                out.append((text, prop.answer(lab), prop.answer(other)))
    return out


@torch.no_grad()
def verify(name: str) -> dict:
    """Compare TransformerBridge (compatibility mode) with the plain Hugging Face model on the same inputs: next-token
    log-probabilities at every position (compatibility mode centers the unembedding, which shifts logits by a constant
    per position, so log-probabilities are the right comparison) and the readout's answer logit difference."""
    from transformers import AutoModelForCausalLM

    model, mem = load_model(name)
    texts = sample_texts(model)
    tl_lp, tl_ld, toks_all = [], [], []
    for text, good, bad in texts:
        toks = model.to_tokens(text, prepend_bos=True)
        logits = model(toks)[0]
        tl_lp.append(logits.log_softmax(-1).cpu())
        tl_ld.append((logits[-1, model.to_single_token(good)] - logits[-1, model.to_single_token(bad)]).item())
        toks_all.append(toks)
    ids = [(model.to_single_token(g), model.to_single_token(b)) for _, g, b in texts]
    del model, logits
    free()

    hf = AutoModelForCausalLM.from_pretrained(MODELS[name], revision=revision(name), torch_dtype=torch.float32).to("cuda")
    hf.eval()
    max_lp, max_ld = 0.0, 0.0
    for toks, lp, ld, (g, b) in zip(toks_all, tl_lp, tl_ld, ids):
        logits = hf(toks.to("cuda")).logits[0]
        max_lp = max(max_lp, (logits.log_softmax(-1).cpu() - lp).abs().max().item())
        max_ld = max(max_ld, abs((logits[-1, g] - logits[-1, b]).item() - ld))
    del hf, logits
    free()
    res = {**mem, "n_texts": len(texts), "max_abs_logprob_diff": max_lp, "max_abs_answer_logit_diff_diff": max_ld,
           "peak_rss_gb": round(peak_rss_gb(), 2)}
    (out_dir(name) / "verify.json").write_text(json.dumps(res, indent=1))
    print(f"[{name}] {res}", flush=True)
    return res


def screen(name: str) -> None:
    model, _ = load_model(name)
    d = out_dir(name) / "screen"
    d.mkdir(exist_ok=True)
    for lab in LABELS + list(MOOD_ANSWERS.values()):
        model.to_single_token(" " + lab)  # raises if an answer word is not a single token for this tokenizer
    cities = screen_cities(model)
    adj = screen_adjectives(model)
    fil = screen_fillers(model)
    save_jsonl(cities, d / "city_screen.jsonl")
    save_jsonl(adj, d / "adj_screen.jsonl")
    save_jsonl(fil, d / "filler_screen.jsonl")
    kept = {lab: sorted(r["city"] for r in cities if r["keep"] and r["label"] == lab) for lab in LABELS}
    buckets = {}
    for r in cities:
        if r["keep"]:
            buckets.setdefault(r["ntok"], {lab: 0 for lab in LABELS})[r["label"]] += 1
    pairable = sum(min(b.values()) for b in buckets.values())
    adj_kept = {lab: sum(r["keep"] and r["label"] == lab for r in adj) for lab in ("good", "bad")}
    summary = {"cities_kept": {lab: len(v) for lab, v in kept.items()}, "city_buckets": buckets, "pairable_per_side": pairable,
               "adjectives_kept": adj_kept, "adj_neutral_ref": adj[0]["ref"]}
    (d / "summary.json").write_text(json.dumps({**summary, "cities": kept}, indent=1))
    print(f"[{name}] {summary}", flush=True)
    del model
    free()


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    for n in names:
        {"verify": verify, "screen": screen}[cmd](n)
