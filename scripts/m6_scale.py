"""Milestone 6: run the frozen protocol (git tag method-freeze-v1; BRIEF.md decision log) on a scaling model.

  python scripts/m6_scale.py verify pythia-160m [pythia-410m ...]   # TransformerLens vs plain Hugging Face, memory
  python scripts/m6_scale.py screen pythia-160m [...]               # city, adjective and filler screens
  python scripts/m6_scale.py bf16check llama-3.1-8b                 # bf16 vs fp32 numerics
  python scripts/m6_scale.py run pythia-160m [calib behavior knockout accept patch probe]   # the frozen protocol

`run` steps (BRIEF.md decision log, method freeze + "option 2"): calibrate control C for this model; country behavior
(all 8 conditions, question readout); knockout (country with both readouts, mood with the continuation readout); mood
acceptance test (consistent and mixed); mood patching (P, U, U-cap, M, K, C_mid; 12 evenly spaced layers); mood probes
(P and U at k=16). Filler lengths k in {1, 4, 16}. Pythia models use the shared ladder items (outputs/models/pythia-ladder).

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
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import punctsum.dataset as dataset  # noqa: E402
from punctsum.behavior import pair_bootstrap, passage_label_pairs, score  # noqa: E402
from punctsum.dataset import CONDITIONS  # noqa: E402

KS = (1, 4, 16)
INTERNAL = ["P", "U", "Ucap", "M", "K", "C_mid"]  # internal-analysis conditions after the freeze's strikes
PUNCTUATED = ["P", "M", "K", "C_mid"]
KNOCKOUTS = ["none", "s0words", "s0all", "punctonly"]
C_GRID = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26]
C_TARGETS = {"C_lo": 0.5, "C_mid": 1.0, "C_hi": 1.5}

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


@torch.no_grad()
def bf16_check(name: str) -> dict:
    """How much does computing in bf16 change results compared with fp32 (decision log: report this for the modern
    model)? Same passages as `verify`; compares the readout's answer logit difference and per-token log-probabilities."""
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        model = load(MODELS[name], revision=revision(name), dtype=dtype, stage_on_cpu=True)
        texts = sample_texts(model)
        lds, lps = [], []
        for text, good, bad in texts:
            logits = model(model.to_tokens(text, prepend_bos=True))[0].float()
            lds.append((logits[-1, model.to_single_token(good)] - logits[-1, model.to_single_token(bad)]).item())
            lps.append(logits.log_softmax(-1).cpu())
        out[str(dtype)] = (lds, lps)
        del model, logits
        free()
    (a, la), (b, lb) = out["torch.float32"], out["torch.bfloat16"]
    d = np.abs(np.array(a) - np.array(b))
    res = {"n_texts": len(a), "answer_diff_fp32_mean": float(np.mean(a)), "max_abs_answer_diff_change": float(d.max()),
           "mean_abs_answer_diff_change": float(d.mean()),
           "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
           "max_abs_logprob_change": max(float((x - y).abs().max()) for x, y in zip(la, lb))}
    (out_dir(name) / "bf16_check.json").write_text(json.dumps(res, indent=1))
    print(f"[{name}] bf16 vs fp32: {res}", flush=True)
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


def items_dir(name: str) -> Path:
    """Shared item sets: the Pythia ladder's (decision log, option 2); GPT-2's items for GPT-2 and GPT-Neo-125M (same
    tokenizer, so identical inputs); otherwise the model's own."""
    if name.startswith("pythia"):
        return PROJECT_ROOT / "outputs" / "models" / "pythia-ladder"
    if name in ("gpt2", "gpt-neo-125m"):
        return PROJECT_ROOT / "outputs" / "models" / "gpt2-items"
    return PROJECT_ROOT / "outputs" / "models" / name


def ladder_pairs(name: str, kind: str) -> list[dict]:
    return load_jsonl(items_dir(name) / f"pairs_{kind}.jsonl")


def by_k(pairs: list[dict], per_k: int | None = None) -> list[dict]:
    return [p for k in KS for p in [q for q in pairs if q["k"] == k][:per_k]]


def max_tokens_for(model) -> int:
    gb = torch.cuda.memory_allocated() / 1e9
    return 32768 if gb < 20 else 16384 if gb < 40 else 8192


def pair_means(rows: list[dict], key: str = "diff") -> dict[tuple, float]:
    acc: dict[tuple, list[float]] = {}
    for r in rows:
        acc.setdefault((r["group"], r["cond"], r["knockout"], r["pair_id"]), []).append(r[key])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def ratio_ci(num: dict, den: dict, n_boot: int = 2000):
    ids = sorted(num)
    a, b = np.array([num[i] for i in ids]), np.array([den[i] for i in ids])
    idx = np.random.default_rng(0).integers(0, len(ids), size=(n_boot, len(ids)))
    bs = a[idx].mean(1) / b[idx].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def step_calib(model, name, out):
    """Choose control C's substitution rates for this model: 0.5x / 1x / 1.5x U's word-NLL increase over P."""
    pairs = [p for p in ladder_pairs(name, "country") if p["split"] == "train"]
    calib = [p for k in KS for p in [q for q in pairs if q["k"] == k][:40]]
    conds = ["P", "U"] + [f"C@{r}" for r in C_GRID]
    items = passage_label_pairs(calib, conds)
    res = score(model, items, max_tokens=max_tokens_for(model))
    nll: dict[str, list[float]] = {}
    for (_, _, c), r in zip(items, res):
        nll.setdefault(c, []).append(r["word_nll"])
    base, u = np.mean(nll["P"]), np.mean(nll["U"]) - np.mean(nll["P"])
    deltas = np.maximum.accumulate([np.mean(nll[f"C@{r}"]) - base for r in C_GRID])  # enforce monotone for interpolation
    rates = {n: round(float(np.interp(m * u, deltas, C_GRID)), 4) for n, m in C_TARGETS.items()}
    capped = [n for n, m in C_TARGETS.items() if m * u > deltas[-1]]
    info = {"u_delta": float(u), "grid": C_GRID, "deltas": deltas.tolist(), "rates": rates, "capped_at_grid_max": capped}
    (out / "calibration.json").write_text(json.dumps(info, indent=1))
    print(f"[{name}] calibration: U +{u:.3f} nats/word; rates {rates}" + (f"; CAPPED {capped}" if capped else ""), flush=True)


def load_rates(name, out):
    rates = json.loads((out / "calibration.json").read_text())["rates"]
    dataset.C_STRENGTHS.update(rates)  # render_condition reads the module-level rates
    return rates


def step_behavior(model, name, out):
    """Country behavior, all 8 conditions, question readout, 400 pairs per k (as GPT-2, Section 10.5)."""
    pairs = by_k(ladder_pairs(name, "country"))
    items = passage_label_pairs(pairs, CONDITIONS)
    res = score(model, items, max_tokens=max_tokens_for(model))
    rows = [{"group": "country", "pair_id": p["pair_id"], "k": p["k"], "label": lab, "cond": c, "knockout": "none", **r}
            for (p, lab, c), r in zip(items, res)]
    save_jsonl(rows, out / "behavior.jsonl")
    summary = summarize_behavior(rows, pairs, KS)
    (out / "behavior_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] behavior: P {summary['all|P']['diff'][0]:+.2f}, U delta {summary['all|U']['delta'][0]:+.2f}, "
          f"C_mid delta {summary['all|C_mid']['delta'][0]:+.2f}", flush=True)


def summarize_behavior(rows, pairs, ks_all) -> dict:
    pm, nll = pair_means(rows), pair_means(rows, "word_nll")
    k_of = {p["pair_id"]: p["k"] for p in pairs}
    summary = {}
    for ks, kn in [((k,), f"k={k}") for k in ks_all] + [(tuple(ks_all), "all")]:
        sel = lambda c, d: {pid: v for (g, cc, ko, pid), v in d.items() if cc == c and k_of[pid] in ks}  # noqa: E731
        base, nb = sel("P", pm), sel("P", nll)
        u = np.mean([sel("U", nll)[i] - nb[i] for i in nb])
        for c in CONDITIONS:
            d = sel(c, pm)
            summary[f"{kn}|{c}"] = {"diff": pair_bootstrap(d), "delta": pair_bootstrap({i: d[i] - base[i] for i in d}),
                                    "xU": float(np.mean([sel(c, nll)[i] - nb[i] for i in nb]) / u)}
    return summary


def knockout_rows(model, pairs, group, readout, conds=INTERNAL, knockouts=KNOCKOUTS):
    rows = []
    for ko in knockouts:
        items = passage_label_pairs(pairs, PUNCTUATED if ko == "punctonly" else conds)
        for (p, lab, c), r in zip(items, score(model, items, max_tokens=max_tokens_for(model), readout=readout, knockout=ko)):
            rows.append({"group": group, "pair_id": p["pair_id"], "k": p["k"], "label": lab, "cond": c, "knockout": ko,
                         "diff": r["diff"]})
    return rows


def summarize_knockout(rows, pairs, groups, ks_all=KS):
    pm = pair_means(rows)
    k_of = {p["pair_id"]: p["k"] for p in pairs}
    summary = {}
    for g in groups:
        for ks, kn in [((k,), f"k={k}") for k in ks_all] + [(tuple(ks_all), "all")]:
            for c in INTERNAL:
                sel = {ko: {pid: v for (gg, cc, kk, pid), v in pm.items() if gg == g and cc == c and kk == ko and k_of[pid] in ks}
                       for ko in KNOCKOUTS}
                cell = {ko: pair_bootstrap(v) for ko, v in sel.items() if v}
                if sel["none"] and sel["s0words"] and sel["s0all"]:
                    cell["retention_s0words"] = ratio_ci(sel["s0words"], sel["none"])
                    cell["s0punct_share"] = ratio_ci({i: sel["s0words"][i] - sel["s0all"][i] for i in sel["s0all"]},
                                                     sel["none"])
                summary[f"{g}|{kn}|{c}"] = cell
    return summary


def step_knockout(model, name, out):
    """Knockouts (BRIEF.md 4.7): country with both readouts, mood with the continuation readout; 100 pairs per k."""
    country = by_k(ladder_pairs(name, "country"), 100)
    mood = by_k(ladder_pairs(name, "mood"), 100)
    rows = (knockout_rows(model, country, "country/question", "question")
            + knockout_rows(model, country, "country/continuation", "continuation")
            + knockout_rows(model, mood, "mood", "continuation"))
    save_jsonl(rows, out / "knockout.jsonl")
    summary = summarize_knockout(rows, country + mood, ["country/question", "country/continuation", "mood"])
    (out / "knockout_summary.json").write_text(json.dumps(summary, indent=1))
    m = summary["mood|all|P"]
    print(f"[{name}] knockout mood P: none {m['none'][0]:+.2f}, retention {m['retention_s0words'][0]:.2f}, "
          f"s0 punct share {m['s0punct_share'][0]:.2f}; U retention {summary['mood|all|U']['retention_s0words'][0]:.2f}", flush=True)


def step_accept(model, name, out):
    """Mood acceptance test with the pre-registered criteria (C1: pair acc >= 0.90 and AUC >= 0.80; C2: s0words
    retention >= 0.25), P, continuation readout, pooled over k. Consistent: from the knockout run; mixed: run here."""
    result = {}
    for variant, kind in (("consistent", "mood"), ("mixed", "mood_mixed")):
        if variant == "consistent":
            rows = [r for r in load_jsonl(out / "knockout.jsonl") if r["group"] == "mood" and r["cond"] == "P"]
            pairs = by_k(ladder_pairs(name, "mood"), 100)
        else:
            pairs = by_k(ladder_pairs(name, kind), 100)
            rows = knockout_rows(model, pairs, "mood_mixed", "continuation", ["P"], ["none", "s0words"])
            save_jsonl(rows, out / "accept_mixed.jsonl")
        pm = pair_means(rows)
        none = {pid: v for (g, c, ko, pid), v in pm.items() if ko == "none"}
        s0w = {pid: v for (g, c, ko, pid), v in pm.items() if ko == "s0words"}
        member = [(int(r["label"] == "good"), r["diff"] if r["label"] == "good" else -r["diff"])
                  for r in rows if r["knockout"] == "none"]
        pair_acc = sum(v > 0 for v in none.values()) / len(none)
        auc = float(roc_auc_score([y for y, _ in member], [x for _, x in member]))
        ret = ratio_ci(s0w, none)
        result[variant] = {"pair_acc": pair_acc, "auc": auc, "retention": ret, "C1": pair_acc >= 0.90 and auc >= 0.80,
                           "C2": ret[0] >= 0.25}
        print(f"[{name}] accept {variant}: pair acc {pair_acc:.3f}, AUC {auc:.3f}, retention {ret[0]:.2f} -> "
              f"C1 {'PASS' if result[variant]['C1'] else 'FAIL'}, C2 {'PASS' if result[variant]['C2'] else 'FAIL'}", flush=True)
    (out / "accept.json").write_text(json.dumps(result, indent=1))


def step_patch(model, name, out):
    """Mood patching: P, U, U-cap, M, K, C_mid; k in {1, 4, 16}; 50 pairs; 12 evenly spaced layers."""
    from m5_patching import run as patch_run
    from m5_summary import summarize

    n_layers = model.cfg.n_layers
    layers = sorted({int(round(x)) for x in np.linspace(1, n_layers, 12)})
    pairs = ladder_pairs(name, "mood")
    d = out / "patching"
    for k in KS:
        for c in INTERNAL:
            patch_run(model, "mood", k, c, 50, max_tokens_for(model), pairs=pairs, out=d, layers=layers)
    summ = {f.stem.removeprefix("patching_"): summarize(json.loads(f.read_text())) for f in sorted(d.glob("patching_*.json"))}
    (out / "patching_summary.json").write_text(json.dumps(summ, indent=1))


def step_probe(model, name, out):
    """Mood probes, P and U at k=16 (presence only; probes saturate). The GPU part (caching activations) runs here; the
    fitting (same code as GPT-2: m3_probes.run on the cache) runs in a separate CPU process, so the GPU can move on."""
    import subprocess

    from m3_probes import cached_activations

    pairs = [p for p in ladder_pairs(name, "mood") if p["k"] == 16]
    for c in ("P", "U"):
        cached_activations(model, pairs, c, 16, out / "probes" / "acts" / f"mood_k16_{c}.npz")
    log = open(out / "probes_fit.log", "w")
    subprocess.Popen([sys.executable, __file__, "fitprobes", name], stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    print(f"[{name}] probe activations cached; fitting started in a separate CPU process (log: probes_fit.log)", flush=True)


def fit_probes(name: str) -> None:
    from m3_probes import run as probe_run

    out = out_dir(name)
    t = time.time()
    pairs = ladder_pairs(name, "mood")
    for c in ("P", "U"):
        probe_run(None, "mood", 16, c, pairs=pairs, out=out / "probes")
    (out / "probes_fit_seconds.txt").write_text(f"{time.time() - t:.0f}\n")


def step_long(model, name, out):
    """Long-distance condition (decision log): property sentence + PG19 excerpt + readout. Behavior (country, all
    conditions, question readout) and knockout (none / s0words / s0all; country with both readouts, mood with the
    continuation readout); 100 pairs per level. L900 for every model; L1500 only if the context window allows it."""
    n_ctx = model.cfg.n_ctx
    levels = ["L900"] + (["L1500"] if n_ctx >= 2048 else [])
    d = items_dir(name)
    country = [p for lv in levels for p in load_jsonl(d / f"pairs_country_{lv}.jsonl")]
    mood = [p for lv in levels for p in load_jsonl(d / f"pairs_mood_{lv}.jsonl")]
    mt = max_tokens_for(model)
    items = passage_label_pairs(country, CONDITIONS)
    res = score(model, items, max_tokens=mt)
    longest = max(r["tokens"] for r in res)
    if longest > n_ctx:
        raise ValueError(f"{name}: a long-distance passage has {longest} tokens, over the {n_ctx}-token window")
    brows = [{"group": "country", "pair_id": p["pair_id"], "k": p["k"], "label": lab, "cond": c, "knockout": "none", **r}
             for (p, lab, c), r in zip(items, res)]
    save_jsonl(brows, out / "long_behavior.jsonl")
    kos = ["none", "s0words", "s0all"]
    krows = (knockout_rows(model, country, "country/question", "question", knockouts=kos)
             + knockout_rows(model, country, "country/continuation", "continuation", knockouts=kos)
             + knockout_rows(model, mood, "mood", "continuation", knockouts=kos))
    save_jsonl(krows, out / "long_knockout.jsonl")
    summary = {"levels": levels, "longest_tokens": longest, "n_ctx": n_ctx,
               "behavior": summarize_behavior(brows, country, levels),
               "knockout": summarize_knockout(krows, country + mood, ["country/question", "country/continuation", "mood"],
                                              levels)}
    (out / "long_summary.json").write_text(json.dumps(summary, indent=1))
    for lv in levels:
        b, m = summary["behavior"], summary["knockout"]
        print(f"[{name}] long {lv}: country P {b[f'{lv}|P']['diff'][0]:+.2f}, U delta {b[f'{lv}|U']['delta'][0]:+.2f}, "
              f"C_mid delta {b[f'{lv}|C_mid']['delta'][0]:+.2f}; mood P retention "
              f"{m[f'mood|{lv}|P']['retention_s0words'][0]:.2f}, U retention {m[f'mood|{lv}|U']['retention_s0words'][0]:.2f}",
              flush=True)


STEPS = {"calib": step_calib, "behavior": step_behavior, "knockout": step_knockout, "accept": step_accept,
         "patch": step_patch, "probe": step_probe, "long": step_long}


def run(name: str, steps: list[str]) -> None:
    out = out_dir(name)
    model, mem = load_model(name)
    log_path = out / "run_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log["load"] = mem
    for st in steps:
        if st != "calib":
            load_rates(name, out)
        t = time.time()
        STEPS[st](model, name, out)
        log[st] = {"seconds": round(time.time() - t, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log_path.write_text(json.dumps(log, indent=1))
        print(f"[{name}] step {st} done in {log[st]['seconds']:.0f}s", flush=True)
    del model
    free()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3:] or list(STEPS))
    else:
        for n in sys.argv[2:]:
            {"verify": verify, "screen": screen, "fitprobes": fit_probes, "bf16check": bf16_check}[cmd](n)
