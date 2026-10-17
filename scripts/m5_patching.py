"""Milestone 5: activation patching heatmaps (BRIEF.md section 4.6).

  python scripts/m5_patching.py --prop mood --k 4 --cond P
  python scripts/m5_patching.py --all            # every property x k in {1, 4, 16} x internal condition

For each of the first N pairs at filler length k: clean = one member, corrupted = the other (alternating by pair, so
neither label is always clean). For every landmark row (punctsum/probe_positions.py) and layer 1-12, patch the clean
residual at that position into the corrupted run and record the readout's logit difference (continuation readout).
Recovery per cell = mean(LD_patched - LD_corrupted) / mean(LD_clean - LD_corrupted) over pairs, with a 95% bootstrap
CI over pairs. Span rows patch several positions at once (the diffusion test, BRIEF.md 4.6):
  span.s0_punct       all punctuation in the property sentence (absent without punctuation)
  span.s0_after_prop  every s0 token after the last property word
  span.filler_punct   all punctuation in the filler sentences
  span.fillers        every filler token
  span.passage_after  every passage token after the last property word (s0 rest + fillers)
  span.readout_pre    every readout token except the final one Layer 0 is skipped: there a patch either changes nothing (same token in both members) or swaps the
property word itself. Runs on the best available device (punctsum.model.default_device: CUDA, MPS, CPU). Writes
outputs/m5/.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_probes import BLUES, INK, INK_MUTED, INK_ON_DARK, NEUTRAL, load_pairs  # noqa: E402

from punctsum import PROJECT_ROOT  # noqa: E402
from punctsum.dataset import INTERNAL_CONDITIONS, property_of, render_condition  # noqa: E402
from punctsum.model import load, peak_rss_gb, tokenize_with_readout  # noqa: E402
from punctsum.patching import patch_pair  # noqa: E402
from punctsum.probe_positions import probe_positions  # noqa: E402
from punctsum.roles import label_tokens  # noqa: E402

N_PAIRS = 50
SPANS = ["span.s0_punct", "span.s0_after_prop", "span.filler_punct", "span.fillers", "span.passage_after",
         "span.readout_pre"]
PUNCT_KINDS = {"comma", "period", "misplaced_comma", "misplaced_period"}


def spans_for(rows: list[dict]) -> dict[str, list[int]]:
    last_prop = max(t for t, r in enumerate(rows) if r["kind"] == "property")
    s0 = [t for t, r in enumerate(rows) if r["segment"] == "s0"]
    fill = [t for t, r in enumerate(rows) if r["segment"].startswith("f")]
    ro = [t for t, r in enumerate(rows) if r["segment"] == "readout"]
    spans = {
        "span.s0_punct": [t for t in s0 if rows[t]["kind"] in PUNCT_KINDS],
        "span.s0_after_prop": [t for t in s0 if t > last_prop],
        "span.filler_punct": [t for t in fill if rows[t]["kind"] in PUNCT_KINDS],
        "span.fillers": fill,
        "span.passage_after": [t for t in s0 + fill if t > last_prop],
        "span.readout_pre": ro[:-1],
    }
    return {n: v for n, v in spans.items() if v}
LAYERS = list(range(1, 13))
N_BOOT = 2000
OUT = PROJECT_ROOT / "outputs" / "m5"


def run(model, prop_name: str, k: int, cond: str, n_pairs: int = N_PAIRS, max_tokens: int = 4096) -> None:
    t0 = time.time()
    tag = f"{prop_name}_k{k}_{cond}"
    pairs = [p for p in load_pairs(prop_name) if p["k"] == k][:n_pairs]
    records = []  # per pair: ld_clean, ld_corr, {row: [LD per layer]}
    examples: dict[str, str] = {}
    for i, p in enumerate(pairs):
        prop = property_of(p)
        clean_lab, corr_lab = (prop.labels[0], prop.labels[1]) if i % 2 == 0 else (prop.labels[1], prop.labels[0])
        enc = {}
        for lab in (clean_lab, corr_lab):
            pieces = render_condition(p, lab, cond, model)
            _, toks, pot, n_ro = tokenize_with_readout(model, pieces, prop.readouts["continuation"])
            enc[lab] = (toks[0].tolist(), label_tokens(prop.passage_for(p, lab), pieces, pot, n_ro))
        clean, corr = enc[clean_lab][0], enc[corr_lab][0]
        pos = probe_positions(enc[corr_lab][1], k)
        for n, t in pos.items():
            examples.setdefault(n, model.to_string(corr[t]))
        spans = spans_for(enc[corr_lab][1])
        rows = list(pos) + list(spans)
        targets = {**pos, **spans}
        cells = [(l, targets[r]) for r in rows for l in LAYERS]
        answers = (model.to_single_token(prop.answer(clean_lab)), model.to_single_token(prop.answer(corr_lab)))
        ldc, ldr, lds = patch_pair(model, clean, corr, cells, answers, max_tokens)
        records.append({"pair_id": p["pair_id"], "ld_clean": ldc, "ld_corr": ldr,
                        "pos": {**pos, **{s: 10_000 + j for j, s in enumerate(SPANS)}},  # spans sort last
                        "patched": {r: lds[j * len(LAYERS):(j + 1) * len(LAYERS)] for j, r in enumerate(rows)}})

    # Rows in passage order (mean token position), as in the probe heatmaps.
    names = sorted({r for rec in records for r in rec["patched"]},
                   key=lambda r: np.mean([rec["pos"][r] for rec in records if r in rec["pos"]]))
    gap = np.array([rec["ld_clean"] - rec["ld_corr"] for rec in records])
    rng = np.random.default_rng(0)
    R, L = len(names), len(LAYERS)
    rec_mean, lo, hi, n_used = np.zeros((R, L)), np.zeros((R, L)), np.zeros((R, L)), []
    rows_out = []
    for i, r in enumerate(names):
        idx = [j for j, rec in enumerate(records) if r in rec["patched"]]
        n_used.append(len(idx))
        num = np.array([[records[j]["patched"][r][l] - records[j]["ld_corr"] for l in range(L)] for j in idx])  # [n, L]
        den = gap[idx]
        boot = rng.integers(0, len(idx), size=(N_BOOT, len(idx)))
        bs = num[boot].mean(1) / den[boot].mean(1)[:, None]  # [N_BOOT, L]
        rec_mean[i] = num.mean(0) / den.mean()
        lo[i], hi[i] = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
        for l in range(L):
            rows_out.append({"row": r, "example": examples.get(r, ""), "layer": LAYERS[l], "recovery": rec_mean[i, l],
                             "ci_lo": lo[i, l], "ci_hi": hi[i, l], "n_pairs": len(idx)})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"patching_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    summary = {"tag": tag, "rows": names, "layers": LAYERS, "recovery": rec_mean.tolist(), "ci_lo": lo.tolist(),
               "ci_hi": hi.tolist(), "n_pairs": n_used, "mean_gap": float(gap.mean()),
               "mean_ld_clean": float(np.mean([r["ld_clean"] for r in records])),
               "mean_ld_corr": float(np.mean([r["ld_corr"] for r in records]))}
    (OUT / f"patching_{tag}.json").write_text(json.dumps(summary, indent=1))
    plot(rec_mean, lo > 0, names, examples, n_used, tag, gap.mean(), OUT / f"patching_{tag}.png")
    print(f"[{tag}] {len(pairs)} pairs, {R} rows; mean clean-corrupted gap {gap.mean():+.2f}; "
          f"runtime {time.time() - t0:.0f}s; peak RSS {peak_rss_gb():.2f} GB", flush=True)


def plot(rec, sig, names, examples, n_used, tag, gap, path):
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    R, L = rec.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * L, 0.9 + 0.30 * R), dpi=150)
    rgba = cmap(np.clip(rec, 0, 1))
    rgba[~sig] = matplotlib.colors.to_rgba(NEUTRAL)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for i in range(R):
        for j in range(L):
            v = rec[i, j]
            txt = f"{v:.2f}".replace("0.", ".") if abs(v) < 1 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color=INK_ON_DARK if (sig[i, j] and v >= 0.6) else (INK if sig[i, j] else INK_MUTED))
    ax.set_xticks(range(L), [f"L{l}" for l in LAYERS], fontsize=7, color=INK_MUTED)
    ax.set_yticks(range(R), [f"{n}  {examples[n]!r}" if n in examples else n for n in names], fontsize=7, color=INK)
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(range(R), [f"n={n}" for n in n_used], fontsize=6, color=INK_MUTED)
    ax2.tick_params(length=0)
    prev = None
    for i, n in enumerate(names):
        seg = n.split(".")[0]
        if prev is not None and seg != prev:
            ax.axhline(i - 0.5, color="#ffffff", lw=2)
        prev = seg
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("patched layer (residual stream after block)", fontsize=7, color=INK_MUTED)
    prop, k, cond = tag.split("_", 2)  # condition names may contain "_" (C_mid)
    ax.set_title(f"Patching recovery: {prop}, condition {cond}, {k.replace('k', 'k=')} "
                 f"(clean→corrupted, mean gap {gap:+.2f} logits)", fontsize=8.5, color=INK, loc="left")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.12)
    cb.set_label("share of the logit gap recovered (gray = 95% CI includes 0)", fontsize=6.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", choices=["mood", "country"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--cond", default="P")
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="tokens per patching batch; results don't depend on it (larger = faster on big GPUs)")
    args = ap.parse_args()
    model = load("gpt2")
    if args.all:
        for prop in ("mood", "country"):
            for k in (1, 4, 16):
                for cond in INTERNAL_CONDITIONS:
                    if (OUT / f"patching_{prop}_k{k}_{cond}.json").exists():  # resume: skip finished configurations
                        print(f"[{prop}_k{k}_{cond}] exists, skipped", flush=True)
                        continue
                    run(model, prop, k, cond, args.n_pairs, args.max_tokens)
    else:
        run(model, args.prop, args.k, args.cond, args.n_pairs, args.max_tokens)


if __name__ == "__main__":
    main()
