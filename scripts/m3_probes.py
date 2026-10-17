"""Milestone 3: linear probes by (layer x position) with a pair-level permutation null, and heatmaps.

  python scripts/m3_probes.py --prop mood --k 4 --cond P
  python scripts/m3_probes.py --prop country --k 4 --cond P --check-c
  python scripts/m3_probes.py --all     # every property x k in {1, 4, 16} x internal condition, model loaded once

For every passage of the given property, filler length and condition: render, append the continuation readout,
label tokens, map landmark rows to token positions (punctsum/probe_positions.py), and cache the residual stream at
those positions at every depth (outputs/m3/acts/*.npz, float16). Then, for each (row, layer): ridge probe trained
logistic probe trained on the train split (held-out templates, fillers and cities/adjectives in test), accuracy on the
test split, and a pair-level permutation null on the test labels (10,000 permutations; punctsum/probes.py). Writes a CSV (the table view), a JSON summary and a heatmap PNG to outputs/m3/.
Mood pairs (consistent valence, 400 per filler length) are generated once into outputs/m3/pairs_mood.jsonl.
"""

import argparse
import csv
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from punctsum import PROJECT_ROOT  # noqa: E402
from punctsum.dataset import INTERNAL_CONDITIONS, load_jsonl, property_of, render_condition, save_jsonl  # noqa: E402
from punctsum.model import load, peak_rss_gb, resid_at, tokenize_with_readout  # noqa: E402
from punctsum.mood import make_mood_pairs, neutral_fillers  # noqa: E402
from punctsum.probe_positions import probe_positions  # noqa: E402
from punctsum.probes import PROBE_C, fit_predict, pair_flip_null  # noqa: E402
from punctsum.roles import label_tokens  # noqa: E402

SEED = 0
N_PERM = 10000
# Rows present in too few passages (e.g. a 5th misplaced mark in condition M) are not probed.
MIN_TEST, MIN_TRAIN = 40, 80
OUT = PROJECT_ROOT / "outputs" / "m3"
# Sequential blue ramp (dataviz reference palette, steps 100 -> 700) for accuracy 0.5 -> 1.0; cells that do not beat
# the permutation null are drawn in the neutral gray instead.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#f0efec"
INK, INK_MUTED, INK_ON_DARK = "#1f1f1f", "#6b6b6b", "#ffffff"


def load_pairs(prop: str) -> list[dict]:
    if prop == "country":
        return load_jsonl(PROJECT_ROOT / "outputs" / "m2" / "pairs.jsonl")
    path = OUT / "pairs_mood.jsonl"
    if not path.exists():
        adj = load_jsonl(PROJECT_ROOT / "outputs" / "m2b" / "adj_screen.jsonl")
        fillers = neutral_fillers(load_jsonl(PROJECT_ROOT / "outputs" / "m2b" / "filler_screen.jsonl"))
        save_jsonl(make_mood_pairs(adj, fillers, 400, SEED, "consistent"), path)
    return load_jsonl(path)


def collect(model, pairs: list[dict], cond: str, k: int):
    """Activations at landmark rows. Returns acts [n, R, L+1, d] (float16), mask [n, R], row names, labels,
    pair ids, splits, and an example token string per row."""
    per_passage, seqs, meta, examples = [], [], [], {}
    for p in pairs:
        prop = property_of(p)
        for label in prop.labels:
            pieces = render_condition(p, label, cond, model)
            _, toks, pot, n_ro = tokenize_with_readout(model, pieces, prop.readouts["continuation"])
            rows = label_tokens(prop.passage_for(p, label), pieces, pot, n_ro)
            pos = probe_positions(rows, k)
            ids = toks[0].tolist()
            for name, t in pos.items():
                examples.setdefault(name, model.to_string(ids[t]))
            per_passage.append(pos)
            seqs.append(ids)
            meta.append((prop.labels.index(label), p["pair_id"], p["split"]))
    names = sorted({n for pos in per_passage for n in pos}, key=lambda n: np.mean([pos[n] for pos in per_passage if n in pos]))
    positions = [[pos.get(n, 0) for n in names] for pos in per_passage]
    mask = np.array([[n in pos for n in names] for pos in per_passage])
    acts = resid_at(model, seqs, positions).astype(np.float16)
    y = np.array([m[0] for m in meta])
    return acts, mask, names, y, np.array([m[1] for m in meta]), np.array([m[2] for m in meta]), examples


def plot(acc, sig, names, examples, n_test, title, path):
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    R, L = acc.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * L, 0.9 + 0.30 * R), dpi=150)
    rgba = cmap(np.clip((acc - 0.5) / 0.5, 0, 1))
    rgba[~sig] = matplotlib.colors.to_rgba(NEUTRAL)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for i in range(R):
        for j in range(L):
            dark = sig[i, j] and acc[i, j] >= 0.80
            ax.text(j, i, f"{acc[i, j]:.2f}"[1:] if acc[i, j] < 1 else "1.0", ha="center", va="center", fontsize=6.5,
                    color=INK_ON_DARK if dark else (INK if sig[i, j] else INK_MUTED))
    ax.set_xticks(range(L), [f"L{j}" for j in range(L)], fontsize=7, color=INK_MUTED)
    ax.set_yticks(range(R), [f"{n}  {examples[n]!r}" for n in names], fontsize=7, color=INK)
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(range(R), [f"n={n}" for n in n_test], fontsize=6, color=INK_MUTED)
    ax2.tick_params(length=0)
    prev = None
    for i, n in enumerate(names):  # separators between segments
        seg = n.split(".")[0]
        if prev is not None and seg != prev:
            ax.axhline(i - 0.5, color="#ffffff", lw=2)
        prev = seg
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("residual stream depth (L0 = embeddings, L12 = last block)", fontsize=7, color=INK_MUTED)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0.5, 1.0))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.12)
    cb.set_label("test accuracy (gray = not above permutation null, 95th pct)", fontsize=6.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(model, prop: str, k: int, cond: str, check_c: bool = False) -> None:
    t0 = time.time()
    pairs = [p for p in load_pairs(prop) if p["k"] == k]
    tag = f"{prop}_k{k}_{cond}"
    cache = OUT / "acts" / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        acts, mask, names, y, pid, split = z["acts"], z["mask"], list(z["names"]), z["y"], z["pid"], z["split"]
        examples = json.loads(str(z["examples"]))
    else:
        acts, mask, names, y, pid, split, examples = collect(model, pairs, cond, k)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, acts=acts, mask=mask, names=np.array(names), y=y, pid=pid, split=split,
                 examples=json.dumps(examples))
    print(f"[{tag}] activations {acts.shape} ({acts.nbytes / 1e9:.2f} GB) at {time.time() - t0:.0f}s", flush=True)

    counts = {n: (int((mask[:, i] & (split == "train")).sum()), int((mask[:, i] & (split == "test")).sum()))
              for i, n in enumerate(names)}
    dropped = [n for n, (a, b) in counts.items() if a < MIN_TRAIN or b < MIN_TEST]
    keep = [i for i, n in enumerate(names) if n not in dropped]
    acts, mask, names = acts[:, keep], mask[:, keep], [names[i] for i in keep]
    if dropped:
        print(f"[{tag}] rows dropped (train/test passages): " + ", ".join(f"{n} {counts[n]}" for n in dropped), flush=True)
    R, L = len(names), acts.shape[2]
    acc, p95, nullmean = np.zeros((R, L)), np.zeros((R, L)), np.zeros((R, L))
    n_test = []
    rows_out = []
    for i, name in enumerate(names):
        tr, te = mask[:, i] & (split == "train"), mask[:, i] & (split == "test")
        n_test.append(int(te.sum()))
        for l in range(L):
            pred = fit_predict(acts[tr, i, l].astype(np.float32), y[tr], acts[te, i, l].astype(np.float32))
            correct = pred == y[te]
            null = pair_flip_null(correct, pid[te], N_PERM, SEED)
            a = float(correct.mean())
            acc[i, l], p95[i, l], nullmean[i, l] = a, np.percentile(null, 95), null.mean()
            rows_out.append({"row": name, "example": examples[name], "layer": l, "acc": a, "null_p95": p95[i, l],
                             "null_mean": nullmean[i, l], "n_train": int(tr.sum()), "n_test": int(te.sum())})
    sig = acc > p95

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"probes_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    summary = {"tag": tag, "rows": names, "acc": acc.tolist(), "null_p95": p95.tolist(), "null_mean": nullmean.tolist(),
               "n_test": n_test, "n_perm": N_PERM, "dropped_rows": {n: counts[n] for n in dropped}}

    if check_c:
        # Is the fixed regularization strength reasonable? Compare test accuracy for other C values on sample
        # cells (a sanity check of the default, not a per-cell choice made on test data).
        rng = np.random.default_rng(SEED)
        cells = [(i, l) for i in rng.choice(R, size=min(6, R), replace=False) for l in (2, 4, 8, 12)]
        comp = []
        for i, l in cells:
            tr, te = mask[:, i] & (split == "train"), mask[:, i] & (split == "test")
            row = {"row": names[i], "layer": l}
            for C in (0.01, 0.1, 1.0):
                pred = fit_predict(acts[tr, i, l].astype(np.float32), y[tr], acts[te, i, l].astype(np.float32), C)
                row[f"C={C}"] = float((pred == y[te]).mean())
            comp.append(row)
        summary["c_check"] = comp
        print(f"C check (default C={PROBE_C}), test accuracy on {len(comp)} sample cells:")
        for c in comp:
            print(f"  {c['row']:18} L{c['layer']:<2} " + "  ".join(f"{k} {v:.3f}" for k, v in c.items() if k.startswith("C=")))

    (OUT / f"probes_{tag}.json").write_text(json.dumps(summary, indent=1))
    prop_note = "held-out templates, fillers, adjectives" if prop == "mood" else "held-out templates, fillers, cities"
    title = (f"Probe accuracy: {prop}, condition {cond}, k={k} filler sentences "
             f"(test = {prop_note}; null = {N_PERM:,} pair-level label permutations)")
    plot(acc, sig, names, examples, n_test, title, OUT / f"heatmap_{tag}.png")
    print(f"[{tag}] null p95 range {p95.min():.3f}..{p95.max():.3f}; significant cells {sig.sum()}/{sig.size}; "
          f"runtime {time.time() - t0:.0f}s; peak RSS {peak_rss_gb():.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", choices=["mood", "country"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--cond", default="P")
    ap.add_argument("--check-c", action="store_true", help="compare probe strength C values on sample cells")
    ap.add_argument("--all", action="store_true", help="every property x k in {1, 4, 16} x internal condition")
    args = ap.parse_args()
    model = load("gpt2")
    if args.all:
        for prop in ("mood", "country"):
            for k in (1, 4, 16):
                for cond in INTERNAL_CONDITIONS:
                    run(model, prop, k, cond)
    else:
        run(model, args.prop, args.k, args.cond, args.check_c)


if __name__ == "__main__":
    main()
