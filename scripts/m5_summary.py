"""Milestone 5 summary: the routes the readout uses, per property x k x condition.

From outputs/m5/patching_*.json, for each configuration take the peak recovery over layers 1-12 (and its layer, and the
95% CI lower bound at that layer) of:
  s0_punct        span: all punctuation in the property sentence (true or misplaced)
  passage_after   span: every passage token after the last property word (s0 rest + fillers)
  fillers         span: every filler token
  best_position   the single non-property, non-readout passage position with the highest peak (relocation test)
  prop_mid        sum of the property rows' recovery at L7 (how much is still read straight from the property words)
Writes outputs/m5/summary.csv and outputs/m5/summary_{prop}.png (one heatmap panel per measure: rows = conditions,
columns = k; gray = 95% CI lower bound at the peak layer <= 0).
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from punctsum import PROJECT_ROOT  # noqa: E402
from punctsum.dataset import INTERNAL_CONDITIONS  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "m5"
KS = (1, 4, 16)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL, BLANK, INK, INK_MUTED, INK_ON_DARK = "#f0efec", "#ffffff", "#1f1f1f", "#6b6b6b", "#ffffff"
MEASURES = [("s0_punct", "span.s0_punct"), ("passage_after", "span.passage_after"), ("fillers", "span.fillers")]
VMAX = 0.4  # color scale top (recoveries above are drawn darkest)


def peak(s, row):
    i = s["rows"].index(row)
    rec, lo = np.array(s["recovery"][i]), np.array(s["ci_lo"][i])
    j = int(np.argmax(rec))
    return float(rec[j]), s["layers"][j], float(lo[j])


def summarize(s):
    out = {}
    for name, row in MEASURES:
        out[name] = peak(s, row) if row in s["rows"] else None
    is_prop = lambda r: ".prop" in r and not r.endswith("+1")  # noqa: E731  (the property word itself)
    cands = [r for r in s["rows"] if not r.startswith(("span.", "ro.")) and not is_prop(r)]
    best = max(cands, key=lambda r: peak(s, r)[0])
    out["best_position"] = (*peak(s, best), best)
    # Property words' recovery at ~58% depth (layer 7 of 12 in GPT-2, where the summary route peaks), so models with
    # different depths are compared at the same relative depth.
    lm = int(np.argmin([abs(l / max(s["layers"]) - 7 / 12) for l in s["layers"]]))
    props = [r for r in s["rows"] if ".prop" in r and not r.endswith("+1")]
    out["prop_mid"] = float(sum(s["recovery"][s["rows"].index(r)][lm] for r in props))
    return out


def main():
    rows_out, table = [], {}
    for prop in ("mood", "country"):
        for k in KS:
            for c in INTERNAL_CONDITIONS:
                path = OUT / f"patching_{prop}_k{k}_{c}.json"
                if not path.exists():
                    continue
                s = json.loads(path.read_text())
                sm = summarize(s)
                table[(prop, k, c)] = sm
                row = {"prop": prop, "k": k, "cond": c, "gap": s["mean_gap"], "prop_L7_sum": sm["prop_mid"]}
                for name, _ in MEASURES:
                    v = sm[name]
                    row.update({f"{name}_peak": v[0] if v else "", f"{name}_layer": v[1] if v else "",
                                f"{name}_ci_lo": v[2] if v else ""})
                b = sm["best_position"]
                row.update({"best_position": b[3], "best_position_peak": b[0], "best_position_layer": b[1],
                            "best_position_ci_lo": b[2]})
                rows_out.append(row)
    if not rows_out:
        print("no patching results yet")
        return
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    panels = MEASURES + [("best_position", None)]
    titles = {"s0_punct": "property sentence's punctuation (span)", "passage_after": "all passage tokens after the property",
              "fillers": "all filler tokens (span)", "best_position": "best single passage position (relocation test)"}
    for prop in ("mood", "country"):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6), dpi=150, gridspec_kw={"wspace": 0.35})
        for ax, (name, _) in zip(axes, panels):
            vals = np.full((len(INTERNAL_CONDITIONS), len(KS)), np.nan)
            sig = np.zeros_like(vals, dtype=bool)
            labels = np.empty(vals.shape, dtype=object)
            for i, c in enumerate(INTERNAL_CONDITIONS):
                for j, k in enumerate(KS):
                    sm = table.get((prop, k, c))
                    if sm is None or sm[name] is None:
                        continue
                    v = sm[name]
                    vals[i, j], sig[i, j] = v[0], v[2] > 0
                    labels[i, j] = f"{v[0]:.2f}\nL{v[1]}" + (f"\n{v[3]}" if name == "best_position" else "")
            rgba = cmap(np.clip(np.nan_to_num(vals) / VMAX, 0, 1))
            rgba[~sig] = matplotlib.colors.to_rgba(NEUTRAL)
            rgba[np.isnan(vals)] = matplotlib.colors.to_rgba(BLANK)
            ax.imshow(rgba, aspect="auto", interpolation="nearest")
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    if np.isnan(vals[i, j]):
                        ax.text(j, i, "—", ha="center", va="center", fontsize=6, color=INK_MUTED)
                        continue
                    dark = sig[i, j] and vals[i, j] >= 0.25
                    ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=5.2,
                            color=INK_ON_DARK if dark else (INK if sig[i, j] else INK_MUTED))
            ax.set_xticks(range(len(KS)), [f"k={k}" for k in KS], fontsize=7, color=INK)
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(INTERNAL_CONDITIONS)), INTERNAL_CONDITIONS, fontsize=7, color=INK)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(titles[name], fontsize=7.5, color=INK, loc="left", pad=16)
        fig.suptitle(f"{prop}: peak patching recovery (share of the logit gap, best layer shown; gray = CI includes 0; "
                     f"— = no such tokens)", fontsize=8.5, color=INK, x=0.01, ha="left", y=1.06)
        fig.savefig(OUT / f"summary_{prop}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"summarized {len(rows_out)} configurations")


if __name__ == "__main__":
    main()
