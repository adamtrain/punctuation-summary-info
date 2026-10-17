"""Milestone 3 summary: how each condition's probe accuracy differs from P at the same position.

For each property and filler length k, a heatmap with rows = positions (P's landmark rows) and columns = conditions;
each cell is the mean over layers 1-12 of acc(condition, row) - acc(P, row). Comparing the same row across conditions
avoids the lexical-variability confound between different rows (BRIEF.md section 10.8). Rows a condition lacks (e.g.
punctuation in U) are blank. Diverging palette: blue = more decodable than in P, red = less, gray midpoint = same.
Writes outputs/m3/summary_{prop}.png and outputs/m3/summary.csv.
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

OUT = PROJECT_ROOT / "outputs" / "m3"
KS = (1, 4, 16)
CONDS = [c for c in INTERNAL_CONDITIONS if c != "P"]
# Diverging blue <-> red with the neutral gray midpoint (dataviz reference palette poles).
DIVERGING = ["#a8302f", "#e34948", "#f2a6a5", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]
BLANK, INK, INK_MUTED = "#ffffff", "#1f1f1f", "#6b6b6b"
LIM = 0.15  # color scale: +-0.15 accuracy


def load(prop, k, cond):
    s = json.loads((OUT / f"probes_{prop}_k{k}_{cond}.json").read_text())
    return dict(zip(s["rows"], np.array(s["acc"])))


def main():
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    out_rows = []
    for prop in ("mood", "country"):
        fig, axes = plt.subplots(1, len(KS), figsize=(6.4 * len(KS), 9.0), dpi=150,
                                 gridspec_kw={"wspace": 0.85})  # room for each panel's own row labels
        for ax, k in zip(axes, KS):
            p = load(prop, k, "P")
            rows = list(p)
            delta = np.full((len(rows), len(CONDS)), np.nan)
            for j, c in enumerate(CONDS):
                acc = load(prop, k, c)
                for i, r in enumerate(rows):
                    if r in acc:
                        delta[i, j] = (acc[r][1:] - p[r][1:]).mean()
                        out_rows.append({"prop": prop, "k": k, "cond": c, "row": r, "delta_L1_12": delta[i, j],
                                         "p_acc_L1_12": p[r][1:].mean()})
            rgba = cmap(np.clip((np.nan_to_num(delta) + LIM) / (2 * LIM), 0, 1))
            rgba[np.isnan(delta)] = matplotlib.colors.to_rgba(BLANK)
            ax.imshow(rgba, aspect="auto", interpolation="nearest")
            for i in range(len(rows)):
                for j in range(len(CONDS)):
                    v = delta[i, j]
                    ax.text(j, i, "—" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=6,
                            color=INK_MUTED if np.isnan(v) or abs(v) < 0.03 else INK)
            ax.set_xticks(range(len(CONDS)), CONDS, fontsize=7, color=INK)
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(rows)), [f"{r}  (P {p[r][1:].mean():.2f})" for r in rows], fontsize=6.5, color=INK)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"k={k}", fontsize=8, color=INK_MUTED, loc="left", pad=18)
        fig.suptitle(f"{prop}: probe accuracy relative to P at the same position (mean over L1–L12; "
                     f"blue = higher, red = lower; — = position absent)", fontsize=9, color=INK, x=0.01, ha="left")
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(-LIM, LIM))
        cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
        cb.set_label("accuracy difference vs P", fontsize=7, color=INK_MUTED)
        cb.ax.tick_params(labelsize=6, colors=INK_MUTED)
        cb.outline.set_visible(False)
        fig.savefig(OUT / f"summary_{prop}.png", bbox_inches="tight")
        plt.close(fig)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote summary figures and {len(out_rows)} rows")


if __name__ == "__main__":
    main()
