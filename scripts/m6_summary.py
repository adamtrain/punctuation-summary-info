"""Cross-model summary of the scaling runs (BRIEF.md Section 10): one column per model, from each model's saved
summaries in outputs/models/<model>/. Writes outputs/models/summary.md (the table used in the brief) and summary.json.

  python scripts/m6_summary.py
"""

import json

from punctsum import PROJECT_ROOT
from punctsum.behavior import pair_bootstrap

OUT = PROJECT_ROOT / "outputs" / "models"
MODELS = ["gpt2", "gpt-neo-125m", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "llama-3.1-8b"]
KS = (1, 4, 16)


def load(model, name):
    p = OUT / model / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, signed=True, ci=None):
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci and signed else f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def continuation_delta(m, cond="U", file="knockout.jsonl", k=None):
    """Country, continuation readout, no knockout: cond's margin minus P's, per pair (mean of the two labels), with a
    pair-bootstrap CI. From the knockout rows, the only place the country continuation readout is scored
    (knockout.jsonl for k = 1, 4, 16; long_knockout.jsonl for the long-distance levels, selected by k)."""
    path = OUT / m / file
    if not path.exists():
        return None
    acc: dict = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if (r["group"] == "country/continuation" and r["knockout"] == "none" and r["cond"] in ("P", cond)
                and (k is None or r["k"] == k)):
            acc.setdefault((r["cond"], r["pair_id"]), []).append(r["diff"])
    mean = {k: sum(v) / len(v) for k, v in acc.items()}
    return pair_bootstrap({pid: mean[(cond, pid)] - mean[("P", pid)] for c, pid in mean if c == "P"})


def column(m):
    b, k, a, p, lg = (load(m, n) for n in ("behavior_summary.json", "knockout_summary.json", "accept.json",
                                           "patching_summary.json", "long_summary.json"))
    if b is None:
        return None
    col = {}
    col["country P margin (question)"] = f(b["all|P"]["diff"][0])
    col["U Δ vs P (pooled)"] = f(b["all|U"]["delta"][0], ci=b["all|U"]["delta"][1:])
    col["U Δ at k = 1 / 4 / 16"] = " / ".join(f(b[f"k={kk}|U"]["delta"][0]) for kk in KS)
    col["Ucap Δ"] = f(b["all|Ucap"]["delta"][0])
    col["C_mid Δ (matched control)"] = f(b["all|C_mid"]["delta"][0], ci=b["all|C_mid"]["delta"][1:])
    col["M Δ / K Δ"] = f"{f(b['all|M']['delta'][0])} / {f(b['all|K']['delta'][0])}"
    if k:
        cd, cm = continuation_delta(m), continuation_delta(m, "C_mid")
        col["country continuation readout: U Δ vs P"] = f(cd[0], ci=cd[1:])
        col["country continuation readout: C_mid Δ"] = f(cm[0], ci=cm[1:])
        md = lambda c, key: k[f"mood|all|{c}"][key]  # noqa: E731
        col["mood P margin"] = f(md("P", "none")[0])
        col["mood P s0words retention"] = f(md("P", "retention_s0words")[0], False)
        col["mood s0-punctuation share (P)"] = f(md("P", "s0punct_share")[0], False, md("P", "s0punct_share")[1:])
        col["mood s0-punctuation share (M / K)"] = (f"{f(md('M', 's0punct_share')[0], False)} / "
                                                   f"{f(md('K', 's0punct_share')[0], False)}")
        col["mood U retention"] = f(md("U", "retention_s0words")[0], False)
    if p:
        col["patching: s0 punct span, P (k=1/4/16)"] = " / ".join(
            f(p[f"mood_k{kk}_P"]["s0_punct"][0], False) for kk in KS)
        col["patching: s0 punct peak layer / depth (k=16, P)"] = (f"L{p['mood_k16_P']['s0_punct'][1]}")
        col["patching: post-property passage, U (k=1/4/16)"] = " / ".join(
            f(p[f"mood_k{kk}_U"]["passage_after"][0], False) for kk in KS)
    if a:
        col["consistent acceptance (AUC; C1/C2)"] = (f"{a['consistent']['auc']:.3f}; "
                                                    f"{'pass' if a['consistent']['C1'] else 'fail'}/"
                                                    f"{'pass' if a['consistent']['C2'] else 'fail'}")
        col["mixed acceptance (AUC; C1/C2)"] = (f"{a['mixed']['auc']:.3f}; {'pass' if a['mixed']['C1'] else 'fail'}/"
                                               f"{'pass' if a['mixed']['C2'] else 'fail'}")
    if lg:
        for lv in lg["levels"]:
            lb, lk = lg["behavior"], lg["knockout"]
            col[f"{lv}: country P margin"] = f(lb[f"k={lv}|P"]["diff"][0])
            col[f"{lv}: U Δ / C_mid Δ"] = f"{f(lb[f'k={lv}|U']['delta'][0])} / {f(lb[f'k={lv}|C_mid']['delta'][0])}"
            cd = continuation_delta(m, "U", "long_knockout.jsonl", lv)
            col[f"{lv}: continuation readout U Δ"] = f(cd[0], ci=cd[1:]) if cd else "—"
            col[f"{lv}: mood retention P / U"] = (f"{f(lk[f'mood|k={lv}|P']['retention_s0words'][0], False)} / "
                                                 f"{f(lk[f'mood|k={lv}|U']['retention_s0words'][0], False)}")
            col[f"{lv}: mood s0-punctuation share (P)"] = f(lk[f"mood|k={lv}|P"]["s0punct_share"][0], False)
    return col


def main():
    cols = {m: c for m in MODELS if (c := column(m))}
    rows = list(dict.fromkeys(r for c in cols.values() for r in c))
    lines = ["| measure | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {r} | " + " | ".join(c.get(r, "—") for c in cols.values()) + " |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(cols, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
