"""Which token positions get a probe (the rows of the heatmaps; BRIEF.md section 4.5).

Rows are structural landmarks, defined from the role labels so they line up across passages and conditions:
  s0.prop{i}         last token of the i-th property word (city, or i-th mood adjective)
  s0.prop{i}+1       the word right after it
  {seg}.pre_comma{j} word before the j-th true comma boundary; {seg}.comma{j} the comma; {seg}.post_comma{j} the
                     first word after it (in conditions without punctuation the boundary words still exist)
  {seg}.pre_period   last word of the sentence; {seg}.period the period
  {seg}.first        first word of a filler sentence
  {seg}.mis{n}       n-th misplaced punctuation mark in the segment (condition M); {seg}.pre_mis{n} the word before it
  ro.q{i}, ro.final  readout tokens; ro.final is where the answer is predicted
Segments: s0 (property sentence), f1 (first filler), flast (last filler, when k >= 2). Only positions from the first
property token onward are included, so no row can be decided before the label-carrying word appears.
"""

PUNCT = {"comma", "period"}


def probe_positions(rows: list[dict], k: int) -> dict[str, int]:
    """Map row name -> token index for one rendered passage (rows from roles.label_tokens)."""
    out: dict[str, int] = {}
    seg_name = {"s0": "s0", "f1": "f1", f"f{k}": "flast" if k >= 2 else "f1"}
    first_prop = min(t for t, r in enumerate(rows) if r["kind"] == "property")
    counters: dict[str, dict[str, int]] = {}
    prev_prop: str | None = None
    tokens = [(t, r) for t, r in enumerate(rows) if r["word_final"] and t >= first_prop]
    for idx, (t, r) in enumerate(tokens):
        seg, kind = r["segment"], r["kind"]
        if seg == "readout":
            out[f"ro.{kind}"] = t
            continue
        if seg not in seg_name:
            continue
        name = seg_name[seg]
        c = counters.setdefault(seg, {"prop": 0, "comma": 0, "mis": 0})
        nxt = tokens[idx + 1][1] if idx + 1 < len(tokens) else None
        if prev_prop is not None and kind != "property":
            out.setdefault(f"{prev_prop}+1", t)
        prev_prop = None
        if kind == "property":
            c["prop"] += 1
            prev_prop = f"{name}.prop{c['prop']}"
            out[prev_prop] = t
        elif kind == "initial":
            out[f"{name}.first"] = t
        elif kind == "clause_final":
            out[f"{name}.pre_comma{c['comma'] + 1}"] = t
        elif kind == "comma":
            out[f"{name}.comma{c['comma'] + 1}"] = t
        elif kind == "clause_initial":
            c["comma"] += 1
            out[f"{name}.post_comma{c['comma']}"] = t
        elif kind == "sent_final":
            out[f"{name}.pre_period"] = t
        elif kind == "period":
            out[f"{name}.period"] = t
        elif kind.startswith("misplaced_"):
            c["mis"] += 1
            out[f"{name}.mis{c['mis']}"] = t
        if nxt is not None and nxt["kind"].startswith("misplaced_") and nxt["segment"] == seg:
            out[f"{name}.pre_mis{c['mis'] + 1}"] = t
    return out
