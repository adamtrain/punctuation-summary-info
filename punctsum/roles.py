"""Per-token role labels (BRIEF.md section 4.3).

Every token gets:
  segment   "bos", "s0" (property sentence), "f1".."fk" (filler sentences), or "readout"
  kind      what the token is within its segment:
              words: property > sent_final > clause_final > initial > clause_initial > content
                     (property = the label-carrying word: the city, or a mood adjective)
              punctuation: comma, period, misplaced_comma, misplaced_period
              readout: q0..qN, with the last readout token labelled "final"
  role      f"{segment}.{kind}", the default grouping for heatmaps
  word_final  True for the last token of a word (the position that has seen the whole word)
  flags     conj, capitalized, substituted, before_punct, before_cap, first_punct
  after_property  True for tokens after the last property token (the only positions probes may use)
"""

from punctsum.conditions import CONJUNCTIONS, Passage, Piece

WORD_KIND_PRIORITY = ["property", "sent_final", "clause_final", "initial", "clause_initial"]


def _piece_info(passage: Passage, pieces: list[Piece]) -> list[dict]:
    info = []
    first_punct_seen = False
    for i, pc in enumerate(pieces):
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        flags = set()
        if pc.kind == "word":
            sent = passage.sentences[pc.sent]
            clause_len = len(sent[pc.clause])
            last_word = pc.pos == clause_len - 1
            tags = {
                "property": pc.property,
                "sent_final": last_word and pc.clause == len(sent) - 1,
                "clause_final": last_word and pc.clause < len(sent) - 1,
                "initial": pc.initial,
                "clause_initial": pc.pos == 0 and pc.clause > 0,
            }
            kind = next((k for k in WORD_KIND_PRIORITY if tags[k]), "content")
            if pc.text.lower() in CONJUNCTIONS:
                flags.add("conj")
            if pc.capitalized:
                flags.add("capitalized")
            if pc.text.lower() != pc.original.lower():
                flags.add("substituted")
            if nxt is not None and nxt.kind != "word":
                flags.add("before_punct")
            if nxt is not None and nxt.kind == "word" and nxt.capitalized:
                flags.add("before_cap")
        else:
            kind = ("misplaced_" if pc.misplaced else "") + pc.kind
            if not first_punct_seen:
                flags.add("first_punct")
                first_punct_seen = True
        info.append({"segment": "s0" if pc.sent == 0 else f"f{pc.sent}", "kind": kind, "flags": flags})
    return info


def label_tokens(passage: Passage, pieces: list[Piece], piece_of_token: list[int], n_readout: int) -> list[dict]:
    """Labels for the passage tokens (piece_of_token, BOS first) followed by n_readout readout tokens."""
    info = _piece_info(passage, pieces)
    last_prop_tok = max(t for t, pi in enumerate(piece_of_token) if pi >= 0 and pieces[pi].property)
    rows = []
    for t, pi in enumerate(piece_of_token):
        if pi < 0:
            rows.append({"segment": "bos", "kind": "bos", "word_final": True, "flags": []})
            continue
        nxt = piece_of_token[t + 1] if t + 1 < len(piece_of_token) else None
        rows.append({
            "segment": info[pi]["segment"],
            "kind": info[pi]["kind"],
            "word_final": nxt != pi,
            "flags": sorted(info[pi]["flags"]),
        })
    for i in range(n_readout):
        kind = "final" if i == n_readout - 1 else f"q{i}"
        rows.append({"segment": "readout", "kind": kind, "word_final": True, "flags": []})
    for t, r in enumerate(rows):
        r["role"] = f"{r['segment']}.{r['kind']}"
        r["after_property"] = t > last_prop_tok
    return rows
