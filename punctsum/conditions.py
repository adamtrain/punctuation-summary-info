"""Structured passages and the text conditions rendered from them.

A passage is stored as words grouped into clauses and sentences rather than as a string, so every
condition is rendered from the same source and we always know which word is sentence-initial, which
is a proper noun, and where punctuation belongs. A rendered condition is a list of Pieces (words and
punctuation marks) that keep this metadata, so tokens can later be mapped back to roles.
See BRIEF.md section 4.2 for why each condition exists.

Conditions:
  P     punctuated, normal casing
  U     no punctuation; sentence-initial words lowercased unless proper nouns
  Ucap  no punctuation; casing unchanged
  C     P with some words replaced by random neutral words (OOD control; strength = sub_prob)
  M     U with the same number of commas/periods as P inserted at wrong (non-boundary) places
  K     P with `restarts` mid-sentence words capitalized, mimicking dictation tools that capitalize
        the first word after a pause (research question 2)
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace

# Sentence -> comma-separated clauses -> words.
Sentences = list[list[list[str]]]

# Words that can mark a clause boundary on their own; never substituted, so control C doesn't remove
# boundary cues other than punctuation.
CONJUNCTIONS = {"and", "then", "but", "so", "or"}


@dataclass
class Passage:
    sentences: Sentences
    proper_nouns: set[str] = field(default_factory=set)  # keep their capital in every condition
    property_words: set[str] = field(default_factory=set)  # the label-carrying word(s), e.g. the city
    # Label-carrying words given by position (sentence, clause, word index), for properties whose words may
    # also occur elsewhere in the passage (e.g. mood adjectives).
    property_positions: set[tuple[int, int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class Piece:
    text: str
    kind: str  # "word", "comma" or "period"
    sent: int
    clause: int
    pos: int  # word index within its clause; for punctuation, index of the word it follows
    proper: bool = False
    property: bool = False  # carries the label (the city, or a mood adjective)
    initial: bool = False  # first word of a sentence
    original: str = ""  # source word before any change (substitution, decapitalization)
    misplaced: bool = False  # punctuation not at a true clause/sentence boundary
    capitalized: bool = False  # spuriously capitalized mid-sentence word (condition K)


@dataclass
class Substitution:
    """Settings for control C. `keeps_token_count(w)` says whether word w is a single token (pool words
    must be too), so a swap keeps the token count; it depends on the tokenizer, so the caller supplies it."""

    prob: float
    pool: list[str]
    keeps_token_count: Callable[[str], bool]


def base_words(p: Passage) -> list[Piece]:
    out = []
    for si, sent in enumerate(p.sentences):
        for ci, clause in enumerate(sent):
            for wi, w in enumerate(clause):
                prop = w in p.property_words or (si, ci, wi) in p.property_positions
                # Property words are also protected like proper nouns: no substitution, recasing or adjacent
                # misplaced punctuation in any condition.
                out.append(Piece(w, "word", si, ci, wi, proper=w in p.proper_nouns or prop, property=prop,
                                 initial=ci == 0 and wi == 0, original=w))
    return out


def true_boundary_after(words: list[Piece], i: int) -> str | None:
    """The punctuation that belongs after words[i] in normal text, if any."""
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else None
    if nxt is None or nxt.sent != w.sent:
        return "period"
    if nxt.clause != w.clause:
        return "comma"
    return None


def _punct(kind: str, after: Piece, misplaced: bool = False) -> Piece:
    return Piece("," if kind == "comma" else ".", kind, after.sent, after.clause, after.pos, misplaced=misplaced)


def _with_true_punct(words: list[Piece]) -> list[Piece]:
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if kind := true_boundary_after(words, i):
            out.append(_punct(kind, w))
    return out


def _decap(words: list[Piece]) -> list[Piece]:
    return [
        replace(w, text=w.text[0].lower() + w.text[1:]) if w.initial and not w.proper else w for w in words
    ]


def _substitute(words: list[Piece], rng: random.Random, sub: Substitution) -> list[Piece]:
    out = []
    for w in words:
        eligible = not (w.proper or w.initial or w.text.lower() in CONJUNCTIONS) and sub.keeps_token_count(w.text)
        if eligible and rng.random() < sub.prob:
            new = rng.choice([c for c in sub.pool if c != w.text])
            w = replace(w, text=new)
        out.append(w)
    return out


def _misplace(words: list[Piece], rng: random.Random) -> list[Piece]:
    """Insert as many commas and periods as P would have, but only into gaps that are not true
    boundaries and not next to the property word (so the city's immediate context matches U)."""
    kinds = [k for i in range(len(words)) if (k := true_boundary_after(words, i))]
    gaps = [
        i
        for i in range(len(words) - 1)
        if true_boundary_after(words, i) is None and not (words[i].property or words[i + 1].property)
    ]
    if len(gaps) < len(kinds):
        raise ValueError(f"passage too short to misplace {len(kinds)} marks in {len(gaps)} gaps")
    chosen = rng.sample(gaps, len(kinds))
    rng.shuffle(kinds)
    marks = dict(zip(chosen, kinds))
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if i in marks:
            out.append(_punct(marks[i], w, misplaced=True))
    return out


def _restart_caps(words: list[Piece], rng: random.Random, n: int) -> list[Piece]:
    """Capitalize n random lowercase, non-initial words. Unlike control C, this deliberately ignores
    token counts: a dictation tool capitalizes whatever word comes next."""
    candidates = [i for i, w in enumerate(words) if not (w.initial or w.proper) and w.text[:1].islower()]
    chosen = set(rng.sample(candidates, min(n, len(candidates))))
    return [
        replace(w, text=w.text[0].upper() + w.text[1:], capitalized=True) if i in chosen else w
        for i, w in enumerate(words)
    ]


def render(
    p: Passage,
    condition: str,
    rng: random.Random | None = None,
    sub: Substitution | None = None,
    restarts: int = 0,
) -> list[Piece]:
    words = base_words(p)
    if condition == "P":
        return _with_true_punct(words)
    if condition == "U":
        return _decap(words)
    if condition == "Ucap":
        return words
    if condition == "C":
        return _with_true_punct(_substitute(words, rng, sub))
    if condition == "M":
        return _misplace(_decap(words), rng)
    if condition == "K":
        return _with_true_punct(_restart_caps(words, rng, restarts))
    raise ValueError(f"unknown condition {condition!r}")


def to_text(pieces: list[Piece]) -> tuple[str, list[tuple[int, int]]]:
    """Join pieces into text (space before words, none before punctuation); return each piece's
    character span."""
    parts, spans, pos = [], [], 0
    for i, pc in enumerate(pieces):
        if pc.kind == "word" and i > 0:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(pc.text)))
        parts.append(pc.text)
        pos += len(pc.text)
    return "".join(parts), spans


# Convenience wrappers returning plain strings (used by the milestone 1 script).
def render_P(p: Passage) -> str:
    return to_text(render(p, "P"))[0]


def render_U(p: Passage) -> str:
    return to_text(render(p, "U"))[0]


def render_Ucap(p: Passage) -> str:
    return to_text(render(p, "Ucap"))[0]
