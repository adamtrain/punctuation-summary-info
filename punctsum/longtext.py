"""Long-distance filler from natural text: contiguous excerpts of PG19 books (BRIEF.md decision log, long-distance).

Books are the official PG19 test split (DeepMind, Apache-2.0; public-domain Project Gutenberg texts) in data/pg19_test/.
Each excerpt is normalized so that commas and periods are the only punctuation, then parsed into the same
sentence -> clause -> word structure as the templated fillers, so every condition (P, U, U-cap, M, K, C) applies unchanged.

Normalization: `?` `!` -> period; `;` `:` dashes -> comma; quotes, parentheses, brackets, underscores and asterisks dropped;
apostrophes and hyphens inside words kept; periods after common abbreviations (Mr., Mrs., Dr., ...) removed so they don't
end sentences. A sentence is skipped (and the excerpt continues with the next one) if it contains any other character,
has fewer than 3 words, is a heading (mostly capitals), or mentions a label-related word (France, Japan, their adjectives,
any study city, or the mood answer words "great"/"terrible").
"""

import random
import re
from pathlib import Path

from punctsum import PROJECT_ROOT
from punctsum.materials import CITY_CANDIDATES

BOOKS = PROJECT_ROOT / "data" / "pg19_test"
ABBREVIATIONS = ["Mr", "Mrs", "Ms", "Dr", "St", "Mt", "Jr", "Sr", "Prof", "Capt", "Col", "Gen", "Lt", "Rev", "Hon", "No",
                 "vol", "etc", "viz", "vs", "ch", "p", "pp"]
LEAK_WORDS = {"france", "french", "frenchman", "frenchmen", "japan", "japanese", "great", "terrible", "greatly",
              "greater", "greatest", "terribly"}
LEAK_WORDS |= {w.lower() for cities in CITY_CANDIDATES.values() for c in cities for w in c.split()}
ALLOWED = re.compile(r"^[A-Za-z0-9 ,.'\-]+$")


def normalize(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t)
    for ab in ABBREVIATIONS:
        t = re.sub(rf"\b{ab}\.", ab, t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"[?!]+", ".", t)
    t = re.sub(r"\s*(--|—|–|;|:)\s*", ", ", t)
    t = re.sub(r"[\"()\[\]_*{}]", "", t)
    t = re.sub(r"(^|\s)'+|'+(\s|$)", r"\1\2", t)  # quote-apostrophes at word edges; in-word apostrophes stay
    t = re.sub(r"\s+([,.])", r"\1", t)
    t = re.sub(r",\s*,+", ",", t)
    t = re.sub(r",\s*\.", ".", t)
    return re.sub(r"\s+", " ", t).strip()


def sentences_of(paragraph: str) -> list[str]:
    return [s.strip() for s in normalize(paragraph).split(".") if s.strip()]


def parse_sentence(s: str) -> list[list[str]] | None:
    """Sentence -> clauses -> words, or None if the sentence must be skipped."""
    if not ALLOWED.match(s):
        return None
    clauses = [[w for w in c.split() if w.strip("'-")] for c in s.split(",")]
    clauses = [c for c in clauses if c]
    words = [w for c in clauses for w in c]
    if len(words) < 3:
        return None
    if sum(w.isupper() and len(w) > 1 for w in words) > len(words) / 2:  # headings
        return None
    if any(w.lower().strip("'-") in LEAK_WORDS for w in words):
        return None
    return clauses


def proper_nouns(sentences: list[list[list[str]]]) -> set[str]:
    """Words capitalized mid-sentence whose lowercase form never occurs in the excerpt (so "America" counts but "He",
    capitalized mid-sentence only because a quote was removed, doesn't), plus "I" forms. Proper nouns keep their capital in
    every condition."""
    lower = {w for sent in sentences for clause in sent for w in clause if w[:1].islower()}
    proper = set()
    for sent in sentences:
        for ci, clause in enumerate(sent):
            for wi, w in enumerate(clause):
                if (ci, wi) != (0, 0) and w[:1].isupper() and w.lower() not in lower:
                    proper.add(w)
    proper |= {w for sent in sentences for clause in sent for w in clause if w == "I" or w.startswith("I'")}
    return proper


def paragraphs(book: Path) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", book.read_text(errors="ignore")) if p.strip()]


def excerpt(book: Path, n_tokens: int, count_tokens, rng: random.Random) -> dict:
    """A contiguous excerpt of about n_tokens tokens (P rendering, counted with count_tokens), starting at a random
    paragraph between 10% and 50% of the way into the book (so there is room for a full excerpt)."""
    paras = paragraphs(book)
    start = rng.randrange(len(paras) // 10, max(len(paras) // 10 + 1, len(paras) // 2))
    kept, skipped, text = [], 0, ""
    for para in paras[start:]:
        for s in sentences_of(para):
            parsed = parse_sentence(s)
            if parsed is None:
                skipped += 1
                continue
            candidate = text + (" " if text else "") + ", ".join(" ".join(c) for c in parsed) + "."
            if count_tokens(candidate) > n_tokens:
                return {"book": book.stem, "start_paragraph": start, "sentences": kept, "skipped": skipped,
                        "kept_count": len(kept), "proper": sorted(proper_nouns(kept)), "tokens": count_tokens(text)}
            kept.append(parsed)
            text = candidate
    raise ValueError(f"book {book.stem} too short from paragraph {start}")
