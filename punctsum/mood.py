"""Mood property (BRIEF.md section 11.2): a label spread over three evaluative adjectives in the property
sentence, so no single token gives the answer.

The property sentence has three clauses, each judging one aspect with one adjective ("At the hotel the room was
lovely all week, the staff was rude to us, and the breakfast was nice every morning"). Every adjective has at
least two words after it before the clause boundary, so the adjective and the boundary are distinct positions
(the same rule as the city in the country property).

Two variants, registered as separate properties:
  mood_mixed       two adjectives of one valence and one of the other; the label is the majority. Pair members
                   differ in exactly one adjective (one majority slot flips), which flips the majority. Any single
                   adjective agrees with the label only 2/3 of the time.
  mood_consistent  all three adjectives share the label's valence (Tigges et al.-style fallback); members differ
                   in all three slots.
Labels are "good"/"bad"; the readout's answer tokens are " great" / " terrible". GPT-2 small prefers " good" over
" bad" after almost any text (after "The food was awful. Overall, it was", " bad" is not in its top 12), whereas
" great"/" terrible" respond symmetrically (outputs/m2b/screen_report.txt, BRIEF.md decision log).
"""

import random

import torch

from punctsum.conditions import Passage, render, to_text
from punctsum.dataset import (
    FILLER_LENGTHS,
    PROPERTIES,
    TEST_FRACTION,
    Property,
    split_indices,
)
from punctsum.materials import FILLERS, FIXED_PROPER
from punctsum.model import single_token_word

V = "{V}"
MOOD_LABELS = ["good", "bad"]
MOOD_ANSWERS = {"good": "great", "bad": "terrible"}

MOOD_TEMPLATES: list[list[list[str]]] = [
    [["At", "the", "hotel", "the", "room", "was", V, "all", "week"], ["the", "staff", "was", V, "to", "us"], ["and", "the", "breakfast", "was", V, "every", "morning"]],
    [["At", "the", "restaurant", "the", "food", "was", V, "that", "night"], ["the", "waiter", "was", V, "with", "us"], ["and", "the", "music", "was", V, "in", "the", "background"]],
    [["In", "the", "new", "apartment", "the", "kitchen", "was", V, "from", "the", "start"], ["the", "neighbors", "were", V, "most", "days"], ["and", "the", "street", "was", V, "at", "night"]],
    [["At", "the", "concert", "the", "band", "was", V, "on", "stage"], ["the", "crowd", "was", V, "around", "us"], ["and", "the", "sound", "was", V, "near", "the", "front"]],
    [["On", "the", "train", "the", "seat", "was", V, "for", "hours"], ["the", "conductor", "was", V, "with", "everyone"], ["and", "the", "view", "was", V, "from", "the", "window"]],
    [["At", "the", "party", "the", "food", "was", V, "on", "the", "table"], ["the", "guests", "were", V, "all", "evening"], ["and", "the", "music", "was", V, "in", "the", "yard"]],
    [["At", "the", "museum", "the", "exhibit", "was", V, "on", "the", "first", "floor"], ["the", "guide", "was", V, "with", "our", "group"], ["and", "the", "cafe", "was", V, "near", "the", "exit"]],
    [["At", "the", "beach", "the", "water", "was", V, "that", "afternoon"], ["the", "lifeguard", "was", V, "with", "the", "kids"], ["and", "the", "snack", "bar", "was", V, "by", "the", "pier"]],
    [["In", "the", "class", "the", "teacher", "was", V, "most", "weeks"], ["the", "textbook", "was", V, "for", "the", "course"], ["and", "the", "classroom", "was", V, "in", "the", "mornings"]],
    [["On", "the", "camping", "trip", "the", "campsite", "was", V, "by", "the", "lake"], ["the", "tent", "was", V, "at", "night"], ["and", "the", "trail", "was", V, "up", "the", "hill"]],
    [["At", "the", "new", "job", "the", "office", "was", V, "on", "most", "days"], ["the", "manager", "was", V, "in", "meetings"], ["and", "the", "coffee", "was", V, "in", "the", "break", "room"]],
    [["On", "the", "flight", "the", "seat", "was", V, "the", "whole", "way"], ["the", "crew", "was", V, "with", "passengers"], ["and", "the", "meal", "was", V, "on", "the", "tray"]],
    [["At", "the", "gym", "the", "equipment", "was", V, "on", "most", "days"], ["the", "trainer", "was", V, "with", "beginners"], ["and", "the", "locker", "room", "was", V, "after", "class"]],
    [["At", "the", "wedding", "the", "ceremony", "was", V, "in", "the", "garden"], ["the", "band", "was", V, "at", "the", "reception"], ["and", "the", "cake", "was", V, "after", "dinner"]],
    [["At", "the", "bookstore", "the", "selection", "was", V, "on", "the", "shelves"], ["the", "clerk", "was", V, "at", "the", "counter"], ["and", "the", "reading", "corner", "was", V, "in", "the", "back"]],
    [["On", "the", "bus", "tour", "the", "driver", "was", V, "during", "the", "tour"], ["the", "guide", "was", V, "at", "every", "stop"], ["and", "the", "bus", "was", V, "on", "the", "highway"]],
    [["At", "the", "dentist", "the", "waiting", "room", "was", V, "that", "morning"], ["the", "dentist", "was", V, "during", "the", "checkup"], ["and", "the", "receptionist", "was", V, "at", "the", "desk"]],
    [["At", "the", "cooking", "class", "the", "chef", "was", V, "with", "the", "group"], ["the", "kitchen", "was", V, "at", "the", "school"], ["and", "the", "dessert", "was", V, "at", "the", "end"]],
    [["At", "the", "theater", "the", "play", "was", V, "in", "the", "first", "act"], ["the", "actors", "were", V, "on", "stage"], ["and", "the", "seats", "were", V, "in", "the", "balcony"]],
    [["At", "the", "picnic", "the", "sandwiches", "were", V, "in", "the", "basket"], ["the", "park", "was", V, "that", "day"], ["and", "the", "company", "was", V, "all", "afternoon"]],
    [["At", "the", "car", "rental", "the", "car", "was", V, "on", "the", "road"], ["the", "agent", "was", V, "at", "the", "counter"], ["and", "the", "pickup", "was", V, "at", "the", "airport"]],
    [["At", "the", "zoo", "the", "animals", "were", V, "in", "their", "enclosures"], ["the", "keeper", "was", V, "with", "visitors"], ["and", "the", "gift", "shop", "was", V, "near", "the", "gate"]],
    [["At", "the", "spa", "the", "massage", "was", V, "that", "afternoon"], ["the", "therapist", "was", V, "with", "me"], ["and", "the", "pool", "was", V, "on", "the", "roof"]],
    [["At", "the", "library", "the", "study", "room", "was", V, "on", "the", "second", "floor"], ["the", "librarian", "was", V, "with", "students"], ["and", "the", "wifi", "was", V, "in", "the", "building"]],
]

# Generic evaluative adjectives that fit any noun. "good", "bad", "great" and "terrible" are excluded: they are the
# label or answer words, which would let the readout copy instead of aggregate.
ADJ_CANDIDATES = {
    "good": ["nice", "pleasant", "lovely", "wonderful", "excellent", "fantastic", "delightful", "superb", "terrific",
             "perfect", "amazing", "outstanding", "enjoyable", "charming", "splendid", "brilliant", "fabulous",
             "impressive", "decent", "fine", "solid", "marvelous"],
    "bad": ["awful", "horrible", "unpleasant", "dreadful", "poor", "disappointing", "lousy", "mediocre",
            "miserable", "annoying", "frustrating", "disgusting", "atrocious", "abysmal", "appalling", "dismal",
            "pathetic", "unbearable", "shabby", "boring", "dull",
            # milder negatives, added so the kept negatives can match the positives' strength
            "bland", "messy", "tedious", "awkward", "stressful", "sloppy", "chaotic", "crowded", "tiring", "lame",
            "gloomy", "inadequate", "forgettable", "uninspiring", "uncomfortable", "unfriendly", "dirty", "noisy"],
}

SCREEN_NOUNS = ["food", "room", "trip", "service", "show"]
VALENCE_PROMPT = "The {noun} was {adj}. Overall, it was"
FILLER_PROMPT = " Overall, it was"  # filler valence uses the same great/terrible answer tokens
# An adjective is kept if it is a single token and moves the readout (logit(" great") - logit(" terrible")) away from
# the mean of these neutral adjectives in the right direction by an amount inside ADJ_BAND. The band is the same for
# both valences, so positive and negative words have similar strength and a 2-vs-1 majority is not overturned by one
# strong word.
NEUTRAL_REF = ["ordinary", "typical", "average", "normal"]
ADJ_BAND = (1.2, 2.6)
N_NEUTRAL_FILLERS = 64  # most neutral fillers kept for mood passages (test split then has 16 >= max k)


def _good_bad(model) -> tuple[int, int]:
    return model.to_single_token(" " + MOOD_ANSWERS["good"]), model.to_single_token(" " + MOOD_ANSWERS["bad"])


@torch.no_grad()
def _valence(model, adj: str) -> float:
    good, bad = _good_bad(model)
    diffs = []
    for noun in SCREEN_NOUNS:
        lg = model(model.to_tokens(VALENCE_PROMPT.format(noun=noun, adj=adj)))[0, -1]
        diffs.append((lg[good] - lg[bad]).item())
    return sum(diffs) / len(diffs)


@torch.no_grad()
def screen_adjectives(model) -> list[dict]:
    """Each candidate's valence: answer logit diff minus the mean for neutral adjectives (rel)."""
    ref = sum(_valence(model, a) for a in NEUTRAL_REF) / len(NEUTRAL_REF)
    rows = []
    for label, adjs in ADJ_CANDIDATES.items():
        sign = 1 if label == "good" else -1
        for adj in adjs:
            d = _valence(model, adj)
            single = single_token_word(model, adj)
            strength = sign * (d - ref)
            rows.append({"adj": adj, "label": label, "diff": d, "rel": d - ref, "ref": ref, "single_token": single,
                         "keep": single and ADJ_BAND[0] <= strength <= ADJ_BAND[1]})
    return rows


@torch.no_grad()
def screen_fillers(model) -> list[dict]:
    """Valence of each filler sentence on its own (good-bad logit diff after " Overall, it was")."""
    good, bad = _good_bad(model)
    rows = []
    for j, f in enumerate(FILLERS):
        text = to_text(render(Passage([f], proper_nouns=set(FIXED_PROPER)), "P"))[0]
        lg = model(model.to_tokens(text + FILLER_PROMPT))[0, -1]
        rows.append({"filler": j, "text": text, "diff": (lg[good] - lg[bad]).item()})
    return rows


def neutral_fillers(filler_screen: list[dict]) -> list[int]:
    """Indices (into materials.FILLERS) of the N_NEUTRAL_FILLERS fillers closest to the median valence. The median,
    not zero, because GPT-2 has an overall lean toward " good" or " bad" after any text."""
    med = sorted(r["diff"] for r in filler_screen)[len(filler_screen) // 2]
    ranked = sorted(filler_screen, key=lambda r: abs(r["diff"] - med))
    return sorted(r["filler"] for r in ranked[:N_NEUTRAL_FILLERS])


def _split_list(items: list[str], rng: random.Random) -> dict[str, list[str]]:
    items = sorted(items)
    rng.shuffle(items)
    n_test = max(1, round(len(items) * TEST_FRACTION))
    return {"test": items[:n_test], "train": items[n_test:]}


def make_mood_pairs(adj_screen: list[dict], fillers: list[int], n_per_length: int, seed: int, variant: str) -> list[dict]:
    """Pairs for variant "mixed" or "consistent". Templates, fillers and adjectives are split train/test."""
    rng = random.Random(seed)
    t_split = split_indices(MOOD_TEMPLATES, seed)
    f_split = {fillers[i]: s for i, s in split_indices([FILLERS[j] for j in fillers], seed + 1).items()}
    adj = {lab: _split_list([r["adj"] for r in adj_screen if r["keep"] and r["label"] == lab], random.Random(seed + 2))
           for lab in MOOD_LABELS}
    pairs = []
    for k in FILLER_LENGTHS:
        n_test = round(n_per_length * TEST_FRACTION)
        for i in range(n_per_length):
            split = "test" if i < n_test else "train"
            # Distinct adjectives within a passage, so no word repeats.
            pos, neg = rng.sample(adj["good"][split], 3), rng.sample(adj["bad"][split], 3)
            if variant == "mixed":
                minority = rng.randrange(3)  # slot holding the minority valence in the "good" member
                flip = rng.choice([s for s in range(3) if s != minority])  # majority slot that flips
                good = [neg[0] if s == minority else pos[s] for s in range(3)]
                bad = list(good)
                bad[flip] = neg[1]
                valence = {"good": ["bad" if s == minority else "good" for s in range(3)],
                           "bad": ["bad" if s in (minority, flip) else "good" for s in range(3)]}
            elif variant == "consistent":
                good, bad = pos, neg
                valence = {"good": ["good"] * 3, "bad": ["bad"] * 3}
            else:
                raise ValueError(variant)
            pairs.append({
                "property": f"mood_{variant}",
                "pair_id": f"{variant}-k{k}-{i:05d}",
                "k": k,
                "split": split,
                "s0": rng.choice([j for j, s in t_split.items() if s == split]),
                "fillers": rng.sample([j for j, s in f_split.items() if s == split], k),
                "adjs": {"good": good, "bad": bad},
                "valence": valence,
            })
    return pairs


def passage_for_mood(pair: dict, label: str) -> Passage:
    adjs = iter(pair["adjs"][label])
    s0, positions = [], set()
    for ci, clause in enumerate(MOOD_TEMPLATES[pair["s0"]]):
        new = []
        for wi, w in enumerate(clause):
            if w == V:
                w = next(adjs)
                positions.add((0, ci, wi))
            new.append(w)
        s0.append(new)
    sentences = [s0] + [FILLERS[j] for j in pair["fillers"]]
    return Passage(sentences, proper_nouns=set(FIXED_PROPER), property_positions=positions)


# Continuation readout chosen by scripts/m2b_mood_screen.py (largest margins of 4 candidates for both variants).
# No question readout: GPT-2 small barely answers "good or bad?" questions (pair-level margins +0.09 to +0.35),
# so it was struck for mood (BRIEF.md decision log).
MOOD_READOUTS = {"continuation": " and overall the whole experience was"}

for _variant in ("mixed", "consistent"):
    PROPERTIES[f"mood_{_variant}"] = Property(f"mood_{_variant}", MOOD_LABELS, MOOD_READOUTS, passage_for_mood,
                                              answers=MOOD_ANSWERS)
