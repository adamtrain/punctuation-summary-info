"""Hand-written pilot materials for the milestone 1 checks (not the milestone 2 generator)."""

from punctsum.conditions import Passage

CITIES = {
    "Colorado": ["Denver", "Boulder", "Aspen", "Vail", "Durango", "Telluride", "Pueblo", "Greeley"],
    "Texas": ["Houston", "Dallas", "Austin", "Lubbock", "Amarillo", "Galveston", "Waco", "Abilene"],
}
STATES = list(CITIES)

# Clause 1 sets the property (the city); later sentences are label-independent filler.
TEMPLATES = [
    [
        [["We", "drove", "into", "{CITY}", "and", "parked", "near", "the", "trailhead"]],
        [["The", "weather", "was", "fine"]],
        [["We", "ate", "lunch"], ["then", "went", "back"]],
    ],
    [
        [["My", "sister", "flew", "to", "{CITY}", "for", "a", "conference"]],
        [["The", "hotel", "was", "quiet"]],
        [["She", "slept", "early"], ["then", "read", "a", "book"]],
    ],
    [
        [["Last", "spring", "we", "moved", "to", "{CITY}"]],
        [["The", "boxes", "took", "weeks", "to", "unpack"]],
        [["Our", "neighbors", "were", "friendly"], ["and", "the", "kids", "made", "friends"]],
    ],
]

READOUT = "\n\nQuestion: Which state was this in?\nAnswer:"

# Replacement words for control C: common, concrete or abstract, and unrelated to place, climate,
# landscape or region, so a substitution can't leak or contradict the label.
NEUTRAL_POOL = [
    "window", "paper", "seven", "green", "table", "chair", "music", "pencil", "letter", "number",
    "yellow", "blue", "garden", "kitchen", "button", "candle", "carpet", "basket", "bottle", "ladder",
    "mirror", "pocket", "ribbon", "rocket", "silver", "spoon", "ticket", "violin", "wallet", "yesterday",
    "quickly", "loud", "soft", "heavy", "empty", "honest", "simple", "gentle", "bright", "narrow",
    "listen", "borrow", "follow", "carry", "print", "fold", "count", "paint", "laugh", "whisper",
    "idea", "story", "promise", "answer", "lesson", "method", "habit", "detail", "signal", "pattern",
]


def make_passage(template, city: str) -> Passage:
    sents = [[[w.replace("{CITY}", city) for w in clause] for clause in sent] for sent in template]
    return Passage(sents, proper_nouns={city}, property_words={city})


def all_passages() -> list[tuple[Passage, str, str]]:
    """(passage, city, state) for every template x city."""
    return [
        (make_passage(t, c), c, s) for t in TEMPLATES for s, cities in CITIES.items() for c in cities
    ]
