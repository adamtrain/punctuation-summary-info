"""Text materials for the milestone 2 dataset: property-sentence templates, filler sentences, candidate
cities, the readout suffix, and the neutral word pool for control C.

Design rules (BRIEF.md section 4.1):
- The property sentence (s0) names a city, never as its first word and never as the last word of its
  clause: at least two words follow the city before any boundary, so the city token and the
  boundary positions are always distinct.
- Filler sentences carry no information about the label. As a structural guarantee, every template and
  filler sentence is used equally often with both labels (passages come in Colorado/Texas pairs), so
  even an unintended association cannot predict the label.
- No proper nouns in filler except "I".
- Each clause is a list of words; a sentence is a list of clauses (commas go between clauses).
"""

CITY = "{CITY}"

# s0 templates: (clauses). Single-clause ones end in a period; two-clause ones have a comma after the
# city clause.
S0_TEMPLATES: list[list[list[str]]] = [
    [["We", "drove", "into", CITY, "and", "parked", "near", "the", "old", "library"]],
    [["My", "sister", "flew", "to", CITY, "for", "a", "work", "conference", "last", "month"]],
    [["Last", "spring", "we", "moved", "to", CITY, "with", "our", "two", "cats"]],
    [["Our", "team", "spent", "a", "week", "in", CITY, "on", "a", "training", "course"]],
    [["I", "took", "the", "early", "train", "to", CITY, "to", "visit", "my", "uncle"]],
    [["The", "band", "played", "a", "show", "in", CITY, "on", "a", "Friday", "night"]],
    [["My", "parents", "bought", "a", "small", "house", "in", CITY, "near", "a", "park"]],
    [["We", "stopped", "in", CITY, "for", "a", "quick", "lunch", "with", "friends"]],
    [["Her", "brother", "found", "a", "new", "job", "in", CITY, "at", "a", "bank"]],
    [["The", "conference", "was", "held", "in", CITY, "at", "a", "large", "hotel"]],
    [["I", "spent", "the", "summer", "in", CITY, "working", "at", "a", "bookstore"]],
    [["Our", "cousins", "drove", "up", "to", CITY, "for", "the", "long", "weekend"]],
    [["We", "arrived", "in", CITY, "late", "at", "night"], ["then", "we", "checked", "into", "the", "motel"]],
    [["My", "friend", "lives", "in", CITY, "with", "her", "husband"], ["and", "she", "works", "from", "home"]],
    [["The", "meeting", "took", "place", "in", CITY, "last", "Tuesday"], ["but", "nobody", "took", "notes"]],
    [["I", "rented", "a", "car", "in", CITY, "at", "the", "airport"], ["then", "drove", "to", "the", "office"]],
    [["We", "spent", "three", "days", "in", CITY, "with", "my", "aunt"], ["and", "played", "cards", "every", "evening"]],
    [["Our", "flight", "landed", "in", CITY, "right", "on", "time"], ["so", "we", "caught", "the", "early", "bus"]],
    [["They", "opened", "a", "bakery", "in", CITY, "last", "year"], ["and", "business", "has", "been", "steady"]],
    [["My", "grandmother", "grew", "up", "in", CITY, "as", "a", "child"], ["but", "she", "rarely", "talks", "about", "it"]],
    [["We", "booked", "a", "room", "in", CITY, "for", "two", "nights"], ["then", "changed", "our", "plans"]],
    [["The", "wedding", "was", "in", CITY, "at", "a", "small", "chapel"], ["and", "the", "food", "was", "excellent"]],
    [["I", "met", "my", "roommate", "in", CITY, "during", "college"], ["then", "we", "lost", "touch"]],
    [["He", "bought", "a", "used", "bicycle", "in", CITY, "from", "a", "neighbor"], ["and", "fixed", "the", "brakes", "himself"]],
]

# Proper nouns that appear in templates and must keep their capital in every condition.
FIXED_PROPER = {"I", "Friday", "Tuesday"}

FILLERS: list[list[list[str]]] = [
    # single clause
    [["The", "coffee", "was", "still", "warm"]],
    [["My", "phone", "needed", "a", "charge"]],
    [["The", "kids", "watched", "a", "movie"]],
    [["We", "read", "the", "paper", "together"]],
    [["Nobody", "remembered", "the", "password"]],
    [["The", "printer", "jammed", "again"]],
    [["I", "made", "a", "list", "of", "chores"]],
    [["The", "meeting", "ran", "a", "little", "long"]],
    [["Our", "neighbor", "walked", "his", "dog"]],
    [["The", "bread", "was", "a", "bit", "stale"]],
    [["We", "played", "a", "game", "of", "chess"]],
    [["The", "radio", "played", "old", "songs"]],
    [["I", "answered", "a", "few", "emails"]],
    [["The", "cat", "slept", "on", "the", "couch"]],
    [["We", "counted", "the", "spare", "coins"]],
    [["The", "lamp", "flickered", "twice"]],
    [["My", "sister", "called", "after", "dinner"]],
    [["The", "soup", "needed", "more", "salt"]],
    [["We", "folded", "the", "clean", "laundry"]],
    [["The", "clock", "on", "the", "wall", "was", "slow"]],
    [["I", "wrote", "a", "short", "letter"]],
    [["The", "door", "creaked", "when", "it", "opened"]],
    [["We", "shared", "a", "plate", "of", "noodles"]],
    [["The", "teacher", "graded", "the", "tests"]],
    [["My", "shoes", "were", "untied"]],
    [["The", "baby", "finally", "fell", "asleep"]],
    [["We", "cleaned", "the", "kitchen", "floor"]],
    [["The", "bus", "was", "a", "few", "minutes", "late"]],
    [["I", "lost", "my", "favorite", "pen"]],
    [["The", "cake", "was", "too", "sweet"]],
    # two clauses
    [["We", "ate", "lunch"], ["then", "went", "back"]],
    [["The", "store", "was", "busy"], ["so", "we", "waited", "outside"]],
    [["I", "opened", "the", "window"], ["and", "the", "noise", "came", "in"]],
    [["The", "movie", "was", "long"], ["but", "the", "ending", "was", "good"]],
    [["She", "finished", "her", "book"], ["then", "started", "another", "one"]],
    [["The", "kettle", "whistled"], ["and", "I", "made", "some", "tea"]],
    [["We", "missed", "the", "first", "bus"], ["so", "we", "took", "the", "next", "one"]],
    [["The", "lights", "went", "out"], ["then", "came", "back", "on"]],
    [["My", "brother", "fixed", "the", "sink"], ["and", "cleaned", "up", "after"]],
    [["The", "dog", "barked", "at", "the", "mailman"], ["then", "ran", "inside"]],
    [["I", "forgot", "my", "keys"], ["so", "I", "knocked", "on", "the", "door"]],
    [["The", "restaurant", "was", "full"], ["but", "they", "found", "us", "a", "table"]],
    [["We", "washed", "the", "dishes"], ["and", "dried", "them", "by", "hand"]],
    [["The", "children", "drew", "pictures"], ["then", "hung", "them", "up"]],
    [["The", "phone", "rang", "twice"], ["and", "then", "it", "stopped"]],
    [["I", "bought", "some", "apples"], ["then", "walked", "back", "home"]],
    [["The", "class", "ended", "early"], ["so", "we", "went", "for", "coffee"]],
    [["He", "tried", "the", "new", "recipe"], ["but", "it", "did", "not", "work"]],
    [["We", "painted", "the", "fence"], ["and", "left", "it", "to", "dry"]],
    [["The", "laptop", "was", "slow"], ["so", "I", "restarted", "it"]],
    [["My", "friend", "sent", "a", "message"], ["then", "called", "a", "minute", "later"]],
    [["The", "game", "was", "close"], ["but", "our", "team", "lost"]],
    [["I", "tidied", "my", "desk"], ["and", "sorted", "the", "mail"]],
    [["The", "train", "stopped", "briefly"], ["then", "continued", "on"]],
    [["We", "watched", "the", "news"], ["and", "went", "to", "bed"]],
    [["The", "parcel", "arrived"], ["but", "the", "box", "was", "damaged"]],
    [["I", "set", "an", "alarm"], ["then", "fell", "asleep", "quickly"]],
    [["The", "music", "was", "loud"], ["so", "we", "moved", "to", "another", "room"]],
    [["She", "baked", "some", "cookies"], ["and", "shared", "them", "with", "us"]],
    [["The", "printer", "was", "out", "of", "ink"], ["so", "I", "used", "another", "one"]],
    # added in milestone 2 so the test split has 20 fillers
    [["The", "towel", "was", "still", "damp"]],
    [["My", "pencil", "broke", "again"]],
    [["The", "neighbors", "had", "a", "party"]],
    [["We", "sorted", "the", "old", "photos"]],
    [["The", "fridge", "made", "a", "strange", "noise"]],
    [["I", "checked", "the", "calendar", "twice"]],
    [["The", "plant", "needed", "some", "water"]],
    [["Our", "car", "needed", "new", "tires"]],
    [["The", "movie", "started", "a", "bit", "late"]],
    [["The", "milk", "smelled", "a", "little", "sour"]],
    [["The", "kids", "built", "a", "fort"], ["then", "knocked", "it", "down"]],
    [["I", "called", "the", "dentist"], ["but", "nobody", "answered"]],
    [["We", "moved", "the", "couch"], ["and", "swept", "underneath", "it"]],
    [["The", "tea", "was", "too", "hot"], ["so", "I", "waited", "a", "minute"]],
    [["My", "brother", "borrowed", "my", "jacket"], ["then", "forgot", "to", "return", "it"]],
    [["The", "website", "was", "down"], ["so", "we", "tried", "again", "later"]],
    [["She", "lost", "her", "umbrella"], ["but", "found", "it", "the", "next", "day"]],
    [["We", "set", "the", "table"], ["and", "lit", "a", "candle"]],
    [["The", "printer", "finally", "worked"], ["then", "ran", "out", "of", "paper"]],
    [["I", "fixed", "the", "shelf"], ["and", "hung", "a", "picture", "above", "it"]],
]

# Label pair: France vs Japan. Colorado vs Texas was dropped because GPT-2 small reliably knows only 5
# Colorado cities (outputs/m2/property_screen.txt, BRIEF.md decision log). "Nice" is excluded because it
# is also a common English word.
LABEL_NOUN = "country"
CITY_CANDIDATES = {
    "France": [
        "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Lille", "Nantes", "Strasbourg",
        "Montpellier", "Rennes", "Grenoble", "Dijon", "Cannes", "Avignon", "Reims", "Le Havre", "Rouen",
        "Toulon", "Calais", "Versailles", "Metz", "Limoges", "Amiens", "Perpignan", "Biarritz", "Chamonix",
        "Lourdes",
    ],
    "Japan": [
        "Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Sapporo", "Kobe", "Hiroshima", "Fukuoka",
        "Sendai", "Nara", "Nagasaki", "Kawasaki", "Chiba", "Kanazawa", "Niigata", "Okayama", "Kumamoto",
        "Kagoshima", "Matsuyama", "Hakodate", "Nikko", "Kamakura", "Shizuoka", "Kitakyushu", "Hamamatsu",
        "Yokosuka", "Otaru",
    ],
}
LABELS = list(CITY_CANDIDATES)

READOUT = "\n\nQuestion: Which country was this in?\nAnswer:"
KNOWLEDGE_PROMPT = "The city of {CITY} is in the country of"
# Separator-free continuation readout (BRIEF.md risk 6.4): the passage runs on in words into the answer, so
# no condition receives a newline or other separator from the readout. Chosen in
# scripts/m2_readout_screen.py (highest P margin of 5 candidates).
READOUT_CONT = " and all of this was in the country of"

# Replacement words for control C: common, and unrelated to place, climate, landscape or region.
NEUTRAL_POOL = [
    "window", "paper", "seven", "green", "table", "chair", "music", "pencil", "letter", "number",
    "yellow", "blue", "garden", "kitchen", "button", "candle", "carpet", "basket", "bottle", "ladder",
    "mirror", "pocket", "ribbon", "rocket", "silver", "spoon", "ticket", "violin", "wallet", "yesterday",
    "quickly", "loud", "soft", "heavy", "empty", "honest", "simple", "gentle", "bright", "narrow",
    "listen", "borrow", "follow", "carry", "print", "fold", "count", "paint", "laugh", "whisper",
    "idea", "story", "promise", "answer", "lesson", "method", "habit", "detail", "signal", "pattern",
]
