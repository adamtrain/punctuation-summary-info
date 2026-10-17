"""Milestone 2: compare candidate label pairs by how many of their cities GPT-2 small reliably knows.

For each label pair (e.g. France vs Japan), every candidate city gets a knowledge margin: the logit
difference between its true label and the other label after "The city of {CITY} is in the {noun} of".
Cities with margin >= 2.0 are kept. A usable pair needs many kept cities on both sides, in matching
token-count buckets (patching pairs need equal token counts). Writes outputs/m2/property_screen.txt.
"""

import collections

import torch

from punctsum import PROJECT_ROOT
from punctsum.model import load

THRESHOLD = 2.0
CANDIDATES = {
    "state": {
        "Colorado": ["Denver", "Boulder", "Aspen", "Vail", "Durango", "Telluride", "Pueblo", "Greeley", "Fort Collins", "Grand Junction", "Aurora", "Loveland", "Longmont"],
        "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "El Paso", "Fort Worth", "Lubbock", "Amarillo", "Galveston", "Waco", "Plano", "Laredo", "Odessa"],
        "California": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Oakland", "San Jose", "Fresno", "Berkeley", "Pasadena", "Santa Monica", "Long Beach", "Palo Alto", "Anaheim", "Bakersfield", "Santa Barbara", "Malibu", "Riverside", "Irvine", "Stockton", "Modesto", "Santa Cruz", "Burbank", "Monterey", "Napa", "Ventura", "Oxnard"],
        "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Tallahassee", "Gainesville", "Pensacola", "Sarasota", "Fort Lauderdale", "Key West", "Daytona Beach", "Boca Raton", "Clearwater", "Fort Myers", "Ocala", "Lakeland", "Kissimmee", "Palm Beach"],
    },
    "country": {
        "France": ["Paris", "Lyon", "Marseille", "Nice", "Bordeaux", "Toulouse", "Lille", "Nantes", "Strasbourg", "Montpellier", "Rennes", "Grenoble", "Dijon", "Cannes", "Avignon", "Reims", "Le Havre", "Rouen", "Toulon", "Calais", "Versailles", "Metz", "Limoges", "Amiens", "Perpignan", "Biarritz", "Chamonix", "Lourdes"],
        "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Sapporo", "Kobe", "Hiroshima", "Fukuoka", "Sendai", "Nara", "Nagasaki", "Kawasaki", "Chiba", "Kanazawa", "Niigata", "Okayama", "Kumamoto", "Kagoshima", "Matsuyama", "Hakodate", "Nikko", "Kamakura", "Shizuoka", "Kitakyushu", "Hamamatsu", "Yokosuka", "Otaru"],
        "Italy": ["Rome", "Milan", "Venice", "Turin", "Bologna", "Genoa", "Palermo", "Verona", "Pisa", "Siena", "Bari", "Catania", "Padua", "Parma", "Modena", "Trieste", "Perugia", "Ravenna", "Como", "Sorrento", "Bergamo", "Lucca", "Amalfi", "Cagliari"],
        "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne", "Stuttgart", "Dresden", "Leipzig", "Bremen", "Hanover", "Nuremberg", "Heidelberg", "Bonn", "Dortmund", "Essen", "Mainz", "Freiburg", "Potsdam", "Kiel", "Augsburg", "Wiesbaden", "Bielefeld", "Rostock", "Regensburg"],
    },
}
PAIRS = [("state", "Colorado", "Texas"), ("state", "California", "Texas"), ("state", "California", "Florida"),
         ("country", "France", "Japan"), ("country", "Italy", "Japan"), ("country", "Germany", "Japan"),
         ("country", "France", "Germany")]


@torch.no_grad()
def margin(model, city: str, noun: str, label: str, other: str) -> float:
    logits = model(model.to_tokens(f"The city of {city} is in the {noun} of"))[0, -1]
    return (logits[model.to_single_token(" " + label)] - logits[model.to_single_token(" " + other)]).item()


def main():
    model = load("gpt2")
    ntok = lambda c: len(model.tokenizer(" " + c, add_special_tokens=False)["input_ids"])  # noqa: E731
    lines = [f"Knowledge screen, GPT-2 small, threshold {THRESHOLD}. 'pairable' = cities usable in equal-token-count pairs.\n"]
    for noun, a, b in PAIRS:
        kept = {}
        for lab, other in ((a, b), (b, a)):
            ms = {c: margin(model, c, noun, lab, other) for c in CANDIDATES[noun][lab]}
            kept[lab] = {c: m for c, m in ms.items() if m >= THRESHOLD}
        buckets = {lab: collections.Counter(ntok(c) for c in kept[lab]) for lab in (a, b)}
        pairable = sum(min(buckets[a][n], buckets[b][n]) for n in set(buckets[a]) | set(buckets[b]))
        lines.append(
            f"{a} vs {b}: kept {len(kept[a])}/{len(CANDIDATES[noun][a])} {a}, {len(kept[b])}/{len(CANDIDATES[noun][b])} {b}; "
            f"pairable per side = {pairable}; buckets {a}={dict(buckets[a])} {b}={dict(buckets[b])}"
        )
        for lab in (a, b):
            lines.append(f"    {lab}: " + ", ".join(f"{c}({m:+.1f})" for c, m in sorted(kept[lab].items(), key=lambda x: -x[1])))
    out = PROJECT_ROOT / "outputs" / "m2" / "property_screen.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
