"""
slot_filler.py
==============
Extracts structured travel slots from a natural language user query.

Slots extracted:
  - origin       : departure city / location
  - destination  : arrival city / location
  - days         : trip duration in days (int)
  - start_date   : specific start date if mentioned (str)
  - budget       : total budget in USD (float)
  - transport    : preferred transport mode (str)
  - moods        : list of mood/activity descriptors (list[str])

Strategy:
  1. spaCy NER  — GPE (cities), MONEY (budget), DATE (dates/durations), CARDINAL (numbers)
  2. Regex      — catches patterns spaCy misses ("$2,000", "5 days", "a week")
  3. Keywords   — transport modes and mood tags from curated lists
"""

import re
from dataclasses import dataclass, field
import spacy


try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run:  python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------

TRANSPORT_KEYWORDS: dict[str, str] = {
    "public transport": "public transport",
    "public transit":   "public transport",
    "subway":           "public transport",
    "metro":            "public transport",
    "bus":              "public transport",
    "train":            "train",
    "rail":             "train",
    "amtrak":           "train",
    "flight":           "flight",
    "fly":              "flight",
    "flying":           "flight",
    "plane":            "flight",
    "airline":          "flight",
    "car":              "car",
    "drive":            "car",
    "driving":          "car",
    "road trip":        "car",
    "rental car":       "car",
    "uber":             "rideshare",
    "lyft":             "rideshare",
    "taxi":             "rideshare",
    "cab":              "rideshare",
    "walk":             "walking",
    "walking":          "walking",
    "on foot":          "walking",
    "bike":             "cycling",
    "cycling":          "cycling",
    "bicycle":          "cycling",
}

MOOD_KEYWORDS: list[str] = [
    "adventure", "adventurous",
    "chill", "relaxed", "relaxing", "laid-back", "slow",
    "foodie", "food", "cuisine", "eating", "dining", "local food", "street food",
    "art", "arts", "museum", "galleries", "cultural", "culture", "history", "historical",
    "nightlife", "bars", "clubs", "party", "drinking",
    "nature", "outdoors", "hiking", "trekking", "parks", "scenic",
    "beach", "ocean", "coast", "seaside", "surf",
    "shopping", "markets", "boutiques",
    "romantic", "couples",
    "family", "kid-friendly", "kids",
    "spiritual", "temples", "churches", "wellness",
    "photography", "scenic views", "architecture",
    "music", "concerts", "live music",
    "sports", "active", "fitness",
]

# Text fragments that hint a city is the ORIGIN (not destination)
ORIGIN_CUES = {"from", "departing", "leaving", "flying from", "traveling from", "starting from"}

# Text fragments that hint a city is the DESTINATION
DEST_CUES   = {"to", "in", "visit", "visiting", "going to", "heading to", "traveling to", "trip to", "explore"}

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TripSlots:
    origin:     str | None       = None
    destination: str | None      = None
    days:        int | None      = None
    start_date:  str | None      = None
    budget:      float | None    = None
    transport:   str | None      = None
    moods:       list[str]       = field(default_factory=list)

    def is_complete(self) -> bool:
        """True if the minimum required slots for generation are filled."""
        return bool(self.destination and self.days)

    def missing(self) -> list[str]:
        """Return names of slots still needed."""
        needed = []
        if not self.destination:
            needed.append("destination")
        if not self.days:
            needed.append("number of days")
        if not self.budget:
            needed.append("budget")
        if not self.transport:
            needed.append("transport mode")
        if not self.moods:
            needed.append("mood / activities")
        return needed

    def __str__(self) -> str:
        return (
            f"TripSlots(\n"
            f"  origin      = {self.origin}\n"
            f"  destination = {self.destination}\n"
            f"  days        = {self.days}\n"
            f"  start_date  = {self.start_date}\n"
            f"  budget      = {self.budget}\n"
            f"  transport   = {self.transport}\n"
            f"  moods       = {self.moods}\n"
            f")"
        )


def _extract_cities(doc) -> tuple[str | None, str | None]:
    """
    Use spaCy GPE/LOC entities + surrounding prepositions to decide
    which city is origin and which is destination.
    """
    origin      = None
    destination = None

    for ent in doc.ents:
        if ent.label_ not in ("GPE", "LOC"):
            continue

        city = ent.text.strip()
        # Look at the token immediately before the entity for cue words
        prev_tokens = {t.text.lower() for t in doc[max(0, ent.start - 3): ent.start]}

        is_origin = bool(prev_tokens & ORIGIN_CUES)
        is_dest   = bool(prev_tokens & DEST_CUES)

        if is_origin and not origin:
            origin = city
        elif is_dest and not destination:
            destination = city
        elif not destination:
            # fallback: first unassigned city → destination
            destination = city

    return origin, destination


def _extract_days(text: str, doc) -> int | None:
    """
    Extract trip duration in days via:
      - regex: "5 days", "a week", "two weeks", "10-day"
      - spaCy DATE entities containing day/week counts
    """
    text_lower = text.lower()

    # Explicit day count: "5 days", "5-day"
    m = re.search(r"(\d+)\s*[-\s]?days?", text_lower)
    if m:
        return int(m.group(1))

    # Week expressions
    m = re.search(r"(\d+)\s*weeks?", text_lower)
    if m:
        return int(m.group(1)) * 7
    if re.search(r"\ba\s+week\b", text_lower):
        return 7
    if re.search(r"\btwo\s+weeks?\b", text_lower):
        return 14

    # spaCy DATE entities — e.g. "for a week", "for 3 nights"
    for ent in doc.ents:
        if ent.label_ == "DATE":
            m = re.search(r"(\d+)\s*nights?", ent.text.lower())
            if m:
                return int(m.group(1))

    return None


def _extract_budget(text: str, doc) -> float | None:
    """
    Extract budget via:
      - spaCy MONEY entities
      - regex: "$2,000", "2000 dollars", "USD 1500", "1.5k"
    """
    # spaCy MONEY
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            raw = re.sub(r"[^\d.]", "", ent.text.replace(",", ""))
            try:
                return float(raw)
            except ValueError:
                continue

    text_lower = text.lower()

    # $1,500 or $1500
    m = re.search(r"\$([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    # "1500 dollars / USD"
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:dollars?|usd|bucks?)", text_lower)
    if m:
        return float(m.group(1).replace(",", ""))

    # "1.5k" or "2k"
    m = re.search(r"([\d.]+)\s*k\b", text_lower)
    if m:
        return float(m.group(1)) * 1000

    return None


def _extract_transport(text: str) -> str | None:
    """Match transport keywords against the lowercased input (longest match wins)."""
    text_lower = text.lower()
    # Sort by length descending so multi-word phrases match before single words
    for phrase in sorted(TRANSPORT_KEYWORDS, key=len, reverse=True):
        if phrase in text_lower:
            return TRANSPORT_KEYWORDS[phrase]
    return None


def _extract_moods(text: str) -> list[str]:
    """Return all mood keywords found in the text (deduplicated, order-preserved)."""
    text_lower = text.lower()
    seen: set[str] = set()
    found: list[str] = []
    for kw in MOOD_KEYWORDS:
        if kw in text_lower and kw not in seen:
            # Normalise multi-word synonyms to canonical form
            canonical = kw.split()[0] if len(kw.split()) > 1 else kw
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)
    return found


def _extract_start_date(doc) -> str | None:
    """Extract a specific start date if mentioned (e.g. 'March 15', 'next Friday')."""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            text = ent.text.lower()
            # Skip vague durations like "5 days", "a week"
            if re.search(r"\d+\s*days?|\d+\s*weeks?|a week", text):
                continue
            return ent.text
    return None



def fill_slots(text: str) -> TripSlots:
    """
    Parse a natural language trip description and return a TripSlots object.
    """
    doc    = _nlp(text)
    slots  = TripSlots()

    slots.origin, slots.destination = _extract_cities(doc)
    slots.days       = _extract_days(text, doc)
    slots.budget     = _extract_budget(text, doc)
    slots.transport  = _extract_transport(text)
    slots.moods      = _extract_moods(text)
    slots.start_date = _extract_start_date(doc)

    return slots


if __name__ == "__main__":
    samples = [
        "I'm flying from San Francisco to New York for 5 days with a $2,000 budget. "
        "I prefer public transport, and I'm in the mood for art, local food, and some chill walks.",

        "Plan a 7-day road trip from Chicago to New Orleans. Budget is around $1.5k. "
        "I love music, nightlife, and Southern cuisine.",

        "Weekend trip to Miami, 3 days, $800. Beach vibes and relaxing.",

        "Traveling to Tokyo for two weeks starting March 15. I enjoy culture, hiking, and street food. "
        "Will be using trains.",
    ]

    for s in samples:
        print("Input :", s)
        print(fill_slots(s))
        print("-" * 60)
