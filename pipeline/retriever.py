"""
RAG retrieval layer for NLPilot.
Optimized for minimal context size — sends only what the LLM needs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.slot_filler import TripSlots

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_TIPS          = 10

# Category queries for diverse retrieval
CATEGORY_QUERIES = {
    "EAT":   ("restaurants, cafes, brunch, dining, food", 4),
    "VISIT": ("museums, art galleries, parks, landmarks, gardens", 3),
    "TOUR":  ("tours, sightseeing, walking tours", 1),
    "DRINK": ("bars, breweries, cocktail lounges", 1),
    "SHOP":  ("shopping, markets, stores, boutiques", 1),
}
# Total: 10 venues


@dataclass
class RetrievedContext:
    venues:          list[dict] = field(default_factory=list)
    tips:            list[dict] = field(default_factory=list)
    retrieval_log:   list[str]  = field(default_factory=list)

    def summary(self) -> str:
        return f"RetrievedContext(venues={len(self.venues)}, tips={len(self.tips)})"

    def to_prompt_text(self) -> str:
        """Minimal context for LLM — short venue lines with inline tips."""
        if not self.venues:
            return ""

        # Build tip lookup
        tip_lookup: dict[str, str] = {}
        for t in self.tips:
            bid = t.get("metadata", {}).get("business_id", "")
            if bid and bid not in tip_lookup:
                tip_lookup[bid] = t.get("document", "")[:80]

        lines = ["VENUES:"]
        for i, v in enumerate(self.venues, 1):
            meta = v.get("metadata", {})
            name = meta.get("name", "Unknown")
            cats = meta.get("categories", "")
            stars = meta.get("stars", "?")
            bid = meta.get("business_id", "")
            action = _classify_venue(cats)

            line = f"[{i}] {name} ({action}) - {cats[:60]}. {stars} stars."
            tip = tip_lookup.get(bid, "")
            if tip:
                line += f' Tip: "{tip}"'
            lines.append(line)

        return "\n".join(lines)


def _classify_venue(categories: str) -> str:
    cats = categories.lower() if categories else ""
    if any(w in cats for w in ["restaurant", "food", "cafe", "coffee", "bakery",
                                "breakfast", "brunch", "pizza", "seafood", "sandwich"]):
        return "EAT"
    if any(w in cats for w in ["bar", "pub", "brewery", "cocktail", "wine", "nightlife"]):
        return "DRINK"
    if any(w in cats for w in ["tour", "sightseeing"]):
        return "TOUR"
    if any(w in cats for w in ["museum", "art", "gallery", "historic", "landmark", "park", "garden"]):
        return "VISIT"
    if any(w in cats for w in ["shopping", "market", "store", "shop", "boutique"]):
        return "SHOP"
    return "VISIT"


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------

_client = None
_ef = None

def _get_client(vectorstore_dir=VECTORSTORE_DIR):
    global _client, _ef
    if _client is None:
        path = Path(vectorstore_dir)
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at '{path.resolve()}'.")
        _ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _client = chromadb.PersistentClient(path=str(path))
    return _client, _ef

def _safe_get(client, name, ef):
    try:
        return client.get_collection(name, embedding_function=ef)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main retrieval
# ---------------------------------------------------------------------------

def retrieve(
    slots: TripSlots,
    categories: list[str],
    vectorstore_dir: str = VECTORSTORE_DIR,
    include_reviews: bool = False,
) -> RetrievedContext:
    client, ef = _get_client(vectorstore_dir)
    context = RetrievedContext()
    dest_lower = (slots.destination or "").lower().strip()
    dest = slots.destination or "the city"
    moods = ", ".join(slots.moods[:3]) if slots.moods else ""

    # Dynamic venue allocation: mood-driven + scaled by trip length
    days = slots.days or 3
    total_venues = min(days * 3, 10)

    # Map mood_mapper categories to retriever groups
    MOOD_TO_GROUP = {
        "restaurants": "EAT", "cafes": "EAT",
        "bars_nightlife": "DRINK",
        "museums_galleries": "VISIT", "parks_nature": "VISIT",
        "hiking_adventure": "VISIT", "beaches": "VISIT",
        "spas_wellness": "VISIT", "sports_fitness": "VISIT",
        "shopping": "SHOP",
        "landmarks_tours": "TOUR", "entertainment": "DRINK",
    }

    # Count how many mapped categories fall into each group
    group_weight = {"EAT": 1, "VISIT": 1, "TOUR": 0, "DRINK": 0, "SHOP": 0}
    for cat in categories:
        group = MOOD_TO_GROUP.get(cat, "VISIT")
        group_weight[group] = group_weight.get(group, 0) + 1

    # Distribute slots proportionally, minimum 1 for EAT and VISIT
    total_weight = max(sum(group_weight.values()), 1)
    venue_slots = {}
    allocated = 0
    for group in ["EAT", "VISIT", "TOUR", "DRINK", "SHOP"]:
        w = group_weight.get(group, 0)
        if w > 0:
            n = max(1, round(total_venues * w / total_weight))
            venue_slots[group] = n
            allocated += n

    # Adjust if over/under allocated
    while allocated > total_venues:
        # Remove from the group with the most slots
        biggest = max(venue_slots, key=venue_slots.get)
        venue_slots[biggest] -= 1
        allocated -= 1
    while allocated < total_venues:
        # Add to EAT (always useful)
        venue_slots["EAT"] = venue_slots.get("EAT", 0) + 1
        allocated += 1

    # ── Venues: diverse category retrieval ────────────────────────────
    venues_col = _safe_get(client, "yelp_venues", ef)
    if venues_col:
        seen_ids = set()

        for cat_group, (cat_desc, _) in CATEGORY_QUERIES.items():
            n_slots = venue_slots.get(cat_group, 0)
            if n_slots == 0:
                continue

            query = f"{cat_desc} in {dest}"
            if moods:
                query += f" {moods}"

            results = None
            if dest_lower:
                for city_filter in [
                    {"city_lower": {"$eq": dest_lower}},
                    {"city": {"$eq": slots.destination}},
                    {"city": {"$eq": slots.destination.title() if slots.destination else ""}},
                ]:
                    try:
                        results = venues_col.query(
                            query_texts=[query],
                            n_results=n_slots + 3,
                            where=city_filter,
                            include=["documents", "metadatas", "distances"],
                        )
                        if results.get("documents", [[]])[0]:
                            break
                        results = None
                    except Exception:
                        results = None

            if results is None:
                # City not in dataset — skip this category, don't fall back to random cities
                continue

            if results:
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                added = 0
                for d, m, dist in zip(docs, metas, dists):
                    if added >= n_slots:
                        break
                    bid = m.get("business_id", "")
                    if bid in seen_ids:
                        continue
                    seen_ids.add(bid)
                    context.venues.append({"document": d, "metadata": m, "distance": dist})
                    added += 1

        if context.venues:
            context.retrieval_log.append(f"Retrieved {len(context.venues)} venues across {len(CATEGORY_QUERIES)} categories.")
        else:
            context.retrieval_log.append(f"No venues found for {dest}.")

    # ── Tips: match to retrieved venues ───────────────────────────────
    tips_col = _safe_get(client, "yelp_tips", ef)
    if tips_col and context.venues:
        bids = [v["metadata"].get("business_id", "") for v in context.venues if v["metadata"].get("business_id")]
        if bids:
            try:
                results = tips_col.query(
                    query_texts=[f"tips for venues in {dest}"],
                    n_results=N_TIPS,
                    where={"business_id": {"$in": bids}},
                    include=["documents", "metadatas"],
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                context.tips = [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
            except Exception as e:
                logger.warning(f"Tips query failed: {e}")

    return context