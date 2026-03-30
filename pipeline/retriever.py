"""
RAG retrieval layer for NLPilot.

Queries ChromaDB collections with the structured trip slots and mapped
activity categories to fetch grounding context for itinerary generation.

Collections queried:
  - yelp_venues       : relevant places in the destination city
  - yelp_reviews      : quality signals for top venues
  - yelp_tips         : concise venue notes
  - travelplanner     : reference itineraries for similar trips

Pipeline position:
  slot_filler  →  mood_mapper  →  [retriever]  →  generator

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.slot_filler import TripSlots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VECTORSTORE_DIR  = "vectorstore"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# How many results to pull from each collection
N_VENUES         = 15
N_REVIEWS        = 10
N_TIPS           = 10
N_REFERENCE_PLANS = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievedContext:
    venues:          list[dict] = field(default_factory=list)
    reviews:         list[dict] = field(default_factory=list)
    tips:            list[dict] = field(default_factory=list)
    reference_plans: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"RetrievedContext(\n"
            f"  venues          = {len(self.venues)}\n"
            f"  reviews         = {len(self.reviews)}\n"
            f"  tips            = {len(self.tips)}\n"
            f"  reference_plans = {len(self.reference_plans)}\n"
            f")"
        )

    def to_prompt_text(self) -> str:
        """
        Format all retrieved context into a single string block
        ready to be injected into the LLM prompt.
        """
        sections: list[str] = []

        # ── Venues ────────────────────────────────────────────────────────
        if self.venues:
            lines = ["=== VENUES ==="]
            for v in self.venues:
                meta = v.get("metadata", {})
                lines.append(
                    f"• {meta.get('name', 'Unknown')} | {meta.get('city', '')} | "
                    f"{meta.get('categories', '')} | "
                    f"★ {meta.get('stars', '?')} ({meta.get('review_count', 0)} reviews)"
                )
            sections.append("\n".join(lines))

        # ── Tips ──────────────────────────────────────────────────────────
        if self.tips:
            lines = ["=== VENUE TIPS ==="]
            for t in self.tips:
                lines.append(f"• {t.get('document', '')}")
            sections.append("\n".join(lines))

        # ── Reviews ───────────────────────────────────────────────────────
        if self.reviews:
            lines = ["=== REVIEWS ==="]
            for r in self.reviews:
                meta = r.get("metadata", {})
                stars = meta.get("stars", "?")
                text  = r.get("document", "")[:300]   # truncate long reviews
                lines.append(f"• [{stars}★] {text}")
            sections.append("\n".join(lines))

        # ── Reference itineraries ─────────────────────────────────────────
        if self.reference_plans:
            lines = ["=== REFERENCE ITINERARIES ==="]
            for p in self.reference_plans:
                meta = p.get("metadata", {})
                lines.append(
                    f"• {meta.get('days', '?')}-day trip to {meta.get('dest', '?')} "
                    f"from {meta.get('org', '?')}\n"
                    f"  Query: {meta.get('query', '')}"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# ChromaDB client — singleton
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None
_ef: SentenceTransformerEmbeddingFunction | None = None


def _get_client(vectorstore_dir: str = VECTORSTORE_DIR):
    global _client, _ef
    if _client is None:
        path = Path(vectorstore_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Vector store not found at '{path.resolve()}'. "
                "Run:  python pipeline/ingest_chromadb.py --data_dir data/processed"
            )
        _ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _client = chromadb.PersistentClient(path=str(path))
    return _client, _ef


def _safe_get_collection(client, name: str, ef):
    """Return collection or None if it doesn't exist yet."""
    try:
        return client.get_collection(name, embedding_function=ef)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _venue_query(slots: TripSlots, categories: list[str]) -> str:
    """Build a semantic query string for venue retrieval."""
    cat_text  = ", ".join(categories[:5]) if categories else "attractions"
    mood_text = ", ".join(slots.moods[:4]) if slots.moods else ""
    dest      = slots.destination or "the destination"
    parts = [f"{cat_text} in {dest}"]
    if mood_text:
        parts.append(f"for a traveler who enjoys {mood_text}")
    if slots.transport:
        parts.append(f"accessible by {slots.transport}")
    return " ".join(parts)


def _plan_query(slots: TripSlots) -> str:
    """Build a semantic query string for TravelPlanner retrieval."""
    dest   = slots.destination or "a city"
    days   = slots.days or "a few"
    budget = f"${slots.budget:.0f}" if slots.budget else "a moderate budget"
    moods  = ", ".join(slots.moods[:3]) if slots.moods else "general sightseeing"
    return (
        f"{days}-day trip to {dest} with {budget} budget, "
        f"interests: {moods}"
    )


def retrieve(
    slots: TripSlots,
    categories: list[str],
    vectorstore_dir: str = VECTORSTORE_DIR,
) -> RetrievedContext:
    """
    Run RAG retrieval across all ChromaDB collections.

    Args:
        slots:           Structured trip slots from slot_filler.fill_slots()
        categories:      Activity category names from mood_mapper.map_moods()
        vectorstore_dir: Path to the ChromaDB persistent store

    Returns:
        RetrievedContext with venues, reviews, tips, and reference plans
    """
    client, ef = _get_client(vectorstore_dir)
    context    = RetrievedContext()

    venue_q = _venue_query(slots, categories)
    plan_q  = _plan_query(slots)

    # ── Yelp venues ───────────────────────────────────────────────────────
    venues_col = _safe_get_collection(client, "yelp_venues", ef)
    if venues_col:
        # Optionally filter by city if the collection has enough data
        where = None
        if slots.destination:
            where = {"city": {"$eq": slots.destination}}

        try:
            results = venues_col.query(
                query_texts=[venue_q],
                n_results=N_VENUES,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # City filter may fail if no matches — fall back to pure semantic search
            results = venues_col.query(
                query_texts=[venue_q],
                n_results=N_VENUES,
                include=["documents", "metadatas", "distances"],
            )

        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        context.venues = [
            {"document": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metadatas, distances)
        ]
    else:
        print("[Retriever] 'yelp_venues' collection not found — skipping.")

    # ── Yelp tips ─────────────────────────────────────────────────────────
    tips_col = _safe_get_collection(client, "yelp_tips", ef)
    if tips_col and context.venues:
        # Fetch tips for the top venue business IDs
        top_bids = [
            v["metadata"].get("business_id", "")
            for v in context.venues[:8]
            if v["metadata"].get("business_id")
        ]
        if top_bids:
            try:
                results = tips_col.query(
                    query_texts=[venue_q],
                    n_results=N_TIPS,
                    where={"business_id": {"$in": top_bids}},
                    include=["documents", "metadatas"],
                )
                docs      = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                context.tips = [
                    {"document": d, "metadata": m}
                    for d, m in zip(docs, metadatas)
                ]
            except Exception:
                pass
    else:
        if not tips_col:
            print("[Retriever] 'yelp_tips' collection not found — skipping.")

    # ── Yelp reviews ──────────────────────────────────────────────────────
    reviews_col = _safe_get_collection(client, "yelp_reviews", ef)
    if reviews_col and context.venues:
        top_bids = [
            v["metadata"].get("business_id", "")
            for v in context.venues[:5]
            if v["metadata"].get("business_id")
        ]
        if top_bids:
            try:
                results = reviews_col.query(
                    query_texts=[venue_q],
                    n_results=N_REVIEWS,
                    where={"business_id": {"$in": top_bids}},
                    include=["documents", "metadatas"],
                )
                docs      = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                context.reviews = [
                    {"document": d, "metadata": m}
                    for d, m in zip(docs, metadatas)
                ]
            except Exception:
                pass
    else:
        if not reviews_col:
            print("[Retriever] 'yelp_reviews' collection not found — skipping.")

    # ── TravelPlanner reference plans ─────────────────────────────────────
    tp_col = _safe_get_collection(client, "travelplanner", ef)
    if tp_col:
        try:
            where = None
            if slots.destination:
                where = {"dest": {"$eq": slots.destination}}

            results = tp_col.query(
                query_texts=[plan_q],
                n_results=N_REFERENCE_PLANS,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            results = tp_col.query(
                query_texts=[plan_q],
                n_results=N_REFERENCE_PLANS,
                include=["documents", "metadatas", "distances"],
            )

        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        context.reference_plans = [
            {"document": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metadatas, distances)
        ]
    else:
        print("[Retriever] 'travelplanner' collection not found — skipping.")

    return context


if __name__ == "__main__":
    from pipeline.slot_filler import fill_slots
    from pipeline.mood_mapper import map_moods

    query = (
        "I'm flying from San Francisco to New York for 5 days "
        "with a $2,000 budget. I prefer public transport and I'm "
        "in the mood for art, local food, and some chill walks."
    )

    slots      = fill_slots(query)
    categories = map_moods(slots.moods, top_k=3)

    print("Slots     :", slots)
    print("Categories:", categories)
    print()

    context = retrieve(slots, categories)
    print(context.summary())
    print()
    print(context.to_prompt_text()[:1000], "...")
