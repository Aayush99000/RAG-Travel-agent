"""
mood_mapper.py
==============
Maps vague mood/activity descriptors (e.g. "chill", "adventurous", "foodie")
to concrete Yelp activity categories using sentence-transformers + cosine similarity.

  categories = map_moods(["chill", "food", "art"], top_k=3)
  # → ["cafes", "restaurants", "museums"]

"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Canonical activity categories
# ---------------------------------------------------------------------------

ACTIVITY_CATEGORIES: dict[str, str] = {
    "restaurants": (
        "dining out at local restaurants, trying regional cuisine, food tours, "
        "tasting menus, ethnic food, street food stalls"
    ),
    "cafes": (
        "coffee shops, cafes, brunch spots, tea houses, bakeries, "
        "casual daytime dining, pastries, slow mornings"
    ),
    "bars_nightlife": (
        "bars, nightlife, cocktail lounges, clubs, rooftop bars, "
        "live music venues, evening entertainment, drinking"
    ),
    "museums_galleries": (
        "art museums, history museums, science museums, art galleries, "
        "exhibitions, cultural institutions, sculptures, installations"
    ),
    "parks_nature": (
        "parks, botanical gardens, nature walks, green spaces, "
        "scenic viewpoints, picnics, outdoor relaxation"
    ),
    "hiking_adventure": (
        "hiking trails, trekking, rock climbing, adventure sports, "
        "zip-lining, kayaking, cycling, outdoor exploration"
    ),
    "beaches": (
        "beach, ocean, seaside, swimming, surfing, snorkeling, "
        "coastal walks, water sports, sunset at the shore"
    ),
    "shopping": (
        "shopping malls, local markets, boutiques, artisan shops, "
        "vintage stores, souvenir hunting, flea markets"
    ),
    "spas_wellness": (
        "spas, wellness centers, yoga studios, meditation, massages, "
        "hot springs, relaxation, self-care, retreats"
    ),
    "landmarks_tours": (
        "famous landmarks, tourist attractions, guided tours, sightseeing, "
        "historic sites, monuments, architecture tours"
    ),
    "entertainment": (
        "concerts, live music, theaters, comedy shows, "
        "cultural performances, festivals, events"
    ),
    "sports_fitness": (
        "gyms, sports facilities, fitness classes, running tracks, "
        "swimming pools, active lifestyle, workout"
    ),
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None
_category_embeddings: np.ndarray | None = None
_category_names: list[str] = list(ACTIVITY_CATEGORIES.keys())


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_category_embeddings() -> np.ndarray:
    """Compute and cache category description embeddings."""
    global _category_embeddings
    if _category_embeddings is None:
        model = _get_model()
        descriptions = list(ACTIVITY_CATEGORIES.values())
        _category_embeddings = model.encode(descriptions, normalize_embeddings=True)
    return _category_embeddings


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def map_moods(moods: list[str], top_k: int = 3) -> list[str]:
    """
    Map a list of mood/activity words to the top-k canonical activity categories.

    Args:
        moods:  List of mood strings from slot_filler (e.g. ["chill", "art", "food"])
        top_k:  Number of top categories to return per mood (deduplicated across all moods)

    Returns:
        Deduplicated list of category names, ranked by relevance.

    Example:
        map_moods(["adventurous", "beach", "food"], top_k=2)
        → ["hiking_adventure", "beaches", "restaurants", "cafes"]
    """
    if not moods:
        return []

    model              = _get_model()
    category_embeddings = _get_category_embeddings()

    # Embed all mood words together and individually
    mood_embeddings = model.encode(moods, normalize_embeddings=True)

    # Cosine similarity: mood_embeddings (n_moods, dim) @ category_embeddings.T (dim, n_cats)
    similarity_matrix = mood_embeddings @ category_embeddings.T  # (n_moods, n_cats)

    # Aggregate: max score each category receives across all moods
    aggregated_scores = similarity_matrix.max(axis=0)  # (n_cats,)

    # Rank categories by score
    ranked_indices = np.argsort(aggregated_scores)[::-1]

    # Return top (top_k * len(moods)) unique categories, capped at total available
    n_return = min(top_k * max(1, len(moods)), len(_category_names))
    top_categories = [_category_names[i] for i in ranked_indices[:n_return]]

    return top_categories


def map_moods_with_scores(moods: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """
    Same as map_moods but returns (category, score) tuples for inspection/debugging.
    """
    if not moods:
        return []

    model               = _get_model()
    category_embeddings = _get_category_embeddings()

    mood_embeddings    = model.encode(moods, normalize_embeddings=True)
    similarity_matrix  = mood_embeddings @ category_embeddings.T
    aggregated_scores  = similarity_matrix.max(axis=0)
    ranked_indices     = np.argsort(aggregated_scores)[::-1]

    n_return = min(top_k * max(1, len(moods)), len(_category_names))
    return [
        (_category_names[i], float(aggregated_scores[i]))
        for i in ranked_indices[:n_return]
    ]

if __name__ == "__main__":
    test_cases = [
        ["chill", "relaxing", "food"],
        ["adventurous", "hiking", "nature"],
        ["art", "culture", "history"],
        ["nightlife", "bars", "music"],
        ["beach", "ocean", "surfing"],
        ["romantic", "spa", "dining"],
    ]

    print(f"Model: {_MODEL_NAME}\n")
    for moods in test_cases:
        results = map_moods_with_scores(moods, top_k=3)
        print(f"Moods      : {moods}")
        print(f"Categories : {[r[0] for r in results]}")
        print(f"Scores     : {[round(r[1], 3) for r in results]}")
        print("-" * 55)