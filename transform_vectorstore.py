"""
transform_vectorstore.py
========================
Patches your existing ChromaDB in-place. No deletion, no full re-ingestion.

What it does:
  1. yelp_venues:   Adds 'city_lower' metadata field for case-insensitive filtering.
                    (Documents and embeddings are untouched.)
  2. travelplanner: Converts raw JSON blob documents into natural language text,
                    adds rich metadata (dest, org, days, query).
                    ChromaDB will re-compute embeddings for updated documents.

Usage:
  python transform_vectorstore.py
  python transform_vectorstore.py --vectorstore_dir path/to/vectorstore

Requirements:
  pip install chromadb sentence-transformers tqdm
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 200


def _batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# 1. Patch yelp_venues: add city_lower to metadata
# ---------------------------------------------------------------------------

def patch_venues(client, ef):
    """Add city_lower field to venue metadata for case-insensitive filtering."""
    try:
        col = client.get_collection("yelp_venues", embedding_function=ef)
    except Exception:
        print("[yelp_venues] Collection not found — skipping.")
        return

    count = col.count()
    print(f"\n[yelp_venues] Patching {count:,} documents (adding city_lower)…")

    # Process in batches
    offset = 0
    patched = 0

    while offset < count:
        batch_size = min(BATCH_SIZE, count - offset)
        results = col.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        ids = results["ids"]
        metadatas = results["metadatas"]

        update_ids = []
        update_metas = []

        for doc_id, meta in zip(ids, metadatas):
            if not meta:
                continue
            # Skip if already has city_lower
            if "city_lower" in meta:
                continue

            city = meta.get("city", "")
            meta["city_lower"] = str(city).lower().strip()
            update_ids.append(doc_id)
            update_metas.append(meta)

        if update_ids:
            col.update(ids=update_ids, metadatas=update_metas)
            patched += len(update_ids)

        offset += batch_size
        print(f"  Processed {min(offset, count):,} / {count:,} (patched {patched:,})", end="\r")

    print(f"\n[yelp_venues] ✓ Patched {patched:,} documents with city_lower field.")


# ---------------------------------------------------------------------------
# 2. Transform travelplanner: JSON blobs → natural language
# ---------------------------------------------------------------------------

def _json_to_natural_language(raw_doc: str, existing_meta: dict) -> tuple[str, dict]:
    """
    Convert a raw JSON document into a natural-language text + rich metadata.
    Handles the nested structure: {"Attractions in X": [...], "Restaurants in X": [...], ...}
    as well as flat trip records: {"query": "...", "org": "...", "dest": "...", ...}
    """
    try:
        data = json.loads(raw_doc)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — return as-is
        return raw_doc, existing_meta

    if not isinstance(data, dict):
        return raw_doc, existing_meta

    parts = []
    meta = dict(existing_meta)  # preserve existing metadata

    # --- Handle flat trip record format ---
    query = data.get("query", "")
    org = data.get("org", "")
    dest = data.get("dest", "")
    days = data.get("days", "")

    if query:
        parts.append(f"Trip request: {query}")
        meta["query"] = str(query)[:500]
    if org:
        meta["org"] = str(org)
    if dest:
        meta["dest"] = str(dest)
        meta["dest_lower"] = str(dest).lower().strip()
    if days:
        meta["days"] = str(days)
    if org and dest:
        parts.append(f"Traveling from {org} to {dest}.")
    elif dest:
        parts.append(f"Traveling to {dest}.")
    if days:
        parts.append(f"Duration: {days} days.")

    constraint = data.get("local_constraint", "")
    if constraint and str(constraint).lower() not in ("none", ""):
        parts.append(f"Constraints: {constraint}.")
        meta["local_constraint"] = str(constraint)

    # --- Handle nested venue/attraction data ---
    # Format: {"Attractions in CityName": [...], "Restaurants in CityName": [...]}
    for key, value in data.items():
        if key in ("query", "org", "dest", "days", "local_constraint",
                    "plan", "annotated_plan", "reference_information",
                    "level", "split"):
            continue

        if isinstance(value, list) and value:
            # This is likely "Attractions in X" or "Restaurants in X"
            category_label = key  # e.g. "Attractions in Stockton"
            venue_names = []
            for item in value[:10]:  # cap at 10 venues per category
                if isinstance(item, dict):
                    name = item.get("Name", "")
                    city = item.get("City", "")
                    if name:
                        venue_names.append(f"{name} ({city})" if city else name)

                        # Try to extract destination from venue data if not set
                        if not dest and city:
                            dest = city
                            meta["dest"] = dest
                            meta["dest_lower"] = dest.lower().strip()

            if venue_names:
                parts.append(f"{category_label}: {', '.join(venue_names)}.")

    # --- Handle reference_information (nested dict of categories) ---
    ref_info = data.get("reference_information", "")
    if isinstance(ref_info, dict):
        for category, items in ref_info.items():
            if isinstance(items, list):
                names = []
                for item in items[:8]:
                    if isinstance(item, dict):
                        name = item.get("Name", "") or item.get("name", "")
                        if name:
                            names.append(name)
                if names:
                    parts.append(f"{category}: {', '.join(names)}.")
    elif isinstance(ref_info, str) and ref_info.strip():
        parts.append(f"Reference: {ref_info[:300]}")

    # --- Handle plan/annotated_plan ---
    plan = data.get("annotated_plan") or data.get("plan")
    if plan:
        if isinstance(plan, list):
            plan_parts = []
            for day_plan in plan[:7]:
                if isinstance(day_plan, dict):
                    plan_parts.append(json.dumps(day_plan)[:200])
                elif isinstance(day_plan, str):
                    plan_parts.append(day_plan[:200])
            if plan_parts:
                parts.append(f"Sample itinerary: {' | '.join(plan_parts)}")
        elif isinstance(plan, str):
            parts.append(f"Sample itinerary: {plan[:500]}")

    # Build final document
    doc = " ".join(parts) if parts else raw_doc[:500]

    return doc, meta


def transform_travelplanner(client, ef):
    """Transform TravelPlanner documents from JSON blobs to natural language."""
    try:
        col = client.get_collection("travelplanner", embedding_function=ef)
    except Exception:
        print("[travelplanner] Collection not found — skipping.")
        return

    count = col.count()
    print(f"\n[travelplanner] Transforming {count:,} documents (JSON → natural language)…")

    offset = 0
    transformed = 0
    skipped = 0

    while offset < count:
        batch_size = min(BATCH_SIZE, count - offset)
        results = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = results["ids"]
        documents = results["documents"]
        metadatas = results["metadatas"]

        update_ids = []
        update_docs = []
        update_metas = []

        for doc_id, doc, meta in zip(ids, documents, metadatas):
            if not doc:
                skipped += 1
                continue

            # Check if already transformed (has dest in metadata)
            if meta and meta.get("dest"):
                skipped += 1
                continue

            new_doc, new_meta = _json_to_natural_language(doc, meta or {})

            # Only update if we actually changed the document
            if new_doc != doc or new_meta != meta:
                update_ids.append(doc_id)
                update_docs.append(new_doc)
                update_metas.append(new_meta)

        if update_ids:
            # This will re-compute embeddings for updated documents
            col.update(
                ids=update_ids,
                documents=update_docs,
                metadatas=update_metas,
            )
            transformed += len(update_ids)

        offset += batch_size
        print(f"  Processed {min(offset, count):,} / {count:,} (transformed {transformed:,}, skipped {skipped:,})", end="\r")

    print(f"\n[travelplanner] ✓ Transformed {transformed:,} documents. Skipped {skipped:,}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transform existing ChromaDB vectorstore in-place."
    )
    parser.add_argument(
        "--vectorstore_dir", type=str, default=VECTORSTORE_DIR,
        help=f"Path to ChromaDB directory (default: {VECTORSTORE_DIR})"
    )
    parser.add_argument(
        "--skip_venues", action="store_true",
        help="Skip venue metadata patching"
    )
    parser.add_argument(
        "--skip_tp", action="store_true",
        help="Skip TravelPlanner transformation"
    )
    args = parser.parse_args()

    path = Path(args.vectorstore_dir)
    if not path.exists():
        print(f"❌ Vector store not found at '{path.resolve()}'")
        return

    print(f"Vector store: {path.resolve()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(path))

    if not args.skip_venues:
        patch_venues(client, ef)

    if not args.skip_tp:
        transform_travelplanner(client, ef)

    print("Transform complete.")
    print("\nVerify by running:  python inspect_vectorstore.py")


if __name__ == "__main__":
    main()