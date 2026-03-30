"""
ingest_chromadb.py
==================
Reads processed JSONL files from data/processed/ and ingests them into
a local ChromaDB vector store under vectorstore/.

Embeddings are generated locally using sentence-transformers
(all-MiniLM-L6-v2 by default — fast and memory-efficient).

Usage:
  python pipeline/ingest_chromadb.py --data_dir data/processed

Requirements:
  pip install chromadb sentence-transformers tqdm
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   
VECTORSTORE_DIR  = "vectorstore"
BATCH_SIZE       = 256                   # ChromaDB add() batch size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _batch(lst: list, size: int):
    """Yield successive chunks of `size` from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _upsert_batched(collection, ids, documents, metadatas) -> None:
    """Upsert in batches to avoid memory spikes on large datasets."""
    total = len(ids)
    for id_batch, doc_batch, meta_batch in zip(
        _batch(ids, BATCH_SIZE),
        _batch(documents, BATCH_SIZE),
        _batch(metadatas, BATCH_SIZE),
    ):
        collection.upsert(
            ids=id_batch,
            documents=doc_batch,
            metadatas=meta_batch,
        )
    print(f"  ✓ {total:,} records upserted into '{collection.name}'")


# ---------------------------------------------------------------------------
# Yelp venues
# ---------------------------------------------------------------------------

def ingest_yelp_venues(data_dir: Path, client: chromadb.Client, ef) -> None:
    src = data_dir / "yelp_venues.jsonl"
    if not src.exists():
        print(f"[Yelp Venues] {src} not found — skipping.")
        return

    print(f"[Yelp Venues] Loading {src} …")
    records = _load_jsonl(src)
    collection = client.get_or_create_collection("yelp_venues", embedding_function=ef)

    ids, documents, metadatas = [], [], []

    for r in tqdm(records, desc="  building venue docs", unit=" venues"):
        bid = r.get("business_id", "")
        if not bid:
            continue

        # Build a rich text chunk the LLM can reason over
        doc = (
            f"Name: {r.get('name', '')}. "
            f"City: {r.get('city', '')}, {r.get('state', '')}. "
            f"Categories: {r.get('categories', '')}. "
            f"Rating: {r.get('stars', '')} stars ({r.get('review_count', 0)} reviews). "
            f"Hours: {json.dumps(r.get('hours') or {})}."
        )

        meta = {
            "business_id":  bid,
            "name":         str(r.get("name") or ""),
            "city":         str(r.get("city") or ""),
            "state":        str(r.get("state") or ""),
            "stars":        float(r.get("stars") or 0),
            "review_count": int(r.get("review_count") or 0),
            "categories":   str(r.get("categories") or ""),
            "is_open":      int(r.get("is_open") or 0),
        }

        ids.append(bid)
        documents.append(doc)
        metadatas.append(meta)

    _upsert_batched(collection, ids, documents, metadatas)


# ---------------------------------------------------------------------------
# Yelp reviews
# ---------------------------------------------------------------------------

def ingest_yelp_reviews(data_dir: Path, client: chromadb.Client, ef) -> None:
    src = data_dir / "yelp_reviews.jsonl"
    if not src.exists():
        print(f"[Yelp Reviews] {src} not found — skipping.")
        return

    print(f"[Yelp Reviews] Loading {src} …")
    records = _load_jsonl(src)
    collection = client.get_or_create_collection("yelp_reviews", embedding_function=ef)

    ids, documents, metadatas = [], [], []

    for r in tqdm(records, desc="  building review docs", unit=" reviews"):
        rid = r.get("review_id", "")
        text = (r.get("text") or "").strip()
        if not rid or not text:
            continue

        ids.append(rid)
        documents.append(text)
        metadatas.append({
            "business_id": str(r.get("business_id") or ""),
            "stars":       float(r.get("stars") or 0),
            "date":        str(r.get("date") or ""),
            "useful":      int(r.get("useful") or 0),
        })

    _upsert_batched(collection, ids, documents, metadatas)


# ---------------------------------------------------------------------------
# Yelp tips
# ---------------------------------------------------------------------------

def ingest_yelp_tips(data_dir: Path, client: chromadb.Client, ef) -> None:
    src = data_dir / "yelp_tips.jsonl"
    if not src.exists():
        print(f"[Yelp Tips] {src} not found — skipping.")
        return

    print(f"[Yelp Tips] Loading {src} …")
    records = _load_jsonl(src)
    collection = client.get_or_create_collection("yelp_tips", embedding_function=ef)

    ids, documents, metadatas = [], [], []

    for i, r in enumerate(tqdm(records, desc="  building tip docs", unit=" tips")):
        text = (r.get("text") or "").strip()
        bid  = r.get("business_id", "")
        if not text or not bid:
            continue

        ids.append(f"{bid}_{i}")
        documents.append(text)
        metadatas.append({
            "business_id":      str(bid),
            "date":             str(r.get("date") or ""),
            "compliment_count": int(r.get("compliment_count") or 0),
        })

    _upsert_batched(collection, ids, documents, metadatas)


# ---------------------------------------------------------------------------
# TravelPlanner
# ---------------------------------------------------------------------------

def ingest_travelplanner(data_dir: Path, client: chromadb.Client, ef) -> None:
    """
    Combines all travelplanner_*.jsonl splits into a single 'travelplanner'
    collection so the retriever has one place to query.
    """
    files = sorted(data_dir.glob("travelplanner_*.jsonl"))
    if not files:
        print("[TravelPlanner] No travelplanner_*.jsonl files found — skipping.")
        return

    collection = client.get_or_create_collection("travelplanner", embedding_function=ef)
    ids, documents, metadatas = [], [], []

    for src in files:
        split = src.stem.replace("travelplanner_", "")
        print(f"[TravelPlanner] Loading {src.name} (split: {split}) …")
        records = _load_jsonl(src)

        for i, r in enumerate(tqdm(records, desc=f"  {split}", unit=" rows")):
            query    = (r.get("query") or "").strip()
            ref_info = (r.get("reference_information") or "").strip()

            if not query:
                continue

            # Embed query + reference_information together so semantic search
            # on a user's trip request retrieves relevant reference plans.
            doc = f"Query: {query}\nReference: {ref_info}" if ref_info else query

            ids.append(f"{split}_{i}")
            documents.append(doc)
            metadatas.append({
                "split":  split,
                "org":    str(r.get("org") or ""),
                "dest":   str(r.get("dest") or ""),
                "days":   str(r.get("days") or ""),
                "level":  str(r.get("level") or ""),
                "query":  query,
            })

    _upsert_batched(collection, ids, documents, metadatas)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest processed JSONL files into ChromaDB for NLPilot RAG pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed JSONL files (default: data/processed).",
    )
    parser.add_argument(
        "--vectorstore_dir",
        type=str,
        default=VECTORSTORE_DIR,
        help=f"Directory to persist ChromaDB (default: {VECTORSTORE_DIR}).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Sentence-transformers model name (default: {EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--skip_venues",   action="store_true", help="Skip Yelp venues ingestion.")
    parser.add_argument(
        "--skip_reviews",  action="store_true", help="Skip Yelp reviews ingestion.")
    parser.add_argument(
        "--skip_tips",     action="store_true", help="Skip Yelp tips ingestion.")
    parser.add_argument(
        "--skip_tp",       action="store_true", help="Skip TravelPlanner ingestion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir       = Path(args.data_dir)
    vectorstore_dir = Path(args.vectorstore_dir)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding model : {args.embedding_model}")
    print(f"Data directory  : {data_dir.resolve()}")
    print(f"Vector store    : {vectorstore_dir.resolve()}\n")

    ef = SentenceTransformerEmbeddingFunction(model_name=args.embedding_model)
    client = chromadb.PersistentClient(path=str(vectorstore_dir))

    if not args.skip_venues:
        ingest_yelp_venues(data_dir, client, ef)
        print()

    if not args.skip_reviews:
        ingest_yelp_reviews(data_dir, client, ef)
        print()

    if not args.skip_tips:
        ingest_yelp_tips(data_dir, client, ef)
        print()

    if not args.skip_tp:
        ingest_travelplanner(data_dir, client, ef)
        print()

    print("Ingestion complete. Collections in ChromaDB:")
    for col in client.list_collections():
        print(f"  • {col.name}  ({col.count():,} docs)")


if __name__ == "__main__":
    main()
