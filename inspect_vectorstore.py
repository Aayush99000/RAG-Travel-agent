"""
inspect_vectorstore.py
======================
Connects to your existing ChromaDB and shows exactly what's stored.
Run this BEFORE making any changes so you know what you're working with.

Usage:
  python inspect_vectorstore.py
  python inspect_vectorstore.py --vectorstore_dir path/to/vectorstore
"""

import argparse
import json
from pathlib import Path

import chromadb

VECTORSTORE_DIR = "vectorstore"


def inspect(vectorstore_dir: str = VECTORSTORE_DIR):
    path = Path(vectorstore_dir)
    if not path.exists():
        print(f"Vector store not found at '{path.resolve()}'")
        return

    client = chromadb.PersistentClient(path=str(path))
    collections = client.list_collections()

    print(f"Vector store: {path.resolve()}")
    print(f"Collections found: {len(collections)}\n")
    print("=" * 70)

    for col in collections:
        count = col.count()
        print(f"Collection: {col.name}")
        print(f"   Documents:  {count:,}")

        if count == 0:
            print("   (empty)")
            continue

        # Peek at first 3 documents
        n_sample = min(3, count)
        sample = col.peek(limit=n_sample)

        ids       = sample.get("ids", [])
        documents = sample.get("documents", [])
        metadatas = sample.get("metadatas", [])

        print(f"\n   --- Sample documents (first {n_sample}) ---")
        for i in range(n_sample):
            doc_id   = ids[i] if i < len(ids) else "?"
            doc_text = documents[i] if i < len(documents) else ""
            meta     = metadatas[i] if i < len(metadatas) else {}

            # Truncate long documents for display
            display_text = doc_text[:300] if doc_text else "(empty)"
            if len(doc_text) > 300:
                display_text += "…"

            print(f"\n   [{i+1}] ID: {doc_id}")
            print(f"       Metadata keys: {list(meta.keys())}")
            print(f"       Metadata: {json.dumps(meta, indent=8, default=str)[:500]}")
            print(f"       Document: {display_text}")

        # Show metadata schema summary
        if metadatas:
            all_keys = set()
            for m in metadatas:
                all_keys.update(m.keys())
            print(f"\n   Metadata schema: {sorted(all_keys)}")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB vectorstore contents.")
    parser.add_argument("--vectorstore_dir", type=str, default=VECTORSTORE_DIR)
    args = parser.parse_args()
    inspect(args.vectorstore_dir)