"""
fetch_datasets.py
=================
Loads and preprocesses both data sources for the NLPilot RAG pipeline:

  1. Yelp Academic Dataset  – local JSON files downloaded from Kaggle / Yelp
  2. TravelPlanner          – HuggingFace dataset (osunlp/TravelPlanner)

Outputs (written to data/processed/):
  - yelp_venues.jsonl          business records relevant to travel (hotels, restaurants, …)
  - yelp_reviews.jsonl         trimmed review records linked to those businesses
  - travelplanner_train.jsonl  training split
  - travelplanner_validation.jsonl

Usage:
  python data/fetch_datasets.py \
      --yelp_dir /path/to/yelp_dataset \
      --output_dir data/processed

Requirements:
  pip install datasets tqdm
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Yelp helpers
# ---------------------------------------------------------------------------

# Categories that are relevant to travel itinerary planning
TRAVEL_CATEGORIES = {
    "hotels", "hotel", "bed & breakfast", "hostels", "resorts", "vacation rentals",
    "restaurants", "food", "bars", "nightlife", "cafes", "coffee & tea",
    "attractions", "museums", "arts", "parks", "landmarks", "tours",
    "shopping", "spas", "fitness", "yoga", "gyms",
    "transportation", "taxis", "car rental",
}


def _is_travel_relevant(categories_str: str) -> bool:
    """Return True if any of the business's categories overlap with travel interests."""
    if not categories_str:
        return False
    cats = {c.strip().lower() for c in categories_str.split(",")}
    return bool(cats & TRAVEL_CATEGORIES)


def load_yelp_businesses(yelp_dir: Path, output_path: Path) -> set:
    """
    Parse yelp_academic_dataset_business.json and keep travel-relevant records.
    Returns the set of business_ids retained (used to filter reviews).
    """
    src = yelp_dir / "yelp_academic_dataset_business.json"
    if not src.exists():
        print(f"[Yelp] Business file not found at {src}. Skipping.")
        return set()

    kept_ids: set = set()
    total = 0

    print(f"[Yelp] Loading businesses from {src} …")
    with src.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="  businesses", unit=" records"):
            line = line.strip()
            if not line:
                continue
            total += 1
            biz = json.loads(line)

            if not _is_travel_relevant(biz.get("categories") or ""):
                continue

            record = {
                "business_id":  biz.get("business_id"),
                "name":         biz.get("name"),
                "city":         biz.get("city"),
                "state":        biz.get("state"),
                "latitude":     biz.get("latitude"),
                "longitude":    biz.get("longitude"),
                "stars":        biz.get("stars"),
                "review_count": biz.get("review_count"),
                "categories":   biz.get("categories"),
                "attributes":   biz.get("attributes"),
                "hours":        biz.get("hours"),
                "is_open":      biz.get("is_open"),
            }
            fout.write(json.dumps(record) + "\n")
            kept_ids.add(record["business_id"])

    print(f"[Yelp] Retained {len(kept_ids):,} / {total:,} businesses.")
    return kept_ids


def load_yelp_reviews(yelp_dir: Path, output_path: Path, kept_ids: set, max_reviews: int = 500_000) -> None:
    """
    Parse yelp_academic_dataset_review.json and keep reviews for retained businesses.
    Caps at max_reviews to keep file size manageable.
    """
    src = yelp_dir / "yelp_academic_dataset_review.json"
    if not src.exists():
        print(f"[Yelp] Reviews file not found at {src}. Skipping.")
        return
    if not kept_ids:
        print("[Yelp] No business IDs to filter reviews against. Skipping reviews.")
        return

    written = 0
    total = 0

    print(f"[Yelp] Loading reviews from {src} …")
    with src.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="  reviews", unit=" records"):
            if written >= max_reviews:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            rev = json.loads(line)

            if rev.get("business_id") not in kept_ids:
                continue

            record = {
                "review_id":   rev.get("review_id"),
                "business_id": rev.get("business_id"),
                "stars":       rev.get("stars"),
                "text":        rev.get("text"),
                "date":        rev.get("date"),
                "useful":      rev.get("useful"),
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"[Yelp] Wrote {written:,} reviews (scanned {total:,}).")


def load_yelp_tips(yelp_dir: Path, output_path: Path, kept_ids: set) -> None:
    """
    Parse yelp_academic_dataset_tip.json and keep tips for retained businesses.
    Tips are shorter user notes — useful as concise venue descriptions for RAG chunks.
    """
    src = yelp_dir / "yelp_academic_dataset_tip.json"
    if not src.exists():
        print(f"[Yelp] Tips file not found at {src}. Skipping tips.")
        return
    if not kept_ids:
        return

    written = 0
    print(f"[Yelp] Loading tips from {src} …")
    with src.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="  tips", unit=" records"):
            line = line.strip()
            if not line:
                continue
            tip = json.loads(line)
            if tip.get("business_id") not in kept_ids:
                continue
            record = {
                "business_id": tip.get("business_id"),
                "text":        tip.get("text"),
                "date":        tip.get("date"),
                "compliment_count": tip.get("compliment_count"),
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"[Yelp] Wrote {written:,} tips.")


# ---------------------------------------------------------------------------
# TravelPlanner helpers
# ---------------------------------------------------------------------------

def load_travelplanner(output_dir: Path) -> None:
    """
    Download the osunlp/TravelPlanner dataset from HuggingFace and save each
    split as a JSONL file.  Fields kept:
        query, org, dest, days, local_constraint, reference_information, plan
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[TravelPlanner] `datasets` package not installed. Run: pip install datasets")
        return

    print("[TravelPlanner] Fetching dataset from HuggingFace (osunlp/TravelPlanner) …")
    try:
        ds = load_dataset("osunlp/TravelPlanner")
    except Exception as exc:
        print(f"[TravelPlanner] Failed to load dataset: {exc}")
        return

    # Fields present in TravelPlanner; gracefully handle missing ones
    FIELDS_TO_KEEP = [
        "query",
        "org",
        "dest",
        "days",
        "local_constraint",
        "reference_information",
        "plan",
        "annotated_plan",
        "level",
    ]

    for split_name, split_data in ds.items():
        out_file = output_dir / f"travelplanner_{split_name}.jsonl"
        print(f"[TravelPlanner] Writing split '{split_name}' ({len(split_data):,} rows) → {out_file}")
        with out_file.open("w", encoding="utf-8") as fout:
            for row in tqdm(split_data, desc=f"  {split_name}", unit=" rows"):
                record = {k: row.get(k) for k in FIELDS_TO_KEEP if k in row}
                fout.write(json.dumps(record) + "\n")
        print(f"[TravelPlanner] ✓ {split_name} saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and preprocess Yelp + TravelPlanner datasets for NLPilot."
    )
    parser.add_argument(
        "--yelp_dir",
        type=str,
        default=None,
        help=(
            "Path to the folder containing the Yelp Academic Dataset JSON files "
            "(e.g. yelp_academic_dataset_business.json). "
            "If omitted, the Yelp step is skipped."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory where processed JSONL files will be written (default: data/processed).",
    )
    parser.add_argument(
        "--max_reviews",
        type=int,
        default=500_000,
        help="Maximum number of Yelp reviews to keep (default: 500000).",
    )
    parser.add_argument(
        "--skip_yelp",
        action="store_true",
        help="Skip the Yelp dataset entirely.",
    )
    parser.add_argument(
        "--skip_travelplanner",
        action="store_true",
        help="Skip the TravelPlanner dataset entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}\n")

    # ── Yelp ──────────────────────────────────────────────────────────────
    if not args.skip_yelp:
        if args.yelp_dir is None:
            print(
                "[Yelp] --yelp_dir not provided. Skipping Yelp dataset.\n"
                "       Re-run with:  --yelp_dir /path/to/yelp_dataset\n"
            )
        else:
            yelp_dir = Path(args.yelp_dir)
            kept_ids = load_yelp_businesses(
                yelp_dir,
                output_dir / "yelp_venues.jsonl",
            )
            load_yelp_reviews(
                yelp_dir,
                output_dir / "yelp_reviews.jsonl",
                kept_ids,
                max_reviews=args.max_reviews,
            )
            load_yelp_tips(
                yelp_dir,
                output_dir / "yelp_tips.jsonl",
                kept_ids,
            )
            print()

    # ── TravelPlanner ──────────────────────────────────────────────────────
    if not args.skip_travelplanner:
        load_travelplanner(output_dir)
        print()

    print("Done. Processed files are in:", output_dir.resolve())


if __name__ == "__main__":
    main()
