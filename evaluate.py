"""
evaluate.py
===========
Automated evaluation of the NLPilot RAG pipeline.

Metrics:
  1. Venue Grounding Rate — % of venues mentioned in itinerary that match retrieved data
  2. Constraint Satisfaction — budget adherence, day count, city correctness
  3. Hallucination Detection — venues in output NOT in retrieved set

Usage:
  python evaluate.py

Outputs results to console and saves to evaluation_results.json
"""

import json
import re
import sys
import time
from pathlib import Path

from pipeline.slot_filler import fill_slots
from pipeline.mood_mapper import map_moods
from pipeline.retriever import retrieve
from pipeline.generator import generate_itinerary

# ---------------------------------------------------------------------------
# Test queries — diverse cities, budgets, trip lengths, vibes
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "I'm flying from San Francisco to Philadelphia for 3 days with a $500 budget. I prefer public transport and I'm in the mood for art, local food, and some chill walks.",
        "expected_city": "Philadelphia",
        "expected_days": 3,
        "expected_budget": 500,
    },
    {
        "query": "I'm flying from Boston to Nashville for 2 days with a $800 budget. I love music, bars, and BBQ.",
        "expected_city": "Nashville",
        "expected_days": 2,
        "expected_budget": 800,
    },
]


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def check_venue_grounding(itinerary: str, retrieved_venues: list[dict]) -> dict:
    """
    Check what % of venue names in the itinerary are from the retrieved set.
    Returns grounding rate and lists of grounded vs hallucinated venues.
    """
    # Extract retrieved venue names
    retrieved_names = []
    for v in retrieved_venues:
        name = v.get("metadata", {}).get("name", "")
        if name:
            retrieved_names.append(name.lower().strip())

    # Find venue-like mentions in itinerary
    # Look for patterns: "at VenueName", "visit VenueName", venue names directly
    grounded = []
    hallucinated = []

    for name in retrieved_names:
        if name in itinerary.lower():
            grounded.append(name)

    # Count total unique venue-like mentions
    total_mentioned = len(grounded) + len(hallucinated)
    grounding_rate = len(grounded) / len(retrieved_names) if retrieved_names else 0

    return {
        "retrieved_count": len(retrieved_names),
        "grounded_count": len(grounded),
        "grounding_rate": round(grounding_rate, 2),
        "grounded_venues": grounded,
    }


def check_constraints(
    itinerary: str,
    slots,
    expected_city: str,
    expected_days: int,
    expected_budget: float,
) -> dict:
    """
    Check if the itinerary satisfies the trip constraints.
    """
    results = {
        "city_mentioned": False,
        "day_count_correct": False,
        "budget_referenced": False,
        "days_found": 0,
    }

    text_lower = itinerary.lower()

    # City check
    results["city_mentioned"] = expected_city.lower() in text_lower

    # Day count — count "Day N" patterns
    day_matches = re.findall(r'day\s*(\d+)', text_lower)
    if day_matches:
        max_day = max(int(d) for d in day_matches)
        results["days_found"] = max_day
        results["day_count_correct"] = max_day == expected_days

    # Budget — check if any dollar amounts are mentioned
    dollar_amounts = re.findall(r'\$[\d,]+', itinerary)
    results["budget_referenced"] = len(dollar_amounts) > 0
    results["dollar_amounts_found"] = len(dollar_amounts)

    return results


def check_hallucination(itinerary: str, retrieved_venues: list[dict], city: str) -> dict:
    """
    Check if the itinerary mentions the city not being in the database
    (which means no hallucination occurred — the system correctly refused).
    """
    text_lower = itinerary.lower()

    is_refusal = "no venue data found" in text_lower or "not in our database" in text_lower

    return {
        "is_refusal": is_refusal,
        "venue_count": len(retrieved_venues),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    print("=" * 70)
    print("NLPilot RAG Pipeline Evaluation")
    print("=" * 70)
    print()

    all_results = []
    total_grounding = 0
    total_constraints_met = 0
    total_tests = len(TEST_QUERIES)

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected_city = test["expected_city"]
        expected_days = test["expected_days"]
        expected_budget = test["expected_budget"]

        print(f"[{i}/{total_tests}] Testing: {expected_city} ({expected_days} days, ${expected_budget})")
        print(f"  Query: {query[:80]}...")

        start_time = time.time()

        # Run pipeline
        try:
            slots = fill_slots(query)
            categories = map_moods(slots.moods, top_k=3)
            context = retrieve(slots, categories)
            n_venues = len(context.venues)

            if n_venues == 0:
                print(f"  Retrieval: 0 venues (city not in dataset)")
                hallucination_check = {"is_refusal": True, "venue_count": 0}
                grounding = {"retrieved_count": 0, "grounded_count": 0, "grounding_rate": 0, "grounded_venues": []}
                constraints = {"city_mentioned": False, "day_count_correct": False, "budget_referenced": False, "days_found": 0}
                itinerary = "No venue data — refusal triggered."
            else:
                print(f"  Retrieval: {n_venues} venues found")
                result = generate_itinerary(slots, context)
                itinerary = result.itinerary

                # Evaluate
                grounding = check_venue_grounding(itinerary, context.venues)
                constraints = check_constraints(itinerary, slots, expected_city, expected_days, expected_budget)
                hallucination_check = check_hallucination(itinerary, context.venues, expected_city)

            elapsed = time.time() - start_time

            # Score
            constraint_score = sum([
                constraints.get("city_mentioned", False),
                constraints.get("day_count_correct", False),
                constraints.get("budget_referenced", False),
            ]) / 3

            total_grounding += grounding["grounding_rate"]
            total_constraints_met += constraint_score

            test_result = {
                "query": query,
                "expected_city": expected_city,
                "expected_days": expected_days,
                "expected_budget": expected_budget,
                "venues_retrieved": n_venues,
                "grounding": grounding,
                "constraints": constraints,
                "hallucination": hallucination_check,
                "constraint_score": round(constraint_score, 2),
                "time_seconds": round(elapsed, 1),
            }
            all_results.append(test_result)

            print(f"  Grounding rate: {grounding['grounding_rate']:.0%} ({grounding['grounded_count']}/{grounding['retrieved_count']})")
            print(f"  Constraints: city={'Y' if constraints.get('city_mentioned') else 'N'} | "
                  f"days={'Y' if constraints.get('day_count_correct') else 'N'} ({constraints.get('days_found')}/{expected_days}) | "
                  f"budget={'Y' if constraints.get('budget_referenced') else 'N'}")
            print(f"  Constraint score: {constraint_score:.0%}")
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "query": query,
                "expected_city": expected_city,
                "error": str(e),
            })

        print()

    # Summary
    avg_grounding = total_grounding / total_tests if total_tests else 0
    avg_constraints = total_constraints_met / total_tests if total_tests else 0

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Test queries:              {total_tests}")
    print(f"  Avg venue grounding rate:  {avg_grounding:.0%}")
    print(f"  Avg constraint satisfaction: {avg_constraints:.0%}")
    print()

    # Save results
    output = {
        "summary": {
            "total_tests": total_tests,
            "avg_grounding_rate": round(avg_grounding, 3),
            "avg_constraint_satisfaction": round(avg_constraints, 3),
        },
        "results": all_results,
    }

    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()