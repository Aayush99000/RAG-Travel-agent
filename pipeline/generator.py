"""
Itinerary generation using a locally running Ollama LLM (Qwen 3).

Pipeline position:
  slot_filler → mood_mapper → retriever → [generator]

How it works:
  1. Builds a structured system prompt describing the task and constraints,
     with real venue data injected directly from the RAG retriever.
  2. Sends the prompt to Ollama running locally (no API key needed).
  3. Supports multi-turn refinement — conversation history is maintained and
     follow-up edits ("make Day 2 more relaxed") are handled in context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pipeline.slot_filler import TripSlots
from pipeline.retriever   import RetrievedContext

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL    = "qwen3:1.7b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE     = 0.7
MAX_TOKENS      = 8000


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GeneratorResult:
    itinerary: str
    history:   list[SystemMessage | HumanMessage | AIMessage] = field(default_factory=list)

    def __str__(self) -> str:
        return self.itinerary


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_retrieved_venues(context: RetrievedContext) -> str:
    """Format RAG context for injection into the system prompt."""
    if not context.venues and not context.tips and not context.reviews:
        return "No specific venue data retrieved — use your best local knowledge for this destination."

    lines: list[str] = []

    if context.venues:
        lines.append("VENUES:")
        for v in context.venues:
            meta = v.get("metadata", {})
            lines.append(
                f"  • {meta.get('name', 'Unknown')} | {meta.get('city', '')} | "
                f"{meta.get('categories', '')} | "
                f"★ {meta.get('stars', '?')} ({meta.get('review_count', 0)} reviews)"
            )

    if context.tips:
        lines.append("\nVENUE TIPS:")
        for t in context.tips:
            lines.append(f"  • {t.get('document', '')}")

    if context.reviews:
        lines.append("\nREVIEWS:")
        for r in context.reviews:
            meta  = r.get("metadata", {})
            stars = meta.get("stars", "?")
            text  = r.get("document", "")[:150]
            lines.append(f"  • [{stars}★] {text}")

    if context.reference_plans:
        lines.append("\nREFERENCE ITINERARIES:")
        for p in context.reference_plans:
            meta = p.get("metadata", {})
            lines.append(
                f"  • {meta.get('days', '?')}-day trip to {meta.get('dest', '?')} "
                f"from {meta.get('org', '?')} | Query: {meta.get('query', '')}"
            )

    return "\n".join(lines)


def _build_system_prompt(
    slots: TripSlots,
    context: RetrievedContext,
    traveler_str: str = "solo traveler",
    dietary_str:  str = "none",
    fitness_str:  str = "moderate",
) -> str:
    budget_str    = f"${slots.budget:,.0f}" if slots.budget else "unspecified"
    transport_str = slots.transport or "any"
    origin_str    = slots.origin or "origin city"
    moods_str     = ", ".join(slots.moods) if slots.moods else "general travel"
    days_str      = str(slots.days) if slots.days else "a few"
    dest_str      = slots.destination or "the destination"

    # Dates
    arrival_date   = slots.start_date or "Day 1"
    if slots.start_date and slots.days:
        try:
            from datetime import datetime, timedelta
            arr = datetime.strptime(slots.start_date, "%B %d")
            dep = arr + timedelta(days=slots.days - 1)
            departure_date = dep.strftime("%B %d")
        except Exception:
            departure_date = f"Day {slots.days}"
    else:
        departure_date = f"Day {slots.days}" if slots.days else "last day"

    retrieved_venues = _format_retrieved_venues(context)

    return f"""\
You are NLPilot, an expert local travel planner with deep knowledge of {dest_str}. \
Your job is to create a hyper-personalized, realistic, day-by-day travel itinerary \
grounded entirely in the real venue data provided below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIP DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Destination     : {dest_str}
- Flying From     : {origin_str} (origin — NOT part of the itinerary)
- Arrival Date    : {arrival_date}
- Departure Date  : {departure_date}
- Duration        : {days_str} days
- Total Budget    : {budget_str}
- Transport       : {transport_str}
- Mood / Vibe     : {moods_str}
- Travelers       : {traveler_str}
- Dietary Needs   : {dietary_str}
- Fitness Level   : {fitness_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribute the total budget of {budget_str} across these categories:
- Accommodation  : ~35% of total budget
- Food & Drinks  : ~30% of total budget
- Activities     : ~20% of total budget
- Local Transport: ~15% of total budget

Use this breakdown to size daily costs realistically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL VENUE DATA (use ONLY these venues)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following venues have been retrieved from our database for {dest_str}.
You MUST prioritize these real venues. Do NOT invent venues not listed here.

{retrieved_venues}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSPORT RULES:
- {origin_str} is ONLY where the traveler flies FROM. Never include it in daily activities.
- NEVER suggest transport between {origin_str} and {dest_str} inside daily plans.
- All transport tips must describe LOCAL travel within {dest_str} ONLY.
- Use realistic local fares: metro ~$2-5, taxi/rideshare ~$10-20, walking ~$0.

VENUE RULES:
- ALL venues must be located in {dest_str}. No venues from other cities.
- Respect dietary needs: {dietary_str}. Never recommend restaurants that conflict with these.
- Respect fitness level: {fitness_str}. Avoid strenuous hikes for low fitness.
- Provide ONE backup option per day in case a venue is unexpectedly closed.

COST RULES:
- Every activity MUST have a non-zero cost estimate including food, drinks, transport, tips.
- Daily subtotal MUST be the sum of all activity costs + transport for that day.
- Grand total MUST equal the sum of all daily subtotals and stay within {budget_str}.

ITINERARY RULES:
- Generate ALL {days_str} days — never stop early or skip a day.
- Each day MUST have Breakfast, Morning, Lunch, Afternoon, Evening, and Dinner slots.
- Assign a unique mood-aligned theme to each day (e.g. "Day 1: Art & Culture").
- Last day ({departure_date}) must account for checkout time and travel to airport — \
keep it light with morning activities only and no evening plans.
- Keep the overall vibe consistent with: {moods_str}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCOMMODATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommend ONE hotel or accommodation in {dest_str} that fits the budget and mood.
Format:
**Recommended Stay:** [Hotel Name] — [brief reason] (~$[nightly rate]/night × {days_str} nights = ~$[total])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (follow exactly for every day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Recommended Stay:** [Hotel Name] — [reason] (~$[nightly rate]/night)

---
## Day [N]: [Mood-Aligned Theme]

**Breakfast**
- [Meal] at [Venue Name] — [brief description] (~$[cost])

**Morning**
- [Activity] at [Venue Name] — [brief description] (~$[cost])
- Getting there: [transport] from [Hotel or prev venue] to [Venue Name] (~$[fare])

**Lunch**
- [Meal] at [Venue Name] — [brief description] (~$[cost])
- Getting there: [transport] from [Morning venue] to [Lunch venue] (~$[fare])

**Afternoon**
- [Activity] at [Venue Name] — [brief description] (~$[cost])
- Getting there: [transport] from [Lunch venue] to [Afternoon venue] (~$[fare])

**Evening**
- [Activity] at [Venue Name] — [brief description] (~$[cost])
- Getting there: [transport] from [Afternoon venue] to [Evening venue] (~$[fare])

**Dinner**
- [Meal] at [Venue Name] — [brief description] (~$[cost])
- Getting there: [transport] from [Evening venue] to [Dinner venue] (~$[fare])

⚠️ **Backup Option:** If [primary venue] is closed, visit [alternative venue] instead.

**Daily Subtotal: ~$[sum of all costs above]**
  - Accommodation : ~$[nightly rate]
  - Food          : ~$[food total]
  - Activities    : ~$[activity total]
  - Transport     : ~$[transport total]
---

(repeat for all {days_str} days)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIP SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**TOTAL ESTIMATED COST: ~$[sum of all daily subtotals]**
  - Accommodation  : ~$[total]
  - Food & Drinks  : ~$[total]
  - Activities     : ~$[total]
  - Local Transport: ~$[total]

**PRACTICAL TIPS:**
1. [Local customs / etiquette tip — e.g. tipping culture, dress codes]
2. [Transport tip — e.g. best metro card to buy, rideshare apps that work locally]
3. [Safety or timing tip — e.g. neighborhoods to avoid at night, rush hour times]
4. [Money tip — e.g. best places to exchange currency, ATM fees]
5. [Packing tip — relevant to activities and mood e.g. walking shoes, layers]
"""


# ---------------------------------------------------------------------------
# LLM client — singleton
# ---------------------------------------------------------------------------

_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
        )
    return _llm


def generate_itinerary(
    slots: TripSlots,
    context: RetrievedContext,
    model: str = OLLAMA_MODEL,
    traveler_str: str = "solo traveler",
    dietary_str:  str = "none",
    fitness_str:  str = "moderate",
) -> GeneratorResult:
    """
    Generate a full day-by-day itinerary from slots + RAG context.

    Args:
        slots:        Structured trip slots from slot_filler.fill_slots()
        context:      Retrieved venue/plan context from retriever.retrieve()
        model:        Ollama model to use
        traveler_str: Group description (e.g. "couple", "family of 4")
        dietary_str:  Dietary restrictions (e.g. "vegetarian", "none")
        fitness_str:  Fitness level ("low", "moderate", "high")

    Returns:
        GeneratorResult with itinerary text and conversation history
    """
    llm = _get_llm()

    system_msg = SystemMessage(
        content=_build_system_prompt(slots, context, traveler_str, dietary_str, fitness_str)
    )
    user_msg = HumanMessage(content="Generate the complete itinerary now.")

    messages = [system_msg, user_msg]

    print(f"[Generator] Calling Ollama ({model}) …")
    response  = llm.invoke(messages)
    itinerary = response.content.strip()

    history = messages + [AIMessage(content=itinerary)]

    return GeneratorResult(itinerary=itinerary, history=history)


def refine_itinerary(
    user_feedback: str,
    previous_result: GeneratorResult,
) -> GeneratorResult:
    """
    Refine an existing itinerary based on a follow-up instruction.

    Args:
        user_feedback:   Natural language refinement request
                         e.g. "Make Day 3 more relaxed"
        previous_result: The GeneratorResult from a previous generate/refine call

    Returns:
        New GeneratorResult with updated itinerary and extended history
    """
    llm      = _get_llm()
    messages = previous_result.history + [HumanMessage(content=user_feedback)]

    print("[Generator] Refining itinerary …")
    response  = llm.invoke(messages)
    itinerary = response.content.strip()

    history = messages + [AIMessage(content=itinerary)]

    return GeneratorResult(itinerary=itinerary, history=history)


if __name__ == "__main__":
    from pipeline.slot_filler import fill_slots
    from pipeline.mood_mapper  import map_moods
    from pipeline.retriever    import retrieve

    query = (
        "I'm flying from San Francisco to New York for 5 days "
        "with a $2,000 budget. I prefer public transport and "
        "I'm in the mood for art, local food, and some chill walks."
    )

    print("Parsing query …")
    slots      = fill_slots(query)
    categories = map_moods(slots.moods, top_k=3)

    print(f"Slots     : {slots}")
    print(f"Categories: {categories}\n")

    print("Retrieving context …")
    context = retrieve(slots, categories)
    print(context.summary(), "\n")

    print("Generating itinerary …\n")
    result = generate_itinerary(slots, context)
    print(result.itinerary)

    print("\n" + "=" * 60)
    print("Refining: make Day 1 more food-focused …\n")
    refined = refine_itinerary("Make Day 1 more food-focused, add a food market visit.", result)
    print(refined.itinerary)
