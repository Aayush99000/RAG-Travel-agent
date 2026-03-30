"""
Itinerary generation using a locally running Ollama LLM (Llama 3).

Pipeline position:
  slot_filler → mood_mapper → retriever → [generator]

How it works:
  1. Builds a structured system prompt describing the task and constraints.
  2. Injects the RAG context (venues, reviews, tips, reference plans) from retriever.py.
  3. Sends the prompt to Ollama running locally (no API key needed).
  4. Supports multi-turn refinement — conversation history is maintained and
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

OLLAMA_MODEL    = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE     = 0.7        # slight creativity for itinerary variety
MAX_TOKENS      = 4096


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GeneratorResult:
    itinerary: str                              # the generated itinerary text
    history:   list[SystemMessage | HumanMessage | AIMessage] = field(default_factory=list)

    def __str__(self) -> str:
        return self.itinerary


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt(slots: TripSlots) -> str:
    """
    Build the system prompt that sets constraints and output format.
    This stays fixed across multi-turn refinements.
    """
    budget_str    = f"${slots.budget:,.0f}" if slots.budget else "unspecified"
    transport_str = slots.transport or "any"
    origin_str    = f" from {slots.origin}" if slots.origin else ""
    moods_str     = ", ".join(slots.moods) if slots.moods else "general travel"
    days_str      = str(slots.days) if slots.days else "a few"
    dest_str      = slots.destination or "the destination"

    return f"""You are NLPilot, an expert travel planner. Your job is to create a detailed, \
day-by-day travel itinerary that is grounded in real venue data.

TRIP DETAILS:
- Destination  : {dest_str}{origin_str}
- Duration     : {days_str} days
- Total Budget : {budget_str}
- Transport    : {transport_str}
- Mood/Vibe    : {moods_str}

RULES YOU MUST FOLLOW:
1. Generate exactly {days_str} days of activities.
2. Each day must have Morning, Afternoon, and Evening slots.
3. Only recommend venues from the CONTEXT section below. Do not invent venues.
4. Respect the total budget — include estimated costs per activity and a daily subtotal.
5. All transport suggestions must match the preferred mode: {transport_str}.
6. Keep the vibe consistent with the traveler's mood: {moods_str}.
7. Be specific — include venue names, neighborhoods, and practical tips.

OUTPUT FORMAT (follow exactly):
---
## Day 1: [Theme]
**Morning**
- [Activity] at [Venue Name] — [brief description] (~$[cost])
- [Transport tip]

**Afternoon**
- [Activity] at [Venue Name] — [brief description] (~$[cost])

**Evening**
- [Activity] at [Venue Name] — [brief description] (~$[cost])

Daily Estimated Cost: ~$[total]

---
(repeat for each day)

TOTAL ESTIMATED COST: ~$[grand total]
TIPS: [2-3 practical travel tips for this trip]
"""


def _build_user_prompt(context: RetrievedContext) -> str:
    """Build the first human turn — includes the RAG context."""
    context_text = context.to_prompt_text()
    if context_text.strip():
        return (
            "Using the venue data and reference itineraries below, "
            "generate the complete day-by-day itinerary.\n\n"
            f"{context_text}"
        )
    return (
        "No specific venue data was retrieved for this destination. "
        "Generate the best itinerary you can based on general knowledge, "
        "staying within the constraints defined."
    )


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
) -> GeneratorResult:
    """
    Generate a full day-by-day itinerary from slots + RAG context.

    Args:
        slots:   Structured trip slots from slot_filler.fill_slots()
        context: Retrieved venue/plan context from retriever.retrieve()
        model:   Ollama model to use (default: llama3)

    Returns:
        GeneratorResult with itinerary text and conversation history
    """
    llm = _get_llm()

    system_msg = SystemMessage(content=_build_system_prompt(slots))
    user_msg   = HumanMessage(content=_build_user_prompt(context))

    messages   = [system_msg, user_msg]

    print(f"[Generator] Calling Ollama ({model}) …")
    response   = llm.invoke(messages)
    itinerary  = response.content.strip()

    # Append AI response to history for multi-turn
    history    = messages + [AIMessage(content=itinerary)]

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
                              "Swap the museum on Day 1 with something outdoors"
                              "I'm on a tighter budget, cut costs where possible"
        previous_result: The GeneratorResult from a previous generate/refine call

    Returns:
        New GeneratorResult with updated itinerary and extended history
    """
    llm      = _get_llm()
    messages = previous_result.history + [HumanMessage(content=user_feedback)]

    print(f"[Generator] Refining itinerary …")
    response  = llm.invoke(messages)
    itinerary = response.content.strip()

    history   = messages + [AIMessage(content=itinerary)]

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
