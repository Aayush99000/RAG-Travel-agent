"""
Itinerary generation using Ollama LLM.
Final version — qwen3:4b on T4 GPU for demo.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Generator

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pipeline.slot_filler import TripSlots
from pipeline.retriever   import RetrievedContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL    = "qwen3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE     = 0.4
MAX_TOKENS      = 8000
TIMEOUT         = 120
MAX_HISTORY_TURNS = 2


@dataclass
class GeneratorResult:
    itinerary: str
    history:   list = field(default_factory=list)
    def __str__(self) -> str:
        return self.itinerary


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def _strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<think>.*?$", "", cleaned, flags=re.MULTILINE | re.DOTALL)
    return cleaned.strip()


def clean_itinerary(text: str) -> str:
    text = _strip_thinking_tags(text)
    original = text

    # Remove emojis
    text = re.sub(
        r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
        r'\U0000200D\U00002600-\U000026FF\U00002700-\U000027BF]',
        '', text
    )

    # Strip preamble before Day 1
    day1 = re.search(r'(#{1,3}\s*Day\s*1|Day\s*1\s*[:\-]|\*\*Day\s*1)', text, re.IGNORECASE)
    if day1:
        text = text[day1.start():]

    # Normalize venue citations
    text = re.sub(r'\(Venue\s*(\d+)\)', r'[Venue \1]', text)

    # Strip sections we don't want — but KEEP "Why This Works"
    for cut_pattern in [
        r'\n#{1,3}\s*Total Budget Breakdown.*?(?=\n#{1,3}\s|\nWhy This Works|\n\*\*Why This Works|$)',
        r'\n\*\*Total Budget Breakdown.*?(?=\n#{1,3}\s|\nWhy This Works|\n\*\*Why This Works|$)',
        r'\n#{1,3}\s*Summary\b.*?(?=\nWhy This Works|\n\*\*Why This Works|$)',
        r'\n\*\*Summary\b.*?(?=\nWhy This Works|\n\*\*Why This Works|$)',
        r'\n#{1,3}\s*Key Notes.*?(?=\nWhy This Works|\n\*\*Why This Works|$)',
        r'\n#{1,3}\s*Pro [Tt]ip.*',
        r'\n\*\*Pro [Tt]ip.*',
    ]:
        text = re.sub(cut_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Strip trailing meta-commentary lines
    for pattern in [
        r'^Note:.*This itinerary.*$',
        r'^Disclaimer:.*$',
        r'^Let me know.*$',
        r'^If you (want|need|have).*$',
        r'^This itinerary (is|was|delivers|prioritizes).*$',
        r'^Feel free.*$',
        r'^Happy travels.*$',
    ]:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove markdown tables (budget breakdown tables)
    text = re.sub(r'^\|.*\|.*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if len(text) < 50:
        return original.strip()
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _build_system_prompt(slots: TripSlots) -> str:
    days = slots.days or 3
    dest = slots.destination or "the destination"
    budget = f"${slots.budget:,.0f}" if slots.budget else "moderate"
    per_day = int(slots.budget / slots.days) if slots.budget and slots.days else 300
    moods = ", ".join(slots.moods) if slots.moods else "general sightseeing"

    return f"""/no_think
You generate travel itineraries. ONLY use venues from the provided list — do NOT invent venues. Budget: {budget} (~${per_day}/day). Vibe: {moods}.

Use this exact format for each day:

## Day 1: [Theme]

- **Morning (9:00 AM - 12:00 PM):**
  [Activity] at *[Venue Name]*
  Why: [2-3 sentences — what makes this place special, what to expect, how it fits the trip vibe]
  Cost: $[amount] ([brief breakdown like "entry fee" or "2-course meal"])

- **Afternoon (1:00 PM - 4:00 PM):**
  [Activity] at *[Venue Name]*
  Getting there: [transport mode and time from morning venue, e.g. "15 min walk" or "Bus route 42, ~10 min"]
  Why: [2-3 sentences]
  Cost: $[amount] ([brief breakdown])

- **Evening (6:00 PM - 9:00 PM):**
  [Activity] at *[Venue Name]*
  Getting there: [transport from afternoon venue]
  Why: [2-3 sentences]
  Cost: $[amount] ([brief breakdown])

- **Daily Total:** $[sum of all three]

---

Write all {days} days for {dest}. Each daily total should be close to ${per_day}. Grand total close to {budget}. Start directly with Day 1. STOP after the last day — do NOT add summaries, key notes, budget breakdowns, or closing remarks."""


def _build_user_prompt(context: RetrievedContext, slots: TripSlots) -> str:
    context_text = context.to_prompt_text()

    if not context_text.strip() or not context.venues:
        dest = slots.destination or "this destination"
        return (
            f"No venue data found for **{dest}** in our database. "
            f"NLPilot only generates itineraries for cities in the Yelp dataset.\n\n"
            f"**Try:** Philadelphia, Tucson, New Orleans, Nashville, Tampa, Boise, Reno."
        )

    budget_str = f"${slots.budget:,.0f}" if slots.budget else "moderate budget"
    days = str(slots.days) if slots.days else "a few"
    dest = slots.destination or "the destination"
    moods = ", ".join(slots.moods) if slots.moods else "general sightseeing"
    per_day = int(slots.budget / slots.days) if slots.budget and slots.days else 300

    return (
        f"Plan a {days}-day trip to {dest}. "
        f"Budget: {budget_str} (~${per_day}/day). Vibe: {moods}. "
        f"Use each venue only once.\n\n"
        f"{context_text}\n\n"
        f"Generate the itinerary now. Start with Day 1."
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_llm = None
_llm_stream = None

def _get_llm(streaming=False):
    global _llm, _llm_stream
    if streaming:
        if _llm_stream is None:
            _llm_stream = ChatOllama(
                model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE, num_predict=MAX_TOKENS, timeout=TIMEOUT,
            )
        return _llm_stream
    else:
        if _llm is None:
            _llm = ChatOllama(
                model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE, num_predict=MAX_TOKENS, timeout=TIMEOUT,
            )
        return _llm


def _trim_history(history):
    if len(history) <= 3:
        return history
    system = [m for m in history if isinstance(m, SystemMessage)]
    conv = [m for m in history if not isinstance(m, SystemMessage)]
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(conv) > max_msgs:
        conv = conv[-max_msgs:]
    return system + conv


# ---------------------------------------------------------------------------
# Generation — streaming
# ---------------------------------------------------------------------------

def generate_itinerary_stream(slots, context):
    if not context.venues:
        dest = slots.destination or "this destination"
        msg = (
            f"No venue data found for **{dest}** in our database. "
            f"NLPilot only generates itineraries for cities in the Yelp dataset.\n\n"
            f"**Try:** Philadelphia, Tucson, New Orleans, Nashville, Tampa."
        )
        yield msg
        yield GeneratorResult(itinerary=msg, history=[])
        return

    llm = _get_llm(streaming=True)
    system_msg = SystemMessage(content=_build_system_prompt(slots))
    user_msg = HumanMessage(content=_build_user_prompt(context, slots))
    messages = [system_msg, user_msg]

    full_text = ""
    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                full_text += token
                yield token
    except Exception as e:
        full_text += f"\n\nGeneration error: {e}"
        yield f"\n\nGeneration error: {e}"

    cleaned = clean_itinerary(full_text)
    history = messages + [AIMessage(content=cleaned)]
    yield GeneratorResult(itinerary=cleaned, history=history)


def generate_itinerary(slots, context, model=OLLAMA_MODEL):
    if not context.venues:
        msg = f"No venue data found for {slots.destination or 'this destination'}."
        return GeneratorResult(itinerary=msg, history=[])

    llm = _get_llm(streaming=False)
    system_msg = SystemMessage(content=_build_system_prompt(slots))
    user_msg = HumanMessage(content=_build_user_prompt(context, slots))
    messages = [system_msg, user_msg]

    try:
        response = llm.invoke(messages)
        itinerary = clean_itinerary(response.content.strip())
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e

    return GeneratorResult(itinerary=itinerary, history=messages + [AIMessage(content=itinerary)])


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def refine_itinerary_stream(user_feedback, previous_result):
    llm = _get_llm(streaming=True)
    history = _trim_history(previous_result.history)
    messages = history + [HumanMessage(content=user_feedback)]

    full_text = ""
    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                full_text += token
                yield token
    except Exception as e:
        full_text += f"\n\nRefinement error: {e}"
        yield f"\n\nRefinement error: {e}"

    cleaned = clean_itinerary(full_text)
    yield GeneratorResult(itinerary=cleaned, history=messages + [AIMessage(content=cleaned)])


def refine_itinerary(user_feedback, previous_result):
    llm = _get_llm(streaming=False)
    history = _trim_history(previous_result.history)
    messages = history + [HumanMessage(content=user_feedback)]

    try:
        response = llm.invoke(messages)
        itinerary = clean_itinerary(response.content.strip())
    except Exception as e:
        raise RuntimeError(f"Refinement failed: {e}") from e

    return GeneratorResult(itinerary=itinerary, history=messages + [AIMessage(content=itinerary)])