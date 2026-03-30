"""
app.py
======
Streamlit frontend for NLPilot — AI-powered travel itinerary planner.

Run:
  streamlit run app.py

Make sure before running:
  - Ollama is running:  ollama serve
  - Llama 3 is pulled:  ollama pull llama3
  - ChromaDB is built:  python pipeline/ingest_chromadb.py --data_dir data/processed
"""

import streamlit as st

from pipeline.slot_filler import fill_slots, TripSlots
from pipeline.mood_mapper import map_moods
from pipeline.retriever   import retrieve
from pipeline.generator   import generate_itinerary, refine_itinerary

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NLPilot — AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "messages":       [],    # chat history [{role, content}]
        "slots":          None,  # TripSlots from last parse
        "categories":     [],    # mapped activity categories
        "gen_result":     None,  # last GeneratorResult (holds LLM history)
        "phase":          "input",  # "input" | "generated" | "refining"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helper: render a slots summary card in the sidebar
# ---------------------------------------------------------------------------

def _render_slots_card(slots: TripSlots):
    st.sidebar.markdown("### Trip Details")
    fields = {
        "Destination":  slots.destination or "—",
        "Origin":       slots.origin or "—",
        "Days":         str(slots.days) if slots.days else "—",
        "Start Date":   slots.start_date or "—",
        "Budget":       f"${slots.budget:,.0f}" if slots.budget else "—",
        "Transport":    slots.transport or "—",
        "Moods":        ", ".join(slots.moods) if slots.moods else "—",
    }
    for label, value in fields.items():
        st.sidebar.markdown(f"**{label}:** {value}")

    missing = slots.missing()
    if missing:
        st.sidebar.warning(f"Still needed: {', '.join(missing)}")
    else:
        st.sidebar.success("Trip details complete")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("✈️ NLPilot")
    st.caption("AI-powered travel itinerary planner — runs 100% locally.")
    st.divider()

    if st.session_state["slots"]:
        _render_slots_card(st.session_state["slots"])
        st.divider()

    if st.session_state["phase"] != "input":
        if st.button("Start New Trip", use_container_width=True):
            for k in ["messages", "slots", "categories", "gen_result", "phase"]:
                st.session_state[k] = [] if k in ("messages", "categories") else None
            st.session_state["phase"] = "input"
            st.rerun()

    st.divider()
    st.caption("Powered by Qwen 3 (1.7b) · Ollama · ChromaDB · spaCy")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.title("🧳 NLPilot — AI Travel Planner")
st.caption(
    "Describe your trip in plain English and get a personalized day-by-day itinerary. "
    "Once generated, you can refine it conversationally."
)

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Phase: INPUT — first query to generate itinerary
# ---------------------------------------------------------------------------

if st.session_state["phase"] == "input":

    # Show example prompts for first-time users
    if not st.session_state["messages"]:
        st.info(
            "**Try something like:**\n\n"
            "- *\"I'm flying from Chicago to Miami for 4 days with a \\$1,200 budget. "
            "I love beaches, nightlife, and street food. I'll use public transport.\"*\n\n"
            "- *\"Plan a 7-day trip to New Orleans from NYC, \\$2,000 budget, "
            "road trip vibes, jazz music, and Southern cuisine.\"*"
        )

    user_input = st.chat_input("Describe your trip…")

    if user_input:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Step 1: Parse slots
            with st.spinner("Parsing your trip details…"):
                slots = fill_slots(user_input)
                st.session_state["slots"] = slots

            # If critical slots are missing, ask for clarification
            if not slots.is_complete():
                missing = slots.missing()
                reply = (
                    f"I need a bit more info to plan your trip. Could you tell me:\n\n"
                    + "\n".join(f"- **{m}**" for m in missing)
                )
                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                # Stay in input phase so user can provide missing info
                st.rerun()

            # Step 2: Map moods → categories
            with st.spinner("Mapping your preferences to activities…"):
                categories = map_moods(slots.moods, top_k=3)
                st.session_state["categories"] = categories

            # Step 3: Retrieve RAG context
            with st.spinner("Searching venue database…"):
                try:
                    context = retrieve(slots, categories)
                    n_venues = len(context.venues)
                    venue_msg = f"Found {n_venues} relevant venues." if n_venues else "No venues found — generating from general knowledge."
                except FileNotFoundError:
                    context  = None
                    venue_msg = "Vector store not found — generating without RAG context."

            # Step 4: Generate itinerary
            with st.spinner(f"Generating your {slots.days}-day itinerary with Qwen 3 (1.7b)…"):
                try:
                    if context:
                        result = generate_itinerary(slots, context)
                    else:
                        from pipeline.retriever import RetrievedContext
                        result = generate_itinerary(slots, RetrievedContext())

                    st.session_state["gen_result"] = result
                    st.session_state["phase"]      = "generated"

                    reply = (
                        f"Here's your **{slots.days}-day itinerary** for "
                        f"**{slots.destination}**! ({venue_msg})\n\n"
                        f"{result.itinerary}\n\n"
                        "---\n"
                        "_You can now refine this — e.g. 'Make Day 2 more relaxed' "
                        "or 'Swap the museum for something outdoors'._"
                    )

                except Exception as e:
                    reply = (
                        f"Failed to generate itinerary. Make sure Ollama is running:\n\n"
                        f"```\nollama serve\n```\n\nError: `{e}`"
                    )

            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()

# ---------------------------------------------------------------------------
# Phase: GENERATED / REFINING — multi-turn refinement
# ---------------------------------------------------------------------------

elif st.session_state["phase"] in ("generated", "refining"):

    user_input = st.chat_input("Refine your itinerary… (e.g. 'Make Day 3 more relaxed')")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Refining your itinerary…"):
                try:
                    refined = refine_itinerary(user_input, st.session_state["gen_result"])
                    st.session_state["gen_result"] = refined
                    st.session_state["phase"]      = "refining"

                    reply = refined.itinerary

                except Exception as e:
                    reply = f"Refinement failed. Error: `{e}`"

            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()
