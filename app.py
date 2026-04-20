"""
app.py — NLPilot Streamlit frontend.
Light theme inspired by rtrvr.ai — white, orange accents, clean and spacious.

CSS strategy: ONE single <style> tag built dynamically per render.
Prevents stale CSS from previous renders persisting in the DOM.
"""

import re
import streamlit as st

from pipeline.slot_filler import fill_slots, TripSlots
from pipeline.mood_mapper import map_moods
from pipeline.retriever   import retrieve, RetrievedContext
from pipeline.generator   import (
    generate_itinerary_stream,
    refine_itinerary_stream,
    GeneratorResult,
    _strip_thinking_tags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_dollars(text: str) -> str:
    text = re.sub(r'\$(\d)', r'\\$\1', text)
    text = re.sub(r'(?<!\\)\$(?!\d)', r'\\$', text)
    return text


def _format_display(text: str) -> str:
    text = re.sub(r'\s*\[V\d+\]', '', text)
    text = re.sub(r'\s*\[Venue\s*\d+\]', '', text)
    text = re.sub(r'\s*\(Venue\s*\d+\)', '', text)
    text = _escape_dollars(text)
    return text


def _render_user_msg(text: str):
    st.markdown(
        f'<div style="background:#FFF7ED; border:1px solid #FED7AA; border-radius:12px; '
        f'padding:16px 20px; margin-bottom:12px; font-family:\'DM Sans\',sans-serif; '
        f'font-size:14px; line-height:1.7; color:#4A3728;">{text}</div>',
        unsafe_allow_html=True,
    )


def _render_retrieval_trace(context: RetrievedContext = None):
    n_v = len(context.venues) if context and context.venues else 0
    n_t = len(context.tips) if context and context.tips else 0
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:12px 0 4px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{"#22C55E" if n_v > 0 else "#D1D5DB"};display:inline-block;"></span>'
        f'<span style="color:#4A3728;font-weight:600;font-size:14px;text-transform:uppercase;letter-spacing:0.08em;">RAG Context</span>'
        f'</div>'
        f'<p style="color:#B8977A;font-size:13px;margin:0 0 8px;">{n_v} venues &middot; {n_t} tips retrieved</p>',
        unsafe_allow_html=True,
    )
    with st.sidebar.expander(f"Retrieved Venues ({n_v})", expanded=False):
        if n_v == 0:
            st.markdown('<div style="color:#FFFFFF;font-size:12px;padding:8px 0;">No venues retrieved yet. Enter a trip to see results.</div>', unsafe_allow_html=True)
        else:
            for i, v in enumerate(context.venues, 1):
                meta = v.get("metadata", {})
                name = meta.get("name", "Unknown")
                cats = meta.get("categories", "")
                bid = meta.get("business_id", "")
                stars = meta.get("stars", "?")
                cat_list = [c.strip() for c in cats.split(",") if c.strip()][:2]
                cat_str = " &middot; ".join(cat_list)
                tip = ""
                for t in context.tips:
                    if t.get("metadata", {}).get("business_id") == bid:
                        tip = t.get("document", "")[:100]
                        break
                tip_html = f'<div style="color:#FDE8D0;font-size:11px;font-style:italic;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.2);">{tip}</div>' if tip else ""
                st.markdown(
                    f'<div style="background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:9px 11px;margin-bottom:6px;">'
                    f'<div style="color:#FFFFFF;font-weight:600;font-size:12px;">{name}</div>'
                    f'<div style="color:#FDE8D0;font-size:11px;margin-top:1px;">{cat_str} &middot; {stars}&#9733;</div>'
                    f'{tip_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


LOADING_ANIMATION = """
<div class="travel-loader-mini">
    <span class="lm-icon lm1">&#9992;&#65039;</span>
    <span class="lm-icon lm2">&#x1F68C;</span>
    <span class="lm-icon lm3">&#x1F6A2;</span>
    <span class="lm-text">Planning your trip...</span>
</div>
<style>
.travel-loader-mini{display:flex;align-items:center;gap:8px;padding:8px 0 4px}
.lm-icon{font-size:16px;animation:lmBounce 1.4s ease-in-out infinite}
.lm1{animation-delay:0s} .lm2{animation-delay:0.2s} .lm3{animation-delay:0.4s}
@keyframes lmBounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.lm-text{color:#B8977A;font-size:13px;font-weight:500;margin-left:4px}
</style>
"""


st.set_page_config(page_title="NLPilot", page_icon="\u2708\uFE0F", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {"messages":[],"slots":None,"categories":[],"gen_result":None,"context":None,"phase":"input"}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
_init_state()

_is_landing = st.session_state["phase"] == "input" and not st.session_state["messages"]


# ---------------------------------------------------------------------------
# CSS — ONE single unified block, built dynamically per state
# ---------------------------------------------------------------------------

# ── Shared base styles (always present) ──
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,500;1,400;1,500&display=swap');
::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:#D1D5DB;border-radius:4px}
#MainMenu,footer,header{visibility:hidden} [data-testid="stToolbar"]{display:none}

/* Sidebar base styling */
section[data-testid="stSidebar"]{background:#FEF7F0!important;border-right:1px solid #F5E6D8!important;width:260px!important;min-width:260px!important}
section[data-testid="stSidebar"]>div{background:#FEF7F0!important}
section[data-testid="stSidebar"]>div>div{background:#FEF7F0!important}
section[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]{display:none!important}
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] .stMarkdown li{color:#B8977A!important;font-family:'DM Sans',sans-serif!important;font-size:13.5px;line-height:1.6}
section[data-testid="stSidebar"] h3{font-family:'DM Sans',sans-serif!important;color:#C4A882!important;font-size:11px!important;font-weight:500!important;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px!important}
section[data-testid="stSidebar"] strong{color:#4A3728!important;font-weight:600}
section[data-testid="stSidebar"] hr{border-color:#F0DFD0!important;margin:14px 0!important}
section[data-testid="stSidebar"] [data-testid="stAlert"]{background:#FFF0E0!important;border:1px solid #FED7AA!important;color:#C2410C!important;border-radius:8px!important;padding:8px 12px!important;font-size:12.5px}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#C4A882!important;font-size:11px!important}
section[data-testid="stSidebar"] .stButton>button{background:#F97316!important;color:#FFF!important;border:none!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:700!important;font-size:13.5px!important;height:42px!important;transition:all .2s ease}
section[data-testid="stSidebar"] .stButton>button:hover{background:#EA580C!important}
section[data-testid="stSidebar"] [data-testid="stExpander"]{background:#F97316!important;border:1px solid #EA580C!important;border-radius:8px!important}
section[data-testid="stSidebar"] .streamlit-expanderHeader{color:#FFFFFF!important;font-weight:600!important;font-size:13px!important;background:#F97316!important}
section[data-testid="stSidebar"] .streamlit-expanderHeader span{color:#FFFFFF!important}
section[data-testid="stSidebar"] .streamlit-expanderHeader svg{fill:#FFFFFF!important}
section[data-testid="stSidebar"] .streamlit-expanderContent{background:#FFFFFF!important;border-radius:0 0 8px 8px!important}

/* Chat message cards */
[data-testid="stChatMessage"]{background:#1C1C21!important;border-top:1px solid #2A2A35!important;border-right:1px solid #2A2A35!important;border-bottom:1px solid #2A2A35!important;border-left:4px solid #F97316!important;border-radius:0 12px 12px 0!important;padding:20px 24px!important;margin-bottom:12px!important;box-shadow:none}
[data-testid="stChatMessage"] p{color:#E4E4E7!important;line-height:1.8!important;font-size:16px!important}
[data-testid="stChatMessage"] li{color:#E4E4E7!important;margin-bottom:8px;font-size:16px;line-height:1.8}
[data-testid="stChatMessage"] h2{font-family:'DM Sans',sans-serif!important;color:#F97316!important;font-size:26px!important;font-weight:700!important;letter-spacing:-0.02em;margin-top:28px!important;margin-bottom:16px!important;padding-bottom:10px;border-bottom:2px solid #F97316}
[data-testid="stChatMessage"] h2:first-of-type{margin-top:8px!important}
[data-testid="stChatMessage"] h1{font-family:'DM Sans',sans-serif!important;color:#E4E4E7!important;font-size:22px!important;font-weight:700!important;margin-top:28px!important}
[data-testid="stChatMessage"] strong{color:#F97316!important;font-weight:600;font-size:16px!important}
[data-testid="stChatMessage"] em{color:#A1A1AA!important;font-size:15px!important}
[data-testid="stChatMessage"] hr{border-color:#2A2A35!important;margin:24px 0!important}
[data-testid="stChatMessage"] code{background:#2A2A35!important;color:#F97316!important;padding:2px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace!important;font-size:14px}
[data-testid="stChatMessage"] pre{background:#141419!important;border:1px solid #2A2A35!important;border-radius:8px!important}
[data-testid="stChatMessage"] a{color:#F97316!important;pointer-events:none!important;text-decoration:none!important;cursor:default!important}
[data-testid="stChatMessage"] a:hover{text-decoration:none!important}

/* Alerts, status, expanders */
[data-testid="stAlert"]{background:#FFFFFF!important;border:1px solid #E5E7EB!important;border-radius:10px!important;color:#111827!important}
[data-testid="stAlert"] p{color:#6B7280!important}
[data-testid="stStatusWidget"]{background:#FFFFFF!important;border:1px solid #E5E7EB!important;border-radius:8px!important}
.main [data-testid="stExpander"]{background:#FFFFFF!important;border:1px solid #E5E7EB!important;border-radius:8px!important}
.main .streamlit-expanderHeader{color:#6B7280!important;font-weight:500!important;font-size:13.5px!important}

/* Chat input base */
[data-testid="stChatInput"]{background:transparent!important;position:relative!important}
[data-testid="stChatInput"]>div>div,[data-testid="stChatInput"]>div>div>div,[data-testid="stChatInput"]>div>div>div>div,[data-testid="stChatInput"]>div>div>div>div>div{background:transparent!important;border:none!important;box-shadow:none!important;outline:none!important}
[data-testid="stChatInput"] [data-baseweb]{background:transparent!important;border:none!important;display:flex!important;align-items:flex-end!important}
[data-testid="stChatInput"] [data-baseweb] div{background:transparent!important;border:none!important}
[data-testid="stChatInput"] textarea{font-family:'DM Sans',sans-serif!important;font-size:15px!important;border:none!important;background:transparent!important;color:#111827!important;caret-color:#111827!important;border-radius:0!important;box-shadow:none!important;outline:none!important}
[data-testid="stChatInput"] textarea:focus{border:none!important;box-shadow:none!important;outline:none!important;background:transparent!important}
[data-testid="stChatInput"] textarea::placeholder{color:#B8977A!important;font-size:15px!important}
[data-testid="stChatInput"] button{background:#F97316!important;color:#FFFFFF!important;font-weight:600!important;border:none!important;white-space:nowrap!important;flex-shrink:0!important;align-self:flex-end!important}
[data-testid="stChatInput"] button svg{display:none!important}
[data-testid="stChatInput"] button:hover{background:#EA580C!important}

/* Avatars */
[data-testid="chatAvatarIcon-user"]{background:#FFF7ED!important;color:#F97316!important}
[data-testid="chatAvatarIcon-assistant"]{background:#2A2A35!important;color:#F97316!important}
a{color:#F97316!important;pointer-events:none!important;text-decoration:none!important;cursor:default!important} a:hover{text-decoration:none!important}
"""

# ── Landing-only overrides ──
_LANDING_CSS = """
@keyframes globeSpin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
section[data-testid="stSidebar"]{width:0!important;min-width:0!important;visibility:hidden!important;overflow:hidden!important}
section[data-testid="stSidebar"]>div{width:0!important;overflow:hidden!important}
html,body,.stApp,.main{background:#FEF7F0!important;color:#111827!important;font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif!important;overflow:hidden!important}
.stApp{max-height:100vh!important;max-height:100dvh!important}
.main .block-container{max-width:720px!important;padding-top:0!important;padding-bottom:0!important}
[data-testid="stChatInput"]>div{border:2px solid #F97316!important;border-radius:28px!important;background:#FFFFFF!important;padding:6px 8px!important;overflow:visible!important;position:relative!important;display:flex!important;align-items:flex-end!important}
[data-testid="stChatInput"]>div::before{content:"\\1F30E"!important;display:block!important;position:absolute;left:18px;bottom:14px;font-size:17px;z-index:10;animation:globeSpin 4s linear infinite;line-height:1}
[data-testid="stChatInput"] textarea{padding:8px 8px 8px 38px!important}
[data-testid="stChatInput"] button{min-width:160px!important;height:42px!important;border-radius:24px!important;padding:0 22px!important;font-size:0!important;margin:0 2px 2px 0!important}
[data-testid="stChatInput"] button::after{content:"\\2708\\FE0F  Let's travel  \\2192"!important;font-size:14px;font-family:'DM Sans',sans-serif;font-weight:600;color:#FFFFFF;white-space:nowrap}
[data-testid="stBottom"]{background:#FEF7F0!important;padding-top:4px!important;padding-bottom:8px!important}
[data-testid="stBottom"]>div{background:#FEF7F0!important}
[data-testid="stBottom"] *{background:#FEF7F0!important}
[data-testid="stBottom"] [data-testid="stChatInput"]{background:transparent!important}
[data-testid="stBottom"] [data-testid="stChatInput"]>div{background:#FFFFFF!important}
[data-testid="stBottomBlockContainer"]{background:#FEF7F0!important;max-width:700px!important;margin:0 auto!important;padding-left:0!important;padding-right:0!important}
"""

# ── Chat-mode overrides ──
_CHAT_CSS = """
section[data-testid="stSidebar"]{width:260px!important;min-width:260px!important;visibility:visible!important;overflow-y:auto!important;overflow-x:hidden!important}
section[data-testid="stSidebar"]>div{width:260px!important;overflow-y:auto!important;overflow-x:hidden!important}
html,body,.stApp,.main{background:#FAFAFA!important;color:#111827!important;font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif!important;overflow:auto!important}
.stApp{max-height:none!important}
.main .block-container{max-width:880px!important;padding-top:20px!important;padding-bottom:0!important}
[data-testid="stChatInput"]>div{border:1px solid #E5E7EB!important;border-radius:12px!important;background:#FFFFFF!important;padding:4px!important;overflow:visible!important;position:relative!important;display:flex!important;align-items:flex-end!important}
[data-testid="stChatInput"]>div::before{display:none}
[data-testid="stChatInput"] textarea{padding:8px 8px 8px 12px!important}
[data-testid="stChatInput"] button{min-width:40px!important;height:36px!important;border-radius:8px!important;padding:0 10px!important;font-size:0!important;position:relative;margin:0 2px 2px 0!important}
[data-testid="stChatInput"] button::after{content:"\\2192";font-size:16px;font-weight:600;color:#FFFFFF}
[data-testid="stBottom"]{background:#FAFAFA!important}
[data-testid="stBottom"]>div{background:#FAFAFA!important}
[data-testid="stBottomBlockContainer"]{background:#FAFAFA!important;max-width:none!important;margin:auto!important}
"""

# Build the single CSS string for this render
_mode_css = _LANDING_CSS if _is_landing else _CHAT_CSS
st.markdown(f"<style>{_BASE_CSS}\n{_mode_css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_slots_card(slots: TripSlots):
    fields = [
        ("Destination",slots.destination,"accent"),
        ("Origin",slots.origin,""),
        ("Duration",f"{slots.days} days" if slots.days else None,""),
        ("Budget",f"${slots.budget:,.0f}" if slots.budget else None,"green"),
        ("Transport",(slots.transport or "").title() or None,""),
        ("Vibes",", ".join(slots.moods) if slots.moods else None,""),
    ]
    rows=""
    for i,(label,value,cls) in enumerate(fields):
        border_top = 'border-top:1px solid #FDF2EA;' if i > 0 else ''
        if value:
            if cls == "accent":
                val_style = 'display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;background:#FFF0E0;color:#C2410C;'
            elif cls == "green":
                val_style = 'display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;background:#ECFDF5;color:#166534;'
            else:
                val_style = 'font-weight:500;color:#4A3728;'
            rows += (f'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:12.5px;{border_top}">'
                     f'<span style="color:#B8977A;font-weight:400;">{label}</span>'
                     f'<span style="{val_style}">{value}</span></div>')
        else:
            rows += (f'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:12.5px;{border_top}">'
                     f'<span style="color:#B8977A;font-weight:400;">{label}</span>'
                     f'<span style="color:#D4C4B0;font-style:italic;">&mdash;</span></div>')
    st.sidebar.markdown(
        f'<div style="background:#FFFFFF;border-top:1px solid #F0DFD0;border-right:1px solid #F0DFD0;'
        f'border-bottom:1px solid #F0DFD0;border-left:3px solid #F97316;border-radius:0 8px 8px 0;'
        f'padding:12px 14px;margin:8px 0;">{rows}</div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(
        '<div style="text-align:left;padding:8px 0 16px;border-bottom:1px solid #F0DFD0;margin-bottom:2px;">'
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:32px;font-weight:700;letter-spacing:-0.03em;">'
        '<span style="color:#F97316;">NL</span><span style="color:#2D1F14;">Pilot</span></div>'
        '<div style="width:38px;height:3.5px;background:#F97316;border-radius:2px;margin-top:5px;"></div>'
        '<div style="font-size:12px;color:#B8977A;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-top:7px;">AI Travel Planner</div>'
        '</div>',unsafe_allow_html=True)
    st.divider()
    if st.session_state["slots"]:
        st.markdown("### Trip Details")
        _render_slots_card(st.session_state["slots"])
        st.divider()
    _render_retrieval_trace(st.session_state.get("context"))
    st.divider()
    if st.session_state["phase"]!="input":
        if st.button("\u2726 New trip",use_container_width=True):
            for k in ["messages","slots","categories","gen_result","context","phase"]:
                st.session_state[k]=[] if k in ("messages","categories") else None
            st.session_state["phase"]="input"
            st.rerun()
        st.divider()
    st.markdown(
        '<div style="text-align:center;padding:12px 0 4px;">'
        '<div style="color:#C4A882;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Powered by</div>'
        '<div style="color:#F97316;font-size:12px;margin-top:3px;font-weight:600;">Ollama &middot; ChromaDB</div>'
        '</div>',unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

for msg in st.session_state["messages"]:
    if msg["role"]=="user":
        _render_user_msg(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(_format_display(msg["content"]))


# ---------------------------------------------------------------------------
# Phase: INPUT
# ---------------------------------------------------------------------------
if st.session_state["phase"]=="input":
    user_input=st.chat_input("Where do you want to go?")

    if not st.session_state["messages"] and not user_input:
        st.markdown(
            '<div style="text-align:center; padding:0;">'
            '<div style="font-family:\'DM Sans\',sans-serif; font-size:64px; font-weight:700; letter-spacing:-0.04em; line-height:1.1;">'
            '<span style="color:#F97316;">NL</span><span style="color:#2D1F14;">Pilot</span>'
            '</div>'
            '<div style="width:56px; height:3px; background:#F97316; border-radius:2px; margin:8px auto 0;"></div>'
            '<p style="color:#4A3728; font-family:\'DM Sans\',sans-serif; font-size:22px; margin:14px 0 4px; line-height:1.4; font-weight:500;">'
            'Describe your trip in plain English.</p>'
            '<p style="color:#7C6F5E; font-family:\'DM Sans\',sans-serif; font-size:18px; margin:0 0 14px; line-height:1.5;">'
            'Real venues. Real reviews. <span style="color:#F97316; font-weight:600;">Retrieved</span> from Yelp and powered by RAG.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:#1C1C21; border-top:1px solid #2A2A35; border-right:1px solid #2A2A35; border-bottom:1px solid #2A2A35; border-left:4px solid #F97316; border-radius:0 14px 14px 0; padding:18px 24px; margin:0 auto; max-width:700px;">'
            '<p style="color:#F97316; font-family:\'DM Sans\',sans-serif; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 14px;">Try something like:</p>'
            '<p style="font-family:\'Lora\',Georgia,serif; color:#C8C8CE; font-size:15px; line-height:1.7; margin:0 0 12px; padding-bottom:12px; border-bottom:1px solid #F9731640;">'
            '<span style="color:#F97316; margin-right:8px;">&#9679;</span>'
            '"Plan a 2 day trip from San Francisco to San Jose for me with a budget of $500. '
            'I prefer public transport and I am in the mood for art, local food and hiking."</p>'
            '<p style="font-family:\'Lora\',Georgia,serif; color:#C8C8CE; font-size:15px; line-height:1.7; margin:0 0 12px; padding-bottom:12px; border-bottom:1px solid #F9731640;">'
            '<span style="color:#F97316; margin-right:8px;">&#9679;</span>'
            '"I\'m flying from San Francisco to Philadelphia for 3 days with a $1000 budget. '
            'I prefer public transport and I\'m in the mood for music, sightseeing and fine dining."</p>'
            '<p style="font-family:\'Lora\',Georgia,serif; color:#C8C8CE; font-size:15px; line-height:1.7; margin:0;">'
            '<span style="color:#F97316; margin-right:8px;">&#9679;</span>'
            '"5-day trip to New Orleans from Chicago, $3,000 budget. '
            'Live music, nightlife, Southern food, and history."</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    if user_input:
        st.session_state["messages"].append({"role":"user","content":user_input})
        _render_user_msg(user_input)
        with st.chat_message("assistant"):
            loader_placeholder=st.empty()
            loader_placeholder.markdown(LOADING_ANIMATION,unsafe_allow_html=True)
            slots=fill_slots(user_input)
            st.session_state["slots"]=slots
            if not slots.is_complete():
                missing=slots.missing()
                reply="I need a bit more info to plan your trip:\n\n"+"\n".join(f"- **{m}**" for m in missing)
                loader_placeholder.empty()
                st.markdown(reply)
                st.session_state["messages"].append({"role":"assistant","content":reply})
                st.rerun()
            categories=map_moods(slots.moods,top_k=3)
            st.session_state["categories"]=categories
            try:
                context=retrieve(slots,categories)
                st.session_state["context"]=context
            except FileNotFoundError:
                context=RetrievedContext()
                st.session_state["context"]=context
            loader_placeholder.empty()
            full_text=""
            result=None
            placeholder=st.empty()
            try:
                for chunk in generate_itinerary_stream(slots,context):
                    if isinstance(chunk,GeneratorResult):
                        result=chunk
                    else:
                        full_text+=chunk
                        display_text=_format_display(_strip_thinking_tags(full_text))
                        placeholder.markdown(display_text+" \u258C")
                if result:
                    placeholder.markdown(_format_display(result.itinerary))
                    st.session_state["gen_result"]=result
                    st.session_state["phase"]="generated"
                    reply=result.itinerary
                else:
                    display_text=_strip_thinking_tags(full_text)
                    placeholder.markdown(_format_display(display_text))
                    reply=display_text
            except Exception as e:
                reply=f"**Generation failed.** Make sure Ollama is running:\n\n```\nollama serve\nollama pull qwen3:4b\n```\n\nError: `{e}`"
                placeholder.markdown(reply)
            st.session_state["messages"].append({"role":"assistant","content":reply})
            st.rerun()


# ---------------------------------------------------------------------------
# Phase: GENERATED / REFINING
# ---------------------------------------------------------------------------
elif st.session_state["phase"] in ("generated","refining"):
    user_input=st.chat_input("Refine your itinerary...")
    if user_input:
        st.session_state["messages"].append({"role":"user","content":user_input})
        _render_user_msg(user_input)
        with st.chat_message("assistant"):
            full_text=""
            result=None
            placeholder=st.empty()
            try:
                for chunk in refine_itinerary_stream(user_input,st.session_state["gen_result"]):
                    if isinstance(chunk,GeneratorResult):
                        result=chunk
                    else:
                        full_text+=chunk
                        display_text=_format_display(_strip_thinking_tags(full_text))
                        placeholder.markdown(display_text+" \u258C")
                if result:
                    placeholder.markdown(_format_display(result.itinerary))
                    st.session_state["gen_result"]=result
                    st.session_state["phase"]="refining"
                    reply=result.itinerary
                else:
                    display_text=_strip_thinking_tags(full_text)
                    placeholder.markdown(_format_display(display_text))
                    reply=display_text
            except Exception as e:
                reply=f"Refinement failed. Error: `{e}`"
                placeholder.markdown(reply)
            st.session_state["messages"].append({"role":"assistant","content":reply})
            st.rerun()