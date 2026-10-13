"""
ahead/pages/assistant.py
========================
"Ask AHEAD" — the conversational assistant page: hero with mascot, trait
chips, suggested prompts and a chat transcript backed by Gemini.
"""

from datetime import datetime
from html import escape
from urllib.parse import quote

import streamlit as st

from ahead.components import go_to
from ahead.config import ICONS_DIR
from ahead.resources import chat_reply, gemini_available, gemini_error_message, icon_uri, static_url
from ahead.theme import is_dark, tokens

WELCOME = "Hi! I'm AHEAD Assistant. What would you like to know about AHEAD or its screening conditions?"

# (material icon, prompt)
SUGGESTIONS = [
    (":material/lightbulb:", "What is AHEAD?"),
    (":material/cardiology:", "Tell me about the screening conditions"),
    (":material/explore:", "How do I use the platform?"),
]

SYSTEM_PROMPT = (
    "You are AHEAD Assistant inside a university healthcare screening prototype called AHEAD "
    "(Advanced Health Early Awareness and Disease Detection). AHEAD screens for diabetes, cardiovascular "
    "disease and chronic kidney disease using machine-learning models trained on public datasets, and shows "
    "the result as an educational screening estimate, never a diagnosis. Answer questions about AHEAD, the "
    "three conditions, common screening measurements (HbA1c, blood glucose, BMI, blood pressure, eGFR, ACR, "
    "creatinine, BUN, hemoglobin), and how to use the website (Overview, Screenings, Data & Analytics, "
    "Settings, Clinical Dashboard for doctors). Give general educational information only. Do not diagnose "
    "a person, do not claim a screening result confirms disease, and do not recommend starting, stopping or "
    "changing prescription medication. For urgent symptoms advise appropriate urgent medical care. "
    "Keep answers concise, warm and patient-friendly; use short paragraphs or bullet points."
)

# (label, svg paths) — rendered as <img> data URIs because st.html strips inline <svg>
TRAITS = [
    ("Evidence-based",
     '<path d="M2 4h6a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h7z"/>'),
    ("Supportive",
     '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l1 1L12 21l7.8-7.6 1-1a5.5 5.5 0 0 0 0-7.8z"/>'),
    ("Practical guidance",
     '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>'),
]


def _trait_icon(paths: str) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{tokens()["accent"]}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )
    return f'<img src="data:image/svg+xml;utf8,{quote(svg)}" alt="">'


# Offline answers for the suggested prompts so the page is useful without an API key.
LOCAL_ANSWERS = {
    "what is ahead": (
        "**AHEAD** (Advanced Health Early Awareness and Disease Detection) is a Senior Design prototype that "
        "screens for **diabetes**, **cardiovascular disease** and **chronic kidney disease** using machine-learning "
        "models trained on public datasets.\n\nYou enter lifestyle, medical-history and clinical values; AHEAD "
        "returns an *educational* screening estimate, practical next steps and the factors it noticed — never a diagnosis."
    ),
    "screening conditions": (
        "AHEAD currently screens three conditions:\n\n"
        "- **Diabetes** — HbA1c, blood glucose, BMI, age, smoking and history of hypertension / heart disease.\n"
        "- **Cardiovascular disease** — general health, activity, sleep, smoking, alcohol, BMI and prior conditions.\n"
        "- **Chronic kidney disease** — blood pressure, fasting glucose, HbA1c, creatinine, BUN, eGFR, urine protein / ACR, "
        "hemoglobin, family history and lifestyle.\n\nEach condition has its own recommended model chosen by benchmark."
    ),
    "how do i use": (
        "1. Open **Screenings** and pick a condition.\n"
        "2. Fill in the three sections of the form and press **Generate Screening Result**.\n"
        "3. Read the gauge, the recommended next steps and the factors AHEAD identified.\n"
        "4. Explore **Data & Analytics** for datasets and model performance, or adjust your **Settings**.\n\n"
        "Doctor accounts also get a **Clinical Dashboard** for uploading patient files."
    ),
}


def _local_answer(prompt: str) -> str | None:
    text = prompt.lower()
    for key, answer in LOCAL_ANSWERS.items():
        if key in text:
            return answer
    return None


def _reply(history: list) -> str:
    """Send the transcript as proper user/model turns with a system instruction."""
    turns = [{"role": "user" if m["role"] == "user" else "model", "text": m["content"]} for m in history[-12:]]
    return chat_reply(SYSTEM_PROMPT, turns)


def _append(role: str, content: str) -> None:
    st.session_state.chat_messages.append({"role": role, "content": content, "time": datetime.now()})


def render_assistant() -> None:
    robot = str(ICONS_DIR / "chatbot.svg")

    # page background (soft waves) in light mode; dark mode keeps the plain dark canvas
    if not is_dark():
        st.html(
            f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: radial-gradient(ellipse at 85% 10%, #EDE9FE, transparent 55%), #F3F7FF !important;
}}
[data-testid="stMain"] {{ background: transparent !important; }}
[data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
</style>
"""
        )

    if st.button("← Back", key="assistant_back", type="tertiary"):
        # return to where the user came from, including a screening that was open
        go_to(st.session_state.get("assistant_return", "overview"), keep_screening=True)

    traits = "".join(f'<div class="assistant-trait">{_trait_icon(paths)}{escape(label)}</div>' for label, paths in TRAITS)
    st.html(
        f"""
<div class="ahead-card assistant-shell" style="padding: 30px 34px 22px;">
    <div class="assistant-hero">
        <div>
            <div class="eyebrow">AHEAD Assistant</div>
            <h1>Ask <span>AHEAD</span></h1>
            <p>Ask general questions about AHEAD, its three screening conditions, health measurements,
            or how to use the platform.</p>
        </div>
        <div class="assistant-mascot"><img src="{icon_uri('chatbot.svg')}" alt="AHEAD Assistant"></div>
    </div>
    <div class="assistant-traits">{traits}</div>
    <div class="assistant-try">Try asking…</div>
</div>
"""
    )

    if not st.session_state.chat_messages:
        _append("assistant", WELCOME)

    # suggestion chips, side by side (wrap on narrow screens)
    with st.container(key="assistant_suggestions", horizontal=True, gap="small"):
        for index, (icon, text) in enumerate(SUGGESTIONS):
            if st.button(f"{text}  →", key=f"suggest_{index}", icon=icon):
                st.session_state.pending_prompt = text
                st.rerun()

    if not gemini_available():
        st.info(
            "AHEAD Assistant needs a Gemini API key to answer. Set `GEMINI_API_KEY` in the environment or "
            "add it to `.streamlit/secrets.toml`, then restart the app."
        )

    # transcript
    for message in st.session_state.chat_messages:
        is_user = message["role"] == "user"
        with st.chat_message("user" if is_user else "assistant", avatar=None if is_user else robot):
            st.markdown(message["content"])
            st.html(f'<div class="assistant-time">{message["time"].strftime("%I:%M %p").lstrip("0")}</div>')

    # input (typed or from a suggestion chip)
    typed = st.chat_input("Ask AHEAD a question…")
    pending = st.session_state.pop("pending_prompt", None)   # always consume a queued suggestion
    prompt = typed or pending
    if prompt:
        _append("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=robot):
            with st.spinner("Thinking…"):
                answer = None
                if gemini_available():
                    try:
                        answer = _reply(st.session_state.chat_messages)
                    except Exception as exc:
                        answer = gemini_error_message(exc)
                if not answer:
                    answer = _local_answer(prompt) or (
                        "AHEAD Assistant is not available right now (no Gemini API key or the service could not be "
                        "reached). Try one of the suggested questions above — the screening tools work normally without me."
                    )
            st.markdown(answer)
        _append("assistant", answer)
        st.rerun()

    if len(st.session_state.chat_messages) > 1:
        if st.button("Clear conversation", key="assistant_clear", type="tertiary"):
            st.session_state.chat_messages = []
            st.rerun()
