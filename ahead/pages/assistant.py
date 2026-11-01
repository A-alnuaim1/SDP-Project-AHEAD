"""
ahead/pages/assistant.py
========================
"Ask AHEAD" — the conversational assistant page: hero with mascot, trait
chips, suggested prompts and a chat transcript backed by Gemini.
"""

import json
import re
from datetime import datetime
from html import escape
from urllib.parse import quote

import streamlit as st

from ahead.components import current_user, go_to, pref
from ahead.config import ICONS_DIR, NAV_ITEMS
from ahead.resources import chat_reply, gemini_available, gemini_error_message, icon_uri, static_url, transcribe_audio
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
    "creatinine, BUN, hemoglobin), and how to use the website (Overview, Screenings, Health & Lifestyle Tracker, "
    "Food Nutrition Analyzer, Medication Monitoring, Notifications, Data & Analytics, Settings, and for doctors the "
    "Clinical Dashboard and Patient Intelligence). Users may speak their questions; their words arrive already "
    "transcribed, and your answer may be read aloud, so prefer plain sentences over tables. "
    "Give general educational information only. Do not diagnose "
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


# ----------------------------------------------------------------------------- app intents
# Answered on-device (no Gemini call): page navigation and the patient's own status.
PAGE_ALIASES = [
    ("screenings", ("screening", "screenings", "assessment", "assessments", "test", "tests")),
    ("wellness", ("lifestyle", "wellness", "health tracker", "tracker", "check-in", "check in")),
    ("food", ("food", "nutrition", "meal", "meals")),
    ("medications", ("medication", "medications", "medicine", "medicines", "meds", "pills")),
    ("notifications", ("notification", "notifications", "reminders")),
    ("analytics", ("analytics", "data", "datasets", "data and analytics", "data & analytics")),
    ("clinical", ("clinical dashboard", "clinical")),
    ("intelligence", ("intelligence", "patient intelligence", "patients")),
    ("settings", ("settings", "account", "profile")),
    ("overview", ("overview", "home", "dashboard")),
]
NAV_PATTERN = re.compile(r"^(?:please\s+|can you\s+|could you\s+)?(?:open|go to|go|show me|show|take me to|navigate to)\s+(?:the\s+|my\s+)?(.+?)(?:\s+page)?[.!?]*$")
PATIENT_ONLY = {"wellness", "food", "medications", "notifications"}


def _allowed_pages(is_doctor: bool) -> set:
    return {key for key, _label, _icon, doctor_only in NAV_ITEMS
            if (not doctor_only or is_doctor) and not (is_doctor and key in PATIENT_ONLY)}


PAGE_SUFFIX = r"(?:\s+(?:page|screen|section|tab|analy[sz]er|tracker|monitoring|dashboard|center|centre|history))*"


def _navigation_target(text: str, is_doctor: bool) -> str | None:
    """Page key for requests like "open medication" / "take me to the food analyzer", else None.

    The whole requested name must be one page's alias (plus words like "page" or "tracker"), the most
    specific alias wins, and a page this user cannot open is never chosen.
    """
    match = NAV_PATTERN.match(text.lower().strip())
    if not match:
        return None
    wanted = re.sub(r"^(?:a|an|the|my)\s+", "", match.group(1).strip())
    best = None
    for key, aliases in PAGE_ALIASES:
        for alias in aliases:
            if re.fullmatch(re.escape(alias) + PAGE_SUFFIX, wanted) and (best is None or len(alias) > len(best[1])):
                best = (key, alias)
    if best and best[0] in _allowed_pages(is_doctor):
        return best[0]
    return None


def _status_answer(text: str, user_id: int) -> str | None:
    """The patient's own lifestyle score or today's doses, read from their account."""
    from ahead.platform_data import lifestyle_history, local_today, scheduled_doses
    lowered = text.lower()
    if not re.search(r"\b(my|i|me)\b", lowered):
        return None
    if re.search(r"\b(lifestyle|wellness)\b", lowered) or re.search(r"\bmy (latest |last )?score\b", lowered):
        latest = lifestyle_history(user_id)
        if not latest:
            return "You haven't saved a lifestyle check-in yet. Open the **Health & Lifestyle Tracker** to record your first one."
        return (f"Your latest lifestyle score is **{latest[0]['scores']['overall']} out of 100**. "
                "Open the Health & Lifestyle Tracker to see what makes it up.")
    if re.search(r"\b(medication|medications|medicine|medicines|meds|doses?|pills?)\b", lowered):
        doses = scheduled_doses(user_id, local_today(), local_today())
        if not doses:
            return "You have no medication doses scheduled today."
        lines = [f"- {d['time']} · {d['name']} {d['dosage']} — {d['status']}" for d in sorted(doses, key=lambda d: d["time"])]
        return "Here are today's doses:\n\n" + "\n".join(lines) + "\n\nYou can record them in **Medication Monitoring**."
    return None


# ----------------------------------------------------------------------------- read aloud
def _speakable(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)        # links -> their text
    text = re.sub(r"[*_`#>]+", "", text)                                 # emphasis, code, headings, quotes
    text = re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", text, flags=re.M)      # list markers
    return re.sub(r"\s+", " ", text).strip()


def _read_aloud(text: str) -> None:
    """Speak through the page's own speechSynthesis (free, on-device) with a Stop control.

    The iframe shares the app's origin, so it can drive window.parent.speechSynthesis, which already
    has the user's click/recording as its activation.
    """
    payload = json.dumps(_speakable(text)).replace("</", "<\\/")
    t = tokens()
    st.iframe(f"""<!doctype html><html><body style="margin:0;font:13px Inter,system-ui,sans-serif;background:transparent">
<div id="bar" style="display:flex;align-items:center;gap:10px;padding:6px 12px;border-radius:12px;
  background:{t['accent_soft']};color:{t['accent']};width:max-content">
  <span id="msg">🔊 Reading the answer aloud…</span>
  <button id="stop" style="border:0;border-radius:8px;padding:4px 10px;background:{t['accent']};color:white;cursor:pointer">Stop</button>
</div>
<script>
const text = {payload};
const host = window.parent && window.parent.speechSynthesis ? window.parent : window;
const synth = host.speechSynthesis;
const msg = document.getElementById('msg'), stop = document.getElementById('stop');
if (!synth || !host.SpeechSynthesisUtterance) {{
  msg.textContent = 'Reading aloud is not supported in this browser.'; stop.remove();
}} else {{
  synth.cancel();
  const u = new host.SpeechSynthesisUtterance(text);
  u.lang = /[\u0600-\u06FF]/.test(text) ? 'ar-SA' : 'en-US';
  u.onend = () => {{ msg.textContent = '🔊 Finished reading.'; stop.remove(); }};
  stop.onclick = () => {{ synth.cancel(); msg.textContent = 'Stopped.'; stop.remove(); }};
  synth.speak(u);
}}
</script></body></html>""", height=46)


def _reply(history: list) -> str:
    """Send the transcript as proper user/model turns with a system instruction."""
    turns = [{"role": "user" if m["role"] == "user" else "model", "text": m["content"]} for m in history[-12:]]
    return chat_reply(SYSTEM_PROMPT, turns)


def _append(role: str, content: str, voice: bool = False) -> None:
    st.session_state.chat_messages.append({"role": role, "content": content, "time": datetime.now(), "voice": voice})


def _sync_speak_pref() -> None:
    from ahead.pages.settings import _save_prefs
    st.session_state.prefs["speak_replies"] = st.session_state["assistant_speak_replies"]
    _save_prefs()


def render_assistant() -> None:
    robot = str(ICONS_DIR / "chatbot.svg")

    # page background (soft waves) in light mode; dark mode keeps the plain dark canvas
    if not is_dark():
        st.html(
            f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{static_url('chatbot_bg.png')}") center / cover no-repeat fixed !important;
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
            or how to use the platform. Type, or tap the microphone in the message bar and just ask.</p>
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
    for index, message in enumerate(st.session_state.chat_messages):
        is_user = message["role"] == "user"
        with st.chat_message("user" if is_user else "assistant", avatar=None if is_user else robot):
            st.markdown(message["content"])
            stamp = message["time"].strftime("%I:%M %p").lstrip("0")
            badge = '<span class="assistant-voice-badge">🎙 Voice</span>' if message.get("voice") else ""
            st.html(f'<div class="assistant-time">{badge}{stamp}</div>')
            if not is_user and index > 0:
                if st.button("Read aloud", key=f"speak_{index}", icon=":material/volume_up:", type="tertiary"):
                    st.session_state.speak_now = message["content"]
                    st.rerun()

    speak_now = st.session_state.pop("speak_now", None)
    if speak_now:
        _read_aloud(speak_now)

    voice_ready = gemini_available()
    if voice_ready:
        with st.container(key="assistant_voice_controls", horizontal=True, gap="small"):
            st.toggle("Read answers to voice questions aloud", value=pref("speak_replies"),
                      key="assistant_speak_replies", on_change=_sync_speak_pref)

    # input: typed, spoken (mic in the bar) or a suggestion chip
    submitted = st.chat_input("Ask AHEAD — type, or tap the mic to speak…", accept_audio=voice_ready)
    pending = st.session_state.pop("pending_prompt", None)   # always consume a queued suggestion
    prompt, spoken = pending, False
    if submitted is not None:
        prompt = submitted if isinstance(submitted, str) else (submitted.text or "").strip()
        audio = None if isinstance(submitted, str) else submitted.audio
        if audio is not None and not prompt:
            with st.spinner("Listening to your question…"):
                try:
                    prompt = transcribe_audio(audio.getvalue(), audio.type or "audio/wav")
                except Exception as exc:
                    _append("assistant", f"I couldn't process that recording. {gemini_error_message(exc)}")
                    st.rerun()
            if not prompt:
                _append("assistant", "I didn't catch any words in that recording. Please try again, a little closer to the microphone, or type your question.")
                st.rerun()
            spoken = True

    if prompt:
        user = current_user()
        _append("user", prompt, voice=spoken)
        target = _navigation_target(prompt, user["is_doctor"])
        if target:
            label = next(label for key, label, _i, _d in NAV_ITEMS if key == target)
            _append("assistant", f"Opening **{label}**…")
            go_to(target)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=robot):
            with st.spinner("Thinking…"):
                answer = None if user["is_doctor"] else _status_answer(prompt, st.session_state.user["id"])
                if not answer and gemini_available():
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
        if spoken and pref("speak_replies"):
            st.session_state.speak_now = answer
        st.rerun()

    if len(st.session_state.chat_messages) > 1:
        if st.button("Clear conversation", key="assistant_clear", type="tertiary"):
            st.session_state.chat_messages = []
            st.rerun()
