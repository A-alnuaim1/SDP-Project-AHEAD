"""
ahead/components.py
===================
Small reusable UI pieces shared by several pages, plus session-state and
navigation helpers (current user, screening history, go_to).
"""

import re
from datetime import datetime, timezone
from html import escape

import streamlit as st

from ahead.config import DISCLAIMER_TEXT, DISEASES, ROLE_DOCTOR, ROLE_PATIENT, model_label
from ahead.resources import available_models, best_model, decision_threshold, features, icon_uri, load_pipeline, model_reason
from ahead.storage import save_screening
from ahead.triage import track_text


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state() -> None:
    """Create every session key the app relies on, once."""
    defaults = {
        "authenticated": False,
        "user": None,
        "nav": "overview",
        "active_screening": None,
        "assistant_return": "overview",
        "chat_messages": [],
        "pending_prompt": None,
        "appearance_mode": "Light",
        "screening_history": [],
        "last_results": {},
        "prefs": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.prefs is None:
        st.session_state.prefs = dict(DEFAULT_PREFS)


DEFAULT_PREFS = {
    "notify_screening": True,
    "notify_system": True,
    "notify_education": False,
    "keep_history": True,
    "speak_replies": True,
}


def pref(name: str) -> bool:
    return bool((st.session_state.get("prefs") or DEFAULT_PREFS).get(name, DEFAULT_PREFS[name]))


def go_to(nav_key: str, screening: str | None = None, keep_screening: bool = False) -> None:
    """Switch page. The open screening is cleared unless keep_screening (e.g. a detour to the assistant)."""
    st.session_state.nav = nav_key
    if not keep_screening:
        st.session_state.active_screening = screening
    st.rerun()


def open_assistant(from_nav: str) -> None:
    """Jump to the assistant and remember where to come back to (keeping any open screening)."""
    st.session_state.assistant_return = from_nav if from_nav != "assistant" else "overview"
    go_to("assistant", keep_screening=True)


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def current_user() -> dict:
    """
    Normalised view of the signed-in user:
    name, first_name, initials, email, role, role_label, is_doctor, institution.
    """
    user = st.session_state.get("user") or {}
    name = (user.get("name") or "AHEAD User").strip()
    clean = re.sub(r"^(dr|prof|mr|mrs|ms)\.?\s+", "", name, flags=re.IGNORECASE).strip() or name
    parts = clean.split()
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        initials = parts[0][0].upper()
    else:
        initials = "U"
    role = user.get("role") or ROLE_PATIENT
    return {
        "name": name,
        "first_name": parts[0] if parts else "there",
        "initials": initials,
        "email": user.get("email", ""),
        "role": role,
        "role_label": "Doctor" if role == ROLE_DOCTOR else "Patient",
        "is_doctor": role == ROLE_DOCTOR,
        "institution": user.get("institution", ""),
    }


def record_screening(disease: str, model_name: str, probability: float, prediction: int, threshold: float = 0.5, values=None) -> None:
    """Add a completed screening to the session history (if the user allows it).
    The history is newest first, like load_screenings, and times are UTC."""
    if not pref("keep_history"):
        return
    st.session_state.screening_history.insert(
        0,
        {
            "time": datetime.now(timezone.utc),
            "disease": disease,
            "label": DISEASES[disease]["label"],
            "model": model_name,
            "probability": probability,
            "prediction": prediction,
            "threshold": threshold,
        }
    )
    owner = (st.session_state.get("user") or {}).get("id")
    if owner:
        save_screening(owner, disease, model_name, probability, prediction, threshold, values)
        from ahead.platform_data import event
        event(owner,'assessment_completed')


# =============================================================================
# HTML BUILDING BLOCKS
# =============================================================================

def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(
        f"""
<div class="clinical-header">
    <div class="header-eyebrow">{escape(eyebrow)}</div>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
</div>
"""
    )


def section_heading(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.html(f'<div class="section-heading"><h2>{escape(title)}</h2>{sub}</div>')


def markdown_literal(text: str) -> str:
    """Escape Markdown punctuation so user text (names, uploaded IDs) shows literally in Markdown and widget labels."""
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", str(text))


def disclaimer(text: str = DISCLAIMER_TEXT, label: str = "Medical disclaimer") -> None:
    st.html(f'<div class="notice disclaimer"><strong>{escape(label)}:</strong> {escape(text)}</div>')


def empty_state(icon: str, title: str, text: str) -> None:
    st.html(
        f"""
<div class="ahead-card empty-state">
    <div class="big" aria-hidden="true">{icon}</div>
    <strong>{escape(title)}</strong>
    {escape(text)}
</div>
"""
    )


def condition_cards(key_prefix: str, button_label: str = "Start screening", note: str | None = None) -> None:
    """Three condition cards tinted in each condition's colour. The whole card is the start button."""
    st.html("<style>" + "".join(
        f'[class*="_{disease}_condition_card"] {{ --condition: {info["colour"]}; }}' for disease, info in DISEASES.items()
    ) + "</style>")
    columns = st.columns(3)
    for column, (disease, info) in zip(columns, DISEASES.items()):
        count = len(features(disease))
        meta = note or f"{count} questions · ~{max(2, round(count / 6))} min"
        with column, st.container(key=f"{key_prefix}_{disease}_condition_card"):
            st.html(
                f"""
<div class="condition-icon" style="background:linear-gradient(135deg, {info['colour']}, color-mix(in srgb, {info['colour']} 72%, white));">
    <img src="{icon_uri(info['icon'])}" alt="">
</div>
<h3 class="condition-title">{escape(info['card_title'])}</h3>
<p class="condition-copy">{escape(info['card_copy'])}</p>
<div class="condition-foot">
    <span class="condition-meta">{escape(meta)}</span>
    <span class="condition-go" aria-hidden="true">{escape(button_label)} →</span>
</div>
"""
            )
            if st.button(f"{button_label} →", key=f"{key_prefix}_{disease}", width="stretch"):
                go_to("screenings", disease)


def stepper(active: int) -> None:
    steps = ["1 · Health Information", "2 · Model Screening", "3 · Results & Guidance"]
    items = "".join(
        f'<div class="step {"active" if index <= active else ""}">{label}</div>'
        for index, label in enumerate(steps, start=1)
    )
    st.html(f'<div class="stepper">{items}</div>')


def result_card(prediction: int, probability: float, disease_label: str, model_name: str, threshold: float,
                audience: str = "patient", subject: str = "This record's values", status: str = "") -> None:
    """The screening result box. audience="doctor" describes *subject* (the record, or values entered on
    Screenings) and the patient in the third person, titled Flagged / Not flagged like the rest of the doctor UI;
    a record's *status* (e.g. Reviewed) replaces that word so the card agrees with the record label and queue."""
    elevated = prediction == 1
    css = "result-elevated" if elevated else "result-lower"
    title = "Elevated Risk Pattern" if elevated else "Lower Predicted Risk"
    if audience == "doctor":
        model_name = model_label(model_name)
        title = (f"{status or 'Flagged'} · elevated risk pattern" if elevated
                 else f"{status or 'Not flagged'} · lower predicted risk")
        text = (
            f"{subject} matched a pattern associated with elevated {disease_label} risk; this is not a "
            "diagnosis and does not confirm that the patient has the condition."
            if elevated
            else f"{subject} did not reach the model's classification level for elevated {disease_label} "
            "risk. This does not rule out disease or replace routine screening for the patient."
        )
    else:
        text = (
            f"The screening model identified a pattern associated with elevated {disease_label} risk. "
            "This result is intended for early awareness and does not confirm that you have the condition."
            if elevated
            else f"Your entered information did not meet the model's classification level for elevated "
            f"{disease_label} risk. This does not rule out disease or replace routine screening."
        )
    st.html(
        f"""
<div class="result-card {css}">
    <div class="result-kicker">Screening Result</div>
    <div class="result-title">{title}</div>
    <div class="result-score">Model screening score: <strong>{probability * 100:.1f}%</strong></div>
    <div class="result-text">{escape(text)}</div>
    <div class="result-meta">Model: {escape(model_name)} · Classification threshold: {threshold * 100:.0f}% ·
    The score is an algorithmic output, not a calibrated clinical probability.</div>
</div>
"""
    )


# =============================================================================
# SCREENING TRACKS (ahead/triage.py sorts; these draw the track the same way on every page)
# =============================================================================

# CSS class per track: .track-panel.<class> and .track-pill.<class> in redesign.css.
TRACK_CLASS = {"urgent": "urgent", "at_risk": "at-risk", "not_flagged": "clear", "no_score": "none"}
_TRACK_ICON = {"urgent": "flagged", "at_risk": "scored", "not_flagged": "review"}


def track_pill_html(track, audience: str = "patient", short: bool = True) -> str:
    """
    The track as a small tinted pill for tables and rows: a triage() result or a bare track name. Shows the short
    label ('Urgent') unless short=False; the full label is the pill's tooltip. The text is escaped.
    """
    text = track_text(track, audience)
    name = track["track"] if isinstance(track, dict) else track
    return (f'<span class="track-pill {TRACK_CLASS[name]}" title="{escape(text["label"])}">'
            f'{escape(text["short"] if short else text["label"])}</span>')


def track_panel_html(result: dict, audience: str = "patient", note: str = "") -> str:
    """
    The track panel's HTML (see track_panel); empty for a result with no score and no urgent rule. Everything shown
    is escaped, including the reasons and *note*.
    """
    track = result["track"]
    if track == "no_score":
        return ""
    text = track_text(result, audience)
    icon = (f'<div class="summary-icon" aria-hidden="true">'
            f'<img src="{icon_uri(f"dashboard/{_TRACK_ICON[track]}.svg")}" alt=""></div>')
    extra = f'<p class="track-panel-note">{escape(note)}</p>' if note else ""
    if track == "not_flagged":   # calm and short: one line
        return (f'<div class="track-panel clear">{icon}<div class="track-panel-body">'
                f'<p class="track-panel-summary"><strong class="track-panel-title">{escape(text["label"])}.</strong> '
                f'{escape(text["summary"])}</p>{extra}</div></div>')
    reasons = "".join(f"<li>{escape(reason[audience])}</li>" for reason in result["reasons"])
    reasons = f'<ul class="track-panel-reasons">{reasons}</ul>' if reasons else ""
    return (f'<div class="track-panel {TRACK_CLASS[track]}">{icon}<div class="track-panel-body">'
            f'<strong class="track-panel-title">{escape(text["label"])}</strong>'
            f'<p class="track-panel-summary">{escape(text["summary"])}</p>{reasons}{extra}</div></div>')


def track_panel(result: dict, audience: str = "patient", note: str = "") -> bool:
    """
    A triage() result as a tinted panel in the dashboard style: urgent in the danger tone with its reasons as a short
    list, at risk in the warning tone, not flagged as one calm line in the success tone. A result with no score and
    no urgent rule draws nothing. *note* adds one line under the summary (escaped). Returns True when drawn.
    """
    body = track_panel_html(result, audience, note)
    if body:
        st.html(body)
    return bool(body)


# =============================================================================
# MODEL SELECTOR (shared by the patient predictor and the clinical dashboard)
# =============================================================================



def model_selector(disease: str, key: str, label: str = "Prediction model", blank_labs: bool = False):
    """
    Selectbox of benchmark models (recommended first) + explanation expander.
    Returns (pipeline, model_name, threshold).  Falls back to the recommended
    model if the chosen pickle has not been trained yet. *blank_labs* mentions the
    lower threshold some models use when optional lab values are left blank.
    Doctors read the models by name, as in the queue, the result card and Patient Intelligence.
    """
    recommended = best_model(disease)
    choices = available_models(disease) or [recommended]
    shown = model_label if current_user()["is_doctor"] else str
    name = st.selectbox(
        label,
        choices,
        format_func=lambda n: f"{shown(n)} — Recommended" if n == recommended else shown(n),
        key=key,
    )
    pipeline = load_pipeline(disease, name)
    if pipeline is None:
        st.warning(
            f"{shown(name)} has not been saved yet — run `python train_model.py` to train every benchmark model. "
            f"Using {shown(recommended)} for now."
        )
        name = recommended
        pipeline = load_pipeline(disease, recommended)
    with st.expander("Why does AHEAD recommend this model?"):
        reason = model_reason(disease)  # opens with the recommended model's class name
        st.write(reason.replace(recommended, shown(recommended), 1))
        st.caption(
            "Alternative models are available for comparison and research. The recommended model "
            "remains the default and is the only one with a tuned decision threshold"
            + (" when every value is entered. If optional lab values are left blank, a model may use a lower "
               "threshold measured for that case." if blank_labs else ".")
        )
    return pipeline, name, decision_threshold(disease, name)
