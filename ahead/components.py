"""
ahead/components.py
===================
Small reusable UI pieces shared by several pages, plus session-state and
navigation helpers (current user, screening history, go_to).
"""

import re
from datetime import datetime
from html import escape

import streamlit as st

from ahead.config import DISCLAIMER_TEXT, DISEASES, ROLE_DOCTOR, ROLE_PATIENT
from ahead.resources import available_models, best_model, decision_threshold, features, icon_uri, load_pipeline, model_reason
from ahead.i18n import tr, arabic
from ahead.storage import save_screening


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
    """Append a completed screening to the session history (if the user allows it)."""
    if not pref("keep_history"):
        return
    st.session_state.screening_history.append(
        {
            "time": datetime.now(),
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
    <h1>{escape(tr(title))}</h1>
    <p>{escape(tr(subtitle))}</p>
</div>
"""
    )


def section_heading(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(tr(subtitle))}</p>" if subtitle else ""
    st.html(f'<div class="section-heading"><h2>{escape(tr(title))}</h2>{sub}</div>')


def kpi_card(label: str, value: str, note: str, icon: str, tone: str = "") -> None:
    st.html(
        f"""
<div class="metric-card">
    <div class="metric-icon {tone}">{icon}</div>
    <div>
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(str(value))}</div>
        <div class="metric-foot">{escape(note)}</div>
    </div>
</div>
"""
    )


def disclaimer(text: str = DISCLAIMER_TEXT, label: str = "Medical disclaimer") -> None:
    st.html(f'<div class="notice disclaimer"><strong>{escape(label)}:</strong> {escape(text)}</div>')


def empty_state(icon: str, title: str, text: str) -> None:
    st.html(
        f"""
<div class="ahead-card empty-state">
    <div class="big">{icon}</div>
    <strong>{escape(title)}</strong>
    {escape(text)}
</div>
"""
    )


def condition_cards(key_prefix: str, button_label: str = "Start screening", note: str | None = None) -> None:
    """Three condition cards (icon, copy, inputs count, start button). Navigates to the predictor on click."""
    columns = st.columns(3)
    for column, (disease, info) in zip(columns, DISEASES.items()):
        count = len(features(disease))
        meta = note or f"{count} inputs · ~{max(2, round(count / 6))} min"
        with column, st.container(key=f"{key_prefix}_{disease}_condition_card"):
            st.html(
                f"""
<div class="condition-stripe" style="background:{info['colour']};"></div>
<div class="condition-icon" style="background:linear-gradient(135deg, {info['colour']}, color-mix(in srgb, {info['colour']} 72%, white));">
    <img src="{icon_uri(info['icon'])}" alt="">
</div>
<h3 class="condition-title">{escape(tr(info['card_title']))}</h3>
<p class="condition-copy">{escape(info['card_copy'])}</p>
<div class="condition-meta">{escape(meta)}</div>
"""
            )
            if st.button(f"{tr(button_label)} →", key=f"{key_prefix}_{disease}", width="stretch", type="primary"):
                go_to("screenings", disease)


def stepper(active: int) -> None:
    steps = ["1 · Health Information", "2 · Model Screening", "3 · Results & Guidance"]
    items = "".join(
        f'<div class="step {"active" if index <= active else ""}">{label}</div>'
        for index, label in enumerate(steps, start=1)
    )
    st.html(f'<div class="stepper">{items}</div>')


def result_card(prediction: int, probability: float, disease_label: str, model_name: str, threshold: float) -> None:
    elevated = prediction == 1
    css = "result-elevated" if elevated else "result-lower"
    title = tr("Elevated Risk Pattern" if elevated else "Lower Predicted Risk")
    text = (
        f"The screening model identified a pattern associated with elevated {disease_label} risk. "
        "This result is intended for early awareness and does not confirm that you have the condition."
        if elevated
        else f"Your entered information did not meet the model's classification level for elevated "
        f"{disease_label} risk. This does not rule out disease or replace routine screening."
    )
    if arabic():
        text = ("تظهر بياناتك مؤشرات تستدعي المراجعة مع مختص. هذه النتيجة ليست تشخيصاً."
                if elevated else "لم يصل النموذج إلى حد التصنيف المرتفع. هذا لا يستبعد وجود المرض.")
    st.html(
        f"""
<div class="result-card {css}">
    <div class="result-kicker">{escape(tr("Screening Result"))}</div>
    <div class="result-title">{title}</div>
    <div class="result-score">{escape(tr("Model screening score"))}: <strong>{probability * 100:.1f}%</strong></div>
    <div class="result-text">{escape(text)}</div>
    <div class="result-meta">Model: {escape(model_name)} · Classification threshold: {threshold * 100:.0f}% ·
    The score is an algorithmic output, not a calibrated clinical probability.</div>
</div>
"""
    )


# =============================================================================
# MODEL SELECTOR (shared by the patient predictor and the clinical dashboard)
# =============================================================================

def model_selector(disease: str, key: str, label: str = "Prediction model"):
    """
    Selectbox of benchmark models (recommended first) + explanation expander.
    Returns (pipeline, model_name, threshold).  Falls back to the recommended
    model if the chosen pickle has not been trained yet.
    """
    recommended = best_model(disease)
    choices = available_models(disease) or [recommended]
    name = st.selectbox(
        label,
        choices,
        format_func=lambda n: f"{n} — Recommended" if n == recommended else n,
        key=key,
    )
    pipeline = load_pipeline(disease, name)
    if pipeline is None:
        st.warning(
            f"{name} has not been saved yet — run `python train_model.py` to train every benchmark model. "
            f"Using {recommended} for now."
        )
        name = recommended
        pipeline = load_pipeline(disease, recommended)
    with st.expander("Why does AHEAD recommend this model?"):
        st.write(model_reason(disease))
        st.caption(
            "Alternative models are available for comparison and research. The recommended model "
            "remains the default and is the only one with a tuned decision threshold."
        )
    return pipeline, name, decision_threshold(disease, name)
