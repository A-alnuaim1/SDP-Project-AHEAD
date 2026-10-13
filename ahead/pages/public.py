"""
ahead/pages/public.py
=====================
Everything a visitor sees before signing in: the login card and the
About / Contact / Help information pages.  All share login.css.
"""

from html import escape
import os

import streamlit as st

from ahead.config import BASE, ROLE_DOCTOR, ROLE_PATIENT
from ahead.resources import icon_uri, static_url
from ahead.storage import authenticate, create_user, load_screenings
from ahead.config import DISEASES
from datetime import datetime
from ahead.platform_data import event


@st.cache_data(show_spinner=False)
def _login_css() -> str:
    return (BASE / "login.css").read_text()


def _public_chrome(active: str) -> None:
    """Background, stylesheet and the top navigation bar shared by public pages."""
    st.html(
        f"""
<style>
{_login_css()}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse at 10% 10%, #DDD6FE, transparent 50%),
        radial-gradient(ellipse at 90% 80%, #CFFAFE, transparent 45%), #F3F7FF !important;
}}
</style>
"""
    )
    links = [("login", "Sign In")] + [(key, content["nav"]) for key, content in PUBLIC_CONTENT.items()]
    nav = "".join(
        f'<a href="?page={key}" target="_self" class="{"active" if key == active else ""}">{label}</a>'
        for key, label in links
    )
    st.html(
        f"""
<div class="login-topbar">
    <div class="login-links">{nav}</div>
</div>
"""
    )


def _footer() -> None:
    st.html(
        """
<div class="login-footer">
    <span>© AHEAD · Senior Design Project</span>
    <span>Educational screening prototype · Not a medical device</span>
</div>
"""
    )


# =============================================================================
# LOGIN
# =============================================================================

def _sign_in(email: str, password: str, account_type: str) -> str | None:
    """Return an error message, or None on success (session is updated)."""
    email = (email or "").strip().lower()
    wanted = ROLE_PATIENT if account_type == "Patient" else ROLE_DOCTOR
    account = authenticate(email, password, wanted)
    if not account:
        return "Incorrect email, password or account type."
    st.session_state.authenticated = True
    st.session_state.user = account
    if wanted == ROLE_PATIENT:
        event(account['id'],'login')
    st.session_state.prefs = {**st.session_state.prefs, **account.get("preferences", {}).get("toggles", {})}
    st.session_state.pref_language = account.get("preferences", {}).get("language", "English")
    st.session_state.appearance_mode = account.get("preferences", {}).get("appearance", "Light")
    st.session_state.screening_history = [
        {"time": datetime.fromisoformat(item["created_at"]), "disease": item["disease"],
         "label": DISEASES[item["disease"]]["label"], "model": item["model"],
         "probability": item["score"], "prediction": item["prediction"], "threshold": item["threshold"]}
        for item in load_screenings(account["id"]) if item["disease"] in DISEASES
    ]
    return None


def render_login() -> None:
    _public_chrome("login")

    with st.container(key="login_shell"):
        left, right = st.columns([0.92, 1.08], gap="small")

        with left:
            feature_rows = "".join(
                f"""
<div class="ahead-feature">
    <div class="ahead-feature-icon"><img src="{icon_uri(icon)}" class="feature-svg"></div>
    <div class="ahead-feature-title">{title}</div>
</div>"""
                for icon, title in [
                    ("shield-check.svg", "Detect disease risk early"),
                    ("chart-simple.svg", "Track your lifestyle and progress"),
                    ("users.svg", "Connect your records with your doctor"),
                ]
            )
            st.html(
                f"""
<div class="ahead-login-brand">
    <div class="ahead-login-logo">
        <div class="ahead-login-logo-icon">✚</div>
        <div class="ahead-login-logo-text">AHEAD</div>
    </div>
    <h1>Detect Early.<br>Prevent Better.<br>Live Healthier.</h1>
    <div class="ahead-login-description">
        AI-powered health screening designed to support earlier awareness and smarter healthcare decisions.
    </div>
    {feature_rows}
    <div class="ahead-login-tagline">Early Detection. Healthier Tomorrows.</div>
</div>
"""
            )

        with right:
            with st.container(key="login_form"):
                st.html('<div class="login-heading"><h2>Welcome to AHEAD</h2><p>Sign in to continue.</p></div>')

                account_type = st.segmented_control(
                    "Account Type",
                    ["Patient", "Doctor / Admin"],
                    default="Patient",
                    selection_mode="single",
                    label_visibility="collapsed",
                    key="login_account_type",
                ) or "Patient"

                # A real form so that pressing Enter submits.
                with st.form("login_submit", border=False):
                    email = st.text_input("Email", placeholder="name@example.com", key="login_email", autocomplete="username")
                    password = st.text_input(
                        "Password", type="password", placeholder="Enter your password",
                        key="login_password", autocomplete="current-password",
                    )
                    submitted = st.form_submit_button("Sign In", width="stretch", type="primary")

                if submitted:
                    error = _sign_in(email, password, account_type)
                    if error:
                        st.error(error)
                    else:
                        st.query_params.clear()
                        st.rerun()

                with st.expander("Create a patient account"):
                    with st.form("patient_registration"):
                        name = st.text_input("Your full name")
                        new_email = st.text_input("Your email")
                        new_password = st.text_input("Choose a password (12+ characters)", type="password")
                        if st.form_submit_button("Create account"):
                            if not name.strip() or "@" not in new_email or len(new_password) < 12:
                                st.error("Provide a name, valid email and password of at least 12 characters.")
                            elif new_email.strip().lower() == os.environ.get("AHEAD_DOCTOR_EMAIL", "").strip().lower():
                                st.error("This email address is reserved.")
                            elif create_user(new_email, new_password, ROLE_PATIENT, name):
                                st.success("Account created. Sign in above.")
                            else:
                                st.error("That email is already registered.")

    _footer()


# =============================================================================
# INFORMATION PAGES
# =============================================================================

PUBLIC_CONTENT = {
    "about": {
        "nav": "About",
        "eyebrow": "ABOUT THE PROJECT",
        "title": "About AHEAD",
        "intro": (
            "Advanced Health Early Awareness and Disease Detection is a Senior Design "
            "machine-learning project focused on early-awareness screening for chronic disease."
        ),
        "cards": [
            ("✚", "What AHEAD does",
             ["AHEAD accepts lifestyle, medical-history and clinical information and uses disease-specific "
              "machine-learning models to generate an early-awareness screening result.",
              "The current system includes diabetes, cardiovascular and chronic kidney disease assessments, "
              "each benchmarked across six model families."]),
            ("i", "How results should be used",
             ["Results are designed for education and early awareness. They are not diagnoses and should not "
              "be used to make medication or treatment decisions.",
              "Concerning symptoms or abnormal laboratory results should be evaluated by a qualified "
              "healthcare professional."]),
        ],
    },
    "contact": {
        "nav": "Contact",
        "eyebrow": "GET IN TOUCH",
        "title": "Contact the AHEAD team",
        "intro": "AHEAD is a university Senior Design prototype. Reach the project team through the channels below.",
        "cards": [
            ("✉", "Project team", ["For questions about the platform, its models or the datasets, contact the "
                                   "Senior Design team through your course supervisor or the project repository."]),
            ("⚑", "Report a problem", ["Found a bug or an incorrect result? Open an issue on the project repository "
                                       "with the screening condition, the model selected and the values entered."]),
            ("♥", "Medical concerns", ["AHEAD cannot answer medical questions. If you are worried about your health, "
                                      "please contact a healthcare professional or local emergency services."]),
        ],
    },
    "help": {
        "nav": "Help",
        "eyebrow": "HELP CENTRE",
        "title": "How to use AHEAD",
        "intro": "A short guide to signing in, running a screening and reading the result.",
        "cards": [
            ("1", "Sign in", ["Choose Patient or Doctor / Admin, then use one of the demo accounts listed under "
                             "\"Demo accounts\" on the sign-in card. No real patient data is stored."]),
            ("2", "Run a screening", ["Open Screenings, pick a condition and complete the three sections of the form. "
                                     "AHEAD scores the answers with the recommended model (you can compare alternatives)."]),
            ("3", "Read the result", ["The gauge shows the model score against its decision threshold. Below it you will find "
                                     "next steps, the factors AHEAD noticed in your inputs, and an AI-generated explanation."]),
            ("4", "Ask the assistant", ["Use the floating assistant button or the AI Assistant page for general questions "
                                       "about AHEAD, the three conditions or common health measurements."]),
            ("5", "Clinicians", ["Doctor accounts also get a Clinical Dashboard to upload a CSV/XLSX of patient records "
                                "and screen a whole file or a single record."]),
            ("!", "Important", ["AHEAD is an educational prototype. It does not diagnose disease and must not be used "
                               "to start, stop or change any treatment."]),
        ],
    },
}


def render_public_page(kind: str) -> None:
    content = PUBLIC_CONTENT[kind]
    _public_chrome(kind)
    grid_class = "three" if len(content["cards"]) % 3 == 0 else ""
    cards = "".join(
        f"""
<div class="public-info-card">
    <div class="icon">{icon}</div>
    <h3>{escape(title)}</h3>
    {''.join(f'<p>{escape(p)}</p>' for p in paragraphs)}
</div>"""
        for icon, title, paragraphs in content["cards"]
    )
    st.html(
        f"""
<div class="public-card">
    <div class="public-eyebrow">{escape(content['eyebrow'])}</div>
    <h1 class="public-title">{escape(content['title'])}</h1>
    <div class="public-intro">{escape(content['intro'])}</div>
    <div class="public-grid {grid_class}">{cards}</div>
</div>
"""
    )
    _footer()
