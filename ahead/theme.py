"""
ahead/theme.py
==============
One place for the AHEAD look: colour tokens (light / dark), the global CSS
that every authenticated page shares, and a Plotly styling helper.

Every custom element uses ``var(--ahead-*)`` so switching the appearance mode
only swaps the token block emitted by :func:`apply_theme`.
"""

from pathlib import Path
import plotly.graph_objects as go
import streamlit as st

from ahead.config import NAV_ITEMS
from ahead.resources import icon_uri


# =============================================================================
# TOKENS
# =============================================================================

LIGHT = {
    "bg": "#F3F7FF",
    "card": "#FFFFFF",
    "soft": "#F3F7FA",
    "border": "#E1E8EF",
    "title": "#172554",
    "text": "#344054",
    "muted": "#6B7A90",
    "accent": "#2563EB",
    "accent_strong": "#1D4ED8",
    "accent_soft": "#EAF0FF",
    "danger": "#EF4444",
    "danger_soft": "#FCEDED",
    "success": "#10B981",
    "success_soft": "#E8F5EF",
    "warning": "#F59E0B",
    "warning_soft": "#FFF6E0",
    "shadow": "0 8px 26px rgba(16,42,67,0.06)",
    "grid": "#E9EEF3",
    "plotly_template": "plotly_white",
}

DARK = {
    "bg": "#0E1B27",
    "card": "#142633",
    "soft": "#1A3040",
    "border": "#27404F",
    "title": "#F1F7FA",
    "text": "#DCE7EC",
    "muted": "#9FB4C0",
    "accent": "#3BC3CB",
    "accent_strong": "#2AA9B2",
    "accent_soft": "rgba(43,195,203,0.14)",
    "danger": "#E07A7A",
    "danger_soft": "rgba(224,122,122,0.14)",
    "success": "#4DBB99",
    "success_soft": "rgba(77,187,153,0.14)",
    "warning": "#E0B65A",
    "warning_soft": "rgba(224,182,90,0.14)",
    "shadow": "0 8px 26px rgba(0,0,0,0.28)",
    "grid": "#25394A",
    "plotly_template": "plotly_dark",
}

APPEARANCE_OPTIONS = {   # mode -> (preview css class, description)
    "Light": ("light", "Clean and bright interface."),
    "Dark": ("dark", "Easy on the eyes in low light."),
}
APPEARANCE_MODES = tuple(APPEARANCE_OPTIONS)


def appearance_mode() -> str:
    mode = st.session_state.get("appearance_mode", "Light")
    if mode not in APPEARANCE_MODES:
        mode = "Light"
        st.session_state.appearance_mode = mode
    return mode


def is_dark() -> bool:
    return appearance_mode() == "Dark"


def tokens() -> dict:
    return DARK if is_dark() else LIGHT


# =============================================================================
# PLOTLY
# =============================================================================

def style_figure(figure: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    """Apply the AHEAD chart look for the current appearance mode."""
    t = tokens()
    figure.update_layout(
        template=t["plotly_template"],
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=t["text"], size=12),
        title_font=dict(size=15, color=t["title"]),
        margin=dict(l=20, r=20, t=50, b=80 if legend else 20),
        showlegend=legend,
        colorway=["#2563EB", "#8B5CF6", "#06B6D4", "#10B981", "#F97316"],
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0, title=""),
    )
    figure.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], title_font=dict(color=t["muted"]))
    figure.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], title_font=dict(color=t["muted"]))
    return figure


def show_chart(figure: go.Figure) -> None:
    """Render a figure full-width without Plotly's mode bar."""
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


# =============================================================================
# GLOBAL CSS
# =============================================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
/* ---------------------------------------------------------------- base */
html, body, .stApp, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp { background: var(--ahead-bg); color: var(--ahead-text); }
[data-testid="stMainBlockContainer"] {
    max-width: 1240px; padding-top: 3.4rem; padding-bottom: 4rem;   /* clears the fixed 3.75rem header */
}
/* the header is a transparent fixed bar; let clicks pass through it except on its own controls */
[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
[data-testid="stHeader"] button { pointer-events: auto; }
/* keep the toolbar (it holds the expand-sidebar control on small screens) but drop Deploy / menu */
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stExpandSidebarButton"] { color: var(--ahead-title) !important; }

h1, h2, h3, h4, h5, h6 { color: var(--ahead-title); }
[data-testid="stMain"] p, [data-testid="stMain"] li, [data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] { color: var(--ahead-text); }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--ahead-muted); }

/* ---------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {
    width: 262px !important; min-width: 262px !important; max-width: 262px !important;
    background: linear-gradient(180deg, #0B2942 0%, #103B54 62%, #0C5058 100%) !important;
}
[data-testid="stSidebar"] > div { width: 262px !important; min-width: 262px !important; }
[data-testid="stSidebarHeader"] { display: none !important; height: 0 !important; padding: 0 !important; }
[data-testid="stSidebarContent"] {
    height: 100vh; box-sizing: border-box; padding: 22px 16px 18px 16px !important;
    display: flex; flex-direction: column; overflow-y: auto !important; overflow-x: hidden !important;
}
[data-testid="stSidebarUserContent"] { padding: 0 !important; height: 100%; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div { height: 100%; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1 1 auto; }
[data-testid="stSidebarUserContent"] [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_bottom) { margin-top: auto; }
[data-testid="stSidebar"] * { color: #F8FAFC; }

.sidebar-brand { display: flex; align-items: center; gap: 12px; margin: 0 0 4px 2px; }
.brand-symbol {
    width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #14BFC0, #83DDD7); color: #073B52 !important;
    font-size: 22px; font-weight: 800; flex: 0 0 auto;
}
.sidebar-brand h2 { margin: 0; color: #FFFFFF !important; font-size: 1.28rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; }
.sidebar-brand p { margin: 3px 0 0; color: #BDD0DB !important; font-size: 0.66rem; line-height: 1.35; }
.sidebar-divider { height: 1px; background: rgba(255,255,255,0.12); margin: 16px 0 12px; }
.sidebar-section-label { color: #8FB0C0 !important; font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; margin: 4px 0 6px 12px; }

.st-key-sidebar_nav { gap: 3px !important; }
.st-key-sidebar_nav .stButton > button {
    width: 100% !important; min-height: 42px !important; height: 42px !important;
    margin: 0 !important; padding: 8px 10px 8px 44px !important;
    background: transparent !important; border: none !important; border-radius: 10px !important;
    box-shadow: none !important; color: #EAF5F7 !important; font-size: 0.83rem !important; font-weight: 520 !important;
    position: relative !important; justify-content: flex-start !important; text-align: left !important;
}
.st-key-sidebar_nav .stButton > button > div { width: 100%; display: flex; justify-content: flex-start; text-align: left; }
.st-key-sidebar_nav .stButton > button p { margin: 0 !important; color: inherit !important; font-size: inherit !important; font-weight: inherit !important; }
.st-key-sidebar_nav .stButton > button:hover { background: rgba(32,176,188,0.16) !important; color: #FFFFFF !important; }
.st-key-sidebar_nav .stButton > button::before {
    content: ""; position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
    width: 19px; height: 19px; background-size: contain; background-repeat: no-repeat; background-position: center;
}

.st-key-sidebar_bottom { gap: 8px !important; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.12); }
.sidebar-user-profile { display: flex; align-items: center; gap: 10px; padding: 2px 2px 0; }
.sidebar-user-avatar {
    width: 38px; height: 38px; min-width: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: #F5FAFC; color: #456984 !important; font-size: 0.74rem; font-weight: 750;
}
.sidebar-user-info { min-width: 0; display: flex; flex-direction: column; gap: 1px; }
.sidebar-user-name { color: #FFFFFF !important; font-size: 0.79rem; font-weight: 650; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px; }
.sidebar-user-role { color: #C3D5DF !important; font-size: 0.68rem; line-height: 1.2; }
.st-key-sidebar_logout .stButton > button {
    width: 100% !important; min-height: 38px !important; margin: 0 !important; padding: 7px 10px 7px 38px !important; position: relative;
    border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important;
    background: rgba(14,151,161,0.55) !important; color: #FFFFFF !important; box-shadow: none !important;
    font-size: 0.76rem !important; font-weight: 600 !important; justify-content: center !important;
}
.st-key-sidebar_logout .stButton > button:hover { background: rgba(18,169,179,0.8) !important; }
.st-key-sidebar_logout .stButton > button::before {
    content: ""; position: absolute; left: 14px; top: 50%; transform: translateY(-50%); width: 17px; height: 17px;
    background-size: contain; background-repeat: no-repeat;
}

/* ---------------------------------------------------------------- generic cards & text */
.ahead-card, .metric-card, .condition-card, .action-card, .result-card, .clinical-header, .settings-panel {
    background: var(--ahead-card); border: 1px solid var(--ahead-border); border-radius: 16px; box-shadow: var(--ahead-shadow);
}
.eyebrow, .header-eyebrow, .home-eyebrow {
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.68rem; font-weight: 740; color: var(--ahead-accent);
}
.section-heading { margin: 22px 0 14px; }
.section-heading h2 { color: var(--ahead-title); font-size: 1.2rem; margin: 0; font-weight: 720; }
.section-heading p { color: var(--ahead-muted); margin: 4px 0 0; font-size: 0.78rem; }

/* page header */
.clinical-header { padding: 28px 32px; margin-bottom: 20px; border-radius: 20px; }
.clinical-header h1 { margin: 8px 0 0; color: var(--ahead-title); font-size: 1.9rem; font-weight: 760; letter-spacing: -0.03em; }
.clinical-header p { max-width: 800px; color: var(--ahead-muted); font-size: 0.93rem; line-height: 1.6; margin: 10px 0 0; }

/* hero — the keyed container is the gradient box, so the action buttons sit inside it */
.st-key-home_hero {
    background: linear-gradient(125deg, #0D304B, #14556B 55%, #187D7D); border-radius: 22px; padding: 34px 38px 30px;
    color: white; box-shadow: 0 16px 40px rgba(11,41,66,0.16); position: relative; overflow: hidden; margin-bottom: 18px;
    gap: 22px !important;
}
.st-key-home_hero::before {
    content: ""; position: absolute; width: 340px; height: 340px; border-radius: 50%;
    border: 68px solid rgba(255,255,255,0.04); right: -120px; top: -160px; pointer-events: none;
}
.home-hero { display: flex; justify-content: space-between; align-items: center; gap: 24px; position: relative; z-index: 1; }
.home-hero .home-eyebrow { color: #BDE5E2; }
.home-hero h1 { color: white; font-size: 2.3rem; margin: 8px 0 6px; font-weight: 770; letter-spacing: -0.04em; }
.home-hero p { max-width: 640px; color: #D9E9EE; font-size: 0.95rem; line-height: 1.6; margin: 0; }
.hero-badge {
    position: relative; z-index: 1; text-align: right; color: #EAF7F7; font-size: 1.05rem; font-weight: 700; line-height: 1.35;
    padding: 14px 18px; border-radius: 14px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    min-width: 170px;
}
.hero-badge small { display: block; color: #BDE5E2; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.st-key-home_hero [data-testid="stHorizontalBlock"] { gap: 12px !important; align-items: center; position: relative; z-index: 1; }
/* white primary + glass secondary read better on the dark gradient than the teal buttons */
.st-key-home_hero [data-testid="stPopoverButton"] {
    background: white !important; color: #0D304B !important; border: 1px solid white !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18) !important;
}
.st-key-home_hero [data-testid="stPopoverButton"]:hover { background: #E9F6F7 !important; filter: none; }
.st-key-home_hero .st-key-hero_ask button {
    background: rgba(255,255,255,0.1) !important; color: white !important; border: 1px solid rgba(255,255,255,0.35) !important;
}
.st-key-home_hero .st-key-hero_ask button:hover { background: rgba(255,255,255,0.18) !important; border-color: white !important; color: white !important; }

/* KPI cards */
.metric-card { padding: 16px 18px; min-height: 108px; display: flex; gap: 12px; align-items: flex-start; min-width: 0; overflow: hidden; }
.metric-card > div { min-width: 0; }
.metric-icon {
    width: 42px; height: 42px; min-width: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    background: var(--ahead-accent-soft); color: var(--ahead-accent); font-weight: 800; font-size: 1rem;
}
.metric-icon.danger { background: var(--ahead-danger-soft); color: var(--ahead-danger); }
.metric-icon.success { background: var(--ahead-success-soft); color: var(--ahead-success); }
.metric-icon.warning { background: var(--ahead-warning-soft); color: var(--ahead-warning); }
.metric-label { color: var(--ahead-muted); text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.64rem; font-weight: 680; line-height: 1.3; }
.metric-value { color: var(--ahead-title); font-size: clamp(1.1rem, 1.6vw, 1.55rem); font-weight: 760; margin-top: 4px; line-height: 1.1; overflow-wrap: anywhere; }
.metric-foot { color: var(--ahead-accent); font-size: 0.68rem; margin-top: 5px; }

/* condition cards — the keyed container is the card, so the button sits inside it */
[class*="_condition_card"] {
    position: relative; overflow: hidden; gap: 0 !important;
    background: var(--ahead-card); border: 1px solid var(--ahead-border); border-radius: 18px; box-shadow: var(--ahead-shadow);
    padding: 26px 22px 20px; height: 100%;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[class*="_condition_card"]:hover { transform: translateY(-3px); box-shadow: 0 16px 34px rgba(16,44,66,0.12); border-color: var(--ahead-accent); }
.condition-stripe { height: 5px; margin: -26px -22px 22px; }   /* pulled up through the card padding */
.condition-icon {
    width: 54px; height: 54px; border-radius: 16px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px; box-shadow: 0 8px 18px rgba(16,44,66,0.16);
}
.condition-icon img { width: 28px; height: 28px; }
.condition-title { color: var(--ahead-title); font-size: 1.05rem; font-weight: 720; margin: 0; letter-spacing: -0.01em; }
.condition-copy { color: var(--ahead-muted); font-size: 0.8rem; line-height: 1.6; margin: 8px 0 0; min-height: 3.2em; }
.condition-meta {
    display: inline-flex; align-items: center; gap: 6px; margin: 14px 0 14px;
    color: var(--ahead-accent); background: var(--ahead-accent-soft); border-radius: 100px; padding: 4px 10px;
    font-size: 0.66rem; font-weight: 650; letter-spacing: 0.02em;
}
[class*="_condition_card"] .stButton { margin-top: auto; }
[class*="_condition_card"] .stButton p { white-space: nowrap; }
[class*="_condition_card"] [data-testid="stElementContainer"]:has(> .stButton) { margin-top: auto; }

/* stepper */
.stepper { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 6px 0 18px; }
.step { padding: 10px; border-radius: 10px; background: var(--ahead-soft); color: var(--ahead-muted); font-size: 0.72rem; font-weight: 650; text-align: center; border: 1px solid var(--ahead-border); }
.step.active { background: var(--ahead-accent-soft); color: var(--ahead-accent); font-weight: 720; border-color: transparent; }

/* forms */
[data-testid="stForm"] {
    background: var(--ahead-card); border: 1px solid var(--ahead-border); border-radius: 18px; padding: 12px 23px 24px; box-shadow: var(--ahead-shadow);
}
.form-section-title { display: flex; gap: 12px; align-items: flex-start; margin: 18px 0 16px; padding-bottom: 11px; border-bottom: 1px solid var(--ahead-border); }
.form-section-number, .action-number {
    width: 28px; height: 28px; min-width: 28px; border-radius: 8px; background: var(--ahead-accent-soft); color: var(--ahead-accent);
    display: inline-flex; align-items: center; justify-content: center; font-size: 0.71rem; font-weight: 760;
}
.form-section-title h3 { margin: 0; color: var(--ahead-title); font-size: 0.98rem; }
.form-section-title p { margin: 3px 0 0; color: var(--ahead-muted); font-size: 0.72rem; }
[data-testid="stMain"] [data-testid="stWidgetLabel"] p { color: var(--ahead-text) !important; font-size: 0.78rem !important; font-weight: 600 !important; }
[data-testid="stMain"] [data-testid="stTextInputRootElement"], [data-testid="stMain"] [data-testid="stNumberInputContainer"],
[data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"],
[data-testid="stMain"] [data-testid="stTextArea"] textarea {
    background: var(--ahead-soft) !important; border: 1px solid var(--ahead-border) !important; border-radius: 10px !important;
}
[data-testid="stMain"] [data-testid="stTextInputRootElement"]:focus-within, [data-testid="stMain"] [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:focus-within { border-color: var(--ahead-accent) !important; }
[data-testid="stMain"] [data-testid="stNumberInputStepDown"], [data-testid="stMain"] [data-testid="stNumberInputStepUp"] {
    background: transparent !important; color: var(--ahead-title) !important;
}

/* buttons in the main area */
[data-testid="stMain"] .stButton > button, [data-testid="stMain"] .stFormSubmitButton > button, [data-testid="stMain"] [data-testid="stPopoverButton"] {
    border-radius: 10px !important; font-weight: 640 !important; min-height: 42px;
}
[data-testid="stMain"] button [data-testid="stMarkdownContainer"], [data-testid="stMain"] button p,
[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] button p { color: inherit !important; }
[data-testid="stMain"] [data-testid^="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--ahead-accent), var(--ahead-accent-strong)) !important; color: #FFFFFF !important; border: none !important;
    box-shadow: 0 6px 16px rgba(11,127,139,0.22) !important;
}
[data-testid="stMain"] [data-testid^="stBaseButton-primary"]:hover { filter: brightness(1.06); }
[data-testid="stMain"] [data-testid^="stBaseButton-primary"]:disabled { filter: none; opacity: 0.55; }
[data-testid="stMain"] [data-testid^="stBaseButton-secondary"] {
    background: var(--ahead-card) !important; color: var(--ahead-title) !important; border: 1px solid var(--ahead-border) !important;
}
[data-testid="stMain"] [data-testid^="stBaseButton-secondary"]:hover { border-color: var(--ahead-accent) !important; color: var(--ahead-accent) !important; }
[data-testid="stMain"] [data-testid^="stBaseButton-tertiary"] { color: var(--ahead-accent) !important; }

/* results */
.result-card { padding: 26px; min-height: 250px; }
.result-elevated { border-top: 4px solid var(--ahead-danger); }
.result-lower { border-top: 4px solid var(--ahead-success); }
.result-kicker { text-transform: uppercase; letter-spacing: 0.1em; color: var(--ahead-muted); font-size: 0.65rem; font-weight: 750; }
.result-title { font-size: 1.45rem; font-weight: 740; margin: 8px 0 4px; }
.result-elevated .result-title { color: var(--ahead-danger); }
.result-lower .result-title { color: var(--ahead-success); }
.result-score { color: var(--ahead-text); font-size: 0.9rem; margin-top: 13px; }
.result-text { color: var(--ahead-muted); font-size: 0.79rem; line-height: 1.6; margin-top: 10px; }
.result-meta { color: var(--ahead-muted); font-size: 0.7rem; margin-top: 14px; }

.action-card { padding: 18px 20px; margin-bottom: 10px; }
.action-number { margin-right: 9px; }
.action-title { color: var(--ahead-title); font-size: 0.91rem; font-weight: 700; }
.action-text { color: var(--ahead-muted); font-size: 0.79rem; line-height: 1.58; margin-top: 8px; }

.factor-chip {
    display: inline-block; margin: 0 8px 8px 0; padding: 7px 12px; border-radius: 100px; font-size: 0.76rem; font-weight: 600;
    background: var(--ahead-warning-soft); color: var(--ahead-warning); border: 1px solid transparent;
}

.notice { border-radius: 12px; padding: 14px 16px; font-size: 0.77rem; line-height: 1.55; margin-top: 16px; border: 1px solid var(--ahead-border); }
.notice.urgent { background: var(--ahead-danger-soft); border-left: 4px solid var(--ahead-danger); color: var(--ahead-text); }
.notice.disclaimer { background: var(--ahead-warning-soft); border-left: 4px solid var(--ahead-warning); color: var(--ahead-text); }
.notice strong { color: var(--ahead-title); }

/* empty state */
.empty-state { text-align: center; padding: 34px 20px; color: var(--ahead-muted); font-size: 0.85rem; }
.empty-state .big { font-size: 2rem; margin-bottom: 8px; }
.empty-state strong { display: block; color: var(--ahead-title); font-size: 0.95rem; margin-bottom: 4px; }

/* ---------------------------------------------------------------- settings */
.settings-shell-title { margin: 0 0 14px; }
.settings-shell-title h1 { margin: 0; color: var(--ahead-title); font-size: 1.65rem; font-weight: 790; letter-spacing: -0.03em; }
.settings-shell-title p { margin: 5px 0 0; color: var(--ahead-muted); font-size: 0.82rem; }
.st-key-settings_tabs [role="tablist"] { gap: 6px; border-bottom: 1px solid var(--ahead-border); }
.st-key-settings_tabs [data-testid="stTab"] { padding: 10px 14px; font-weight: 600; color: var(--ahead-muted); }
.st-key-settings_tabs [data-testid="stTab"][aria-selected="true"] { color: var(--ahead-accent); }
.settings-panel { padding: 20px 22px; margin: 14px 0 12px; }
.settings-panel h2 { color: var(--ahead-title); font-size: 1.05rem; margin: 0 0 4px; }
.settings-panel p { color: var(--ahead-muted); font-size: 0.77rem; line-height: 1.55; margin: 0; }
.settings-profile-banner {
    display: flex; align-items: center; gap: 16px; padding: 16px 18px; margin: 0 0 14px; border-radius: 14px;
    background: var(--ahead-soft); border: 1px solid var(--ahead-border);
}
.settings-avatar {
    width: 58px; height: 58px; min-width: 58px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #8EABC0, #5F8AA6); color: white !important; font-size: 0.95rem; font-weight: 750;
}
.settings-person-name { color: var(--ahead-title); font-size: 0.95rem; font-weight: 730; }
.settings-person-role { color: var(--ahead-accent); font-size: 0.72rem; font-weight: 650; margin-top: 2px; }
.settings-person-email { color: var(--ahead-muted); font-size: 0.72rem; margin-top: 2px; }
.settings-row { display: flex; align-items: center; gap: 12px; padding: 12px 6px; }
.settings-row-icon {
    width: 36px; height: 36px; min-width: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    background: var(--ahead-accent-soft); color: var(--ahead-accent) !important; font-size: 0.95rem; font-weight: 700;
}
.settings-row-title { color: var(--ahead-title); font-size: 0.82rem; font-weight: 680; }
.settings-row-copy { color: var(--ahead-muted); font-size: 0.7rem; margin-top: 2px; }
.st-key-settings_rows [data-testid="stHorizontalBlock"], .st-key-settings_rows_privacy [data-testid="stHorizontalBlock"] { align-items: center !important; border-bottom: 1px solid var(--ahead-border); }
.st-key-settings_rows [data-testid="stHorizontalBlock"]:last-child, .st-key-settings_rows_privacy [data-testid="stHorizontalBlock"]:last-child { border-bottom: none; }
.appearance-preview { height: 96px; border-radius: 10px; border: 1px solid var(--ahead-border); margin-bottom: 10px; overflow: hidden; position: relative; }
.appearance-preview::after { content: ""; position: absolute; left: 34%; top: 18%; width: 46%; height: 10px; border-radius: 6px; background: rgba(120,140,160,0.35); box-shadow: 0 18px 0 rgba(120,140,160,0.25), 0 36px 0 rgba(120,140,160,0.18); }
.appearance-preview.light { background: linear-gradient(90deg, #0D3A55 26%, #F7FAFC 26%); }
.appearance-preview.dark { background: linear-gradient(90deg, #081F30 26%, #142633 26%); }
.appearance-choice { border: 1px solid var(--ahead-border); border-radius: 14px; padding: 12px; background: var(--ahead-card); }
.appearance-choice.active { border: 2px solid var(--ahead-accent); box-shadow: 0 0 0 4px var(--ahead-accent-soft); }
.appearance-choice strong { color: var(--ahead-title); font-size: 0.84rem; display: block; }
.appearance-choice span { display: block; color: var(--ahead-muted); font-size: 0.7rem; margin-top: 2px; }
.appearance-choice .check { float: right; color: var(--ahead-accent); font-weight: 800; }
.settings-danger { background: var(--ahead-danger-soft); border: 1px solid transparent; border-radius: 12px; padding: 14px 16px; margin-top: 8px; }
.settings-danger strong { color: var(--ahead-danger); font-size: 0.82rem; }
.settings-danger p { margin: 3px 0 0; color: var(--ahead-muted); font-size: 0.72rem; }
[data-testid="stMain"] .st-key-settings_logout button { background: linear-gradient(135deg, #D2565C, #B54247) !important; box-shadow: 0 6px 16px rgba(181,66,71,0.22) !important; }

/* ---------------------------------------------------------------- assistant */
.assistant-shell { padding: 6px 0 0; }
.assistant-hero { display: flex; justify-content: space-between; align-items: center; gap: 24px; margin-top: 4px; }
.assistant-hero h1 { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em; margin: 6px 0 10px; color: var(--ahead-title); line-height: 1.05; }
.assistant-hero h1 span { color: var(--ahead-accent); }
.assistant-hero p { color: var(--ahead-muted); font-size: 0.98rem; line-height: 1.6; max-width: 640px; margin: 0; }
.assistant-mascot {
    width: 150px; height: 150px; min-width: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle, var(--ahead-card) 55%, var(--ahead-accent-soft) 100%); box-shadow: var(--ahead-shadow);
}
.assistant-mascot img { width: 96px; height: 96px; }
.assistant-traits { display: flex; gap: 34px; flex-wrap: wrap; margin: 22px 0 6px; padding-bottom: 18px; border-bottom: 1px solid var(--ahead-border); }
.assistant-trait { display: flex; align-items: center; gap: 10px; color: var(--ahead-text); font-size: 0.92rem; font-weight: 600; }
.assistant-trait img { width: 26px; height: 26px; }
.assistant-try { color: var(--ahead-title); font-size: 1.05rem; font-weight: 750; margin: 16px 0 10px; }
.st-key-assistant_suggestions { flex-wrap: wrap; margin-bottom: 4px; }
.st-key-assistant_suggestions .stButton > button {
    min-height: 52px !important; padding: 10px 18px 10px 14px !important; gap: 10px !important;
    background: var(--ahead-card) !important; border: 1px solid var(--ahead-border) !important; border-radius: 14px !important;
    color: var(--ahead-title) !important; font-weight: 600 !important; box-shadow: var(--ahead-shadow) !important;
}
.st-key-assistant_suggestions .stButton > button:hover { border-color: var(--ahead-accent) !important; color: var(--ahead-accent) !important; transform: translateY(-1px); }
.st-key-assistant_suggestions .stButton > button p { font-size: 0.86rem !important; }
.st-key-assistant_suggestions .stButton > button [data-testid="stIconMaterial"] {
    color: var(--ahead-accent) !important; font-size: 1.25rem !important; background: var(--ahead-accent-soft); border-radius: 9px; padding: 5px;
}
.assistant-time { color: var(--ahead-muted); font-size: 0.68rem; margin: -6px 0 4px 4px; }
[data-testid="stChatMessage"] { background: var(--ahead-card); border: 1px solid var(--ahead-border); border-radius: 16px; padding: 12px 16px; box-shadow: var(--ahead-shadow); }
[data-testid="stChatMessage"] p { color: var(--ahead-text); }
[data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessageAvatarAssistant"] { background: var(--ahead-accent-soft); }
[data-testid="stBottomBlockContainer"] { background: transparent; max-width: 1240px; }
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid var(--ahead-border); background: var(--ahead-card); box-shadow: var(--ahead-shadow); }
[data-testid="stChatInput"] textarea { color: var(--ahead-text); }

/* floating assistant button */
.stApp .st-key-open_assistant { position: fixed !important; right: 26px !important; bottom: 24px !important; width: 68px !important; height: 68px !important; padding: 0 !important; margin: 0 !important; z-index: 99999 !important; }
.stApp .st-key-open_assistant .stButton, .stApp .st-key-open_assistant .stElementContainer { width: 68px !important; height: 68px !important; margin: 0 !important; }
.stApp .st-key-open_assistant button {
    width: 68px !important; min-width: 68px !important; height: 68px !important; min-height: 68px !important; padding: 0 !important;
    border-radius: 50% !important; background-color: var(--ahead-card) !important; border: 4px solid var(--ahead-accent) !important;
    box-shadow: 0 8px 24px rgba(14,105,120,0.22) !important; font-size: 0 !important; color: transparent !important;
    background-size: 42px 42px !important; background-position: center !important; background-repeat: no-repeat !important;
}
.stApp .st-key-open_assistant button p { position: absolute !important; width: 1px !important; height: 1px !important; overflow: hidden !important; clip: rect(0 0 0 0) !important; white-space: nowrap !important; }
.stApp .st-key-open_assistant button:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(14,105,120,0.3) !important; }

/* ---------------------------------------------------------------- misc native widgets */
[data-testid="stMain"] [data-testid="stExpander"] > details { border: 1px solid var(--ahead-border) !important; border-radius: 12px; background: var(--ahead-card); }
[data-testid="stMain"] [data-testid="stExpander"] summary, [data-testid="stMain"] [data-testid="stExpander"] summary p { color: var(--ahead-title); }
[data-testid="stMain"] [data-testid="stAlertContainer"] p, [data-testid="stMain"] [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] { color: inherit; }
/* style-only st.html blocks should not take a layout slot */
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > style:only-child) { display: none !important; }
[data-testid="stMain"] [data-testid="stMetric"] { background: var(--ahead-card); border: 1px solid var(--ahead-border); border-radius: 14px; padding: 14px 16px; }
[data-testid="stMain"] [data-testid="stMetricLabel"] p { color: var(--ahead-muted) !important; font-size: 0.72rem !important; }
[data-testid="stMain"] [data-testid="stMetricValue"] { color: var(--ahead-title); font-size: 1.45rem; }
[data-testid="stMain"] hr { border-color: var(--ahead-border); }
[data-testid="stMain"] [data-testid="stAlert"] { border-radius: 12px; }
/* popover menu (Start Screening): light list with dividers and a chevron on each row */
[data-testid="stPopoverBody"] {
    border-radius: 14px; min-width: 300px; padding: 6px 8px !important;
    background: #E9F4F8 !important; border: 1px solid #D3E6EE !important; box-shadow: 0 14px 34px rgba(16,44,66,0.16) !important;
}
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:not(:last-child) { border-bottom: 1px solid #CDE2EA; }
[data-testid="stPopoverBody"] .stButton > button {
    width: 100% !important; justify-content: space-between !important; min-height: 52px; padding: 10px 12px 10px 14px !important;
    background: transparent !important; color: #0E2E5C !important; border: none !important; border-radius: 10px !important;
    font-weight: 620 !important; box-shadow: none !important;
}
[data-testid="stPopoverBody"] .stButton > button > div { justify-content: flex-start !important; text-align: left !important; flex: 1; }
[data-testid="stPopoverBody"] .stButton > button p { text-align: left !important; }
[data-testid="stPopoverBody"] .stButton > button::after {
    content: ""; width: 16px; height: 16px; flex: none; margin-left: 12px;
    background: url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20viewBox%3D%270%200%2024%2024%27%20fill%3D%27none%27%20stroke%3D%27%230E9AA7%27%20stroke-width%3D%272.6%27%20stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27%3E%3Cpath%20d%3D%27m9%206%206%206-6%206%27%2F%3E%3C%2Fsvg%3E") center / contain no-repeat;
}
[data-testid="stPopoverBody"] .stButton > button:hover { background: #D5EAF1 !important; color: #0E2E5C !important; }
@media (max-width: 900px) {
    .home-hero { flex-direction: column; align-items: flex-start; }
    .hero-badge { text-align: left; }
    .assistant-hero { flex-direction: column-reverse; align-items: flex-start; }
}
</style>
"""

DARK_NATIVE_OVERRIDES = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--ahead-bg) !important; color: var(--ahead-text) !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--ahead-bg) !important; }
[data-testid="stMain"] [data-testid="stTextInputRootElement"], [data-testid="stMain"] [data-testid="stNumberInputContainer"],
[data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"], [data-testid="stMain"] [data-testid="stTextArea"] textarea,
[data-testid="stMain"] [data-testid="stChatInput"] {
    background: var(--ahead-soft) !important; color: var(--ahead-title) !important; border-color: var(--ahead-border) !important;
}
[data-testid="stMain"] input, [data-testid="stMain"] textarea, [data-testid="stMain"] [data-testid="stSelectbox"] button { color: var(--ahead-title) !important; }
[data-testid="stMain"] input::placeholder, [data-testid="stMain"] textarea::placeholder { color: var(--ahead-muted) !important; }
[role="listbox"], .react-aria-Popover, [data-testid="stSelectboxVirtualDropdown"] { background: var(--ahead-card) !important; color: var(--ahead-text) !important; border-color: var(--ahead-border) !important; }
[role="option"] { color: var(--ahead-text) !important; }
[role="option"]:hover, [role="option"][data-focused], [role="option"][aria-selected="true"] { background: var(--ahead-soft) !important; }
[data-testid="stMain"] [data-testid="stTab"] { color: var(--ahead-muted) !important; }
[data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] { color: var(--ahead-accent) !important; }
[data-testid="stMain"] [role="tablist"] { border-color: var(--ahead-border) !important; }
[data-testid="stMain"] [data-testid="stAlertContainer"] { background: var(--ahead-soft) !important; color: var(--ahead-text) !important; border-color: var(--ahead-border) !important; }
[data-testid="stMain"] [data-testid="stAlertContainer"] [data-testid^="stAlertContent"], [data-testid="stMain"] [data-testid="stAlertContainer"] p,
[data-testid="stMain"] [data-testid="stAlertContainer"] svg { color: var(--ahead-text) !important; }
[data-testid="stMain"] [data-testid="stAlertContainer"] code { background: var(--ahead-card) !important; color: var(--ahead-accent) !important; }
[data-testid="stMain"] [data-testid="stFileUploader"] section { background: var(--ahead-soft) !important; border-color: var(--ahead-border) !important; }
[data-testid="stMain"] [data-testid="stFileUploader"] * { color: var(--ahead-text); }
[data-testid="stPopoverBody"] { background: var(--ahead-card) !important; border-color: var(--ahead-border) !important; }
[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:not(:last-child) { border-bottom-color: var(--ahead-border); }
[data-testid="stPopoverBody"] .stButton > button { color: var(--ahead-title) !important; }
[data-testid="stPopoverBody"] .stButton > button:hover { background: var(--ahead-soft) !important; color: var(--ahead-title) !important; }
[data-testid="stMain"] [data-testid="stCheckbox"] p, [data-testid="stMain"] [data-testid="stRadio"] p { color: var(--ahead-text) !important; }
[data-testid="stMain"] [data-testid="stButtonGroup"] button { color: var(--ahead-text) !important; border-color: var(--ahead-border) !important; }
[data-testid="stMain"] [data-testid="stButtonGroup"] button[data-selected="true"] { background: var(--ahead-accent) !important; color: white !important; }
[data-testid="stMain"] code { background: var(--ahead-soft) !important; color: var(--ahead-accent) !important; }
[data-testid="stMain"] [data-testid="stExpander"] > details { background: var(--ahead-card) !important; border-color: var(--ahead-border) !important; }
[data-testid="stMain"] [data-testid="stExpander"] summary, [data-testid="stMain"] [data-testid="stExpander"] summary * { color: var(--ahead-title) !important; }
[data-testid="stMain"] button [data-testid="stMarkdownContainer"], [data-testid="stMain"] button p { color: inherit !important; }
</style>
"""


# Pages with a chat input render the main column inside "stAppScrollToBottomContainer"
# instead of "stMain", so every main-column rule is widened to cover both.
_MAIN_SCOPE = ':is([data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"])'
GLOBAL_CSS = GLOBAL_CSS.replace('[data-testid="stMain"]', _MAIN_SCOPE)
DARK_NATIVE_OVERRIDES = DARK_NATIVE_OVERRIDES.replace('[data-testid="stMain"]', _MAIN_SCOPE)


def _token_block(t: dict) -> str:
    return (
        f"--ahead-bg:{t['bg']};--ahead-card:{t['card']};--ahead-soft:{t['soft']};--ahead-border:{t['border']};"
        f"--ahead-title:{t['title']};--ahead-text:{t['text']};--ahead-muted:{t['muted']};"
        f"--ahead-accent:{t['accent']};--ahead-accent-strong:{t['accent_strong']};--ahead-accent-soft:{t['accent_soft']};"
        f"--ahead-danger:{t['danger']};--ahead-danger-soft:{t['danger_soft']};"
        f"--ahead-success:{t['success']};--ahead-success-soft:{t['success_soft']};"
        f"--ahead-warning:{t['warning']};--ahead-warning-soft:{t['warning_soft']};--ahead-shadow:{t['shadow']};"
    )


def inject_global_css() -> None:
    st.html(GLOBAL_CSS)
    st.html("<style>" + (Path(__file__).with_name("redesign.css")).read_text() + "</style>")


def apply_theme(force_light: bool = False) -> None:
    """Emit the colour tokens for the current appearance mode (plus native-widget overrides in dark mode)."""
    dark = (not force_light) and is_dark()
    t = DARK if dark else LIGHT
    st.html(f"<style>:root {{ {_token_block(t)} }}</style>")
    if dark:
        st.html(DARK_NATIVE_OVERRIDES)


def sidebar_dynamic_css(active_key: str) -> str:
    """Per-run CSS: nav icons (as data URIs) and the active item highlight."""
    rules = []
    for key, _label, icon, _doctor_only in NAV_ITEMS:
        rules.append(
            f'.st-key-nav_{key} .stButton > button::before {{ background-image:url("{icon_uri("sidebar/" + icon)}"); }}'
        )
    rules.append(
        f'.st-key-sidebar_logout .stButton > button::before {{ background-image:url("{icon_uri("sidebar/logout.svg")}"); }}'
    )
    rules.append(
        f".st-key-nav_{active_key} .stButton > button, .st-key-nav_{active_key} .stButton > button:hover {{"
        " background: linear-gradient(135deg, #2563EB, #6452D6) !important; color: #FFFFFF !important;"
        " font-weight: 650 !important; box-shadow: 0 4px 12px rgba(3,91,108,0.22) !important; }"
    )
    rules.append(
        f'.stApp .st-key-open_assistant button {{ background-image:url("{icon_uri("chatbot.svg")}") !important; }}'
    )
    return "<style>" + "\n".join(rules) + "</style>"
