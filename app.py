"""
AHEAD — Advanced Health Early Awareness and Disease Detection System
Clinical Screening Portal · Senior Design Project

Entry point: page configuration, routing and the sidebar.  Everything else
lives in the ``ahead`` package:

    ahead/config.py      static configuration (diseases, form layout, labels, nav)
    ahead/resources.py   cached models, metadata, datasets, images, Gemini
    ahead/theme.py       colour tokens, global CSS, Plotly styling
    ahead/components.py  shared UI pieces and session-state helpers
    ahead/screening.py   the guided screening flow
    ahead/pages/         one module per page
"""

from html import escape
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AHEAD | Clinical Screening",
    page_icon=str(Path(__file__).resolve().parent / "assets" / "favicon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from ahead.components import current_user, go_to, init_state, logout, open_assistant  # noqa: E402
from ahead.config import NAV_ITEMS  # noqa: E402
from ahead.pages.analytics import render_analytics  # noqa: E402
from ahead.pages.assistant import render_assistant  # noqa: E402
from ahead.pages.clinical import render_clinical_dashboard  # noqa: E402
from ahead.pages.overview import render_overview  # noqa: E402
from ahead.pages.public import PUBLIC_CONTENT, render_login, render_public_page  # noqa: E402
from ahead.pages.screenings import render_screenings  # noqa: E402
from ahead.pages.settings import render_settings  # noqa: E402
from ahead.theme import apply_theme, inject_global_css, sidebar_dynamic_css  # noqa: E402
from ahead.storage import init_db  # noqa: E402
from ahead.i18n import tr  # noqa: E402
from ahead.platform_data import init_platform, notifications  # noqa: E402
from ahead.pages.wellness import render_wellness  # noqa: E402
from ahead.pages.food import render_food  # noqa: E402
from ahead.pages.medications import render_medications  # noqa: E402
from ahead.pages.notifications import render_notifications  # noqa: E402
from ahead.pages.notifications import refresh_patient_notifications  # noqa: E402
from ahead.pages.doctor_intelligence import render_doctor_intelligence  # noqa: E402
from ahead.pages.voice import render_voice  # noqa: E402

init_db()
init_platform()
init_state()
inject_global_css()

# =============================================================================
# PUBLIC ROUTING  (?page=login|about|contact|help)
# =============================================================================

requested = st.query_params.get("page", "login")
if requested not in PUBLIC_CONTENT:
    requested = "login"

if not st.session_state.authenticated:
    apply_theme(force_light=True)
    if requested == "login":
        render_login()
    else:
        render_public_page(requested)
    st.stop()

# Signed in: public query params are meaningless, drop them.
if "page" in st.query_params:
    st.query_params.clear()

apply_theme()
user = current_user()
if st.session_state.get('pref_language')=='العربية':
    st.html('<style>[data-testid="stMainBlockContainer"], [data-testid="stSidebarContent"] {direction:rtl;text-align:right;} input,textarea {text-align:right;} [data-testid="stHorizontalBlock"] {direction:rtl;}</style>')

# =============================================================================
# SIDEBAR
# =============================================================================

patient_only = {'wellness','food','medications','notifications','voice'}
visible_nav = [item for item in NAV_ITEMS if (not item[3] or user["is_doctor"])
               and (not user["is_doctor"] or item[0] not in patient_only)]
nav_keys = {item[0] for item in visible_nav}
if st.session_state.nav not in nav_keys:
    st.session_state.nav = "overview"
active = st.session_state.nav

with st.sidebar:
    st.html(
        """
<div class="sidebar-brand">
    <div class="brand-symbol">✚</div>
    <div>
        <h2>AHEAD</h2>
        <p>Clinical Screening &amp;<br>Early Awareness Platform</p>
    </div>
</div>
<div class="sidebar-divider"></div>
<div class="sidebar-section-label">Menu</div>
"""
    )

    with st.container(key="sidebar_nav"):
        if not user['is_doctor']:
            refresh_patient_notifications(st.session_state.user['id'])
            unread=sum(not n['read_at'] for n in notifications(st.session_state.user['id']))
            if st.button(f"🔔 {tr('Notifications')} ({unread})",key='notification_badge',width='stretch'):
                go_to('notifications')
        previous_group = None
        for key, label, _icon, _doctor_only in visible_nav:
            group = ('Care & insights' if key in ('clinical','intelligence','analytics') else
                     'My health' if key in ('wellness','food','medications') else
                     'Support & account' if key in ('notifications','voice','assistant','settings') else 'Workspace')
            if group != previous_group:
                st.html(f'<div class="sidebar-section-label">{tr(group)}</div>')
                previous_group = group
            if st.button(tr(label), key=f"nav_{key}", width="stretch"):
                if key == "assistant":
                    open_assistant(active)
                go_to(key)

    st.html(sidebar_dynamic_css(active))

    with st.container(key="sidebar_bottom"):
        st.html(
            f"""
<div class="sidebar-user-profile">
    <div class="sidebar-user-avatar">{escape(user['initials'])}</div>
    <div class="sidebar-user-info">
        <div class="sidebar-user-name">{escape(user['name'])}</div>
        <div class="sidebar-user-role">{user['role_label']}</div>
    </div>
</div>
"""
        )
        if st.button(tr("Log Out"), key="sidebar_logout", width="stretch"):
            logout()

# =============================================================================
# FLOATING ASSISTANT BUTTON
# =============================================================================

if active != "assistant":
    with st.container(key="open_assistant"):
        if st.button(tr("Open AHEAD Assistant"), key="open_assistant_button"):
            open_assistant(active)

# =============================================================================
# PAGES
# =============================================================================

PAGES = {
    "overview": render_overview,
    "screenings": render_screenings,
    "wellness":render_wellness,
    "food":render_food,
    "medications":render_medications,
    "notifications":render_notifications,
    "voice":render_voice,
    "analytics": render_analytics,
    "clinical": render_clinical_dashboard,
    "intelligence":render_doctor_intelligence,
    "assistant": render_assistant,
    "settings": render_settings,
}

if (active in ('clinical','intelligence') and not user['is_doctor']) or (active in patient_only and user['is_doctor']):
    st.error("Doctor access required.")
else:
    PAGES[active]()
