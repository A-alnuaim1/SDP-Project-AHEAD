"""
ahead/pages/settings.py
=======================
Profile · Appearance · Preferences · Account — tabbed settings inspired by
a conventional account-settings layout.
"""

from html import escape

import streamlit as st

from ahead.components import current_user, logout, pref, page_header
from ahead.theme import APPEARANCE_OPTIONS, appearance_mode
from ahead.storage import change_password, delete_records, delete_screenings, update_profile, update_preferences
from ahead.i18n import tr
from ahead.platform_data import doctors_for, grant_doctor, revoke_doctor, ai_consent


def _row(icon: str, title: str, copy: str) -> str:
    return f"""
<div class="settings-row">
    <div class="settings-row-icon">{icon}</div>
    <div>
        <div class="settings-row-title">{escape(title)}</div>
        <div class="settings-row-copy">{escape(copy)}</div>
    </div>
</div>"""


def _panel(title: str, copy: str) -> None:
    st.html(f'<div class="settings-panel"><h2>{escape(title)}</h2><p>{escape(copy)}</p></div>')


def _sync_pref(name: str) -> None:
    st.session_state.prefs[name] = st.session_state[f"pref_widget_{name}"]
    _save_prefs()


def _save_prefs() -> None:
    user = st.session_state.get("user")
    if user:
        update_preferences(user["id"], {
            "toggles": st.session_state.prefs,
            "language": st.session_state.get("pref_language", "English"),
            "appearance": st.session_state.get("appearance_mode", "Light"),
        })


def _toggle_row(icon: str, title: str, copy: str, name: str) -> None:
    """A preference row. The toggle is fed from st.session_state.prefs and writes back on change."""
    left, right = st.columns([0.86, 0.14])
    with left:
        st.html(_row(icon, title, copy))
    with right:
        st.toggle(
            title, value=pref(name), key=f"pref_widget_{name}", label_visibility="collapsed",
            on_change=_sync_pref, args=(name,),
        )


# =============================================================================
# TABS
# =============================================================================

def _profile_tab(user: dict) -> None:
    _panel("Account information", "Review and update the information associated with your AHEAD profile.")
    st.html(
        f"""
<div class="settings-profile-banner">
    <div class="settings-avatar">{escape(user['initials'])}</div>
    <div>
        <div class="settings-person-name">{escape(user['name'])}</div>
        <div class="settings-person-role">{user['role_label']}</div>
        <div class="settings-person-email">{escape(user['email'])}</div>
    </div>
</div>
"""
    )
    with st.form("settings_profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full name", value=user["name"])
            st.text_input("Role", value=user["role_label"], disabled=True)
        with c2:
            email = st.text_input("Email", value=user["email"])
            institution = st.text_input(
                "Institution", value=user["institution"], disabled=not user["is_doctor"],
                placeholder="Not applicable for patient accounts" if not user["is_doctor"] else "",
            )
        if st.form_submit_button("Save changes", type="primary"):
            if not full_name.strip():
                st.error("Please enter your name.")
            elif "@" not in email or "." not in email.split("@")[-1]:
                st.error("Please enter a valid email address.")
            else:
                try:
                    update_profile(st.session_state.user["id"], full_name, email, institution)
                except Exception:
                    st.error("Unable to save this profile. The email may already be used.")
                else:
                    st.session_state.user.update(
                        {"name": full_name.strip(), "email": email.strip().lower(), "institution": institution.strip()}
                    )
                    st.toast("Profile updated.", icon="✅")
                    st.rerun()


def _appearance_tab() -> None:
    _panel("Display mode", "Choose between a bright clinical interface and a darker low-light interface.")
    current = appearance_mode()
    columns = st.columns(3)
    for column, (mode, (css, copy)) in zip(columns, APPEARANCE_OPTIONS.items()):
        active = mode == current
        with column:
            st.html(
                f"""
<div class="appearance-choice {'active' if active else ''}">
    <div class="appearance-preview {css}"></div>
    <strong>{mode}{'<span class="check">✓</span>' if active else ''}</strong>
    <span>{copy}</span>
</div>
"""
            )
            if st.button(
                "Selected" if active else f"Use {mode}", key=f"appearance_{mode}", width="stretch",
                type="primary" if active else "secondary", disabled=active,
            ):
                st.session_state.appearance_mode = mode
                _save_prefs()
                st.rerun()
    st.caption("The display mode applies to this browser session and switches charts and every AHEAD element immediately.")


def _preferences_tab() -> None:
    _panel("Display preferences", "Notification delivery is not available in this prototype.")
    with st.container(key="settings_rows"):
        st.info("No email, text or push notifications are sent.")

    _panel("Data & privacy", "Screening history and doctor review records are saved in the configured local database.")
    with st.container(key="settings_rows_privacy"):
        _toggle_row("◈", "Keep screening history", "Save future results and show them on the Overview page.", "keep_history")
    history = st.session_state.get("screening_history", [])
    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        if st.button(f"Clear session history ({len(history)})", disabled=not history, width="stretch"):
            st.session_state.screening_history = []
            st.session_state.last_results = {}
            delete_screenings(st.session_state.user["id"])
            st.toast("Screening history cleared.")
            st.rerun()

    _panel("Language", "Choose the language for patient screening and navigation. Doctor tools and exported PDF remain in English.")
    st.selectbox("Application language / لغة التطبيق", ["English", "العربية"],
                 key="pref_language", on_change=_save_prefs)
    user=st.session_state.get('user') or {}
    if user.get('role')=='Patient':
        _panel('Care team access','You decide which doctor accounts may view your saved history and assign medication. You can revoke access at any time.')
        with st.form('grant_doctor'):
            address=st.text_input('Doctor account email')
            if st.form_submit_button('Add doctor'):
                if grant_doctor(user['id'],address):st.success('Doctor linked.');st.rerun()
                else:st.error('No doctor account found with this email.')
        for doctor in doctors_for(user['id']):
            st.write(f"{doctor['name']} · {doctor['email']}")
            st.checkbox('Allow sending my linked records to Gemini for optional AI summaries',value=bool(doctor['ai_allowed']),
                 key=f"ai_consent_{doctor['id']}",on_change=lambda did=doctor['id']:ai_consent(user['id'],did,st.session_state[f'ai_consent_{did}']))
            if st.button('Remove access',key=f"revoke_{doctor['id']}"):
                revoke_doctor(user['id'],doctor['id']);st.rerun()
    if user := st.session_state.get("user"):
        if user["role"] == "Doctor" and st.button("Delete my saved clinical records", type="secondary"):
            delete_records(user["id"])
            st.success("Your clinical review records were deleted.")
            st.rerun()


def _account_tab(user: dict) -> None:
    _panel("Security & account", "Manage how you sign in to AHEAD and end your session.")

    with st.expander("🔒  Change password — update your password regularly"):
        with st.form("settings_password_form"):
            current_pw = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                if not current_pw or not new_pw:
                    st.error("Please fill in every field.")
                elif len(new_pw) < 12:
                    st.error("Use at least 12 characters.")
                elif new_pw != confirm_pw:
                    st.error("The new passwords do not match.")
                elif not change_password(st.session_state.user["id"], current_pw, new_pw):
                    st.error("The current password is incorrect.")
                else:
                    st.success("Password updated.")

    with st.expander("🖥  Manage sessions — view and manage active sessions"):
        st.html(_row("●", "This browser", f"Signed in as {user['email']} · active now"))
        st.caption("AHEAD keeps no server-side sessions; signing out ends this one.")

    st.html(
        """
<div class="settings-danger">
    <strong>Log out</strong>
    <p>Sign out of your account. Your session screening history will be cleared.</p>
</div>
"""
    )
    if st.button("Log out", key="settings_logout", type="primary"):
        logout()

    st.html(
        """
<div class="settings-panel">
    <h2>About AHEAD</h2>
    <p>Advanced Health Early Awareness and Disease Detection — a Senior Design machine-learning screening
    prototype for diabetes, cardiovascular disease and chronic kidney disease. It provides educational
    screening estimates and does not diagnose disease.</p>
</div>
"""
    )


# =============================================================================
# PAGE
# =============================================================================

def render_settings() -> None:
    user = current_user()
    page_header('Your account','Settings','Personalize your experience, privacy and care team access.')
    with st.container(key="settings_tabs"):
        profile, appearance, preferences, account = st.tabs(["Profile", "Appearance", "Preferences", "Account"])
    with profile:
        _profile_tab(user)
    with appearance:
        _appearance_tab()
    with preferences:
        _preferences_tab()
    with account:
        _account_tab(user)
