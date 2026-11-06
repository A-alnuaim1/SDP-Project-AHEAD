"""
ahead/pages/overview.py
=======================
The signed-in home page. Patients get their personal dashboard; doctors get a
clinician dashboard built from consent-linked patients: summary cards, the
patients who need a look (urgent results first, sorted by ahead/triage.py),
quick screening and the screenings they ran.
"""

import re
from collections import Counter
from datetime import datetime, timedelta
from html import escape

import pandas as pd
import streamlit as st

from ahead.components import (condition_cards, current_user, disclaimer, go_to, markdown_literal, open_assistant, section_heading,
                              track_pill_html)
from ahead.config import DISEASES, model_label
from ahead.platform_data import analytics, local_now, local_today, plans, scheduled_doses, to_local
from ahead.resources import icon_uri
from ahead.triage import TRACKS, entered_later, missing_tests, triage_saved
from ahead.visuals import summary_card

HONORIFIC = re.compile(r"^(?:(?:dr|doctor|prof|mr|mrs|ms)(?:\.\s*|\s+|$))+", re.IGNORECASE)
# Trailing credentials and suffixes are not surnames ("Layla Hassan MD"); "DO" only in capitals, as Do is also a surname.
CREDENTIALS = {"md", "mbbs", "mbchb", "phd", "dds", "dmd", "frcs", "frcp", "frcpc", "mrcp", "mrcs", "facp", "facs", "mph", "msc", "jr", "sr", "ii", "iii"}
# Particles that belong to the family name after them ("Al Mansoori", "Abu Bakr", "bin Rashid", "van der Berg").
PARTICLES = {"al", "el", "bin", "bint", "ibn", "abu", "abd", "abdul", "abdel", "van", "von", "de", "der", "den", "del", "della", "da", "di", "du", "dos", "das", "la", "le"}
ATTENTION_ROWS = 5


def _focus(tab: str, patient: int | None = None) -> None:
    """Open Patient Intelligence on one tab (and patient); that page reads intelligence_focus."""
    st.session_state["intelligence_focus"] = {"tab": tab, "patient": patient}
    go_to("intelligence")


def _name_words(name: str) -> list[str]:
    """The words of a name without leading titles, a comma or trailing bracket tail, or trailing credentials."""
    words = re.sub(r"\s*\([^)]*\)?\s*$", "", HONORIFIC.sub("", (name or "").split(",")[0].strip())).split()
    while words and (words[-1].replace(".", "").lower() in CREDENTIALS or words[-1].replace(".", "") == "DO"):
        words.pop()
    return words


def _initials(name: str) -> str:
    letters = [m.group() for m in (re.search(r"[^\W\d_]", w) for w in _name_words(name)) if m]
    return (letters[0] + letters[-1] if len(letters) > 1 else "".join(letters) or "?").upper()


def _greeting_name(name: str) -> str:
    words = _name_words(name)
    start = len(words) - 1
    while start > 1 and words[start - 1].lower() in PARTICLES:  # the first word stays a given name
        start -= 1
    return f"Dr. {' '.join(words[start:])}" if words else "Doctor"


def _when(moment: datetime, today) -> str:
    if moment.date() == today:
        return f"today {moment:%H:%M}"
    if moment.date() == today - timedelta(days=1):
        return f"yesterday {moment:%H:%M}"
    return f"{moment.day} {moment:%b}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _care_snapshot(doctor_id: int) -> dict:
    """
    Linked patients, recent assessment counts and the patients needing a look: urgent first, then newest. A patient
    needs a look for an urgent or at-risk result in the last 30 days (sorted from the saved values, score and
    threshold, so an urgent value counts even when the model did not flag it, and a measured lab value in the usual
    range clears a model flag) or a missed dose today. Each item's
    "later" holds, per flagged assessment, the values a newer screening of the same condition entered, so a test the
    row would ask for is dropped once the patient has had it (triage.missing_tests).
    """
    people, assessments = analytics(doctor_id)
    today = local_today()
    later = entered_later(assessments)
    rows, week, flagged = {}, 0, 0
    for a in assessments:  # newest first
        moment = to_local(a["created_at"]).replace(tzinfo=None)
        week += moment.date() > today - timedelta(days=7)
        if moment.date() <= today - timedelta(days=30):
            continue
        track = triage_saved(a["disease"], a["values_json"], a["score"], a["threshold"])
        # Counted by track, as the rows are: a measured lab value can set or clear the model's flag.
        flagged += track["track"] in ("urgent", "at_risk")
        if track["track"] in ("urgent", "at_risk"):
            rows.setdefault(a["owner_id"], {"flags": [], "doses": []})["flags"].append((moment, a, track))
    missed = 0
    for p in people:
        # Only this doctor's active plans, as in Medication Monitoring; doses are read only when one exists.
        active = {x["id"] for x in plans(p["id"], doctor_id) if x["status"] == "active"}
        doses = [d for d in scheduled_doses(p["id"], today, today, doctor_id)
                 if d["plan_id"] in active and d["status"] == "missed"] if active else []
        if doses:
            missed += len(doses)
            rows.setdefault(p["id"], {"flags": [], "doses": []})["doses"] = doses
    names = {p["id"]: p["name"] or p["email"] for p in people}
    for row in rows.values():  # the row leads with the most pressing result, the newest one on that track
        row["flags"].sort(key=lambda flag: TRACKS.index(flag[2]["track"]))
    items = [{"id": pid, "name": names[pid], **row, "urgent": bool(row["flags"]) and row["flags"][0][2]["urgent"],
              "later": {a["id"]: later[a["id"]] for _, a, _ in row["flags"]},
              "latest": max([m for m, *_ in row["flags"]] + [datetime.fromisoformat(f"{d['date']}T{d['time']}") for d in row["doses"]])}
             for pid, row in rows.items() if pid in names]
    items.sort(key=lambda item: (item["urgent"], item["latest"]), reverse=True)
    return {"people": people, "items": items, "week": week, "flagged": flagged, "missed": missed,
            "flagged_ids": [i["id"] for i in items if i["flags"]],
            "review_ids": [i["id"] for i in items if i["flags"]], "urgent_ids": [i["id"] for i in items if i["urgent"]],
            "missed_ids": [i["id"] for i in items if i["doses"]]}


def _reason(track: dict) -> str:
    """The first urgent reason in the doctor's words, saying so plainly when the model did not flag the result."""
    text = track["reasons"][0]["doctor"]
    return f"The model did not flag this result, but {text[:1].lower()}{text[1:]}" if track["flagged"] is False else text


def _attention_row(item: dict, today) -> None:
    lines = []
    if item["flags"]:
        moment, a, track = item["flags"][0]
        label = DISEASES.get(a["disease"], {}).get("label", a["disease"].replace("_", " ").title())
        others = Counter("urgent" if t["urgent"] else "flagged" for *_, t in item["flags"][1:])
        more = "".join(f" · {others[word]} more {word}" for word in ("urgent", "flagged") if others[word])
        lines.append(f"{label} · {a['score'] * 100:.0f}% · {_when(moment, today)}{more}")
        if track["urgent"]:
            lines.append(_reason(track))
        else:   # at risk: a lab the patient left blank and has not entered since is the test to ask for (an urgent
            # row goes to a visit first)
            tests = missing_tests(a["disease"], a["values_json"], item["later"].get(a["id"], ()))
            lines += [test["line"] for test in tests]
    for name, count in Counter(d["name"] for d in item["doses"]).items():
        lines.append(f"{name} · {_plural(count, 'dose')} missed today")
    # The shared track pill (the same words as every other surface): Urgent, or At risk for a flag with no urgent value.
    flag = (track_pill_html("urgent", "doctor") if item["urgent"] else
            track_pill_html("at_risk", "doctor") if item["flags"] else "")
    pills = flag + ('<span class="attention-pill dose">Missed doses</span>' if item["doses"] else "")
    tone = "urgent" if item["urgent"] else "flag" if item["flags"] else "dose"
    with st.container(key=f"attention_row_{tone}_{item['id']}", horizontal=True, vertical_alignment="center"):
        st.html(
            f"""
<div class="attention-main">
    <span class="attention-avatar" aria-hidden="true">{escape(_initials(item['name']))}</span>
    <div class="attention-text"><strong class="attention-name">{escape(item['name'])}</strong>
    {''.join(f'<span class="attention-detail">{escape(line)}</span>' for line in lines)}</div>
    <div class="attention-pills">{pills}</div>
</div>
"""
        )
        # The patient's name in the label, so a list of these buttons is not just "Open, Open, Open" to a screen reader.
        first = (_name_words(item["name"]) or [item["name"]])[0]
        target = "Patient Records & History" if item["flags"] else "Medication Monitoring"
        if st.button(f"Open {markdown_literal(first)}", key=f"home_open_{item['id']}", help=f"Opens {target} in Patient Intelligence"):
            _focus(target, item["id"])
        # At risk with no urgent value: a prevention plan fits, so offer the shortcut to it. Urgent rows go to a visit first.
        if item["flags"] and not item["urgent"] and st.button(
                f"Plan for {markdown_literal(first)}", key=f"home_plan_{item['id']}",
                help="Opens the Prevention Plan tab in Patient Intelligence"):
            _focus("Prevention Plan", item["id"])


def _calm_card(icon: str, title: str, text: str, kind: str = "") -> None:
    st.html(
        f"""
<div class="attention-calm {kind}">
    <div class="summary-icon"><img src="{icon_uri(f'dashboard/{icon}.svg')}" alt=""></div>
    <div><strong>{escape(title)}</strong>
    <p>{escape(text)}</p></div>
</div>
"""
    )


def _render_doctor_home() -> None:
    doctor = st.session_state.user
    care = _care_snapshot(doctor["id"])
    people, items, today = care["people"], care["items"], local_today()
    hour = local_now().hour
    greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18 else "Good evening"
    if not people:
        copy = "Patients appear here once they add your doctor email in their settings."
    elif items:
        copy = f"{_plural(len(items), 'patient')} {'needs' if len(items) == 1 else 'need'} a look today."
    else:
        copy = "Nothing needs your attention right now."
    # ------------------------------------------------------------------ hero
    with st.container(key="home_hero"):
        st.html(
            f"""
<div class="home-hero">
    <div>
        <div class="home-eyebrow">AHEAD clinician dashboard</div>
        <h1>{greeting}, {escape(_greeting_name(doctor.get('name', '')))}</h1>
        <p>{escape(copy)}</p>
    </div>
    <div class="hero-badge"><small>Our mission</small>Small screenings.<br>Big futures.</div>
</div>
"""
        )
        primary_col, upload_col, _spacer = st.columns([0.26, 0.24, 0.5])
        with primary_col:
            if st.button("Patient intelligence", key="hero_intelligence", type="primary", width="stretch"):
                st.session_state.pop("intelligence_focus", None)  # open on the default tab, not an unread earlier focus
                go_to("intelligence")
        with upload_col:
            if st.button("Upload records", key="hero_upload", width="stretch"):
                go_to("clinical")

    # ------------------------------------------------------------------ summary cards
    def across(ids, none):
        # With no one linked, "No flagged results" would read as if patients had been checked.
        return f"Across {_plural(len(ids), 'patient')}" if ids else none if people else "No patients linked yet"

    def only(ids):
        return ids[0] if len(ids) == 1 else None

    urgent = len(care["urgent_ids"])
    cards = [
        ("patients", "Linked patients", len(people), "#8E7CC3",
         "Sharing their records with you" if people else "No patients linked yet", "Open overview", "Overview", None),
        ("flagged", "Flagged · 30 days", care["flagged"], "#C9706E",
         across(care["flagged_ids"], "No flagged results") + (f" · {urgent} urgent" if urgent else ""),
         "Review records", "Patient Records & History", only(care["review_ids"])),
        ("doses", "Missed doses today", care["missed"], "#C98F1E", across(care["missed_ids"], "All caught up today"),
         "Medication monitoring", "Medication Monitoring", only(care["missed_ids"])),
        ("activity", "Assessments · 7 days", care["week"], "#2E9477",
         "Saved by your patients" if people else "No patients linked yet", "View activity", "Overview", None),
    ]
    for col, (icon, label, value, color, note, action, tab, patient) in zip(st.columns(4), cards):
        with col:
            if summary_card(f"doctor_{icon}", label, value, note, action, color, icon):
                _focus(tab, patient)

    # ------------------------------------------------------------------ attention list
    section_heading("Needs your attention", "Urgent and flagged results from the last 30 days, urgent first, and missed doses "
                    "from linked patients.")
    if not people:
        _calm_card("patients", "No linked patients yet",
                   "A patient links to you by adding your doctor email under Settings → Preferences → Care team access. "
                   "Their urgent and flagged results and missed doses then appear here.", "neutral")
    elif not items:
        _calm_card("review", "Nothing needs your attention",
                   f"No urgent or flagged results in the last 30 days and no missed doses today across "
                   f"{_plural(len(people), 'linked patient')}.")
    else:
        for item in items[:ATTENTION_ROWS]:
            _attention_row(item, today)
        # Patient Intelligence has no list of these patients (nor of missed doses), so the rest stay here.
        if len(items) > ATTENTION_ROWS:
            with st.expander(_plural(len(items) - ATTENTION_ROWS, "more patient")):
                for item in items[ATTENTION_ROWS:]:
                    _attention_row(item, today)

    # ------------------------------------------------------------------ screening
    section_heading("Screen a patient", "Run a guided screening during a consultation. Results are saved to your account, not to a patient's record.")
    condition_cards("doctor_home")
    history = st.session_state.get("screening_history", [])
    if history:
        with st.expander("Screenings you ran"):
            table = pd.DataFrame(
                [
                    {
                        "Date": to_local(item["time"].isoformat()),
                        "Condition": item["label"],
                        "Model": model_label(item["model"]),
                        "Score": round(item["probability"] * 100, 1),
                        "Result": "Flagged" if item["prediction"] else "Not flagged",
                    }
                    for item in history[:10]
                ]
            )
            # Real datetimes and numbers (not text) so sorting by a header is chronological / numeric.
            st.dataframe(table, width="stretch", hide_index=True,
                         column_config={"Date": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
                                        "Score": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("Your most recent screenings, newest first (up to ten). Flagged means the model score reached that screening's threshold; it is not a diagnosis.")

    disclaimer(
        "AHEAD provides educational screening estimates only. It does not provide medical diagnoses "
        "and does not replace professional medical evaluation.",
        label="Clinical notice",
    )


def render_overview() -> None:
    user = current_user()
    if user["is_doctor"]:
        _render_doctor_home()
        return

    dashboard_copy = "Build healthier habits, keep track of your care, and make time for prevention."
    # ------------------------------------------------------------------ hero
    with st.container(key="home_hero"):
        st.html(
            f"""
<div class="home-hero">
    <div>
        <div class="home-eyebrow">AHEAD Health Dashboard</div>
        <h1>Welcome back, {escape(user['first_name'])}</h1>
        <p>{escape(dashboard_copy)}</p>
    </div>
    <div class="hero-badge"><small>Our mission</small>Small screenings.<br>Big futures.</div>
</div>
"""
        )
        action_col, ask_col, _spacer = st.columns([0.26, 0.24, 0.5])
        with action_col:
            with st.popover("Start Screening", width="stretch", type="primary"):
                for disease, info in DISEASES.items():
                    if st.button(info["card_title"], key=f"hero_{disease}", width="stretch"):
                        go_to("screenings", disease)
        with ask_col:
            if st.button("Ask AHEAD Assistant", key="hero_ask", width="stretch"):
                open_assistant("overview")

    from ahead.pages.patient_dashboard import render_patient_dashboard
    render_patient_dashboard()
