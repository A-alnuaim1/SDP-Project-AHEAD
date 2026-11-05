"""The SORT step on the doctor side: urgent patients first on the doctor home (shared pill, first reason, the Flagged
card's urgent count), the Track column and caption in Patient Intelligence, and the Clinical Dashboard queue's Track
column, filter, CSV, Records tile, record panel and PDF track. Every check runs on a temporary database."""
import html
import io
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import streamlit
from streamlit.testing.v1 import AppTest

import ahead.pages.clinical as clinical
import ahead.pages.overview as overview
from ahead.clinical_data import prepare_upload
from ahead.components import track_pill_html
from ahead.platform_data import grant_doctor, init_platform
from ahead.resources import best_model, feature_summary, features
from ahead.storage import (authenticate, connection, create_user, init_db, insert_records, list_records, save_screening,
                           update_record)
from ahead.triage import LABELS, TRACKS, track_text, triage

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DOCTOR_LABELS = [LABELS[track]["doctor"][0] for track in TRACKS]   # Urgent, At risk, Not flagged, No score
CLEAR = {"age": 40, "HbA1c_level": 5.5, "blood_glucose_level": 140}   # below every diabetes cut-off
NOT_FLAGGED_BUT = "The model did not flag this result, but the patient's "


def reason(disease, values, score, threshold=0.5):
    """The first urgent reason in the doctor's words, as triage() writes it."""
    return triage(disease, values, score, threshold)["reasons"][0]["doctor"]


def typical_row(disease, **changes):
    stats, age_key = feature_summary(disease), "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    row = {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"] if stats[f]["binary"]
           else round(stats[f]["median"], 2) for f in features(disease) if f != age_key}
    return {"PatientID": "DEMO-001", "DOB": "1980-01-01", **row, **changes}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/app.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        create_user("d@test.com", "safe-password-123", "Doctor", "Dr. Layla Hassan")
        self.d = authenticate("d@test.com", "safe-password-123", "Doctor")

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def page(self, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": self.d, "nav": nav, **state}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    def html(self, at):
        return "\n".join(h.proto.body for h in at.get("html") if not h.proto.body.lstrip().startswith("<style"))

    def frame(self, at, column):
        return next(df.value for df in at.dataframe if column in df.value.columns)

    def widget(self, items, label):
        return next(w for w in items if w.label == label)

    def patient(self, email, name, link=True):
        create_user(email, "safe-password-123", "Patient", name)
        user = authenticate(email, "safe-password-123", "Patient")
        if link:
            grant_doctor(user["id"], "d@test.com")
        return user

    def screening(self, patient, score, values, disease="diabetes", days_ago=0):
        sid = save_screening(patient["id"], disease, best_model(disease), score, int(score >= 0.5), 0.5, values)
        if days_ago:
            with connection() as db:
                db.execute("UPDATE screenings SET created_at=? WHERE id=?",
                           ((datetime.now(timezone.utc) - timedelta(days=days_ago, hours=1)).isoformat(), sid))
        return sid


# =============================================================================
# DOCTOR HOME
# =============================================================================

class DoctorHomeTriageTests(Fixture):
    def rows(self, at):
        """(name, detail lines, pills) of each attention row, in page order."""
        found = re.findall(r'<strong class="attention-name">([^<]+)</strong>(.*?)</div>\s*<div class="attention-pills">(.*?)</div>',
                           self.html(at), re.S)
        return [(html.unescape(name), [html.unescape(d) for d in re.findall(r'attention-detail">([^<]*)<', lines)], pills)
                for name, lines, pills in found]

    def test_urgent_patients_come_first_with_the_shared_pill_and_their_first_reason(self):
        amy, ben, cara = (self.patient(f"{n}@test.com", name) for n, name in (("a", "Amy Lee"), ("b", "Ben Cole"), ("c", "Cara Diaz")))
        eve, fay = self.patient("e@test.com", "Eve Calm"), self.patient("f@test.com", "Fay Old")
        dan = self.patient("x@test.com", "Dan Unlinked", link=False)
        urgent = {"age": 52, "HbA1c_level": 7.1, "blood_glucose_level": 212}
        self.screening(amy, 0.8, CLEAR)                                       # at risk, just now
        self.screening(ben, 0.9, urgent, days_ago=2)                          # urgent and flagged
        self.screening(ben, 0.7, CLEAR, days_ago=1)                           # a newer at-risk result
        self.screening(cara, 0.2, {"age": 60, "HbA1c_level": 7.0}, days_ago=3)   # urgent by a value rule only
        self.screening(eve, 0.2, CLEAR)
        self.screening(fay, 0.9, urgent, days_ago=40)                         # older than 30 days
        self.screening(dan, 0.9, urgent)                                      # never linked
        at = self.page("overview")
        page = self.html(at)
        rows = self.rows(at)
        # Urgent first (newest first among them), then the rest newest first; no one unlinked, calm or older.
        self.assertEqual([name for name, *_ in rows], ["Ben Cole", "Cara Diaz", "Amy Lee"])
        for name in ("Dan", "Eve", "Fay"):
            self.assertNotIn(name, page)
        self.assertIn("3 patients need a look today.", page)
        pill = track_pill_html("urgent", "doctor")
        self.assertEqual(pill, '<span class="track-pill urgent" title="Urgent">Urgent</span>')
        self.assertEqual([pills.strip() for *_, pills in rows],
                         [pill, pill, track_pill_html("at_risk", "doctor")])
        # The row leads with the urgent result although a newer at-risk one exists, and counts the rest.
        ben_lines, cara_lines, amy_lines = (lines for _, lines, _ in rows)
        self.assertRegex(ben_lines[0], r"^Diabetes · 90% · \d+ \w{3} · 1 more flagged$")
        self.assertEqual(ben_lines[1], reason("diabetes", urgent, 0.9))
        self.assertIn("a doctor needs to confirm it with tests", ben_lines[1])
        # A rule-only urgent result says plainly that the model did not flag it.
        self.assertRegex(cara_lines[0], r"^Diabetes · 20% · \d+ \w{3}$")
        self.assertEqual(cara_lines[1], NOT_FLAGGED_BUT + "HbA1c of 7% is at or above 6.5%, a level used when diagnosing "
                                         "diabetes; a doctor needs to confirm it with tests.")
        self.assertRegex(amy_lines[0], r"^Diabetes · 80% · today \d\d:\d\d$")
        self.assertEqual(len(amy_lines), 1)
        self.assertNotRegex(page, r"\b[Yy]our (HbA1c|blood)")
        # The Flagged card counts flagged results (Amy 1, Ben 2) and adds the urgent patients (Ben, Cara).
        card = re.search(r'summary-label">Flagged · 30 days</span><strong class="summary-value">(\d+)</strong>\s*'
                         r'<p class="summary-note">([^<]+)</p>', page)
        self.assertEqual(card.groups(), ("3", "Across 2 patients · 2 urgent"))
        self.assertIn("Urgent and flagged results from the last 30 days, urgent first", page)

    def test_a_single_rule_only_urgent_patient_fills_the_card_note_and_its_click(self):
        cara = self.patient("c@test.com", "Cara Diaz")
        self.screening(cara, 0.2, {"age": 60, "blood_glucose_level": 250})
        page = self.html(self.page("overview"))
        self.assertRegex(page, r'summary-label">Flagged · 30 days</span><strong class="summary-value">0</strong>\s*'
                               r'<p class="summary-note">No flagged results · 1 urgent</p>')
        self.assertEqual(page.count('class="track-pill urgent"'), 1)
        self.assertNotIn("Nothing needs your attention", page)
        # The card opens that one patient's records, like a single flagged patient.
        seen, real = [], overview.go_to

        def spy(nav, *args, **kwargs):
            seen.append(dict(overview.st.session_state["intelligence_focus"]))
            return real(nav, *args, **kwargs)
        at = self.page("overview")
        with mock.patch.object(overview, "go_to", spy):
            self.ok(at.button(key="doctor_flagged").click().run())
        self.assertEqual(seen, [{"tab": "Patient Records & History", "patient": cara["id"]}])

    def test_flagged_results_without_urgent_values_keep_the_card_note_unchanged(self):
        self.screening(self.patient("a@test.com", "Amy Lee"), 0.8, CLEAR)
        page = self.html(self.page("overview"))
        self.assertIn('<p class="summary-note">Across 1 patient</p>', page)
        self.assertNotIn("track-pill urgent", page)
        self.assertIn('track-pill at-risk" title="At risk">At risk', page)

    def test_reason_line_helper(self):
        flagged = triage("kidney", {"GFR": 45}, 0.8, 0.5)
        below = triage("kidney", {"GFR": 45}, 0.2, 0.5)
        self.assertEqual(overview._reason(flagged), flagged["reasons"][0]["doctor"])
        self.assertEqual(overview._reason(below), "The model did not flag this result, but the patient's eGFR of 45 is "
                                                  "below 60, a level used when diagnosing chronic kidney disease; a doctor "
                                                  "needs to confirm it with tests.")


# =============================================================================
# PATIENT INTELLIGENCE
# =============================================================================

class IntelligenceTriageTests(Fixture):
    def test_track_column_caption_and_the_urgent_now_count(self):
        p, q = self.patient("p@test.com", "Alice Mansour"), self.patient("q@test.com", "Quinn Ross")
        kidney = self.screening(p, 0.3, {"Age": 50, "GFR": 45, "ACR": 10}, "kidney", days_ago=2)   # urgent, model below
        flagged = self.screening(p, 0.8, {"age": 50, "HbA1c_level": 7.2}, days_ago=1)                # urgent and flagged
        self.screening(p, 0.2, CLEAR)            # newest diabetes: not flagged, but kidney is still urgent
        self.screening(q, 0.8, {"age": 50, "blood_glucose_level": 230}, days_ago=1)
        self.screening(q, 0.7, CLEAR)            # a later diabetes result replaces q's urgent one
        at = self.page("intelligence", intelligence_focus={"tab": "Patient Records & History", "patient": p["id"]})
        page = self.html(at)
        # Overview: the Flagged tile's note counts patients whose latest result for a condition is urgent (p only).
        self.assertRegex(page, r'summary-label">Flagged</span><strong class="summary-value">3</strong>\s*'
                               r'<p class="summary-note">At or above the model threshold, all time · 1 patient urgent now</p>')
        recent = self.frame(at, "Patient")
        self.assertEqual(list(recent["Track"]), ["At risk", "Not flagged", "Urgent", "Urgent", "Urgent"])
        # Disease Analytics (diabetes) lists the same words.
        self.assertTrue(set(self.frame(at, "Threshold (%)")["Track"]) <= set(DOCTOR_LABELS))
        # Patient Records & History: a Track column beside Result, newest first.
        saved = self.frame(at, "ID")
        self.assertEqual(list(saved.columns), ["ID", "Date", "Condition", "Model", "Score (%)", "Threshold (%)", "Result", "Track"])
        self.assertEqual(list(zip(saved["Condition"], saved["Result"], saved["Track"])),
                         [("Diabetes", "Not flagged", "Not flagged"), ("Diabetes", "Flagged", "Urgent"),
                          ("Kidney Health", "Not flagged", "Urgent")])
        captions = lambda: [c.value for c in at.caption]
        self.assertTrue(any(c.endswith("· Not flagged · Track: Not flagged.") for c in captions()))
        self.widget(at.selectbox, "Open historical assessment").set_value(kidney)
        self.ok(at.run())
        expected = (NOT_FLAGGED_BUT + "eGFR of 45 is below 60, a level used when diagnosing chronic kidney disease; a doctor "
                    "needs to confirm it with tests.")
        self.assertIn(f"score 30.0% against a 50.0% threshold · Not flagged · Track: Urgent. {expected}", "\n".join(captions()))
        self.widget(at.selectbox, "Open historical assessment").set_value(flagged)
        self.ok(at.run())
        caption = next(c for c in captions() if "Track:" in c)
        self.assertTrue(caption.endswith("· Flagged · Track: Urgent. " + reason("diabetes", {"HbA1c_level": 7.2}, 0.8)), caption)
        self.assertNotIn("did not flag", caption)

    def test_no_urgent_patient_leaves_the_flagged_tile_note_as_it_was(self):
        self.screening(self.patient("p@test.com", "Alice Mansour"), 0.8, CLEAR)
        page = self.html(self.page("intelligence"))
        self.assertIn('<p class="summary-note">At or above the model threshold, all time</p>', page)


# =============================================================================
# CLINICAL DASHBOARD
# =============================================================================

class ClinicalTriageTests(Fixture):
    ROWS = [("URG-READY", {"HbA1c_level": "7.1"}, None),     # complete, unscored, urgent before scoring
            ("PLAIN", {}, None),                             # complete, unscored, no rule
            ("AT-RISK", {}, 0.8),
            ("CLEAR", {}, 0.2),
            ("URG-LOW", {"HbA1c_level": "6.8"}, 0.3),        # urgent though the model did not flag it
            ("INCOMPLETE", {"HbA1c_level": "7.5", "bmi": ""}, None),   # incomplete, but its HbA1c is checked and urgent
            ("=CMD()", {"blood_glucose_level": "250"}, 0.9),
            ("BLANK", {"bmi": ""}, None),                    # incomplete, no rule: no track
            ("TYPO", {"blood_glucose_level": "420"}, None)]  # past the model's input range, so incomplete, yet urgent

    def setUp(self):
        super().setUp()
        frame = pd.DataFrame([typical_row("diabetes", PatientID=pid, **changes) for pid, changes, _ in self.ROWS]).astype(str)
        insert_records(self.d["id"], "cohort", "diabetes", prepare_upload(frame, "diabetes", features("diabetes"),
                       feature_summary("diabetes"), ["PatientID"]), source_name="clinic.csv")
        self.records = {r["patient_id"]: r for r in list_records(self.d["id"], "diabetes")}
        for pid, _, score in self.ROWS:
            if score is not None:
                update_record(self.d["id"], self.records[pid]["id"], score=score, threshold=0.5, model=best_model("diabetes"),
                              status="Flagged" if score >= 0.5 else "Ready")
        self.records = {r["patient_id"]: r for r in list_records(self.d["id"], "diabetes")}

    def queue(self, at):
        return self.frame(at, "Model score (%)")

    def filter(self, at, show="All", track="All"):
        self.widget(at.selectbox, "Show").set_value(show)
        self.widget(at.selectbox, "Track").set_value(track)
        self.ok(at.run())
        return [] if not [df for df in at.dataframe if "Model score (%)" in df.value.columns] else list(self.queue(at)["Patient"])

    def test_queue_track_column_filter_tiles_and_csv(self):
        exports, real = [], streamlit.download_button

        def capture(label, data, *args, **kwargs):
            if kwargs.get("file_name", "").endswith("_review.csv"):
                exports.append(data.decode("utf-8-sig"))
            return real(label, data, *args, **kwargs)
        with mock.patch.object(streamlit, "download_button", capture):
            at = self.page("clinical")
        queue = self.queue(at)
        self.assertEqual(list(queue.columns), ["Row", "Patient", "Status", "Track", "Model score (%)", "Issues", "Model"])
        self.assertEqual(dict(zip(queue["Patient"], zip(queue["Status"], queue["Track"]))), {
            "URG-READY": ("Ready", "Urgent"), "PLAIN": ("Ready", "No score"), "AT-RISK": ("Flagged", "At risk"),
            "CLEAR": ("Not flagged", "Not flagged"), "URG-LOW": ("Not flagged", "Urgent"),
            "INCOMPLETE": ("Incomplete", "Urgent"), "=CMD()": ("Flagged", "Urgent"), "BLANK": ("Incomplete", ""),
            "TYPO": ("Incomplete", "Urgent")})
        # The Records tile counts the rows the Track filter calls Urgent; the grid keeps its six tiles.
        tiles = re.findall(r'summary-label">([^<]+)</span><strong class="summary-value">.*?<p class="summary-note">([^<]*)</p>',
                           self.html(at), re.S)
        self.assertEqual(len(tiles), 6)
        self.assertEqual(tiles[0], ("Records", "Rows in this upload · 5 urgent"))
        # The CSV carries the Track column, and the formula guard still covers the patient ID.
        export = pd.read_csv(io.StringIO(exports[-1]), keep_default_na=False)
        self.assertEqual(list(export.columns), list(queue.columns))
        self.assertEqual(list(export["Track"]), list(queue["Track"]))
        self.assertIn("'=CMD()", list(export["Patient"]))
        # The Track filter sits next to Show, offers the shared labels and combines with Show.
        self.assertEqual(self.widget(at.selectbox, "Track").options, ["All", *DOCTOR_LABELS])
        self.assertEqual(self.filter(at, track="Urgent"), ["URG-READY", "URG-LOW", "INCOMPLETE", "=CMD()", "TYPO"])
        self.assertEqual(self.filter(at, "Not flagged", "Urgent"), ["URG-LOW"])
        self.assertEqual(self.filter(at, track="At risk"), ["AT-RISK"])
        self.assertEqual(self.filter(at, "Ready", "No score"), ["PLAIN"])
        self.assertEqual(self.filter(at, "Incomplete", "Urgent"), ["INCOMPLETE", "TYPO"])
        self.assertEqual(self.filter(at, "Incomplete", "At risk"), [])
        self.assertIn("No records in this upload match Show: Incomplete and Track: At risk.", [c.value for c in at.caption])
        self.widget(at.selectbox, "Show").set_value("All")
        update_record(self.d["id"], self.records["AT-RISK"]["id"], score=0.2, status="Ready")
        self.assertEqual(self.filter(at, track="At risk"), [])
        self.assertIn("No records in this upload are on the At risk track.", [c.value for c in at.caption])

    def test_record_view_shows_the_doctor_panel_and_hands_the_track_to_the_pdf(self):
        calls = []

        def fake_report(*args, **kwargs):   # stands in for patient_report; only its track argument is checked here
            calls.append(kwargs)
            return b"%PDF-1.4 test"

        def panels(at):
            return re.findall(r'<div class="track-panel ([\w-]+)">', self.html(at))
        with mock.patch.object(clinical, "patient_report", fake_report):
            at = self.page("clinical")
            # Row 1 is complete but unscored: the urgent rules already apply, in the doctor's words.
            page = self.html(at)
            self.assertEqual(panels(at), ["urgent"])
            self.assertIn("There is no model score for this result, but an entered value is already in a range doctors use "
                          "when diagnosing a condition.", page)
            self.assertIn(html.escape(reason("diabetes", {"HbA1c_level": 7.1}, None)), page)
            self.assertNotRegex(page, r"\bYour\b")
            self.assertEqual(calls, [])   # no score, no PDF
            for pid, expected in [("URG-LOW", "urgent"), ("CLEAR", "clear"), ("AT-RISK", "at-risk"), ("PLAIN", None),
                                  ("INCOMPLETE", "urgent"), ("TYPO", "urgent"), ("BLANK", None)]:
                with self.subTest(pid=pid):
                    record = self.records[pid]
                    at.selectbox(key="clinical_record_diabetes_cohort").set_value(record["id"])
                    self.ok(at.run())
                    self.assertEqual(panels(at), [expected] if expected else [])
                    # Only a reason resting on a value that failed validation asks for a check against the source.
                    notes = re.findall(r'<p class="track-panel-note">([^<]*)</p>', self.html(at))
                    self.assertEqual([html.unescape(n) for n in notes], [
                        "Blood glucose (420 mg/dL) failed the input check above and may be an entry error. Check it "
                        "against the source record before acting on it."] if pid == "TYPO" else [])
                    if record["score"] is None:
                        continue
                    track = calls[-1]["track"]
                    values = json.loads(record["values_json"])
                    self.assertEqual(track, triage("diabetes", values, record["score"], record["threshold"]))
                    self.assertEqual(track_text(track, "doctor")["label"],
                                     {"urgent": "Urgent", "clear": "Not flagged", "at-risk": "At risk"}[expected])
            page = self.html(at)
        self.assertNotIn("track-panel", page)   # an incomplete record with no urgent value has no track
        at.selectbox(key="clinical_record_diabetes_cohort").set_value(self.records["URG-LOW"]["id"])
        with mock.patch.object(clinical, "patient_report", fake_report):
            page = self.html(self.ok(at.run()))
        self.assertIn("The model did not flag this result, but an entered value is already in a range", page)

    def test_values_past_the_input_range_stay_urgent(self):
        """A value too far outside the training range for the model fails validation, yet the urgent rules still sort
        it, and the panel note asks for a check against the source; a blank or a non-number never trips a rule."""
        clear = {"kidney": {"HbA1c": "5.4", "GFR": "80", "ACR": "10", "FastingBloodSugar": "90", "SystolicBP": "120",
                            "DiastolicBP": "80"},   # below every cut-off
                 "diabetes": {"HbA1c_level": "5.5", "blood_glucose_level": "140"}}
        one = " failed the input check above and may be an entry error. Check it against the source record before acting on it."
        for disease, changes, note in [
                ("kidney", {"SystolicBP": "210", "DiastolicBP": "125"}, "Blood pressure (210/125 mmHg)" + one),
                ("kidney", {"ACR": "500", "FastingBloodSugar": "260"},
                 "Urine ACR (500 mg/g) and Fasting blood sugar (260 mg/dL) failed the input check above and may be entry "
                 "errors. Check them against the source record before acting on them."),
                ("kidney", {"ACR": "500", "HbA1c": "7"}, "Urine ACR (500 mg/g)" + one),   # the checked HbA1c is not named
                ("diabetes", {"HbA1c_level": "11.5"}, "HbA1c (11.5%)" + one),
                ("kidney", {"ACR": "abc"}, None), ("diabetes", {"bmi": "", "blood_glucose_level": "lots"}, None)]:
            with self.subTest(disease=disease, changes=changes):
                frame = pd.DataFrame([typical_row(disease, **{**clear[disease], **changes})]).astype(str)
                row = prepare_upload(frame, disease, features(disease), feature_summary(disease), ["PatientID"])[0]
                self.assertTrue(row["issues"])
                record = {"status": "Incomplete", "issues_json": json.dumps(row["issues"]),
                          "values_json": json.dumps(row["values"]), "score": None, "threshold": None}
                track = clinical._track(record, disease)
                if note is None:
                    self.assertIsNone(track)
                    continue
                self.assertEqual((track["track"], track["flagged"]), ("urgent", None))
                self.assertEqual(track, triage(disease, row["values"]))
                self.assertEqual(clinical._unchecked(track, row["issues"]), note)


if __name__ == "__main__":
    unittest.main()
