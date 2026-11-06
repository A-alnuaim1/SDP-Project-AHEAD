"""Clinical Dashboard restyle: third-person result wording, readable values and issues, summary tiles above the
queue, section headings, empty states and readable cohort labels. Every check runs on a temporary database."""
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import streamlit
from streamlit.testing.v1 import AppTest

import ahead.pages.clinical as clinical
from ahead.clinical_data import OPTIONAL_BLANK, issue_text, prepare_upload, shown_value, validate_row
from ahead.platform_data import append_record_prediction, init_platform, to_local
from ahead.resources import best_model, feature_summary, features, icon_uri
from ahead.storage import (authenticate, connection, create_user, init_db, insert_records, list_records, save_scoring_run,
                           update_record)

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PATIENT_ELEVATED = ("The screening model identified a pattern associated with elevated diabetes risk. This result is "
                    "intended for early awareness and does not confirm that you have the condition.")
PATIENT_LOWER = ("Your entered information did not meet the model&#x27;s classification level for elevated diabetes "
                 "risk. This does not rule out disease or replace routine screening.")


def typical_row(disease, **changes):
    stats, age_key = feature_summary(disease), "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    row = {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"] if stats[f]["binary"]
           else round(stats[f]["median"], 2) for f in features(disease) if f != age_key}
    return {"PatientID": "DEMO-001", "DOB": "1980-01-01", **row, **changes}


def walk(node):
    """Elements in page order."""
    for key in sorted(getattr(node, "children", {})):
        child = node.children[key]
        yield child
        yield from walk(child)


def result_cards():
    from ahead.components import empty_state, result_card
    result_card(1, 0.8, "diabetes", "Model", 0.5)
    result_card(0, 0.2, "diabetes", "Model", 0.5)
    empty_state("◉", "Title", "Text")


class ClinicalDesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/clinical.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        create_user("doc@test.local", "test-password-123", "Doctor", "Doc")
        self.doctor = authenticate("doc@test.local", "test-password-123", "Doctor")

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def page(self, disease=None):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": self.doctor, "nav": "clinical"}.items():
            at.session_state[key] = value
        self.ok(at.run())
        if disease:
            self.widget(at.selectbox, "Disease screening").set_value(disease)
            self.ok(at.run())
        return at

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def widget(self, items, label):
        return next(w for w in items if w.label == label)

    def html(self, at):
        return "\n".join(h.proto.body for h in at.get("html") if not h.proto.body.lstrip().startswith("<style"))

    def frame(self, at, column):
        return next(df.value for df in at.dataframe if column in df.value.columns)

    def upload(self, disease, rows, cohort="cohort", source_name=None):
        frame = pd.DataFrame(rows).astype(str)
        insert_records(self.doctor["id"], cohort, disease, prepare_upload(frame, disease, features(disease),
                       feature_summary(disease), ["PatientID"]), source_name=source_name)
        return sorted((r for r in list_records(self.doctor["id"], disease) if r["cohort"] == cohort),
                      key=lambda r: r["source_row"])

    def score(self, record, score, status=None, disease="diabetes"):
        update_record(self.doctor["id"], record["id"], score=score, threshold=0.5, model=best_model(disease),
                      status=status or ("Flagged" if score >= 0.5 else "Ready"))

    def pick(self, at, record):
        self.widget(at.selectbox, "Record").set_value(record["id"])
        return self.ok(at.run())

    def test_doctor_result_card_speaks_about_the_record_and_patient_wording_is_unchanged(self):
        elevated, lower = self.upload("diabetes", [typical_row("diabetes", PatientID="HIGH", bmi="35"),
                                                   typical_row("diabetes", PatientID="LOW")])
        self.score(elevated, 0.8); self.score(lower, 0.2)
        at = self.page()
        html = self.html(at)
        self.assertIn("This record&#x27;s values matched a pattern associated with elevated diabetes risk; this is not "
                      "a diagnosis and does not confirm that the patient has the condition.", html)
        self.assertNotIn("you have the condition", html)
        # Titled Flagged / Not flagged like the record label, the queue status and the Flagged tile.
        self.assertIn('<div class="result-title">Flagged · elevated risk pattern</div>', html)
        self.assertNotIn("Elevated Risk Pattern", html)
        self.assertIn("Row 1 · HIGH · Flagged", self.widget(at.selectbox, "Record").options)
        self.assertIn("Model: Gradient boosting ·", html)   # the model by name, as in Patient Intelligence
        # The risk factors read as the same chips as the Screenings Key factors tab.
        self.assertIn("Risk factors in this record", [c.value for c in at.caption])
        self.assertIn('<span class="factor-chip">Higher BMI</span>', html)
        self.assertFalse([m.value for m in at.markdown if "Risk factors" in m.value])
        self.assertNotIn("discuss with a clinician", " ".join(m.value for m in at.markdown))
        html = self.html(self.pick(at, lower))
        self.assertIn("This record&#x27;s values did not reach the model&#x27;s classification level for elevated "
                      "diabetes risk. This does not rule out disease or replace routine screening for the patient.", html)
        self.assertIn('<div class="result-title">Not flagged · lower predicted risk</div>', html)
        self.assertNotIn("Your entered information", html)
        # Once a decision is saved the card carries it, like the record label, the queue and the tiles.
        for record, status, title in [(elevated, "Reviewed", "Reviewed · elevated risk pattern"),
                                      (lower, "Excluded", "Excluded · lower predicted risk")]:
            update_record(self.doctor["id"], record["id"], status=status)
            html = self.html(self.pick(self.page(), record))
            self.assertEqual(re.findall(r'<div class="result-title">([^<]+)</div>', html), [title])
            self.assertIn(f"Row {record['source_row']} · {record['patient_id']} · {status}",
                          self.widget(self.page().selectbox, "Record").options)
        # The patient screening page keeps today's wording; decorative icons are hidden from screen readers.
        cards = AppTest.from_function(result_cards).run()
        html = self.html(cards)
        self.assertIn(PATIENT_ELEVATED, html)
        self.assertIn(PATIENT_LOWER, html)
        self.assertEqual(re.findall(r'<div class="result-title">([^<]+)</div>', html),
                         ["Elevated Risk Pattern", "Lower Predicted Risk"])
        self.assertIn('<div class="big" aria-hidden="true">◉</div>', html)

    def test_values_read_like_the_patient_form(self):
        stats = feature_summary("kidney")
        self.assertEqual([shown_value("Gender", v, stats["Gender"]) for v in (0, 1, 1.0)], ["Male", "Female", "Female"])
        self.assertEqual([shown_value("Smoking", v, stats["Smoking"]) for v in (0, 1)], ["No", "Yes"])
        self.assertEqual(shown_value("BMI", 27.652077393512773, stats["BMI"]), "27.65")
        self.assertEqual(shown_value("BMI", -0.0001, stats["BMI"]), "0")
        self.assertEqual(shown_value("SystolicBP", 140, stats["SystolicBP"]), "140")
        self.assertEqual(shown_value("SystolicBP", 100.0, stats["SystolicBP"]), "100")
        self.assertEqual([shown_value("BMI", v, stats["BMI"]) for v in (None, "", float("nan"))], ["Not provided"] * 3)
        self.assertEqual(shown_value("GFR", None, stats["GFR"], OPTIONAL_BLANK),
                         "Not provided (the model used a typical value)")
        self.assertEqual(shown_value("BMI", "abc", stats["BMI"]), "abc")  # a failed entry stays as typed
        diabetes = feature_summary("diabetes")
        self.assertEqual([shown_value("smoking_history", v, diabetes["smoking_history"]) for v in ("No Info", "never", "not current")],
                         ["No info", "Never", "Not current"])
        self.assertEqual(shown_value("gender", "M", diabetes["gender"]), "M")
        # The value table and the sensitivity table on the page.
        record = self.upload("kidney", [typical_row("kidney", Gender="1", Smoking="0", BMI="34.987654", GFR="41.123456",
                                                    SystolicBP="152")])[0]
        values = json.loads(record["values_json"]) | {"BMI": 27.652077393512773}
        update_record(self.doctor["id"], record["id"], values_json=json.dumps(values))
        self.score(record, 0.7, disease="kidney")
        # A rounded entry that reads the same as the typical value (27.65 vs 27.652…) is left out of the table.
        drift = [("GFR", values["GFR"], stats["GFR"]["median"], -0.0001), ("Gender", 1, 0, 0.012),
                 ("BMI", values["BMI"], 27.65, 0.0)]
        with mock.patch.object(clinical, "individual_sensitivity", return_value=drift), \
                self.assertNoLogs("streamlit.dataframe_util", "INFO"):  # every table serialises without Arrow fallbacks
            at = self.page("kidney")
        table = self.frame(at, "Value")
        shown = dict(zip(table["Input"], table["Value"]))
        self.assertEqual((shown["Gender"], shown["Smoking"], shown["Body Mass Index (BMI)"], shown["Estimated GFR"]),
                         ("Female", "No", "27.65", "41.12"))
        self.assertFalse([v for v in table["Value"] if re.search(r"\d\.\d{3,}", v)])
        sensitivity = self.frame(at, "Typical value")
        self.assertEqual(list(sensitivity["Input"]), ["Estimated GFR", "Gender"])
        self.assertEqual(list(sensitivity["Entered"]), ["41.12", "Female"])
        self.assertEqual(list(sensitivity["Typical value"]), [f"{stats['GFR']['median']:.2f}", "Male"])
        change = sensitivity["Score change (points)"].iloc[0]
        self.assertEqual((change, math.copysign(1, change)), (0.0, 1.0))  # never '-0.0'

    def test_binary_inputs_are_edited_as_words_and_saved_as_codes(self):
        record = self.upload("kidney", [typical_row("kidney", Gender="1", Smoking="0", SystolicBP="")])[0]
        self.assertEqual(record["status"], "Incomplete")
        real = streamlit.data_editor

        def fill(data, *args, **kwargs):
            table = real(data, *args, **kwargs).copy()
            table.loc[table["Input"] == "SystolicBP", "Value"] = "120"
            return table
        with self.assertNoLogs("streamlit.dataframe_util", "INFO"):
            at = self.page("kidney")
        editor = next(d.value for d in at.dataframe if (d.key or "").startswith("record_editor_"))
        rows = editor.set_index("Input")
        self.assertEqual((rows.loc["Gender", "Value"], rows.loc["Smoking", "Value"]), ("Female", "No"))
        self.assertEqual(rows.loc["SystolicBP", "Field"], "Systolic Blood Pressure (mmHg)")
        self.widget(at.button, "Save and validate record").click()
        with mock.patch.object(streamlit, "data_editor", fill):
            self.ok(at.run())
        saved = list_records(self.doctor["id"], "kidney")[0]
        values = json.loads(saved["values_json"])
        self.assertEqual((saved["status"], values["Gender"], values["Smoking"], values["SystolicBP"]), ("Ready", 1, 0, 120))
        # A mistyped word is asked for again in the same words the editor shows, never 'enter a number' or '0 or 1'.
        stats = feature_summary("kidney")
        for feature, typed, message in [("Smoking", "Yess", "Smoking: enter Yes or No."), ("Smoking", "2", "Smoking: enter Yes or No."),
                                        ("Edema", "0.5", "Edema: enter Yes or No."), ("Gender", "Female?", "Gender: enter Male or Female."),
                                        ("Gender", "inf", "Gender: enter Male or Female.")]:
            clean, issues = validate_row({feature: typed}, "kidney", [feature], stats)
            self.assertEqual((issues, clean[feature]), ([message], typed))
        self.assertEqual(issue_text("Smoking: enter Yes or No."), "Smoking: enter Yes or No")
        self.assertEqual(validate_row({"Smoking": "yes", "Gender": "F"}, "kidney", ["Smoking", "Gender"], stats),
                         ({"Smoking": 1, "Gender": 1}, []))
        self.assertEqual(validate_row({"BMI": "abc"}, "kidney", ["BMI"], stats)[1], ["BMI: enter a number."])

    def test_issues_use_field_labels_without_double_punctuation(self):
        self.assertEqual(issue_text("HbA1c_level: 40 is too far outside the training range for the model to score; "
                                    "it accepts 1.12 to 11.38."),
                         "HbA1c Level (%): 40 is too far outside the training range for the model to score; it accepts 1.12 to 11.38")
        # Issues saved by older versions, including the text the start-up repair matches, still read cleanly.
        self.assertEqual(issue_text("HbA1c_level: 9.5 is outside the training range (3.5–9); verify it."),
                         "HbA1c Level (%): 9.5 is outside the training range (3.5–9); verify it")
        self.assertEqual(issue_text("Missing value for bmi"), "Missing value for bmi")
        self.assertEqual(issue_text("Invalid date of birth; use YYYY-MM-DD."), "Invalid date of birth; use YYYY-MM-DD")
        expected = ["Invalid date of birth; use YYYY-MM-DD", "Age: missing; enter a value before scoring",
                    "Body Mass Index (BMI): missing; enter a value before scoring"]
        bad = typical_row("diabetes", PatientID="BAD", DOB="01/02/1980", bmi="")
        record = self.upload("diabetes", [bad])[0]
        exports = []
        real = streamlit.download_button

        def capture(label, data, *args, **kwargs):
            if kwargs.get("file_name", "").endswith("_review.csv"):
                exports.append(data.decode("utf-8-sig"))
            return real(label, data, *args, **kwargs)
        with mock.patch.object(streamlit, "download_button", capture):
            at = self.page()
        self.assertEqual(at.warning[-1].value, "Correct these inputs before scoring:\n" + "\n".join(f"- {i}" for i in expected))
        queue = self.frame(at, "Model score (%)")
        self.assertEqual(queue["Issues"].iloc[0], " · ".join(expected))
        self.assertEqual(pd.read_csv(io.StringIO(exports[-1]))["Issues"].iloc[0], " · ".join(expected))
        for text in (queue["Issues"].iloc[0], exports[-1], at.warning[-1].value):
            self.assertNotRegex(text, r"\.;|\. ·|bmi:|age:")
        # The import preview joins the same way.
        at.file_uploader[0].set_value(("more.csv", pd.DataFrame([bad]).to_csv(index=False).encode(), "text/csv"))
        self.ok(at.run())
        preview = self.frame(at, "Row")
        self.assertEqual((preview["Status"].iloc[0], preview["Issues"].iloc[0]), ("Incomplete", " · ".join(expected)))
        self.assertIn("0 ready · 1 incomplete. No rows are scored during import.", [c.value for c in at.caption])
        self.assertEqual(json.loads(list_records(self.doctor["id"])[0]["issues_json"])[0], "Invalid date of birth; use YYYY-MM-DD.")
        self.assertEqual(record["status"], "Incomplete")

    def test_summary_tiles_sit_above_the_queue_with_review_counts(self):
        rows = [typical_row("diabetes", PatientID=f"P-{i}") for i in range(1, 8)]
        rows[0]["gender"] = "M"
        records = self.upload("diabetes", rows)
        for record, score, status in [(records[2], 0.2, None), (records[3], 0.8, None), (records[4], 0.9, None),
                                      (records[5], 0.7, "Reviewed"), (records[6], 0.3, "Excluded")]:
            self.score(record, score, status)
        at = self.page()
        html = self.html(at)
        icons = {icon_uri(f"dashboard/{name}.svg"): name for name in ("uploads", "returning", "scored", "flagged", "calendar", "review")}
        tiles = re.findall(r'<div class="summary-inner" style="--tone:(#\w+)">\s*<div class="summary-icon"><img src="([^"]+)" '
                           r'alt=""></div>\s*<span class="summary-label">([^<]+)</span>'
                           r'<strong class="summary-value">(.*?)</strong>.*?<p class="summary-note">([^<]*)</p>', html, re.S)
        self.assertEqual([(label, value, icons.get(src)) for _, src, label, value, _ in tiles], [
            ("Records", "7", "uploads"), ("Incomplete", "1", "returning"), ("Scored", "5 <small>of 7</small>", "scored"),
            ("Flagged", "2", "flagged"), ("Pending review", "3", "calendar"), ("Reviewed", "1", "review")])
        self.assertEqual((tiles[1][4], tiles[-1][4]), ("Missing or invalid inputs", "1 excluded"))
        # The tile, the queue Status and its Show filter all call an unscorable row 'Incomplete'.
        self.assertEqual(list(self.frame(at, "Model score (%)")["Status"]).count("Incomplete"), 1)
        self.assertIn("Incomplete", self.widget(at.selectbox, "Show").options)
        self.assertTrue({tone for tone, *_ in tiles} <= {"#0E9AA7", "#C98F1E", "#8E7CC3", "#C9706E", "#5B8FA3", "#2E9477"})
        for old in ("visual-metric", "#2563EB", "#F59E0B", "#8B5CF6", "Records in review"):
            self.assertNotIn(old, html)
        kinds = [("tile" if "summary-inner" in e.proto.body else "") if getattr(e, "type", "") == "html" else
                 ("queue" if "Model score (%)" in e.value.columns else "") if getattr(e, "type", "") == "dataframe" else ""
                 for e in walk(at.main)]
        self.assertEqual([kind for kind in kinds if kind], ["tile"] * 6 + ["queue"])

    def test_sections_use_the_dashboard_headings(self):
        self.upload("diabetes", [typical_row("diabetes")])
        at = self.page()
        html = self.html(at)
        for title in ("1 · Import and check records", "2 · Review and score", "3 · Individual review"):
            self.assertIn(f'<div class="section-heading"><h2>{title}</h2><p>', html)
        self.assertEqual([h.value for h in at.subheader], [])

    def test_cohorts_are_labelled_by_file_date_and_rows(self):
        self.upload("heart", [typical_row("heart", PatientID=f"H-{i}") for i in range(6)], "heart-aaaaaaaaaaaaaaaa", "heart_a.csv")
        self.upload("heart", [typical_row("heart")], "heart-bbbbbbbbbbbbbbbb")  # saved before file names were kept
        with connection() as db:
            db.execute("UPDATE records SET created_at='2026-11-04T10:05:00+00:00' WHERE cohort='heart-aaaaaaaaaaaaaaaa'")
            db.execute("UPDATE records SET created_at='2026-11-03T08:00:00+00:00' WHERE cohort='heart-bbbbbbbbbbbbbbbb'")
        at = self.page("heart")
        box = self.widget(at.selectbox, "Uploaded cohort")
        self.assertEqual(box.options, ["heart_a.csv · 4 Nov 2026, 14:05 · 6 rows", "Upload · 3 Nov 2026, 12:00 · 1 row"])
        self.assertEqual(box.value, "heart-aaaaaaaaaaaaaaaa")  # the cohort key itself is unchanged
        box.set_value("heart-bbbbbbbbbbbbbbbb")
        self.assertEqual(self.widget(self.ok(at.run()).selectbox, "Uploaded cohort").value, "heart-bbbbbbbbbbbbbbbb")
        # A new upload keeps its file name and opens the queue on it.
        data = pd.DataFrame([typical_row("heart", PatientID="N-1"), typical_row("heart", PatientID="N-2")]).to_csv(index=False).encode()
        at.file_uploader[0].set_value(("heart_b.csv", data, "text/csv"))
        self.ok(at.run())
        next(b for b in at.button if b.key == "clinical_import").click()
        self.ok(at.run())
        cohort = f"heart-{hashlib.sha256(data).hexdigest()[:16]}"
        saved = [r for r in list_records(self.doctor["id"], "heart") if r["cohort"] == cohort]
        self.assertEqual({r["source_name"] for r in saved}, {"heart_b.csv"})
        box = self.widget(at.selectbox, "Uploaded cohort")
        self.assertEqual(box.value, cohort)
        self.assertEqual(box.options[0], f"heart_b.csv · {to_local(saved[0]['created_at']).day} "
                                         f"{to_local(saved[0]['created_at']):%b %Y, %H:%M} · 2 rows")

    def test_older_databases_gain_the_file_name_column(self):
        path = self.temp.name + "/old.db"
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE records (id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL, cohort TEXT NOT NULL,
                          disease TEXT NOT NULL, patient_id TEXT NOT NULL, source_row INTEGER NOT NULL, values_json TEXT NOT NULL,
                          dob TEXT, issues_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Incomplete',
                          note TEXT NOT NULL DEFAULT '', model TEXT, score REAL, threshold REAL,
                          created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            db.execute("""INSERT INTO records(owner_id,cohort,disease,patient_id,source_row,values_json,issues_json,created_at,updated_at)
                          VALUES(1,'diabetes-old','diabetes','P-1',1,'{}','[]','2026-11-01T06:00:00+00:00','2026-11-01T06:00:00+00:00')""")
        os.environ["AHEAD_DB_PATH"] = path
        init_db(); init_db()
        with connection() as db:
            self.assertIn("source_name", {row["name"] for row in db.execute("PRAGMA table_info(records)")})
        old = list_records(1)
        self.assertIsNone(old[0]["source_name"])
        self.assertEqual(clinical._cohort_label(old), "Upload · 1 Nov 2026, 10:00 · 1 row")
        insert_records(1, "diabetes-new", "diabetes", [{"patient_id": "P-2", "values": {}, "issues": []}], source_name="new.csv")
        self.assertEqual({r["cohort"]: r["source_name"] for r in list_records(1)}, {"diabetes-old": None, "diabetes-new": "new.csv"})

    def test_prediction_history_reads_as_local_dates_and_reasons(self):
        record = self.upload("diabetes", [typical_row("diabetes")])[0]
        at = self.page()
        next(b for b in at.button if b.label.startswith("Score 1")).click()
        self.ok(at.run())
        history = self.frame(at, "Reason")
        self.assertEqual(list(history["Reason"]), ["Cohort scoring"])
        self.assertEqual(list_records(self.doctor["id"])[0]["id"], record["id"])
        # Versions from different days: the Date column holds local datetimes, so a header click sorts by date
        # ('1 Dec' as text would come before '10 Nov' and '4 Nov').
        model = best_model("diabetes")
        for score in (0.1, 0.2, 0.3):
            append_record_prediction(self.doctor["id"], record["id"], "diabetes", {}, score, 0.5, model, "corrected")
        with connection() as db:
            db.execute("UPDATE record_predictions SET created_at='2026-11-01T08:00:00+00:00' WHERE source='batch'")
            for score, stamp in [(0.1, "2026-11-04T08:00:00+00:00"), (0.2, "2026-11-10T08:00:00+00:00"),
                                 (0.3, "2026-12-01T21:30:00+00:00")]:
                db.execute("UPDATE record_predictions SET created_at=? WHERE source='corrected' AND score=?", (stamp, score))
        at = self.page()
        element = next(df for df in at.dataframe if "Reason" in df.value.columns)
        history = element.value
        self.assertEqual(list(history.sort_values("Date")["Score (%)"])[1:], [10.0, 20.0, 30.0])
        self.assertEqual(history["Date"].iloc[0], pd.Timestamp(dt.datetime(2026, 12, 2, 1, 30)))  # Asia/Dubai, next day
        self.assertEqual(list(history["Reason"]), ["Corrected record"] * 3 + ["Cohort scoring"])  # newest first
        self.assertEqual(json.loads(element.proto.columns)["Date"]["type_config"],
                         {"type": "datetime", "format": "D MMM YYYY, HH:mm"})

    def test_repeated_uploads_of_one_file_get_distinct_labels(self):
        for cohort, bmi in (("diabetes-aaaaaaaaaaaaaaaa", "30"), ("diabetes-bbbbbbbbbbbbbbbb", "31"),
                            ("diabetes-cccccccccccccccc", "32")):
            self.upload("diabetes", [typical_row("diabetes", bmi=bmi)], cohort, "patients.csv")
        with connection() as db:
            for cohort, stamp in (("aaaaaaaaaaaaaaaa", "2026-11-04T10:05:01+00:00"), ("bbbbbbbbbbbbbbbb", "2026-11-04T10:05:40+00:00"),
                                  ("cccccccccccccccc", "2026-11-04T10:05:50+00:00")):
                db.execute("UPDATE records SET created_at=? WHERE cohort=?", (stamp, f"diabetes-{cohort}"))
        box = self.widget(self.page().selectbox, "Uploaded cohort")
        base = "patients.csv · 4 Nov 2026, 14:05 · 1 row"
        self.assertEqual(box.options, [f"{base} · upload 3", f"{base} · upload 2", base])  # newest first
        self.assertEqual(box.value, "diabetes-cccccccccccccccc")

    def test_empty_queue_filter_and_comparison_say_so(self):
        at = self.page()
        self.assertIn('<div class="ahead-card empty-state">', self.html(at))
        self.assertIn("No records in the review queue yet", self.html(at))
        self.assertFalse([i.value for i in at.info if "Save an upload" in i.value])
        record = self.upload("diabetes", [typical_row("diabetes")])[0]
        self.score(record, 0.3)
        # Every entry reads the same as its typical value once rounded, so no comparison rows remain.
        with mock.patch.object(clinical, "individual_sensitivity", return_value=[("bmi", 27.32, 27.3201, 0.0)]):
            at = self.page()
            self.assertIn("The entered values match the typical training values, so there is no input comparison to show.",
                          [i.value for i in at.info])   # the same element as on doctor Screenings
            self.assertFalse([df for df in at.dataframe if "Typical value" in df.value.columns])
            self.widget(at.selectbox, "Show").set_value("Excluded")
            self.ok(at.run())
        self.assertIn("No excluded records in this upload.", [c.value for c in at.caption])
        self.assertFalse([df for df in at.dataframe if "Model score (%)" in df.value.columns])
        self.widget(at.selectbox, "Show").set_value("Ready")
        self.assertIn("No ready records in this upload.", [c.value for c in self.ok(at.run()).caption])

    def test_scored_records_below_the_threshold_read_not_flagged(self):
        records = self.upload("diabetes", [typical_row("diabetes", PatientID="B-1"), typical_row("diabetes", PatientID="B-2")])
        save_scoring_run(self.doctor["id"], "diabetes", best_model("diabetes"), 0.5,
                         [(records[0]["id"], json.loads(records[0]["values_json"]), 0.05)])
        self.assertEqual([r["status"] for r in list_records(self.doctor["id"])], ["Ready", "Ready"])  # stored as before
        at = self.page()
        queue = self.frame(at, "Model score (%)")
        self.assertEqual(queue[["Status", "Model score (%)", "Model"]].iloc[0].tolist(), ["Not flagged", 5.0, "Gradient boosting"])
        self.assertEqual(queue["Status"].iloc[1], "Ready")
        self.assertEqual(self.widget(at.selectbox, "Record").options, ["Row 1 · B-1 · Not flagged", "Row 2 · B-2 · Ready"])
        self.assertEqual(self.widget(at.selectbox, "Review decision").options, ["Not flagged", "Reviewed", "Excluded"])
        self.assertIn("Not flagged · lower predicted risk", self.html(at))
        for shown, rows in (("Not flagged", [1]), ("Ready", [2])):
            self.widget(at.selectbox, "Show").set_value(shown)
            self.assertEqual(list(self.frame(self.ok(at.run()), "Model score (%)")["Row"]), rows)
        # Saving a decision without changing it keeps the stored status.
        self.widget(at.button, "Save review").click()
        self.ok(at.run())
        self.assertEqual(list_records(self.doctor["id"])[0]["status"], "Ready")
        self.assertEqual(next(b.label for b in at.button if b.label.startswith("Score")),
                         "Score 2 eligible records with Gradient boosting")
        # The model picker and its explanation name the models the same way.
        self.assertEqual(self.widget(at.selectbox, "Prediction model").options[0], "Gradient boosting — Recommended")
        self.assertIn("Gradient boosting is recommended because", "\n".join(m.value for m in at.markdown))
        self.widget(at.selectbox, "Show").set_value("Not flagged")
        update_record(self.doctor["id"], records[0]["id"], status="Reviewed")
        self.assertIn("No records scored below the threshold in this upload.", [c.value for c in self.ok(at.run()).caption])

    def test_import_messages_follow_the_preview(self):
        def save(name, rows):
            at = self.page()
            data = pd.DataFrame(rows).to_csv(index=False).encode()
            at.file_uploader[0].set_value((name, data, "text/csv"))
            self.ok(at.run())
            next(b for b in at.button if b.key == "clinical_import").click()
            return self.ok(at.run())
        clean = [typical_row("diabetes", PatientID="C-1"), typical_row("diabetes", PatientID="C-2")]
        self.assertEqual([t.value for t in save("clean.csv", clean).toast], ["Records saved. All rows are ready to score."])
        mixed = clean + [typical_row("diabetes", PatientID="C-3", bmi="")]
        self.assertEqual([t.value for t in save("mixed.csv", mixed).toast], ["Records saved: 2 ready, 1 incomplete."])
        at = save("clean.csv", clean)   # the same file again is a note, not a caution
        self.assertIn("This file is already in your review queue.", [i.value for i in at.info])
        self.assertFalse([w.value for w in at.warning if "already in your review queue" in w.value])

    def test_record_editor_shows_whole_numbers_and_categories_like_the_value_table(self):
        record = self.upload("diabetes", [typical_row("diabetes", DOB="1960-01-01", smoking_history="No Info", bmi="",
                                                      HbA1c_level="5.8765")])[0]
        self.assertEqual(json.loads(record["values_json"])["age"], 66.0)
        at = self.page()
        editor = next(d.value for d in at.dataframe if (d.key or "").startswith("record_editor_")).set_index("Input")["Value"]
        table = next(df.value for df in at.dataframe if list(df.value.columns) == ["Input", "Value"]).set_index("Input")["Value"]
        self.assertEqual((editor["age"], editor["smoking_history"], editor["bmi"]), ("66", "No info", ""))
        self.assertEqual((table["Age"], table["Smoking History"], table["Body Mass Index (BMI)"]), ("66", "No info", "Not provided"))
        # Other numbers stay as stored, so saving the record unchanged never rounds them.
        self.assertEqual((editor["HbA1c_level"], table["HbA1c Level (%)"]), ("5.8765", "5.88"))
        real = streamlit.data_editor

        def fill(data, *args, **kwargs):
            table = real(data, *args, **kwargs).copy()
            table.loc[table["Input"] == "bmi", "Value"] = "27.5"
            return table
        self.widget(at.button, "Save and validate record").click()
        with mock.patch.object(streamlit, "data_editor", fill):
            self.ok(at.run())
        saved = list_records(self.doctor["id"])[0]
        values = json.loads(saved["values_json"])
        self.assertEqual((saved["status"], values["age"], values["smoking_history"], values["HbA1c_level"], values["bmi"]),
                         ("Ready", 66.0, "No Info", 5.8765, 27.5))


if __name__ == "__main__":
    unittest.main()
