"""Clinical Dashboard upload, validation, review and scoring fixes, checked on a temporary database."""
import io
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
import streamlit
from pypdf import PdfReader
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest, element_tree

from ahead.clinical_data import input_bounds, prepare_upload, validate_row
from ahead.platform_data import init_platform, record_prediction_history
from ahead.config import AGE_FIELD, OPTIONAL_INPUTS, form_fields
from ahead.resources import best_model, feature_summary
from ahead.storage import authenticate, create_user, init_db, insert_records, list_records, save_scoring_run, update_record

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def typical_row(disease, **changes):
    """An upload row: every form column at its typical value, the optional labs and readings blank (the model alone
    sorts it), age from the DOB."""
    stats, age_key, optional = feature_summary(disease), AGE_FIELD[disease], OPTIONAL_INPUTS.get(disease, [])
    row = {f: "" if f in optional else stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"]
           if stats[f]["binary"] else round(stats[f]["median"], 2) for f in form_fields(disease) if f != age_key}
    return {"PatientID": "DEMO-001", "DOB": "1980-01-01", **row, **changes}


class ClinicalFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/clinical.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        create_user("doc@test.local", "test-password-123", "Doctor", "Doc")
        create_user("other@test.local", "test-password-123", "Doctor", "Other")
        self.doctor = authenticate("doc@test.local", "test-password-123", "Doctor")
        self.other = authenticate("other@test.local", "test-password-123", "Doctor")["id"]
        self.required, self.stats = form_fields("diabetes"), feature_summary("diabetes")

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def page(self):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": self.doctor, "nav": "clinical"}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def widget(self, items, label):
        return next(w for w in items if w.label == label)

    def upload(self, disease, rows, cohort="cohort"):
        frame = pd.DataFrame(rows).astype(str)
        insert_records(self.doctor["id"], cohort, disease, prepare_upload(frame, disease, form_fields(disease),
                                                                          feature_summary(disease), ["PatientID"]))
        return sorted(list_records(self.doctor["id"], disease), key=lambda r: r["source_row"])

    def test_valid_rows_are_saved_ready_and_stale_rows_are_repaired(self):
        rows = [typical_row("diabetes", PatientID="OK"), typical_row("diabetes", PatientID="BAD", gender="M"),
                typical_row("diabetes", PatientID="SCORED")]
        self.assertEqual([r["status"] for r in self.upload("diabetes", rows)], ["Ready", "Incomplete", "Ready"])
        # A database written before the fix: valid rows were left 'Incomplete' by the schema default.
        with sqlite3.connect(os.environ["AHEAD_DB_PATH"]) as db:
            db.execute("UPDATE records SET status='Incomplete'")
            db.execute("UPDATE records SET score=0.4 WHERE patient_id='SCORED'")
            db.execute("PRAGMA user_version=0")
        init_db(); init_db()
        statuses = {r["patient_id"]: r["status"] for r in list_records(self.doctor["id"])}
        self.assertEqual(statuses, {"OK": "Ready", "BAD": "Incomplete", "SCORED": "Incomplete"})
        with sqlite3.connect(os.environ["AHEAD_DB_PATH"]) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)  # later reruns skip the repair scans

    def test_upload_keeps_text_ids_and_bad_files_show_an_error(self):
        rows = [typical_row("diabetes", PatientID=pid) for pid in ("00042", "", " 00043 ")]
        at = self.page()
        at.file_uploader[0].set_value(("cohort.csv", pd.DataFrame(rows).to_csv(index=False).encode(), "text/csv"))
        self.ok(at.run())
        next(b for b in at.button if b.key == "clinical_import").click()
        self.ok(at.run())
        records = sorted(list_records(self.doctor["id"], "diabetes"), key=lambda r: r["source_row"])
        self.assertEqual([r["patient_id"] for r in records], ["00042", "Row 2", "00043"])
        self.assertEqual({r["status"] for r in records}, {"Ready"})
        archive = io.BytesIO()
        zipfile.ZipFile(archive, "w").writestr("hello.txt", "hi")
        for name, data in [("bad.xlsx", b"PK\x03\x04garbage"), ("zip.xlsx", archive.getvalue()),
                           ("legacy.xlsx", bytes.fromhex("D0CF11E0A1B11AE1") + bytes(600))]:
            at = self.page()
            at.file_uploader[0].set_value((name, data, "application/octet-stream"))
            self.ok(at.run())
            self.assertIn("Could not read this file", " ".join(e.value for e in at.error))

    def test_invalid_values_stay_as_entered_until_corrected(self):
        values = {"gender": "M", "smoking_history": "smoker", "HbA1c_level": "abc", "blood_glucose_level": "140.7"}
        clean, issues = validate_row(dict(values), "diabetes", list(values), self.stats)
        self.assertEqual(clean, values)
        self.assertEqual(len(issues), 4)
        # Saving again without changes must not truncate 140.7 or clear the issue.
        again, issues = validate_row(dict(clean), "diabetes", list(values), self.stats)
        self.assertEqual(again["blood_glucose_level"], "140.7")
        self.assertIn("blood_glucose_level: enter a whole number.", issues)
        fixed, issues = validate_row({"blood_glucose_level": "141"}, "diabetes", ["blood_glucose_level"], self.stats)
        self.assertEqual((fixed["blood_glucose_level"], issues), (141, []))
        self.assertIn("age: enter a number.", validate_row({"age": "inf"}, "diabetes", ["age"], self.stats, "1980-01-01")[1])
        record = self.upload("diabetes", [typical_row("diabetes", **values)])[0]
        at = self.page()
        editor = next(d.value for d in at.dataframe if (d.key or "").startswith("record_editor_"))
        shown = dict(zip(editor["Input"], editor["Value"]))
        self.assertEqual((shown["gender"], shown["HbA1c_level"], shown["blood_glucose_level"]), ("M", "abc", "140.7"))
        self.assertEqual(dict(zip(editor["Input"], editor["Issue"]))["HbA1c_level"], "enter a number.")
        self.widget(at.button, "Save and validate record").click()
        self.ok(at.run())
        saved = list_records(self.doctor["id"])[0]
        self.assertEqual((json.loads(saved["values_json"])["blood_glucose_level"], saved["status"]), ("140.7", "Incomplete"))
        self.assertEqual(saved["id"], record["id"])

    def test_values_use_the_patient_form_bounds(self):
        low, high = input_bounds("HbA1c_level", self.stats["HbA1c_level"])
        self.assertTrue(low < self.stats["HbA1c_level"]["min"] and high > self.stats["HbA1c_level"]["max"])
        _, issues = validate_row({"HbA1c_level": "9.5", "blood_glucose_level": "320", "age": "85"}, "diabetes",
                                 ["HbA1c_level", "blood_glucose_level", "age"], self.stats)
        self.assertEqual(issues, [])
        _, issues = validate_row({"HbA1c_level": "40"}, "diabetes", ["HbA1c_level"], self.stats)
        self.assertIn("too far outside the training range for the model to score", issues[0])
        self.assertIn(f"accepts {low:g} to {high:g}", issues[0])
        self.assertNotIn("verify it", issues[0])

    def test_widened_bounds_stop_at_physical_limits(self):
        for disease, feature, bad, good in [("heart", "PhysicalHealthDays", [31, 38], 30), ("heart", "MentalHealthDays", [31], 30),
                                            ("heart", "SleepHours", [0, 25], 24), ("heart", "BMI", [0, 9.5], 10),
                                            ("diabetes", "bmi", [0, 0.5], 10.5), ("kidney", "creatinine", [0, 0.1, 15.5], 0.2),
                                            ("kidney", "bmi", [0, 9.9], 12)]:
            stats = feature_summary(disease)
            for value in bad:
                self.assertTrue(validate_row({feature: value}, disease, [feature], stats)[1], (feature, value))
            self.assertEqual(validate_row({feature: good}, disease, [feature], stats)[1], [], feature)
        heart, kidney = feature_summary("heart"), feature_summary("kidney")
        self.assertEqual(input_bounds("SleepHours", heart["SleepHours"]), (1, 24))
        # The patient form needs float bounds for float inputs (number_input rejects mixed int/float).
        self.assertTrue(all(isinstance(b, float) for b in input_bounds("bmi", kidney["bmi"])))
        self.assertTrue(all(isinstance(b, float) for b in input_bounds("creatinine", kidney["creatinine"])))

    def test_old_range_issues_are_rechecked_on_start_up(self):
        old = "HbA1c_level: {:g} is outside the training range (3.5–9); verify it."
        gender = "gender: choose a category from the training data."
        insert_records(self.doctor["id"], "legacy", "diabetes", [
            {"patient_id": pid, "values": {"HbA1c_level": value, "gender": "Female"}, "issues": issues}
            for pid, value, issues in [("OK", 9.5, [old.format(9.5)]), ("FAR", 40.0, [old.format(40)]),
                                       ("MIXED", 9.5, [gender, old.format(9.5)]), ("EXCLUDED", 9.5, [old.format(9.5)])]])
        update_record(self.doctor["id"], next(r["id"] for r in list_records(self.doctor["id"]) if r["patient_id"] == "EXCLUDED"),
                      status="Excluded")
        with sqlite3.connect(os.environ["AHEAD_DB_PATH"]) as db:
            db.execute("PRAGMA user_version=0")  # saved before the wider bounds
        init_db(); init_db()
        records = {r["patient_id"]: (r["status"], json.loads(r["issues_json"])) for r in list_records(self.doctor["id"])}
        self.assertEqual(records["OK"], ("Ready", []))
        self.assertEqual(records["MIXED"], ("Incomplete", [gender]))
        self.assertEqual(records["EXCLUDED"], ("Excluded", []))
        self.assertEqual(records["FAR"][0], "Incomplete")
        self.assertIn("too far outside the training range", records["FAR"][1][0])

    def test_batch_scoring_stores_scores_and_history_for_owned_records_only(self):
        first, second = self.upload("diabetes", [typical_row("diabetes"), typical_row("diabetes")])
        save_scoring_run(self.doctor["id"], "diabetes", "Model", 0.5, [(first["id"], {"age": 40}, 0.7)],
                         [(second["id"], {"age": "x"}, ["age: enter a number."])])
        save_scoring_run(self.other, "diabetes", "Model", 0.5, [(second["id"], {"age": 40}, 0.9)])
        records = {r["id"]: r for r in list_records(self.doctor["id"])}
        self.assertEqual((records[first["id"]]["status"], records[first["id"]]["score"]), ("Flagged", 0.7))
        self.assertEqual((records[second["id"]]["status"], records[second["id"]]["score"]), ("Incomplete", None))
        self.assertEqual(len(record_prediction_history(self.doctor["id"], first["id"])), 1)
        self.assertEqual(record_prediction_history(self.other, second["id"]), [])

    def test_scoring_review_and_edits_keep_the_selected_record(self):
        records = self.upload("diabetes", [typical_row("diabetes", PatientID=f"P-{i}") for i in range(1, 4)]
                              + [typical_row("diabetes", PatientID="P-4", gender="M")])
        # The browser keeps the 'Record' label it last saw and resends it on every rerun, taking a new label only when
        # the server sets the value; AppTest rebuilds the label itself, so replay what the browser would hold.
        held, real_state, real_editor = {}, element_tree.Selectbox._widget_state, streamlit.data_editor

        def browser_state(widget):
            return WidgetState(id=widget.id, string_value=held["label"]) if widget.label == "Record" and held \
                else real_state.fget(widget)

        def track(at):
            proto = self.widget(at.selectbox, "Record").proto
            if proto.set_value or not held:
                held["label"] = proto.raw_value if proto.set_value else proto.options[proto.default]
            return at
        run = lambda at: track(self.ok(at.run()))

        def pick(at, record_id):
            held["label"] = self.widget(at.selectbox, "Record").format_func(record_id)
            return run(at)

        def stays_on(at, record_id):  # any next interaction, here the Show filter, resends the browser's label
            self.widget(at.selectbox, "Show").set_value("Reviewed" if self.widget(at.selectbox, "Show").value == "All" else "All")
            self.assertEqual(self.widget(run(at).selectbox, "Record").value, record_id)

        def fix_gender(data, *args, **kwargs):
            table = real_editor(data, *args, **kwargs).copy()
            table.loc[table["Input"] == "gender", "Value"] = "Female"
            return table
        with mock.patch.object(element_tree.Selectbox, "_widget_state", property(browser_state)):
            at = pick(track(self.page()), records[1]["id"])
            next(b for b in at.button if b.label.startswith("Score 3")).click()
            stays_on(run(at), records[1]["id"])
            self.assertEqual(sum(r["score"] is not None for r in list_records(self.doctor["id"])), 3)
            self.assertEqual(len(record_prediction_history(self.doctor["id"], records[0]["id"])), 1)
            for record, decision in [(records[1], "Reviewed"), (records[2], "Excluded")]:
                self.widget(pick(at, record["id"]).selectbox, "Review decision").set_value(decision)
                self.widget(at.button, "Save review").click()
                run(at)
                self.assertIn(decision, held["label"])
                stays_on(at, record["id"])
            self.widget(pick(at, records[3]["id"]).button, "Save and validate record").click()
            with mock.patch.object(streamlit, "data_editor", fix_gender):
                run(at)
            self.assertEqual(held["label"], "Row 4 · P-4 · Ready")
            stays_on(at, records[3]["id"])

    def test_patient_pdf_shows_pending_until_a_doctor_review(self):
        record = self.upload("diabetes", [typical_row("diabetes")])[0]
        update_record(self.doctor["id"], record["id"], score=0.8, threshold=0.5, status="Flagged", model=best_model("diabetes"))
        import ahead.pages.clinical as clinical
        pages, real = [], clinical.patient_report

        def capture(*args, **kwargs):
            pdf = real(*args, **kwargs)
            pages.append(PdfReader(io.BytesIO(pdf)).pages[0].extract_text())
            return pdf
        with mock.patch.object(clinical, "patient_report", capture):
            self.page()
            update_record(self.doctor["id"], record["id"], status="Reviewed")
            self.page()
        self.assertEqual(len(pages), 2)
        self.assertIn("Pending review", pages[0])
        self.assertNotIn("Pending review", pages[1])
        self.assertIn("Reviewed", pages[1])

    def test_changing_dob_derives_age_on_the_first_save(self):
        for disease, new_dob in [("diabetes", "1990-06-01"), ("heart", "1960-01-01")]:
            age_key = "AgeCategory" if disease == "heart" else "age"
            self.upload(disease, [typical_row(disease, DOB="1978-01-01")], cohort=f"{disease}-dob")
            at = self.page()
            self.widget(at.selectbox, "Disease screening").set_value(disease)
            self.ok(at.run())
            self.widget(at.toggle, "Edit this record").set_value(True)
            self.ok(at.run())
            next(t for t in at.text_input if t.label.startswith("Date of birth")).set_value(new_dob)
            self.widget(at.button, "Save and validate record").click()
            self.ok(at.run())
            record = list_records(self.doctor["id"], disease)[0]
            expected = validate_row({}, disease, [age_key], feature_summary(disease), new_dob)[0][age_key]
            self.assertEqual((record["status"], record["issues_json"], record["dob"]), ("Ready", "[]", new_dob))
            self.assertEqual(json.loads(record["values_json"])[age_key], expected)

    def test_a_saved_dob_typo_never_leaves_a_stale_age(self):
        invalid = "Invalid date of birth; use YYYY-MM-DD."
        for disease, typo, new_dob in [("diabetes", "1990-13-45", "1990-06-01"), ("kidney", "2090-01-01", "1990-06-01"),
                                       ("heart", "2015-01-01", "1960-01-01")]:  # bad date, future date, under 18
            age_key = AGE_FIELD[disease]
            self.upload(disease, [typical_row(disease)], cohort=f"{disease}-typo")
            at = self.page()
            self.widget(at.selectbox, "Disease screening").set_value(disease)
            self.ok(at.run())
            for dob, status, issues in [(typo, "Incomplete", [invalid, f"{age_key}: missing; enter a value before scoring."]),
                                        (new_dob, "Ready", []), (new_dob, "Ready", [])]:
                if not any(b.label == "Save and validate record" for b in at.button):  # Ready records open on the toggle
                    self.widget(at.toggle, "Edit this record").set_value(True)
                    self.ok(at.run())
                next(t for t in at.text_input if t.label.startswith("Date of birth")).set_value(dob)
                self.widget(at.button, "Save and validate record").click()
                self.ok(at.run())
                record = list_records(self.doctor["id"], disease)[0]
                self.assertEqual((record["status"], json.loads(record["issues_json"])), (status, issues), (disease, dob))
            expected = validate_row({}, disease, [age_key], feature_summary(disease), new_dob)[0][age_key]
            self.assertEqual(json.loads(record["values_json"])[age_key], expected)

    def test_an_explicitly_edited_age_that_disagrees_with_a_new_dob_is_flagged(self):
        self.upload("diabetes", [typical_row("diabetes")])
        real = streamlit.data_editor

        def edit_age(data, *args, **kwargs):
            table = real(data, *args, **kwargs).copy()
            table.loc[table["Input"] == "age", "Value"] = "50"
            return table
        at = self.page()
        self.widget(at.toggle, "Edit this record").set_value(True)
        self.ok(at.run())
        next(t for t in at.text_input if t.label.startswith("Date of birth")).set_value("1990-06-01")
        self.widget(at.button, "Save and validate record").click()
        with mock.patch.object(streamlit, "data_editor", edit_age):
            self.ok(at.run())
        record = list_records(self.doctor["id"])[0]
        self.assertEqual(record["status"], "Incomplete")
        self.assertIn("age: entered age conflicts with DOB; verify both.", json.loads(record["issues_json"]))
        self.assertEqual(json.loads(record["values_json"])["age"], 50)

    def test_a_new_dob_never_replaces_or_loses_an_age_it_did_not_derive(self):
        self.upload("diabetes", [typical_row("diabetes", DOB="", age="30")])
        at = self.page()
        self.widget(at.toggle, "Edit this record").set_value(True)
        self.ok(at.run())
        conflict, invalid = "age: entered age conflicts with DOB; verify both.", "Invalid date of birth; use YYYY-MM-DD."
        for dob, status, issues in [("1980-01-01", "Incomplete", [conflict]), ("01/01/1980", "Incomplete", [invalid]),
                                    ("", "Ready", [])]:
            next(t for t in at.text_input if t.label.startswith("Date of birth")).set_value(dob)
            self.widget(at.button, "Save and validate record").click()
            self.ok(at.run())
            record = list_records(self.doctor["id"])[0]
            self.assertEqual((record["status"], json.loads(record["issues_json"])), (status, issues), dob)
            self.assertEqual(json.loads(record["values_json"])["age"], 30, dob)


if __name__ == "__main__":
    unittest.main()
