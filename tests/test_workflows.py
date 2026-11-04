"""Focused checks for age handling, validation, account isolation and PDF output."""

import os
import tempfile
import unittest
from datetime import date
from io import BytesIO

import pandas as pd
from pypdf import PdfReader

from ahead.clinical_data import age_category, age_from_dob, prepare_upload, validate_row
from ahead.reports import patient_report
from ahead.storage import (authenticate, change_password, create_user, init_db, insert_records,
                           list_records, update_record)


class ClinicalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = os.path.join(self.temp.name, "test.sqlite3")
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db()
        self.stats = {
            "age": {"kind": "number", "min": 0, "max": 80, "integer": True, "binary": False, "median": 40},
            "gender": {"kind": "category", "options": ["Male", "Female"], "default": "Male"},
        }

    def tearDown(self):
        self.temp.cleanup()
        os.environ.pop("AHEAD_DB_PATH", None)

    def test_missing_age_and_invalid_value_are_not_ready(self):
        data = pd.DataFrame([{"PatientID": "A", "gender": "Female"},
                             {"PatientID": "B", "gender": "other", "DOB": "2000-01-01"}])
        rows = prepare_upload(data, "diabetes", ["age", "gender"], self.stats, ["PatientID"])
        self.assertIn("age: missing", "; ".join(rows[0]["issues"]))
        self.assertFalse(any("age: missing" in issue for issue in rows[1]["issues"]))
        self.assertTrue(any("gender:" in issue for issue in rows[1]["issues"]))
        fixed, issues = validate_row({"age": "35", "gender": "female"}, "diabetes",
                                     ["age", "gender"], self.stats)
        self.assertEqual((fixed["age"], fixed["gender"], issues), (35, "Female", []))

    def test_age_and_out_of_range(self):
        self.assertEqual(age_from_dob("2004-09-23", date(2026, 9, 22)), 21)
        self.assertEqual(age_category(83, ["Age 18 to 24", "Age 80 or older"]), "Age 80 or older")
        _, issues = validate_row({"age": 120, "gender": "Male"}, "diabetes",
                                 ["age", "gender"], self.stats)
        self.assertTrue(any("outside the training range" in issue for issue in issues))

    def test_accounts_and_scoped_queue(self):
        self.assertTrue(create_user("doctor1@example.com", "doctor-passphrase-1", "Doctor", "One"))
        self.assertTrue(create_user("doctor2@example.com", "doctor-passphrase-2", "Doctor", "Two"))
        first = authenticate("doctor1@example.com", "doctor-passphrase-1", "Doctor")
        second = authenticate("doctor2@example.com", "doctor-passphrase-2", "Doctor")
        self.assertIsNone(authenticate("doctor1@example.com", "wrong", "Doctor"))
        insert_records(first["id"], "batch1", "diabetes",
                       [{"patient_id": "DEMO", "values": {"age": 30}, "issues": []}])
        record = list_records(first["id"])[0]
        self.assertFalse(list_records(second["id"]))
        update_record(second["id"], record["id"], status="Reviewed")
        self.assertEqual(list_records(first["id"])[0]["status"], "Ready")
        self.assertTrue(change_password(first["id"], "doctor-passphrase-1", "new-doctor-passphrase"))

    def test_report_has_score_and_limitation(self):
        pdf = patient_report("heart", "DEMO", {"AgeCategory": "Age 25 to 29"}, 0.72,
                             0.6, "LogisticRegression", {"recall_disease": 0.6, "precision_disease": 0.17})
        text = PdfReader(BytesIO(pdf)).pages[0].extract_text()
        self.assertIn("72.0%", text)
        self.assertIn("not a diagnosis", text)


if __name__ == "__main__":
    unittest.main()
