"""What the Screenings page and the Clinical Dashboard pass to patient_report, and what the PDF then prints.
The real report still builds; each page's own reference to it is wrapped so the arguments can be read back."""
import inspect
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import ahead.pages.clinical as clinical
import ahead.screening as screening
from ahead.clinical_data import prepare_upload
from ahead.platform_data import init_platform, to_local
from ahead.reports import patient_report
from ahead.resources import best_model, feature_summary, features
from ahead.storage import authenticate, create_user, init_db, insert_records, list_records, update_record
from ahead.triage import track_text, triage

APP = str(Path(__file__).resolve().parents[1] / "app.py")
# Lab values above the rule-based cut-offs, so the PDF has factors to print.
LABS = {"HbA1c_level": 6.4, "blood_glucose_level": 140}


def typical_row(disease, **changes):
    stats, age_key = feature_summary(disease), "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    row = {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"] if stats[f]["binary"]
           else round(stats[f]["median"], 2) for f in features(disease) if f != age_key}
    return {"PatientID": "DEMO-001", "DOB": "1980-01-01", **row, **changes}


class ReportSpy:
    """Stands in for patient_report: keeps the arguments as the real function binds them (positional ones by name,
    defaults filled in) and the first page of the PDF it builds as one line of text."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = inspect.signature(patient_report).bind(*args, **kwargs)
        bound.apply_defaults()
        pdf = patient_report(*args, **kwargs)
        text = " ".join(PdfReader(io.BytesIO(pdf)).pages[0].extract_text().split())
        self.calls.append((dict(bound.arguments), text))
        return pdf

    @property
    def last(self):
        return self.calls[-1]


class PdfWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/pdf.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        create_user("p@test.local", "test-password-123", "Patient", "Amy Lee")
        create_user("d@test.local", "test-password-123", "Doctor", "Dr. Layla Hassan")
        self.patient = authenticate("p@test.local", "test-password-123", "Patient")
        self.doctor = authenticate("d@test.local", "test-password-123", "Doctor")
        # The AHEAD Insight tab would call Gemini; nothing leaves the machine in these checks.
        patcher = mock.patch.object(screening, "gemini_available", lambda: False)
        patcher.start(); self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def page(self, user, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user, "nav": nav, **state}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    # ------------------------------------------------------------------ Screenings page
    def screen(self, user, labs=LABS):
        """Fill in the diabetes form with typical values and *labs*, submit it, and return what the
        'Download screening PDF' button was built from."""
        at = self.page(user, "screenings", active_screening="diabetes")
        for feature, value in labs.items():
            at.number_input(key=f"diabetes_{feature}").set_value(value)
        spy = ReportSpy()
        with mock.patch.object(screening, "patient_report", spy):
            next(b for b in at.button if b.label == "Generate Screening Result").click()
            self.ok(at.run())
        self.assertIn("Download screening PDF", [b.proto.label for b in at.get("download_button")])
        self.assertEqual(len(spy.calls), 1)
        args, text = spy.last
        self.assertEqual({k: args["values"][k] for k in labs}, labs)   # the submitted form, not a stored result
        self.assertEqual(args["patient_id"], "Session screening")
        return args, text

    def test_patient_self_screening_pdf_is_prepared_for_the_patient_without_a_review_status(self):
        args, text = self.screen(self.patient)
        self.assertEqual(args["audience"], "patient")
        self.assertIsNone(args["status"])
        self.assertEqual(args["name"], "Amy Lee")
        self.assertEqual(args["factors"], screening.identify_risk_factors("diabetes", args["values"]))
        self.assertIn("Higher entered HbA1c", args["factors"])
        self.assertIn("Prepared for", text)
        self.assertIn("Amy Lee", text)
        self.assertIn("Higher entered HbA1c", text)
        for absent in ("Review status", "Pending review", "Reviewing doctor"):
            self.assertNotIn(absent, text)

    def test_doctor_screening_pdf_names_the_screening_type_and_no_patient(self):
        args, text = self.screen(self.doctor)
        self.assertEqual(args["audience"], "doctor")
        self.assertIsNone(args["status"])
        self.assertEqual(args["name"], "")
        self.assertEqual(args["factors"], screening.identify_risk_factors("diabetes", args["values"]))
        self.assertIn("Screening type Session screening", text)
        for absent in ("Pending review", "Review status", "Prepared for", "Layla Hassan"):
            self.assertNotIn(absent, text)

    def test_screening_pdf_prints_the_track_of_the_same_result(self):
        # Labs in the urgent range: the band under the result panel names the next step and its reasons.
        labs = {"HbA1c_level": 7.1, "blood_glucose_level": 212}
        for user, audience, label in ((self.patient, "patient", "See a doctor soon"), (self.doctor, "doctor", "Urgent")):
            with self.subTest(audience=audience):
                args, text = self.screen(user, labs)
                track = args["track"]
                self.assertEqual(track, triage("diabetes", args["values"], args["score"], args["threshold"]))
                self.assertEqual((track["track"], track_text(track, audience)["label"]), ("urgent", label))
                self.assertIn(f"NEXT STEP {label}", text)
                for reason in track["reasons"]:
                    self.assertIn(" ".join(reason[audience].split()), text)

    # ------------------------------------------------------------------ Clinical Dashboard
    def upload(self, rows, source_name, disease="diabetes"):
        frame = pd.DataFrame(rows).astype(str)
        insert_records(self.doctor["id"], "cohort", disease,
                       prepare_upload(frame, disease, features(disease), feature_summary(disease), ["PatientID"]),
                       source_name=source_name)
        return sorted(list_records(self.doctor["id"], disease), key=lambda r: r["source_row"])

    def score(self, record, score, status, disease="diabetes"):
        update_record(self.doctor["id"], record["id"], score=score, threshold=0.5, status=status,
                      model=best_model(disease))

    def dashboard(self, record_id=None, disease="diabetes"):
        """Open the Clinical Dashboard on *disease* (and *record_id* if given) and return the arguments of its
        patient PDF."""
        spy = ReportSpy()
        with mock.patch.object(clinical, "patient_report", spy):
            at = self.page(self.doctor, "clinical")
            if disease != "diabetes":
                next(s for s in at.selectbox if s.label == "Disease screening").set_value(disease)
                self.ok(at.run())
            if record_id is not None:
                at.selectbox(key=f"clinical_record_{disease}_cohort").set_value(record_id)
                self.ok(at.run())
        self.assertIn("Download patient PDF", [b.proto.label for b in at.get("download_button")])
        return spy.last

    def test_scored_record_is_pending_review_and_names_its_uploaded_file(self):
        record = self.upload([typical_row("diabetes", PatientID="P-1", HbA1c_level="6.4")], "march_cohort.csv")[0]
        self.score(record, 0.8, "Flagged")
        args, text = self.dashboard()
        self.assertEqual((args["audience"], args["patient_id"]), ("doctor", "P-1"))
        self.assertEqual((args["status"], args["reviewer"], args["reviewed_at"]), ("", "", None))
        self.assertEqual(args["source"], "march_cohort.csv, row 1")
        self.assertEqual(args["factors"], screening.identify_risk_factors("diabetes", args["values"]))
        self.assertIn("Higher entered HbA1c", args["factors"])
        self.assertIn("Review status Pending review", text)
        self.assertIn("march_cohort.csv, row 1", text)
        self.assertNotIn("Reviewing doctor", text)

    def test_upload_without_a_file_name_reads_uploaded_file_and_its_row(self):
        # Saved before file names were kept; row 2 was scored below the threshold, which is still not a review.
        records = self.upload([typical_row("diabetes", PatientID=f"P-{i}") for i in (1, 2)], None)
        self.score(records[0], 0.8, "Flagged")
        self.score(records[1], 0.2, "Ready")
        args, text = self.dashboard(records[1]["id"])
        self.assertEqual(args["patient_id"], "P-2")
        self.assertEqual((args["status"], args["reviewer"], args["reviewed_at"]), ("", "", None))
        self.assertEqual(args["source"], "Uploaded file, row 2")
        self.assertIn("Uploaded file, row 2", text)
        self.assertIn("Pending review", text)

    def test_reviewed_record_names_the_doctor_and_the_review_time(self):
        record = self.upload([typical_row("diabetes", PatientID="P-1")], "march_cohort.csv")[0]
        self.score(record, 0.8, "Flagged")
        note = "Repeat HbA1c in three months."
        update_record(self.doctor["id"], record["id"], status="Reviewed", note=note)
        saved = list_records(self.doctor["id"])[0]
        args, text = self.dashboard()
        self.assertEqual((args["status"], args["note"]), ("Reviewed", note))
        self.assertEqual(args["reviewer"], "Dr. Layla Hassan")
        self.assertEqual(args["reviewed_at"], saved["updated_at"])
        moment = to_local(saved["updated_at"])   # stored as UTC, printed in Dubai time
        self.assertIn(f"Reviewed on {moment.day} {moment:%b %Y, %H:%M}", text)
        self.assertIn("Reviewing doctor Dr. Layla Hassan", text)
        self.assertIn(note, text)
        self.assertNotIn("Pending review", text)
        # Excluded is a decision too: the doctor and the time of that save, with no "Reviewed on".
        update_record(self.doctor["id"], record["id"], status="Excluded")
        saved = list_records(self.doctor["id"])[0]
        args, text = self.dashboard()
        self.assertEqual((args["status"], args["reviewer"], args["reviewed_at"]),
                         ("Excluded", "Dr. Layla Hassan", saved["updated_at"]))
        self.assertIn("Review status Excluded", text)
        self.assertNotIn("Reviewed on", text)
        self.assertNotIn("Pending review", text)

    def test_record_pdf_prints_the_track_of_the_record_on_one_page(self):
        # Every kidney input, reviewed with a note: urine ACR, HbA1c and fasting blood sugar are in a diagnostic
        # range, so the band is Urgent, as in the queue's Track column, and the copy still fits one page.
        record = self.upload([typical_row("kidney", PatientID="K-1")], "kidney_cohort.csv", "kidney")[0]
        self.score(record, 0.8, "Flagged", "kidney")
        update_record(self.doctor["id"], record["id"], status="Reviewed", note="Discussed with the patient.")
        args, text = self.dashboard(disease="kidney")
        track = args["track"]
        self.assertEqual(track, triage("kidney", args["values"], args["score"], args["threshold"]))
        self.assertEqual([reason["feature"] for reason in track["reasons"]], ["ACR", "HbA1c", "FastingBloodSugar"])
        self.assertEqual(args["audience"], "doctor")
        band = text.split("NEXT STEP Urgent")[1].split("Entered values")[0]
        for reason in track["reasons"]:   # the value with its unit, which only the band prints
            self.assertIn(reason["value_text"], band)
        self.assertIn("Discussed with the patient.", text)
        self.assertIn("Page 1 of 1", text)


if __name__ == "__main__":
    unittest.main()
