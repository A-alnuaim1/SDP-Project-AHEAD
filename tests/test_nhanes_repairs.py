"""Repairs after the switch to NHANES models: older kidney screenings still sort by their HbA1c and fasting sugar, scores
from retired models are never compared with retrained ones, every benchmark model flags at its own tuned threshold,
and every surface (doctor home tile, result card, score ring, PDF, next steps, AHEAD Insight prompt, missing-lab note,
patient home table, upload template, prevention buttons) follows the track rather than the model's flag alone. Page
checks run on a temporary database; nothing leaves the machine."""
import html
import inspect
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import ahead.pages.doctor_intelligence as intelligence
import ahead.pages.overview as overview
import ahead.screening as screening
from ahead.config import DISEASES, FIELD_LABELS, FORM_QUESTIONS, OPTIONAL_INPUTS, form_fields
from ahead.pages.assistant import LOCAL_ANSWERS, SYSTEM_PROMPT
from ahead.pages.clinical import example_row
from ahead.platform_data import grant_doctor, init_platform
from ahead.reports import patient_report
from ahead.resources import (best_model, decision_threshold, feature_summary, load_pipeline, meta, model_test_metrics,
                             same_series)
from ahead.storage import authenticate, create_user, init_db, save_screening
from ahead.triage import LEGACY_RULES, RULES, triage

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DOCS = Path(__file__).resolve().parents[1] / "docs" / "data_sources.md"
KIDNEY = {"age": 50, "gender": "Male", "bmi": 27.0, "smoking_history": "never", "physically_active": 1,
          "hypertension": 0, "diabetes": 0, "heart_disease": 0, "family_kidney": 0, "systolic_bp": None,
          "diastolic_bp": None, "creatinine": None, "egfr": None, "uacr": None}
DIABETES = {"gender": "Female", "age": 50, "smoking_history": "never", "bmi": 27.0, "hypertension": 0,
            "heart_disease": 0, "HbA1c_level": None, "blood_glucose_level": None}


def pdf_text(data: bytes) -> str:
    return " ".join(" ".join((page.extract_text() or "").split()) for page in PdfReader(BytesIO(data)).pages)


class TriageRepairTests(unittest.TestCase):
    def test_older_kidney_screenings_keep_their_hba1c_and_fasting_sugar_rules(self):
        legacy = {"Age": 55, "GFR": 75, "ACR": 12, "HbA1c": 9.1, "FastingBloodSugar": 190}
        result = triage("kidney", legacy, 0.8, 0.5)
        self.assertEqual(result["track"], "urgent")
        self.assertEqual([r["feature"] for r in result["reasons"]], ["HbA1c", "FastingBloodSugar"])
        self.assertIsNone(result["lab"])
        self.assertIn("a level doctors use when diagnosing diabetes", result["reasons"][0]["patient"])
        self.assertEqual(triage("kidney", {k: v for k, v in legacy.items() if k != "ACR"}, 0.2, 0.5)["track"], "urgent")
        # Below the cut-offs, the older screening still sorts by its eGFR and ACR.
        self.assertEqual(triage("kidney", legacy | {"HbA1c": 6.0, "FastingBloodSugar": 110}, 0.8, 0.5)["track"],
                         "not_flagged")
        # The current form never has these keys, and they are not current rules.
        self.assertFalse({r.feature for r in LEGACY_RULES["kidney"]} & set(form_fields("kidney")))
        self.assertFalse({r.feature for r in LEGACY_RULES["kidney"]} & {r.feature for r in RULES["kidney"]})

    def test_prediabetes_hba1c_never_rounds_onto_the_urgent_cut_off(self):
        result = triage("diabetes", {"HbA1c_level": 6.499, "age": 50}, 0.3, 0.14)
        self.assertEqual(result["track"], "at_risk")
        for audience in ("patient", "doctor"):
            self.assertNotIn("6.5%", result["text"][audience]["summary"].split("(")[0])
            self.assertIn("6.499%", result["text"][audience]["summary"])

    def test_kidney_reasons_never_say_to_wait_three_months_first(self):
        result = triage("kidney", {"age": 50, "gender": "Male", "creatinine": 6.0}, 0.5, 0.11)
        reason = result["reasons"][0]
        self.assertIn("repeating this test soon to rule out a short-term change", reason["patient"])
        self.assertIn("repeat promptly to exclude AKI", reason["doctor"])
        self.assertNotIn("repeating the test in about 3 months", reason["patient"])


class ModelRepairTests(unittest.TestCase):
    def test_every_model_flags_at_its_own_tuned_threshold(self):
        for disease in DISEASES:
            tuned = meta(disease)["model_thresholds"]
            with self.subTest(disease=disease):
                for name, threshold in tuned.items():
                    self.assertEqual(decision_threshold(disease, name), threshold)
                    self.assertEqual(model_test_metrics(disease, name)["threshold"], threshold)
                    self.assertLess(threshold, 0.5)   # 0.5 would flag almost no one after unweighted training
                self.assertEqual(decision_threshold(disease), meta(disease)["decision_threshold"])
                final = meta(disease)["final_test_metrics"]
                best = model_test_metrics(disease, best_model(disease))
                self.assertEqual({k: best[k] for k in final}, final)

    def test_scores_from_a_retired_model_of_the_same_name_are_not_compared(self):
        old = {"id": 1, "disease": "heart", "model": "LogisticRegression", "threshold": 0.6, "score": 0.7}
        new = {"id": 2, "disease": "heart", "model": "LogisticRegression", "threshold": 0.08, "score": 0.05}
        self.assertFalse(same_series(new, old))
        self.assertTrue(same_series(new, dict(new, id=3, score=0.2)))
        self.assertFalse(same_series(new, dict(new, model="SVM")))
        self.assertEqual(intelligence._same_model([new, old]), [(new, None)])
        newer = dict(new, id=3, score=0.09)
        self.assertEqual(intelligence._same_model([newer, new, old]), [(newer, new)])


class CopyRepairTests(unittest.TestCase):
    def test_diabetes_copy_says_who_it_is_for_and_what_a_glucose_does(self):
        info = DISEASES["diabetes"]
        for text in (info["page_subtitle"], info["card_copy"]):
            self.assertIn("not been told by a doctor that they have diabetes", text)
            self.assertNotIn("counts for more", text)
        self.assertIn("200 mg/dL", info["page_subtitle"])
        self.assertIn("never told they have diabetes", screening.DOCTOR_SUBTITLES["diabetes"])
        self.assertEqual(FIELD_LABELS["blood_glucose_level"], "Blood Glucose Level (mg/dL, any time of day)")
        answer = LOCAL_ANSWERS["screening conditions"]
        self.assertIn("a blood glucose result counts only at 200 mg/dL or more", answer)
        self.assertNotIn("count for more", answer)
        self.assertNotIn("counts for more", answer)
        self.assertIn("180/120", answer)
        self.assertNotIn("count for more than the model's", SYSTEM_PROMPT)

    def test_kidney_copy_says_which_test_values_count(self):
        subtitle = DISEASES["kidney"]["page_subtitle"]
        self.assertNotIn("count for more", subtitle)
        self.assertIn("An eGFR below 60 (entered, or worked out from your creatinine) or a urine ACR of 30 mg/g or "
                      "more is flagged for a doctor", subtitle)
        self.assertNotIn("creatinine) below", subtitle)   # creatinine itself is never 'below 60'

    def test_history_questions_match_the_nhanes_variable(self):
        self.assertEqual(FORM_QUESTIONS["diabetes"]["heart_disease"],
                         "Has a doctor told you that you have coronary heart disease?")
        self.assertEqual(FORM_QUESTIONS["diabetes"]["hypertension"],
                         "Has a doctor told you that you have high blood pressure?")
        self.assertNotIn("angina", FORM_QUESTIONS["kidney"]["heart_disease"])

    def test_docs_say_how_the_kidney_pipeline_fills_blank_activity(self):
        pre = load_pipeline("kidney", best_model("kidney")).named_steps["preprocessor"]
        statistic = next(dict(zip(cols, (step.named_steps["imputer"] if hasattr(step, "named_steps") else step)
                                       .statistics_))["physically_active"]
                         for _, step, cols in pre.transformers_ if "physically_active" in list(cols))
        self.assertEqual(statistic, 0.0)
        text = " ".join(DOCS.read_text().split())
        self.assertIn("fills those blanks with the training median, 0 (not active)", text)
        self.assertNotIn("most common answer (active)", text)

    def test_disease_analytics_caption_drops_the_retired_blank_lab_threshold(self):
        source = inspect.getsource(intelligence._disease)
        self.assertNotIn("some models use a lower threshold when HbA1c", source)


class ScreeningRepairTests(unittest.TestCase):
    def test_next_steps_follow_the_track_a_lab_value_set(self):
        at_risk = triage("diabetes", DIABETES | {"HbA1c_level": 6.0}, 0.05, 0.14)
        self.assertEqual(screening.get_next_steps("diabetes", 0, track=at_risk), screening.NEXT_STEPS["diabetes"][1])
        cleared = triage("kidney", KIDNEY | {"egfr": 95, "uacr": 8}, 0.5, 0.11)
        self.assertEqual(screening.get_next_steps("kidney", 1, track=cleared), screening.NEXT_STEPS["kidney"][0])
        doctor = triage("diabetes", DIABETES | {"HbA1c_level": 5.2}, 0.48, 0.14)
        self.assertEqual(screening.get_next_steps("diabetes", 1, True, doctor), screening.DOCTOR_NEXT_STEPS["diabetes"][0])
        self.assertEqual(screening.get_next_steps("diabetes", 1), screening.NEXT_STEPS["diabetes"][1])   # no track

    def test_insight_prompt_names_a_track_a_lab_value_set(self):
        label, line, rule = screening._insight_track(0, triage("diabetes", DIABETES | {"HbA1c_level": 6.0}, 0.05, 0.14),
                                                     False)
        self.assertEqual(label, "At risk (set by a measured lab value)")
        self.assertIn('AHEAD track: "At risk". HbA1c of 6% is in the prediabetes range', line)
        self.assertIn("Never describe the result as reassuring or low risk.", rule)
        label, line, rule = screening._insight_track(1, triage("kidney", KIDNEY | {"egfr": 95, "uacr": 8}, 0.5, 0.11),
                                                     True)
        self.assertEqual((label, rule), ("Not flagged (set by a measured lab value)", ""))
        self.assertIn("eGFR 95 and urine ACR 8 mg/g are in the usual range", line)

    def test_missing_lab_note_does_not_call_a_lab_set_result_an_estimate(self):
        values = KIDNEY | {"age": 66, "gender": "Female", "creatinine": 1.6}
        self.assertEqual(triage("kidney", values)["reasons"][0]["feature"], "egfr")
        note = screening.missing_note("kidney", {"values": values})
        self.assertEqual(note.split("\n\n")[0], "You left urine ACR blank. Ask your doctor about urine ACR.")
        self.assertNotIn("its estimate", note)
        self.assertNotIn("A blood test for creatinine", note)
        doctor = screening.missing_note("kidney", {"values": values}, doctor=True)
        self.assertEqual(doctor.split("\n\n")[0], "Urine ACR was left blank. Ask for a urine ACR test.")
        # With no lab entered the result is the model's estimate, and the note still says so.
        self.assertIn("its estimate from your answers", screening.missing_note("kidney", {"values": KIDNEY}))

    def test_upload_template_leaves_rules_only_values_blank(self):
        for disease in DISEASES:
            stats = feature_summary(disease)
            row = example_row(disease, form_fields(disease), stats)
            with self.subTest(disease=disease):
                for field in OPTIONAL_INPUTS[disease]:
                    self.assertEqual(row[field], "" if field not in ("HadHeartAttack", "HadAngina") else "No", field)
                values = {k: v for k, v in row.items() if k not in ("PatientID", "DOB")}
                self.assertEqual(triage(disease, values, 0.9, 0.5)["track"], "at_risk")


class ReportRepairTests(unittest.TestCase):
    def report(self, disease, values, score, threshold, audience="patient"):
        track = triage(disease, values, score, threshold)
        return track, pdf_text(patient_report(disease, "P-1", values, score, threshold, best_model(disease), {},
                                              audience=audience, track=track))

    def test_kidney_pdf_prints_the_egfr_worked_out_from_creatinine(self):
        values = KIDNEY | {"age": 66, "gender": "Female", "creatinine": 1.6, "uacr": 12}
        track, text = self.report("kidney", values, 0.2, 0.11)
        self.assertEqual(track["track"], "urgent")
        self.assertIn("eGFR 35.3 (from creatinine)", text)
        self.assertNotIn("eGFR Not provided", text)

    def test_pdf_headline_is_the_track_with_the_model_flag_under_it(self):
        _, text = self.report("diabetes", DIABETES | {"HbA1c_level": 6.0}, 0.05, 0.14)
        self.assertIn("At risk The model's score is below the classification threshold.", text)
        self.assertIn("My result was flagged. What should we look at first?", text)
        _, text = self.report("heart", {**{k: None for k in form_fields("heart")}, "HadHeartAttack": "Yes"}, 0.03, 0.08)
        self.assertIn("See a doctor soon The model's score is below the classification threshold.", text)


class PageRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/repairs.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        create_user("p@test.local", "test-password-123", "Patient", "Amy Lee")
        create_user("d@test.local", "test-password-123", "Doctor", "Dr. Layla Hassan")
        self.patient = authenticate("p@test.local", "test-password-123", "Patient")
        self.doctor = authenticate("d@test.local", "test-password-123", "Doctor")
        patcher = mock.patch.object(screening, "gemini_available", lambda: False)
        patcher.start(); self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def page(self, user, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user, "nav": nav, **state}.items():
            at.session_state[key] = value
        at.run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def test_doctor_home_counts_results_by_track(self):
        grant_doctor(self.patient["id"], "d@test.local")
        # Model-flagged but cleared by HbA1c 5.2; at risk from HbA1c 6.0 with a low score; urgent with a low score.
        save_screening(self.patient["id"], "diabetes", "LogisticRegression", 0.9, 1, 0.14, DIABETES | {"HbA1c_level": 5.2})
        care = overview._care_snapshot(self.doctor["id"])
        self.assertEqual((care["flagged"], care["flagged_ids"], care["review_ids"]), (0, [], []))
        save_screening(self.patient["id"], "diabetes", "LogisticRegression", 0.05, 0, 0.14, DIABETES | {"HbA1c_level": 6.0})
        save_screening(self.patient["id"], "diabetes", "LogisticRegression", 0.05, 0, 0.14, DIABETES | {"HbA1c_level": 7.1})
        care = overview._care_snapshot(self.doctor["id"])
        self.assertEqual(care["flagged"], 2)
        self.assertEqual(care["flagged_ids"], care["review_ids"])
        self.assertEqual(care["flagged_ids"], [self.patient["id"]])

    def test_result_card_and_ring_name_the_model_when_a_lab_or_history_sets_the_track(self):
        model = best_model("diabetes")
        threshold = decision_threshold("diabetes", model)
        result = {"diabetes": {"values": DIABETES | {"HbA1c_level": 6.0}, "probability": 0.05, "prediction": 0,
                               "model": model, "threshold": threshold, "missing": []}}
        at = self.page(self.patient, "screenings", active_screening="diabetes", last_results=result)
        page = html.unescape("\n".join(h.proto.body for h in at.get("html")))
        self.assertIn('<div class="result-title">Model estimate</div>', page)
        self.assertIn('<span class="visual-badge">Model: not flagged</span>', page)
        self.assertIn('class="track-panel at-risk"', page)
        model = best_model("heart")
        threshold = decision_threshold("heart", model)
        stats = feature_summary("heart")
        values = {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["median"]
                  for f in form_fields("heart") if f in stats} | {"HadHeartAttack": "Yes", "HadAngina": "No"}
        result = {"heart": {"values": values, "probability": 0.03, "prediction": 0, "model": model,
                            "threshold": threshold, "missing": []}}
        at = self.page(self.patient, "screenings", active_screening="heart", last_results=result)
        page = html.unescape("\n".join(h.proto.body for h in at.get("html")))
        self.assertIn('<div class="result-title">Model estimate</div>', page)
        self.assertIn('<span class="visual-badge">Score does not apply</span>', page)
        self.assertNotIn("Lower Predicted Risk", page)

    def test_patient_home_table_shows_the_track(self):
        save_screening(self.patient["id"], "diabetes", "LogisticRegression", 0.05, 0, 0.14, DIABETES | {"HbA1c_level": 6.0})
        at = self.page(self.patient, "overview")
        table = next(df.value for df in at.dataframe if "Result" in df.value.columns)
        self.assertEqual(list(table["Result"]), ["At risk"])


class ResultCardOverrideTests(unittest.TestCase):
    def test_urgent_by_a_measured_value_the_model_did_not_flag_is_titled_as_the_models_estimate(self):
        from ahead.components import model_overridden
        from ahead.triage import triage
        kidney = {"age": 70, "gender": "Female", "creatinine": 1.6}
        urgent = triage("kidney", kidney, score=0.19, threshold=0.22)
        self.assertEqual(urgent["track"], "urgent")
        self.assertEqual(model_overridden(urgent), "lab")    # card reads 'Model estimate', not 'Lower Predicted Risk'
        flagged = triage("kidney", kidney, score=0.4, threshold=0.22)
        self.assertEqual(model_overridden(flagged), "")      # model and track agree: the usual title
        plain = triage("kidney", {"age": 40, "gender": "Male"}, score=0.1, threshold=0.22)
        self.assertEqual(model_overridden(plain), "")


if __name__ == "__main__":
    unittest.main()
