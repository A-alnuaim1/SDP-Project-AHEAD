"""The switch of diabetes and kidney to NHANES models: eGFR from creatinine (CKD-EPI 2021), the tracks a measured lab
value sets, model inputs that never hold a lab value, the new kidney form, and older saved records (the retired kidney
form's keys, the old diabetes smoking answers) that still display, sort and print without being re-scored. Page checks
run on a temporary database; nothing leaves the machine."""
import html
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import ahead.screening as screening
from ahead.clinical_data import SMOKED_100, model_frame, model_inputs, shown_value, validate_row
from ahead.config import DATA_DIR, MODEL_INPUTS, OPTIONAL_INPUTS, form_fields, is_legacy
from ahead.platform_data import grant_doctor, init_platform
from ahead.prevention import facts, fit_text, smoking_status
from ahead.reports import patient_report
from ahead.resources import best_model, feature_summary, features, load_pipeline
from ahead.storage import connection, init_db, insert_records, list_records, load_screenings, save_screening
from ahead.triage import egfr_from_creatinine, missing_tests, open_urgent, prepared_values, triage, triage_saved

APP = str(Path(__file__).resolve().parents[1] / "app.py")
FLAGGED, CLEAR = (0.8, 0.5), (0.2, 0.5)
LABS = {"HbA1c_level", "blood_glucose_level", "creatinine", "egfr", "uacr", "systolic_bp", "diastolic_bp"}
# A kidney screening saved by the retired form (Sara in the demo): eGFR 48 and ACR 40 under the old keys.
LEGACY_SARA = {"Age": 66, "Gender": 1, "BMI": 29.4, "Smoking": 1, "GFR": 48, "ACR": 40, "SerumCreatinine": 1.4,
               "SystolicBP": 150, "DiastolicBP": 92, "HbA1c": None, "FastingBloodSugar": None, "Edema": 0}
SARA = {"age": 66, "gender": "Female", "bmi": 29.4, "smoking_history": "former", "physically_active": 0,
        "hypertension": 1, "diabetes": 0, "heart_disease": 0, "family_kidney": 1, "systolic_bp": 150,
        "diastolic_bp": 92, "creatinine": 1.4, "egfr": 48, "uacr": 40}


def answers(disease, **changes):
    """A form's typical answers, optional labs and readings blank."""
    stats = feature_summary(disease)
    values = {f: None if f in OPTIONAL_INPUTS.get(disease, []) else stats[f]["default"] if stats[f]["kind"] == "category"
              else stats[f]["mode"] if stats[f]["binary"] else stats[f]["median"] for f in form_fields(disease)}
    return values | changes


class EgfrTests(unittest.TestCase):
    def test_known_values(self):
        # 2021 race-free CKD-EPI: 142 x min(Scr/k,1)^a x max(Scr/k,1)^-1.200 x 0.9938^age (x 1.012 female).
        for creatinine, age, sex, expected in ((1.0, 50, "Male", 91.7), (0.8, 60, "Female", 84.3),
                                               (1.6, 66, "Female", 35.3), (0.7, 30, "Female", 119.2),
                                               (1.2, 40, "male", 78.4), (2.5, 75, "Male", 26.1)):
            with self.subTest(creatinine=creatinine, age=age, sex=sex):
                self.assertAlmostEqual(egfr_from_creatinine(creatinine, age, sex), expected, places=1)
        # The retired kidney form coded sex as 0 (male) / 1 (female); text from an upload works too.
        self.assertEqual(egfr_from_creatinine(0.8, 60, 1), egfr_from_creatinine("0.8", "60", "Female"))
        self.assertEqual(egfr_from_creatinine(1.0, 50, 0), egfr_from_creatinine(1.0, 50, "M"))

    def test_matches_the_nhanes_research_file(self):
        # The training labels use the research file's eGFR column: CKD-EPI 2021 on CDC-corrected creatinine.
        data = pd.read_csv(DATA_DIR / "nhanes" / "nhanes_1999_2023_research.csv.gz",
                           usecols=["age", "sex", "creatinine_mg_dl", "egfr_calculated"]).dropna().head(500)
        for row in data.itertuples():
            self.assertAlmostEqual(egfr_from_creatinine(row.creatinine_mg_dl, row.age, row.sex), row.egfr_calculated,
                                   delta=0.051)

    def test_no_estimate_without_what_it_needs(self):
        for args in ((None, 50, "Male"), ("", 50, "Male"), (0, 50, "Male"), (-1, 50, "Male"), (1.0, 17, "Male"),
                     (1.0, None, "Male"), (1.0, 50, None), (1.0, 50, "Other"), (1.0, 50, ""), ("abc", 50, "Male")):
            self.assertIsNone(egfr_from_creatinine(*args), args)


class TrackRuleTests(unittest.TestCase):
    def test_diabetes_prediabetes_range_is_at_risk_whatever_the_model_says(self):
        for score in (FLAGGED, CLEAR, ()):
            for hba1c in (5.7, 6.0, 6.49):
                with self.subTest(score=score, hba1c=hba1c):
                    result = triage("diabetes", {"age": 50, "HbA1c_level": hba1c}, *score)
                    self.assertEqual((result["track"], result["reasons"]), ("at_risk", []))
                    self.assertEqual(result["lab"]["features"], ["HbA1c_level"])
                    patient = result["text"]["patient"]["summary"]
                    self.assertIn("prediabetes range (5.7% to 6.4%)", patient)
                    self.assertIn("prevention plan agreed with your doctor", patient)
                    self.assertIn("not a diagnosis", patient)
                    self.assertIn("prediabetes range", result["text"]["doctor"]["summary"])
        not_flagged = triage("diabetes", {"HbA1c_level": 6.1}, *CLEAR)["text"]["patient"]["summary"]
        self.assertIn("The model's estimate did not flag this result, but a measured value counts for more", not_flagged)

    def test_a_measured_hba1c_below_5_7_clears_a_model_flag(self):
        result = triage("diabetes", {"age": 50, "HbA1c_level": 5.4}, *FLAGGED)
        self.assertEqual((result["track"], result["flagged"]), ("not_flagged", True))
        self.assertIn("Your HbA1c of 5.4% is below the prediabetes range", result["text"]["patient"]["summary"])
        self.assertIn("so it is not flagged", result["text"]["patient"]["summary"])
        self.assertIn("takes precedence over the model score", result["text"]["doctor"]["summary"])
        # 5.699 never prints as 5.7 under 'below 5.7'.
        self.assertIn("5.699%", triage("diabetes", {"HbA1c_level": 5.699}, *FLAGGED)["text"]["patient"]["summary"])
        self.assertEqual(fit_text(triage("diabetes", {"HbA1c_level": 6.0}, *CLEAR))[:5], "HbA1c")

    def test_without_hba1c_the_model_decides(self):
        for values in ({"age": 50}, {"age": 50, "HbA1c_level": None, "blood_glucose_level": 150}):
            self.assertEqual(triage("diabetes", values, *FLAGGED)["track"], "at_risk")
            self.assertEqual(triage("diabetes", values, *CLEAR)["track"], "not_flagged")
            self.assertIsNone(triage("diabetes", values, *FLAGGED)["lab"])
        # Urgent values still win, whatever the model and the other lab.
        self.assertEqual(triage("diabetes", {"HbA1c_level": 6.5}, *CLEAR)["track"], "urgent")
        self.assertEqual(triage("diabetes", {"HbA1c_level": 5.2, "blood_glucose_level": 210}, *CLEAR)["track"], "urgent")

    def test_a_child_in_the_prediabetes_range_needs_a_doctor(self):
        result = triage("diabetes", {"age": 15, "HbA1c_level": 6.0}, *CLEAR)
        self.assertEqual((result["track"], [r["kind"] for r in result["reasons"]]), ("urgent", ["adults_only"]))
        # A measured value under 5.7% clears a flagged child too.
        self.assertEqual(triage("diabetes", {"age": 15, "HbA1c_level": 5.2}, *FLAGGED)["track"], "not_flagged")

    def test_kidney_without_labs_is_the_model_path(self):
        values = answers("kidney")
        self.assertEqual(triage("kidney", values, *FLAGGED)["track"], "at_risk")
        self.assertEqual(triage("kidney", values, *CLEAR)["track"], "not_flagged")
        self.assertEqual(triage("kidney", values)["track"], "no_score")
        # One test alone does not set the track: both are needed for 'usual range'.
        for one in ({"egfr": 85}, {"uacr": 10}, {"creatinine": 0.9, "gender": None}):
            self.assertEqual(triage("kidney", values | one, *FLAGGED)["track"], "at_risk", one)

    def test_kidney_labs_past_a_cut_off_are_urgent_with_a_repeat_test(self):
        for values, features in (({"egfr": 48}, ["egfr"]), ({"uacr": 40}, ["uacr"]),
                                 ({"egfr": 48, "uacr": 40}, ["egfr", "uacr"])):
            with self.subTest(values=values):
                result = triage("kidney", answers("kidney") | values, *CLEAR)
                self.assertEqual(result["track"], "urgent")
                self.assertEqual([r["feature"] for r in result["reasons"]], features)
                for reason in result["reasons"]:   # repeat soon first, never "wait 3 months"
                    self.assertIn("repeating this test soon to rule out a short-term change and again after about 3 "
                                  "months", reason["patient"])
                    self.assertIn("repeat promptly to exclude AKI; chronicity is confirmed at 3 months", reason["doctor"])
        # eGFR from creatinine when eGFR is blank: a 66-year-old woman with creatinine 1.6 has an eGFR of about 35.
        derived = triage("kidney", answers("kidney", age=66, gender="Female", creatinine=1.6), *CLEAR)
        reason = derived["reasons"][0]
        self.assertEqual((derived["track"], reason["feature"], reason["value_text"]), ("urgent", "egfr", "35.3"))
        self.assertIn("worked out from your creatinine of 1.6 mg/dL", reason["patient"])
        self.assertIn("CKD-EPI 2021", reason["doctor"])
        # The laboratory's own eGFR wins over the estimate.
        lab = triage("kidney", answers("kidney", age=66, gender="Female", creatinine=1.6, egfr=72, uacr=10), *FLAGGED)
        self.assertEqual(lab["track"], "not_flagged")
        # Severe blood pressure stays urgent. HbA1c and fasting sugar are only on older saved screenings
        # (triage.LEGACY_RULES), which still sort as they did.
        self.assertEqual(triage("kidney", {"systolic_bp": 182}, *CLEAR)["track"], "urgent")
        self.assertEqual(triage("kidney", {"HbA1c": 7.5, "FastingBloodSugar": 200}, *CLEAR)["track"], "urgent")

    def test_kidney_labs_in_the_usual_range_are_not_flagged(self):
        for score in (FLAGGED, CLEAR, ()):
            result = triage("kidney", answers("kidney", egfr=85, uacr=12), *score)
            self.assertEqual((result["track"], result["lab"]["features"]), ("not_flagged", ["egfr", "uacr"]))
            patient = result["text"]["patient"]["summary"]
            self.assertIn("are both in the usual range", patient)
            self.assertIn("repeated once a year", patient)
            self.assertIn("Repeat both tests yearly when diabetes or hypertension is present",
                          result["text"]["doctor"]["summary"])
        derived = triage("kidney", answers("kidney", age=50, gender="Male", creatinine=0.9, uacr=12), *FLAGGED)
        self.assertEqual(derived["track"], "not_flagged")
        self.assertIn("worked out from your creatinine of 0.9 mg/dL", derived["text"]["patient"]["summary"])


class ModelInputTests(unittest.TestCase):
    def test_model_inputs_never_hold_a_lab_value(self):
        for disease in MODEL_INPUTS:
            self.assertEqual(features(disease), list(MODEL_INPUTS[disease]))
            self.assertFalse(LABS & set(features(disease)), disease)
            pipeline = load_pipeline(disease, best_model(disease))
            self.assertFalse(LABS & set(map(str, pipeline.named_steps["preprocessor"].feature_names_in_)), disease)
            values = answers(disease, smoking_history="current")
            row = model_inputs(disease, values)
            self.assertEqual(set(row), set(MODEL_INPUTS[disease]))
            self.assertEqual(row["smoked_100"], 1)
            # Whatever the labs say, the model's score is the same.
            with_labs = values | {"HbA1c_level": 9.9, "blood_glucose_level": 300, "creatinine": 4.0, "egfr": 15,
                                  "uacr": 900, "systolic_bp": 190, "diastolic_bp": 125}
            scores = pipeline.predict_proba(model_frame(disease, [values, with_labs], features(disease)))[:, 1]
            self.assertEqual(scores[0], scores[1])
            self.assertEqual(screening.model_row(disease, with_labs), screening.model_row(disease, values))

    def test_answers_map_to_the_model_columns(self):
        self.assertEqual(SMOKED_100, {"never": 0, "former": 1, "current": 1, "ever": 1, "not current": 1})
        for answer, coded in (("never", 0), ("former", 1), ("current", 1), ("ever", 1), ("not current", 1),
                              ("No Info", None), (None, None), ("", None)):
            row = model_inputs("diabetes", {"smoking_history": answer})
            self.assertTrue(np.isnan(row["smoked_100"]) if coded is None else row["smoked_100"] == coded, answer)
        row = model_inputs("kidney", {"gender": "female", "age": "52", "bmi": 31.5, "physically_active": "No",
                                      "hypertension": "Yes", "diabetes": 1, "heart_disease": 0})
        self.assertEqual({k: row[k] for k in ("gender", "age", "bmi", "physically_active", "hypertension", "diabetes",
                                              "heart_disease")},
                         {"gender": "Female", "age": 52.0, "bmi": 31.5, "physically_active": 0, "hypertension": 1,
                          "diabetes": 1, "heart_disease": 0})
        self.assertTrue(np.isnan(model_inputs("kidney", {"gender": "Other"})["gender"]))
        # Heart has no mapping: its answers are its model inputs.
        self.assertEqual(model_inputs("heart", {"BMI": 30}), {"BMI": 30})

    def test_the_kidney_form_is_the_short_form(self):
        self.assertEqual(form_fields("kidney"), ["age", "gender", "bmi", "smoking_history", "physically_active",
                                                 "hypertension", "diabetes", "heart_disease", "family_kidney",
                                                 "systolic_bp", "diastolic_bp", "creatinine", "egfr", "uacr"])
        self.assertNotIn("family_kidney", MODEL_INPUTS["kidney"])   # shown to the doctor, never a model input
        stats = feature_summary("kidney")
        self.assertEqual(stats["smoking_history"]["options"], ["never", "former", "current"])
        self.assertEqual(feature_summary("diabetes")["smoking_history"]["options"], ["never", "former", "current"])
        # Uploads take the old diabetes smoking answers, and leave labs blank without an issue.
        clean, issues = validate_row({"smoking_history": "No Info", "HbA1c_level": "", "blood_glucose_level": None},
                                     "diabetes", ["smoking_history", "HbA1c_level", "blood_glucose_level"],
                                     feature_summary("diabetes"))
        self.assertEqual((clean, issues), ({"smoking_history": "No Info", "HbA1c_level": None,
                                            "blood_glucose_level": None}, []))
        clean, issues = validate_row({"egfr": "", "family_kidney": "yes", "bmi": ""}, "kidney",
                                     ["egfr", "family_kidney", "bmi"], stats)
        self.assertEqual(issues, ["bmi: missing; enter a value before scoring."])
        self.assertEqual((clean["egfr"], clean["family_kidney"]), (None, 1))


class LegacyRecordTests(unittest.TestCase):
    def test_legacy_kidney_keys_sort_print_and_ask_like_the_new_ones(self):
        self.assertTrue(is_legacy("kidney", LEGACY_SARA))
        self.assertFalse(is_legacy("kidney", SARA))
        self.assertFalse(is_legacy("diabetes", {"smoking_history": "ever"}))
        old, new = triage_saved("kidney", json.dumps(LEGACY_SARA), 0.81, 0.5), triage_saved("kidney", SARA, 0.3, 0.11)
        self.assertEqual(old["track"], "urgent")
        self.assertEqual([r["feature"] for r in old["reasons"]], [r["feature"] for r in new["reasons"]])
        self.assertEqual([r["patient"] for r in old["reasons"]], [r["patient"] for r in new["reasons"]])
        self.assertEqual(prepared_values("kidney", LEGACY_SARA)["egfr"], 48)
        self.assertEqual(missing_tests("kidney", LEGACY_SARA), [])
        # A newer screening with the new keys clears the older urgent reasons it re-entered.
        rows = [{"disease": "kidney", "values_json": json.dumps(SARA | {"egfr": 75, "uacr": 10}), "score": .2,
                 "threshold": .11},
                {"disease": "kidney", "values_json": json.dumps(LEGACY_SARA), "score": .81, "threshold": .5}]
        self.assertEqual(open_urgent(rows), [])
        self.assertEqual(len(open_urgent(rows[1:])), 1)

    def test_legacy_values_display_and_feed_the_plan(self):
        stats = feature_summary("kidney")
        from ahead.clinical_data import field_info
        self.assertEqual(shown_value("Gender", 1, field_info(stats, "Gender")), "Female")
        self.assertEqual(shown_value("Smoking", 1, field_info(stats, "Smoking")), "Yes")
        self.assertEqual(shown_value("GFR", 48.0, field_info(stats, "GFR")), "48")
        self.assertEqual(smoking_status("kidney", LEGACY_SARA), "current")
        self.assertEqual(smoking_status("kidney", SARA), "former")
        self.assertEqual(facts("kidney", LEGACY_SARA), ["BMI 29.4", "current smoker"])
        self.assertEqual(facts("kidney", SARA), ["BMI 29.4", "former smoker", "not active"])
        self.assertEqual(facts("diabetes", {"bmi": 31.2, "smoking_history": "not current"}), ["BMI 31.2", "former smoker"])
        diabetes = feature_summary("diabetes")["smoking_history"]
        self.assertEqual([shown_value("smoking_history", v, diabetes) for v in ("ever", "not current", "No Info")],
                         ["Ever", "Not current", "No info"])

    def test_legacy_records_print(self):
        for disease, values, score, threshold in (("kidney", LEGACY_SARA, 0.81, 0.5),
                                                  ("diabetes", {"gender": "Other", "age": 58, "smoking_history": "ever",
                                                                "bmi": 31.2, "hypertension": 1, "heart_disease": 0,
                                                                "HbA1c_level": None, "blood_glucose_level": 182},
                                                   0.55, 0.41)):
            for audience in ("doctor", "patient"):
                with self.subTest(disease=disease, audience=audience):
                    track = triage_saved(disease, values, score, threshold)
                    pdf = patient_report(disease, "P-1", values, score, threshold, "XGBoost", {}, audience=audience,
                                         track=track, **({"name": "Sara"} if audience == "patient" else {}))
                    text = " ".join(" ".join(p.extract_text().split()) for p in PdfReader(BytesIO(pdf)).pages)
                    self.assertIn("NEXT STEP", text)
                    if disease == "kidney":
                        self.assertIn("Gender Female", text)
                        self.assertIn("Estimated GFR 48", text)
                        self.assertIn("Smoking Yes", text)


class PageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/switch.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        init_db(); init_platform()
        self.d = self.user("d@test.com", "Doctor", "Dr. Layla Hassan")
        self.p = self.user("p@test.com", "Patient", "Sara Khalil")
        grant_doctor(self.p["id"], "d@test.com")
        patcher = mock.patch.object(screening, "gemini_available", lambda: False)
        patcher.start(); self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def user(self, email, role, name):
        with connection() as db:
            uid = db.execute("INSERT INTO users(email,password_hash,role,name) VALUES(?,?,?,?)", (email, "-", role, name)).lastrowid
        return {"id": uid, "email": email, "role": role, "name": name, "institution": "", "preferences": {}}

    def page(self, user, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user, "nav": nav, **state}.items():
            at.session_state[key] = value
        at.run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def test_kidney_screening_without_labs_gets_a_score_and_is_saved(self):
        at = self.page(self.p, "screenings", active_screening="kidney")
        labels = [w.label for w in list(at.selectbox) + list(at.number_input)]
        self.assertIn("Has a doctor told you that you have high blood pressure?", labels)
        self.assertIn("Does a parent, brother or sister have kidney disease?", labels)
        next(b for b in at.button if b.label == "Generate Screening Result").click().run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        saved = load_screenings(self.p["id"])
        self.assertEqual(len(saved), 1)
        values = json.loads(saved[0]["values_json"])
        self.assertEqual(set(values), set(form_fields("kidney")))
        self.assertTrue(all(values[f] is None for f in ("creatinine", "egfr", "uacr")))
        result = at.session_state["last_results"]["kidney"]
        expected = load_pipeline("kidney", best_model("kidney")).predict_proba(
            model_frame("kidney", [values], features("kidney")))[0, 1]
        self.assertAlmostEqual(result["probability"], expected, places=12)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any(w.startswith("You left creatinine and eGFR, and urine ACR blank.") for w in warnings), warnings)

    def test_diabetes_form_offers_never_former_current(self):
        at = self.page(self.p, "screenings", active_screening="diabetes")
        smoking = next(s for s in at.selectbox if s.key == "diabetes_smoking_history")
        self.assertEqual(list(smoking.options), ["current", "former", "never"])
        smoking.set_value("former")
        at.number_input(key="diabetes_HbA1c_level").set_value(6.1)
        next(b for b in at.button if b.label == "Generate Screening Result").click().run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        panel = next(html.unescape(h.proto.body) for h in at.get("html") if 'class="track-panel' in h.proto.body)
        self.assertIn("prediabetes range", panel)
        self.assertIn('class="track-panel at-risk"', panel)

    def test_legacy_screenings_render_for_the_doctor_and_the_patient(self):
        save_screening(self.p["id"], "kidney", "XGBoost", 0.81, 1, 0.5, LEGACY_SARA)
        save_screening(self.p["id"], "diabetes", "GradientBoosting", 0.55, 1, 0.41,
                       {"gender": "Female", "age": 58, "smoking_history": "ever", "bmi": 31.2, "hypertension": 1,
                        "heart_disease": 0, "HbA1c_level": None, "blood_glucose_level": 182})
        at = self.page(self.d, "intelligence", intelligence_focus={"tab": "Patient Records & History",
                                                                   "patient": self.p["id"]})
        tab = next(t for t in at.tabs if t.label == "Patient Records & History")
        picker = next(s for s in tab.selectbox if s.label == "Open historical assessment")
        for sid in [r["id"] for r in load_screenings(self.p["id"])]:
            picker.set_value(sid); at.run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            tab = next(t for t in at.tabs if t.label == "Patient Records & History")
            inputs = next(df.value for df in tab.dataframe if "Input" in df.value.columns)
            shown = dict(zip(inputs["Input"], inputs["Value"]))
            if "Estimated GFR" in shown:
                self.assertEqual((shown["Gender"], shown["Smoking"], shown["Estimated GFR"]), ("Female", "Yes", "48"))
            else:
                self.assertEqual(shown["Smoking History"], "ever")
        self.page(self.d, "overview")
        home = self.page(self.p, "overview")
        self.assertTrue(any("eGFR of 48" in html.unescape(h.proto.body) for h in home.get("html")))

    def test_legacy_clinical_records_are_never_re_scored(self):
        rows = [{"patient_id": "OLD-1", "values": LEGACY_SARA, "dob": None, "issues": []},
                {"patient_id": "NEW-1", "values": SARA | {"egfr": None, "uacr": None, "creatinine": None}, "dob": None,
                 "issues": []}]
        insert_records(self.d["id"], "kidney-old", "kidney", rows, source_name="mixed.csv")
        old = next(r for r in list_records(self.d["id"], "kidney") if r["patient_id"] == "OLD-1")
        with connection() as db:   # scored by the retired model when it was uploaded
            db.execute("UPDATE records SET score=0.81,threshold=0.5,model='XGBoost',status='Flagged' WHERE id=?", (old["id"],))
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": self.d, "nav": "clinical"}.items():
            at.session_state[key] = value
        at.run()
        next(s for s in at.selectbox if s.label == "Disease screening").set_value("kidney").run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        button = next(b for b in at.button if b.label.startswith("Score "))
        self.assertTrue(button.label.startswith("Score 1 eligible record"), button.label)
        button.click().run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        records = {r["patient_id"]: r for r in list_records(self.d["id"], "kidney")}
        self.assertEqual((records["OLD-1"]["score"], records["OLD-1"]["model"], records["OLD-1"]["status"]),
                         (0.81, "XGBoost", "Flagged"))
        self.assertEqual(json.loads(records["OLD-1"]["values_json"]), LEGACY_SARA)
        self.assertIsNotNone(records["NEW-1"]["score"])
        self.assertEqual(records["NEW-1"]["model"], best_model("kidney"))
        # The legacy record's view says why, shows its own inputs, and its track is still urgent.
        picker = next(s for s in at.selectbox if s.label == "Record")
        picker.set_value(records["OLD-1"]["id"]).run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertTrue(any("earlier version of this condition's form" in i.value for i in at.info))
        values = next(df.value for df in at.dataframe if list(df.value.columns) == ["Input", "Value"])
        self.assertIn("Estimated GFR", list(values["Input"]))
        self.assertTrue(any('class="track-panel urgent"' in h.proto.body for h in at.get("html")))


if __name__ == "__main__":
    unittest.main()
