"""Doctor-facing wording on Screenings, the AI Assistant and Settings, with the patient wording left exactly as it was.
Every check runs on a temporary database with Gemini switched off, so nothing leaves the machine."""
import html
import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

import ahead.pages.assistant as assistant
import ahead.screening as screening
from ahead.platform_data import grant_doctor, init_platform
from ahead.resources import best_model, decision_threshold, feature_summary, features, missing_inputs_profile
from ahead.storage import authenticate, create_user, init_db, load_screenings

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CLINIC = "Find a nearby clinic (map search)"
PATIENT_ELEVATED = ("The screening model identified a pattern associated with elevated diabetes risk. This result is "
                    "intended for early awareness and does not confirm that you have the condition.")
# Screenings run by a doctor are not attached to a record, so the card speaks about the entered values.
DOCTOR_ELEVATED = ("The entered values matched a pattern associated with elevated diabetes risk; this is not a "
                   "diagnosis and does not confirm that the patient has the condition.")
SENSITIVITY = [("BMI", 35.0, 27.652077393512773, 0.031), ("Smoking", 1, 0, -0.012),
               ("AlcoholConsumption", 9.95, 9.95150288308271, 0.0), ("GFR", 41.123456, 66.56, -0.00001)]
PATIENT_HEADER = "Personalize your experience, privacy and care team access."


def typical(disease):
    stats = feature_summary(disease)
    return {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"] if stats[f]["binary"]
            else stats[f]["median"] for f in features(disease)}


def scored(disease, prediction, probability):
    model = best_model(disease)
    return {disease: {"values": typical(disease), "probability": probability, "prediction": prediction, "model": model,
                      "threshold": decision_threshold(disease, model), "missing": []}}


class DoctorWordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/wording.db"
        init_db(); init_platform()
        for email, role, name in [("p@test.local", "Patient", "Pat Test"), ("d@test.local", "Doctor", "Dr Test")]:
            create_user(email, "safe-password-123", role, name)
        self.patient = authenticate("p@test.local", "safe-password-123", "Patient")
        self.doctor = authenticate("d@test.local", "safe-password-123", "Doctor")
        grant_doctor(self.patient["id"], "d@test.local")
        for module in (screening, assistant):
            patcher = mock.patch.object(module, "gemini_available", lambda: False)
            patcher.start(); self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    def page(self, user, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user, "nav": nav, **state}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def text(self, at):
        """Everything a reader sees, with HTML entities decoded."""
        parts = [h.proto.body for h in at.get("html")] + [m.value for m in at.markdown] + [c.value for c in at.caption]
        parts += [a.value for kind in (at.info, at.warning, at.error, at.success) for a in kind]
        parts += [s.value for s in at.subheader]
        return html.unescape("\n".join(parts))

    def links(self, at):
        return [link.proto.label for link in at.get("link_button")]

    # ------------------------------------------------------------------ screenings
    def test_doctor_screening_result_is_worded_for_a_clinician(self):
        at = self.page(self.doctor, "screenings", active_screening="diabetes", last_results=scored("diabetes", 1, 0.8))
        text = self.text(at)
        self.assertIn(DOCTOR_ELEVATED, text)
        self.assertIn("Points to review", text)
        for title, body in screening.DOCTOR_NEXT_STEPS["diabetes"][1]:
            self.assertIn(title, text); self.assertIn(body, text)
        self.assertIn("as clinically indicated", text)
        self.assertIn("Factors identified from the entered values", text)
        self.assertIn("Showing the most recent result for this screening.", text)
        self.assertIn("Optional. Recent laboratory values; leave blank any that are not available.", text)
        self.assertIn("Red flags:", text)
        for patient_copy in ("you have the condition", "Your entered information", "Book an appointment",
                             "Recommended Next Steps", "Factors Identified From Your Inputs", "you don't have",
                             "Personalized explanation of your screening", "Your result: model input sensitivity",
                             "Seek urgent medical care"):
            self.assertNotIn(patient_copy, text)
        self.assertEqual(self.links(at), [])
        self.assertNotIn("this record", text.lower())   # the picker says these screenings are not attached to one
        self.assertIn('<div class="result-title">Flagged · elevated risk pattern</div>', "\n".join(h.proto.body for h in at.get("html")))
        # Models and model columns read as words, as in the queue and Patient Intelligence.
        self.assertIn("Model: Gradient boosting ·", "\n".join(h.proto.body for h in at.get("html")))
        self.assertTrue(any(i.value.startswith("Gradient boosting test-set precision") for i in at.info))
        self.assertEqual(next(s for s in at.selectbox if s.label == "Prediction model").options[0], "Gradient boosting — Recommended")
        bars = self.importance(at, "Diabetes")
        self.assertIn("HbA1c Level (%)", bars)
        self.assertTrue({"Gender: Female", "Smoking History: never"} & set(bars))
        self.assertFalse([b for b in bars if "_" in b])
        # The history trend would join different people's screenings, so doctors get a note instead of the tab.
        self.assertEqual([t.label for t in at.tabs], ["Points to review", "Key factors", "AHEAD Insight", "Model information"])
        self.assertIn("No score history is shown for doctor accounts", text)
        self.assertEqual(len(at.subheader), 0)

        lower_result = scored("kidney", 0, 0.1)   # the typical kidney row is urgent (ACR, HbA1c and FBS), so clear it
        lower_result["kidney"]["values"].update(GFR=75, ACR=10, HbA1c=5.4, FastingBloodSugar=90)
        lower = self.page(self.doctor, "screenings", active_screening="kidney", last_results=lower_result)
        text = self.text(lower)
        self.assertIn("This does not rule out disease or replace routine screening for the patient.", text)
        self.assertIn(screening.DOCTOR_NEXT_STEPS["kidney"][0][0][0], text)
        self.assertIn("leave blank any others that are not available", text)
        self.assertNotIn("you don't have", text)
        self.assertEqual(self.links(lower), [])

    def importance(self, at, label):
        """The bar labels of the 'Leading Model Features' chart."""
        charts = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
        return next(c for c in charts if c["layout"]["title"]["text"] == f"{label} — Leading Model Features")["data"][0]["y"]

    def headings(self, at):
        return re.findall(r'<div class="section-heading"><h2>([^<]+)</h2>(?:<p>([^<]*)</p>)?',
                          "\n".join(h.proto.body for h in at.get("html")))

    def test_doctor_model_tab_reads_like_the_clinical_dashboard(self):
        with mock.patch.object(screening, "individual_sensitivity", return_value=SENSITIVITY):
            at = self.page(self.doctor, "screenings", active_screening="kidney", last_results=scored("kidney", 1, 0.8))
        headings = self.headings(at)
        self.assertEqual([title for title, _ in headings], [
            "Prediction model", "Screening result", "Points to review", "Factors identified from the entered values",
            "AHEAD Insight", "Model interpretation", "Model input sensitivity for these values"])
        self.assertTrue(all(subtitle for _, subtitle in headings))
        self.assertTrue(headings[-1][1].startswith("For each row below, one entered value was replaced"))
        self.assertFalse([m.value for m in at.markdown if m.value.startswith("#")])
        # Rounded words and numbers; a row that only differs by rounding (9.95 vs 9.9515…) is left out.
        table = next(df.value for df in at.dataframe if "Typical value" in df.value.columns)
        self.assertEqual(list(table.columns), ["Input", "Entered", "Typical value", "Score change (points)"])
        self.assertEqual(table.values.tolist(), [["Body Mass Index (BMI)", "35", "27.65", 3.1],
                                                 ["Smoking", "Yes", "No", -1.2], ["Estimated GFR", "41.12", "66.56", 0.0]])
        self.assertEqual(str(table["Score change (points)"].iloc[-1]), "0.0")   # never '-0.0'
        chart = next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if "sensitivity" in c.proto.spec)
        self.assertEqual(list(chart["data"][0]["y"]), ["Body Mass Index (BMI)", "Smoking", "Estimated GFR"])
        self.assertEqual(chart["data"][0]["marker"]["color"], ["#C9706E", "#5B8FA3", "#5B8FA3"])
        self.assertEqual(chart["layout"]["xaxis"]["title"]["text"], "Score change (points)")   # as the table column
        entered = next(df.value for df in at.dataframe if list(df.value.columns) == ["Input", "Value"]).set_index("Input")["Value"]
        stats = feature_summary("kidney")
        self.assertEqual(entered["Gender"], ("Male", "Female")[int(stats["Gender"]["mode"])])
        self.assertEqual((entered["Smoking"], entered["Presence of Edema"]), ("No", "No"))
        self.assertEqual(entered["Body Mass Index (BMI)"], f"{stats['BMI']['median']:.2f}")
        self.assertFalse([v for v in entered if re.fullmatch(r"[01]\.0|\d+\.\d{3,}", v)])

        # An optional lab left blank says the model filled it in, as in Patient Intelligence.
        result = scored("kidney", 1, 0.8)
        result["kidney"]["values"] |= {"HemoglobinLevels": None}
        result["kidney"]["missing"] = ["HemoglobinLevels"]
        at = self.page(self.doctor, "screenings", active_screening="kidney", last_results=result)
        entered = next(df.value for df in at.dataframe if list(df.value.columns) == ["Input", "Value"]).set_index("Input")["Value"]
        self.assertEqual(entered["Hemoglobin Level"], "Not provided (the model used a typical value)")
        self.assertFalse([v for v in entered if v == "Not entered"])
        # Every remaining row reads the same once rounded: the doctor is told so, with no empty chart.
        with mock.patch.object(screening, "individual_sensitivity", return_value=SENSITIVITY[2:3]):
            at = self.page(self.doctor, "screenings", active_screening="kidney", last_results=scored("kidney", 1, 0.8))
        self.assertIn("The entered values match the typical training values, so there is no input comparison to show.",
                      [i.value for i in at.info])
        self.assertFalse([c for c in at.get("plotly_chart") if "sensitivity" in c.proto.spec])

    def test_patient_model_tab_is_unchanged_apart_from_bar_contrast(self):
        with mock.patch.object(screening, "individual_sensitivity", return_value=SENSITIVITY):
            at = self.page(self.patient, "screenings", active_screening="kidney", last_results=scored("kidney", 1, 0.8))
        titles = [title for title, _ in self.headings(at)]
        self.assertIn("Screening Result", titles)
        self.assertIn("Model Interpretation", titles)
        self.assertIn("#### Prediction model", [m.value for m in at.markdown])
        table = next(df.value for df in at.dataframe if "Comparison value" in df.value.columns)
        self.assertEqual(list(table.columns), ["Input", "Your value", "Comparison value", "Score change (percentage points)"])
        self.assertEqual(len(table), 4)
        self.assertEqual(table["Comparison value"].iloc[0], "27.652077393512773")
        chart = next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if "sensitivity" in c.proto.spec)
        self.assertEqual(chart["data"][0]["marker"]["color"], ["#C9706E", "#5B8FA3", "#C9706E", "#5B8FA3"])
        self.assertTrue(any("Entered Value" in df.value.columns for df in at.dataframe))
        html_text = "\n".join(h.proto.body for h in at.get("html"))
        self.assertIn('<div class="result-title">Elevated Risk Pattern</div>', html_text)
        model = best_model("kidney")
        self.assertIn(f"Model: {model} ·", html_text)   # patients keep the model names they had
        self.assertEqual(next(s for s in at.selectbox if s.label == "Prediction model").options[0], f"{model} — Recommended")
        # The importance chart reads field labels for patients too, never column keys.
        bars = self.importance(at, "Kidney Health")
        self.assertIn("Estimated GFR", bars)
        self.assertNotIn("SerumCreatinine", bars)
        heart = self.page(self.patient, "screenings", active_screening="heart", last_results=scored("heart", 1, 0.8))
        bars = self.importance(heart, "Cardiovascular")
        self.assertTrue(any(re.fullmatch(r"(Age Category|History of \w+|Sex|General Health|Smoking Status): .+", b) for b in bars))
        self.assertFalse([b for b in bars if "_" in b or re.match(r"[A-Z][a-z]+[A-Z]", b)])

    def test_doctor_submission_and_missing_labs_use_neutral_wording(self):
        at = self.page(self.doctor, "screenings", active_screening="diabetes")
        next(b for b in at.button if b.label == "Generate Screening Result").click()
        text = self.text(self.ok(at.run()))
        self.assertIn("HbA1c Level and Blood Glucose Level were left blank, so the model used typical values", text)
        self.assertIn("repeat the screening with it when it is available", text)
        self.assertNotIn("You left", text)
        self.assertNotIn("when you have them", text)
        self.assertEqual(self.links(at), [])
        self.assertEqual(len(load_screenings(self.doctor["id"])), 1)   # still saved to the doctor's own account

        values = typical("kidney") | {"GFR": None}
        no_score = {"kidney": {"values": values, "no_score": True, "missing": ["GFR"], "model": best_model("kidney")}}
        text = self.text(self.page(self.doctor, "screenings", active_screening="kidney", last_results=no_score))
        self.assertIn("Estimated GFR was left blank.", text)
        self.assertIn("the factors below come only from the entered values", text)
        self.assertIn("Consider kidney-function assessment", text)
        self.assertNotIn("What you can do next", text)
        self.assertNotIn(CLINIC, text)

        model = best_model("diabetes")
        profile = missing_inputs_profile("diabetes", model, ["HbA1c_level"])
        result = {"model": model, "missing": ["HbA1c_level"], "threshold": profile["threshold"], "profile": profile}
        self.assertTrue(screening.missing_note("diabetes", result, doctor=True).startswith("HbA1c Level was left blank"))
        self.assertTrue(screening.missing_note("diabetes", result).startswith("You left HbA1c Level blank"))

    def test_patient_screening_wording_is_unchanged(self):
        at = self.page(self.patient, "screenings", active_screening="diabetes", last_results=scored("diabetes", 1, 0.8))
        text = self.text(at)
        self.assertIn(PATIENT_ELEVATED, text)
        self.assertIn("Recommended Next Steps", text)
        self.assertIn("Book an appointment with a primary-care doctor or diabetes clinic to review this result "
                      "together with your symptoms, medical history and previous tests.", text)
        self.assertIn("Factors Identified From Your Inputs", text)
        self.assertIn("Personalized explanation of your screening.", text)
        self.assertIn("Showing your most recent result for this screening. Submit the form again to update it.", text)
        self.assertIn("Optional. Values from a recent blood test; leave any you don't have blank.", text)
        self.assertIn("Seek urgent medical care", text)
        self.assertEqual([s.value for s in at.subheader], ["Your result: model input sensitivity"])
        self.assertEqual(self.links(at), [CLINIC])
        self.assertEqual([t.label for t in at.tabs],
                         ["Recommendations", "Key factors", "AHEAD Insight", "Model information", "History"])
        for doctor_copy in ("Points to review", "This record", "Red flags", "No score history"):
            self.assertNotIn(doctor_copy, text)

        values = typical("kidney") | {"GFR": None}
        no_score = {"kidney": {"values": values, "no_score": True, "missing": ["GFR"], "model": best_model("kidney")}}
        at = self.page(self.patient, "screenings", active_screening="kidney", last_results=no_score)
        text = self.text(at)
        self.assertIn("You left Estimated GFR blank.", text)
        self.assertIn("the factors below come only from your answers", text)
        self.assertIn("What you can do next", text)
        self.assertEqual(self.links(at), [CLINIC])

    def test_screenings_picker_explains_doctor_results_are_not_linked_to_a_patient(self):
        doctor = self.text(self.page(self.doctor, "screenings"))
        self.assertIn("Screenings run here are not attached to a patient record.", doctor)
        patient = self.text(self.page(self.patient, "screenings"))
        self.assertIn("Select a condition to begin a guided early-awareness assessment.", patient)
        self.assertNotIn("not attached to a patient record", patient)

    def test_insight_prompt_follows_the_role(self):
        args = ("diabetes", 1, 0.8, ["Higher BMI"], {"bmi": 31, "HbA1c_level": None}, 0.5, 0.5)
        with mock.patch.object(screening, "generate_text", side_effect=lambda prompt: prompt):
            patient, doctor = screening.generate_ai_guidance(*args), screening.generate_ai_guidance(*args, doctor=True)
        self.assertIn("### Understanding Your Result", patient)
        self.assertIn("patient-friendly", patient)
        self.assertIn("### Points to Consider", doctor)
        self.assertIn("for a clinician", doctor)
        self.assertNotIn("Your", doctor.split("Use exactly these headings:")[1].split("Requirements:")[0])

    # ------------------------------------------------------------------ assistant
    def chips(self, at):
        return [b.label.replace("  →", "") for b in at.button if (b.key or "").startswith("suggest_")]

    def test_assistant_suggestions_notice_and_answers_follow_the_role(self):
        patient = self.page(self.patient, "assistant")
        self.assertEqual(self.chips(patient), [text for _icon, text in assistant.SUGGESTIONS])
        self.assertNotIn("identifiable patient records", self.text(patient))

        doctor = self.page(self.doctor, "assistant")
        self.assertEqual(self.chips(doctor), ["How do patients link to me?", "How do I score an uploaded file?",
                                              "What does Patient Intelligence show?"])
        self.assertIn(assistant.DOCTOR_NOTICE, [i.value for i in doctor.info])
        expected = {"suggest_0": "Care team access", "suggest_1": "Save records to review queue",
                    "suggest_2": "Clinical Insights"}
        for key, phrase in expected.items():
            next(b for b in doctor.button if b.key == key).click()
            self.ok(doctor.run())
            self.assertIn(phrase, doctor.session_state.chat_messages[-1]["content"])

        doctor.chat_input[0].set_value("How do I use the platform?")
        answer = self.ok(doctor.run()).session_state.chat_messages[-1]["content"]
        self.assertIn("Patient Intelligence", answer)
        self.assertIn("medication plans", answer)
        patient.chat_input[0].set_value("How do I use the platform?")
        self.assertEqual(self.ok(patient.run()).session_state.chat_messages[-1]["content"],
                         assistant.LOCAL_ANSWERS["how do i use"])

    def test_assistant_times_are_dubai_local_time(self):
        moment = datetime(2026, 11, 4, 14, 5, tzinfo=ZoneInfo("Asia/Dubai"))
        with mock.patch.object(assistant, "local_now", return_value=moment):
            at = self.page(self.doctor, "assistant")
        self.assertEqual(at.session_state.chat_messages[0]["time"], moment)
        self.assertIn('<div class="assistant-time">2:05 PM</div>', [h.proto.body for h in at.get("html")])
        self.assertEqual(self.page(self.patient, "assistant").session_state.chat_messages[0]["time"].tzinfo,
                         ZoneInfo("Asia/Dubai"))

    def test_assistant_sends_the_role_specific_system_prompt(self):
        sent = []
        with mock.patch.object(assistant, "chat_reply", side_effect=lambda system, turns: sent.append(system) or "ok"):
            assistant._reply([{"role": "user", "content": "Hi"}])
            assistant._reply([{"role": "user", "content": "Hi"}], is_doctor=True)
        self.assertEqual(sent[0], assistant.SYSTEM_PROMPT)
        self.assertIn("patient-friendly", sent[0])
        self.assertIn("clinician", sent[1])
        self.assertNotIn("patient-friendly", sent[1])
        self.assertIn("Patient Intelligence", sent[1])

    # ------------------------------------------------------------------ settings
    def test_settings_copy_follows_the_role(self):
        patient = self.page(self.patient, "settings")
        text = self.text(patient)
        self.assertIn(PATIENT_HEADER, text)
        self.assertIn("Care team access", text)
        doctor = self.page(self.doctor, "settings")
        text = self.text(doctor)
        self.assertNotIn("care team", text.lower())
        self.assertIn("Manage your profile, appearance, data and sign-in.", text)
        create_user("d2@test.local", "safe-password-123", "Doctor", "Dr Second")
        next(t for t in patient.text_input if t.label == "Doctor account email").set_value("d2@test.local")
        next(b for b in patient.button if b.label == "Add doctor").click()
        self.assertEqual([t.value for t in self.ok(patient.run()).toast], ["Doctor linked."])
        self.assertFalse(patient.success)
        for at in (patient, doctor):
            text = self.text(at)
            self.assertIn("<h2>Notifications</h2>", "\n".join(h.proto.body for h in at.get("html")))
            self.assertNotIn("Display preferences", text)
            appearance = next(t for t in at.tabs if t.label == "Appearance")
            self.assertEqual(len(appearance.columns), 2)
            self.assertTrue(all(column.button for column in appearance.columns))   # no empty third column


if __name__ == "__main__":
    unittest.main()
