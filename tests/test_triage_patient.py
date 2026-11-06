"""The SORT step on the patient side: the track panel under every Screenings result, the next steps and AHEAD Insight
prompt that follow it, who can see it (or a button to share it), the Care team access focus in Settings, the urgent
banner on the patient home, and the track handed to the screening PDF. Every check runs on a temporary database with
Gemini switched off (or its text generator replaced), so nothing leaves the machine."""
import html
import os
import tempfile
import unittest
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

import ahead.screening as screening
from ahead.components import markdown_literal
from ahead.pages.patient_dashboard import open_urgent
from ahead.platform_data import grant_doctor, init_platform, local_today, save_plan
from ahead.config import OPTIONAL_INPUTS, form_fields
from ahead.resources import best_model, decision_threshold, feature_summary
from ahead.storage import authenticate, connection, create_user, init_db, load_screenings, save_screening
from ahead.triage import track_text, triage

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SHARE = "Share with your doctor"
LINKED = "Your linked doctor, Dr. Layla Hassan, can see this result in AHEAD."
NOT_FLAGGED_BUT = ("The model did not flag this result, but a value you entered is already in a range doctors use when "
                   "diagnosing a condition.")
ODD_NAME = "Dr. Zaid <b>Omar</b> *Said*"   # sorts after Layla; Markdown and HTML must print literally
HBA1C = "Your HbA1c of 7.1% is at or above 6.5%, a level doctors use when diagnosing diabetes."
# Kidney values below every urgent cut-off, both tests in the usual range.
KIDNEY_CLEAR = {"age": 50, "egfr": 75, "uacr": 10, "systolic_bp": 120, "diastolic_bp": 80}


def typical(disease):
    """The form's typical answers, optional labs and readings blank (the model alone sorts the result)."""
    stats = feature_summary(disease)
    return {f: None if f in OPTIONAL_INPUTS.get(disease, []) else stats[f]["default"] if stats[f]["kind"] == "category"
            else stats[f]["mode"] if stats[f]["binary"] else stats[f]["median"] for f in form_fields(disease)}


def scored(disease, probability, **values):
    """A stored Screenings result (as the page keeps it in last_results), flagged when probability >= threshold."""
    model = best_model(disease)
    threshold = decision_threshold(disease, model)
    return {disease: {"values": typical(disease) | values, "probability": probability,
                      "prediction": int(probability >= threshold), "model": model, "threshold": threshold,
                      "missing": []}}


class PatientTriageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/triage.db"
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

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def page(self, user, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user, "nav": nav, **state}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    def screen(self, user, last_results):
        disease = next(iter(last_results))
        return self.page(user, "screenings", active_screening=disease, last_results=last_results)

    @staticmethod
    def panels(node):
        """The track panels drawn under *node* (the page, or one container), HTML entities decoded."""
        return [html.unescape(h.proto.body) for h in node.get("html") if 'class="track-panel' in h.proto.body]

    def panel(self, node):
        panels = self.panels(node)
        self.assertEqual(len(panels), 1, panels)
        return panels[0]

    @staticmethod
    def buttons(at):
        return [b.label for b in at.button]

    @staticmethod
    def captions(at):
        return [c.value.replace("\\", "") for c in at.caption]   # Markdown escapes in names read as plain text

    # ------------------------------------------------------------------ Screenings
    def test_each_track_has_its_panel_under_the_result_card(self):
        urgent = self.panel(self.screen(self.patient, scored("diabetes", 0.9, HbA1c_level=7.1)))
        self.assertIn('class="track-panel urgent"', urgent)
        self.assertIn(f'<strong class="track-panel-title">{track_text("urgent")["label"]}</strong>', urgent)
        self.assertIn("See a doctor soon", urgent)
        self.assertIn("The model flagged this result, and a value you entered", urgent)
        self.assertIn(HBA1C, urgent)
        self.assertIn("A doctor needs to confirm this with tests.", urgent)

        at = self.screen(self.patient, scored("diabetes", 0.9))
        at_risk = self.panel(at)
        self.assertIn('class="track-panel at-risk"', at_risk)
        self.assertIn('<strong class="track-panel-title">At risk</strong>', at_risk)
        self.assertIn("a prevention plan agreed with your doctor can help", at_risk)
        self.assertNotIn("<li>", at_risk)
        # The panel sits under the result card and above the model's precision note.
        bodies = [h.proto.body for h in at.get("html")]
        card = next(i for i, b in enumerate(bodies) if 'class="result-card' in b)
        self.assertGreater(next(i for i, b in enumerate(bodies) if 'class="track-panel' in b), card)

        clear = self.panel(self.screen(self.patient, scored("diabetes", 0.01)))
        self.assertIn('class="track-panel clear"', clear)
        self.assertIn("Not flagged.", clear)
        self.assertIn("The model did not flag this result.", clear)

    def test_not_flagged_but_urgent_says_so_plainly(self):
        at = self.screen(self.patient, scored("diabetes", 0.01, HbA1c_level=7.1))
        panel = self.panel(at)
        self.assertIn('class="track-panel urgent"', panel)
        self.assertIn(NOT_FLAGGED_BUT, panel)
        self.assertIn(HBA1C, panel)
        self.assertIn("This is not a diagnosis.", panel)
        # The result card is titled as the model's estimate, never 'Lower Predicted Risk' over 'See a doctor soon'.
        cards = "\n".join(h.proto.body for h in at.get("html"))
        self.assertIn('<div class="result-title">Model estimate</div>', cards)
        self.assertNotIn("Lower Predicted Risk", cards)

    def test_a_past_heart_attack_is_urgent_and_the_heart_card_is_not_a_future_risk(self):
        page = "\n".join(h.proto.body for h in self.screen(self.patient, scored("heart", 0.03)).get("html"))
        self.assertIn("did not reach the model&#x27;s level for being similar to people in a CDC health survey", page)
        self.assertIn("It is not your chance of having a heart attack in the next 10 years.", page)
        self.assertNotIn("calibrated clinical probability", page)
        at = self.screen(self.patient, scored("heart", 0.03, HadHeartAttack="Yes"))
        panel = self.panel(at)
        self.assertIn('class="track-panel urgent"', panel)
        self.assertIn("You told us you have had a heart attack. AHEAD's heart screening is for people who have not; "
                      "please see your doctor about your heart health.", panel)
        self.assertIn("The model's score does not apply", panel)
        self.assertIn(SHARE, self.buttons(at))   # urgent: who can see it, never the plan shortcut
        self.assertNotIn("Open My plan", self.buttons(at))
        # The form asks the questions in plain words; a doctor reads the short labels and clinical wording.
        at = self.page(self.patient, "screenings", active_screening="heart")
        labels = [s.label for s in at.selectbox]
        for question in ("Has a doctor ever told you that you had a heart attack?",
                         "Has a doctor ever told you that you have angina or coronary heart disease?",
                         "Has a doctor ever told you that you had a stroke?"):
            self.assertIn(question, labels)
        at = self.screen(self.doctor, scored("heart", 0.2, HadAngina="Yes", HadStroke="Yes"))
        panel = self.panel(at)
        self.assertIn("The patient reports known angina or coronary heart disease and a previous stroke (established "
                      "cardiovascular disease)", panel)
        self.assertIn("History of Heart Attack", [s.label for s in self.page(self.doctor, "screenings",
                                                                            active_screening="heart").selectbox])

    def test_a_kidney_result_without_labs_is_sorted_by_the_model(self):
        # The kidney score no longer needs lab results (the retired no-score path is gone): blank labs leave the track
        # to the model, and a severe blood pressure reading is still urgent on its own.
        for probability, css in ((0.9, "at-risk"), (0.01, "clear")):
            panel = self.panel(self.screen(self.patient, scored("kidney", probability)))
            self.assertIn(f'class="track-panel {css}"', panel)
        panel = self.panel(self.screen(self.patient, scored("kidney", 0.01, systolic_bp=185, diastolic_bp=95)))
        self.assertIn('class="track-panel urgent"', panel)
        self.assertIn("get emergency care now", panel)

    def test_share_button_without_a_doctor_and_caption_with_one(self):
        for result in (scored("diabetes", 0.9, HbA1c_level=7.1), scored("diabetes", 0.9)):
            at = self.screen(self.patient, result)
            self.assertIn(SHARE, self.buttons(at))
            self.assertFalse([c for c in self.captions(at) if "can see this result" in c])
        at = self.screen(self.patient, scored("diabetes", 0.01))   # not flagged: nothing to share
        self.assertNotIn(SHARE, self.buttons(at))

        grant_doctor(self.patient["id"], "d@test.local")
        for result in (scored("diabetes", 0.9, HbA1c_level=7.1), scored("diabetes", 0.9)):
            at = self.screen(self.patient, result)
            self.assertNotIn(SHARE, self.buttons(at))
            self.assertIn(LINKED, self.captions(at))
        create_user("d2@test.local", "test-password-123", "Doctor", ODD_NAME)
        grant_doctor(self.patient["id"], "d2@test.local")
        at = self.screen(self.patient, scored("diabetes", 0.9))
        self.assertIn(f"Your linked doctors, Dr. Layla Hassan and {ODD_NAME}, can see this result in AHEAD.",
                      self.captions(at))
        caption = next(c.value for c in at.caption if "linked doctors" in c.value)
        self.assertIn(markdown_literal(ODD_NAME), caption)   # escaped, so the name prints literally

        # With history off nothing was saved, so the linked doctor cannot see it either.
        result = scored("diabetes", 0.9)
        result["diabetes"]["saved"] = False
        at = self.screen(self.patient, result)
        self.assertNotIn(SHARE, self.buttons(at))
        self.assertTrue(any(c.startswith("This result was not saved because Keep screening history is off in "
                                         "Settings, so your linked doctors,") for c in self.captions(at)))

    def test_share_button_opens_settings_on_care_team_access(self):
        at = self.screen(self.patient, scored("diabetes", 0.9, HbA1c_level=7.1))
        next(b for b in at.button if b.label == SHARE).click()
        self.ok(at.run())
        self.assertEqual(at.session_state["nav"], "settings")
        self.assertFocused(at, True)

        # Linking the doctor reruns the page; the patient stays on the same tab and section.
        next(t for t in at.text_input if t.label == "Doctor account email").set_value("d@test.local")
        next(b for b in at.button if b.label == "Add doctor").click()
        self.ok(at.run())
        self.assertEqual([t.value for t in at.toast], ["Doctor linked."])
        self.assertFocused(at, True)

        # Leaving Settings ends the focus: the next visit reads as usual.
        at.session_state["nav"] = "overview"
        self.ok(at.run())
        at.session_state["nav"] = "settings"
        self.assertFocused(self.ok(at.run()), False)
        # A doctor never gets the focus, even when the request is set.
        self.assertFocused(self.page(self.doctor, "settings", settings_care_team_request=True), False, doctor=True)

    def assertFocused(self, at, focused, doctor=False):
        (tabs,) = at.get("tab_container")
        self.assertEqual([t.label for t in tabs.children.values()], ["Profile", "Appearance", "Preferences", "Account"])
        self.assertEqual(tabs.proto.tab_container.default_tab_index, 2 if focused else 0)
        preferences = next(t for t in at.tabs if t.label == "Preferences")
        titles = [html.unescape(h.proto.body.split("<h2>")[1].split("</h2>")[0]) for h in preferences.get("html")
                  if 'class="settings-panel"' in h.proto.body]
        if doctor:
            self.assertNotIn("Care team access", titles)
        elif focused:
            self.assertEqual(titles[0], "Care team access")
            self.assertIn("Add your doctor's AHEAD account email to share your saved screening results with them.",
                          [i.value for i in at.info])
        else:
            self.assertEqual(titles, ["Notifications", "Data & privacy", "Care team access"])

    def test_doctor_screening_uses_doctor_wording_and_no_share_line(self):
        at = self.screen(self.doctor, scored("diabetes", 0.01, HbA1c_level=7.1))
        panel = self.panel(at)
        self.assertIn(f'<strong class="track-panel-title">{track_text("urgent", "doctor")["label"]}</strong>', panel)
        self.assertIn("The model did not flag this result, but an entered value is already in a range", panel)
        self.assertIn("The patient's HbA1c of 7.1% is at or above 6.5%", panel)
        self.assertNotIn("See a doctor soon", panel)
        for patient_copy in (" you", "You ", "Your "):
            self.assertNotIn(patient_copy, panel)
        self.assertNotIn(SHARE, self.buttons(at))
        self.assertFalse([c for c in self.captions(at) if "can see this result" in c or "can see your results" in c])
        at = self.screen(self.doctor, scored("diabetes", 0.9))
        self.assertIn('<strong class="track-panel-title">At risk</strong>', self.panel(at))
        self.assertNotIn(SHARE, self.buttons(at))

    def test_pdf_receives_the_track_of_the_submitted_result(self):
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return b"%PDF-1.4\n"

        at = self.page(self.patient, "screenings", active_screening="diabetes")
        at.number_input(key="diabetes_HbA1c_level").set_value(7.1)
        at.number_input(key="diabetes_blood_glucose_level").set_value(140)
        with mock.patch.object(screening, "patient_report", spy):
            next(b for b in at.button if b.label == "Generate Screening Result").click()
            self.ok(at.run())
        self.assertEqual(len(calls), 1)
        result = at.session_state["last_results"]["diabetes"]
        expected = triage("diabetes", result["values"], result["probability"], result["threshold"])
        self.assertEqual(calls[0]["track"], expected)
        self.assertEqual(calls[0]["track"]["track"], "urgent")
        self.assertTrue(result["saved"])
        self.assertIn(SHARE, self.buttons(at))

    # ------------------------------------------------------------------ patient home
    def save(self, disease, values, score, days_ago=0, threshold=0.5):
        """A saved screening from *days_ago* local days ago (local noon, stored as UTC)."""
        row = save_screening(self.patient["id"], disease, best_model(disease), score, int(score >= threshold),
                             threshold, values)
        moment = datetime.combine(local_today() - timedelta(days=days_ago), time(12), tzinfo=ZoneInfo("Asia/Dubai"))
        with connection() as db:
            db.execute("UPDATE screenings SET created_at=? WHERE id=?", (moment.astimezone(timezone.utc).isoformat(), row))
        return moment

    def home(self):
        at = self.page(self.patient, "overview")
        try:
            return at, at.get_by_key("home_track_banner")
        except KeyError:
            return at, None

    def test_home_banner_shows_only_for_a_recent_urgent_result(self):
        at, banner = self.home()
        self.assertIsNone(banner)
        self.save("diabetes", {"age": 50, "HbA1c_level": 7.1, "blood_glucose_level": 140}, 0.2, days_ago=40)
        self.assertIsNone(self.home()[1])   # too old
        self.save("diabetes", {"age": 50, "HbA1c_level": 5.5, "blood_glucose_level": 140}, 0.9, days_ago=3)
        self.assertIsNone(self.home()[1])   # recent but at risk, not urgent

        moment = self.save("diabetes", {"age": 50, "HbA1c_level": 7.1, "blood_glucose_level": 212}, 0.2, days_ago=29)
        self.assertIsNone(self.home()[1])   # an urgent result older than the latest one for the same condition
        with connection() as db:
            db.execute("DELETE FROM screenings WHERE score=0.9")
        at, banner = self.home()
        self.assertIsNotNone(banner)
        panel = self.panel(banner)
        self.assertIn('class="track-panel urgent"', panel)
        self.assertIn("See a doctor soon", panel)
        self.assertIn(NOT_FLAGGED_BUT.replace("a value you entered is", "some values you entered are"), panel)
        self.assertIn(HBA1C, panel)   # the first reason only
        self.assertNotIn("blood glucose of 212 mg/dL", panel)
        self.assertIn(f"From your diabetes screening on {moment.day} {moment:%b %Y}.", panel)
        self.assertEqual(len(self.panels(at)), 1)
        self.assertIn(SHARE, [b.label for b in banner.get("button")])

        # A later result for another condition does not hide it; a linked doctor is named instead of the button.
        self.save("heart", {}, 0.1, days_ago=1)
        grant_doctor(self.patient["id"], "d@test.local")
        at, banner = self.home()
        self.assertIsNotNone(banner)
        self.assertEqual([b.label for b in banner.get("button")], [])
        self.assertIn(LINKED, [c.value.replace("\\", "") for c in banner.get("caption")])

        # Thirty local days ago (today counts as the first) is outside the window.
        with connection() as db:
            db.execute("DELETE FROM screenings")
        self.save("diabetes", {"age": 50, "HbA1c_level": 7.1}, 0.2, days_ago=30)
        self.assertIsNone(self.home()[1])
        # A newer, not urgent result for the same condition replaces the urgent one.
        self.save("kidney", KIDNEY_CLEAR | {"egfr": 52.3}, 0.9, days_ago=2)
        at, banner = self.home()
        self.assertIn("From your kidney health screening on", self.panel(banner))
        self.assertIn("Your eGFR of 52.3 is below 60", self.panel(banner))
        self.save("kidney", KIDNEY_CLEAR, 0.1, days_ago=0)
        self.assertIsNone(self.home()[1])

    def test_home_banner_names_a_second_urgent_condition(self):
        kidney = self.save("kidney", KIDNEY_CLEAR | {"uacr": 45}, 0.9, days_ago=5)
        self.save("diabetes", {"age": 50, "HbA1c_level": 7.1}, 0.9, days_ago=1)
        at, banner = self.home()
        panel = self.panel(banner)
        self.assertIn("From your diabetes screening on", panel)
        self.assertIn(f'Your kidney health screening on {kidney.day} {kidney:%b %Y} is also marked "See a doctor soon".',
                      panel)
        self.assertNotIn("did not re-check", panel)   # the newest diabetes screening is the one shown
        self.assertEqual(len(self.panels(at)), 1)

    def test_a_later_screening_with_the_value_left_blank_keeps_the_banner(self):
        """A blank value was not re-checked, so only a later screening that re-entered it clears the urgent reason."""
        moment = self.save("diabetes", {"age": 50, "HbA1c_level": 7.8, "blood_glucose_level": 240}, 0.2, days_ago=3)
        self.save("diabetes", {"age": 50, "HbA1c_level": None, "blood_glucose_level": ""}, 0.12, days_ago=2)
        panel = self.panel(self.home()[1])
        self.assertIn(f"From your diabetes screening on {moment.day} {moment:%b %Y}. Your later diabetes screening did "
                      "not re-check this value.", panel)
        self.assertIn("Your HbA1c of 7.8% is at or above 6.5%", panel)

        # HbA1c re-entered below the cut-off: the glucose reason, never re-checked, is the one shown.
        self.save("diabetes", {"age": 50, "HbA1c_level": 5.5}, 0.12, days_ago=1)
        panel = self.panel(self.home()[1])
        self.assertNotIn("HbA1c", panel)
        self.assertIn("Your blood glucose of 240 mg/dL is at or above 200 mg/dL", panel)
        self.assertIn("Your later diabetes screenings did not re-check this value.", panel)

        # Glucose re-entered below the cut-off too: nothing urgent stands.
        self.save("diabetes", {"age": 50, "HbA1c_level": None, "blood_glucose_level": 140}, 0.12, days_ago=0)
        self.assertIsNone(self.home()[1])

    def test_open_urgent_keeps_each_reason_until_its_values_are_re_entered(self):
        def row(disease, values, score=0.1):
            return save_screening(self.patient["id"], disease, best_model(disease), score, int(score >= 0.5), 0.5,
                                  values)

        row("kidney", KIDNEY_CLEAR | {"uacr": 45, "systolic_bp": 185, "diastolic_bp": 125})
        row("kidney", KIDNEY_CLEAR | {"systolic_bp": None})   # ACR re-checked; only one blood-pressure reading
        row("diabetes", {"age": 50, "HbA1c_level": 5.5})
        (found, result), = open_urgent(load_screenings(self.patient["id"]))
        self.assertEqual(found["disease"], "kidney")
        self.assertEqual([r["features"] for r in result["reasons"]], [["systolic_bp", "diastolic_bp"]])
        self.assertTrue(result["urgent"])
        row("kidney", KIDNEY_CLEAR)   # both readings re-entered below the cut-offs
        self.assertEqual(open_urgent(load_screenings(self.patient["id"])), [])
        # An unreadable row neither trips a rule nor re-checks one.
        urgent = row("kidney", KIDNEY_CLEAR | {"egfr": 41})
        with connection() as db:
            db.execute("UPDATE screenings SET values_json='{not json' WHERE id=?", (row("kidney", {}),))
        (found, result), = open_urgent(load_screenings(self.patient["id"]))
        self.assertEqual((found["id"], [r["feature"] for r in result["reasons"]]), (urgent, ["egfr"]))

    # ------------------------------------------------------------------ guidance follows the track
    def recommendations(self, at, doctor=False):
        tab = next(t for t in at.tabs if t.label == ("Points to review" if doctor else "Recommendations"))
        return html.unescape("\n".join(h.proto.body for h in tab.get("html")))

    def test_next_steps_follow_an_urgent_track_the_model_did_not_flag(self):
        for user, doctor in ((self.patient, False), (self.doctor, True)):
            steps = screening.DOCTOR_NEXT_STEPS if doctor else screening.NEXT_STEPS
            for disease, result in (("diabetes", scored("diabetes", 0.01, HbA1c_level=7.1)),
                                    ("kidney", scored("kidney", 0.01, uacr=258))):
                with self.subTest(disease=disease, doctor=doctor):
                    at = self.screen(user, result)
                    self.assertFalse(at.session_state["last_results"][disease]["prediction"])
                    self.assertIn('class="track-panel urgent"', self.panel(at))
                    text = self.recommendations(at, doctor)
                    for title, body in steps[disease][1]:
                        self.assertIn(title, text); self.assertIn(body, text)
                    for title, _ in steps[disease][0]:
                        self.assertNotIn(title, text)
                    self.assertNotIn("routine", text.lower())
            # Not flagged and no rule: the routine steps stay.
            at = self.screen(user, scored("diabetes", 0.01))
            self.assertIn(steps["diabetes"][0][0][0], self.recommendations(at, doctor))
            at = self.screen(user, scored("kidney", 0.01, **KIDNEY_CLEAR))
            self.assertIn('class="track-panel clear"', self.panel(at))
            self.assertIn(steps["kidney"][0][0][0], self.recommendations(at, doctor))

    def test_insight_prompt_carries_an_urgent_track(self):
        prompts = []

        def fake(prompt):
            prompts.append(prompt)
            return "### Understanding Your Result\nText."

        with mock.patch.object(screening, "gemini_available", lambda: True), \
                mock.patch.object(screening, "generate_text", fake):
            at = self.screen(self.patient, scored("diabetes", 0.01, HbA1c_level=7.1))
            at.button(key="diabetes_insight").click()
            self.ok(at.run())
        (prompt,) = prompts
        self.assertIn("Screening result: Not flagged by the model, but entered values are in a range doctors use when "
                      "diagnosing a condition\nAHEAD track: \"See a doctor soon\". The patient's HbA1c of 7.1% is at or "
                      "above 6.5%", prompt)
        self.assertNotIn("Lower screening risk", prompt)
        self.assertIn('Requirements:\n- AHEAD has put this result on its "See a doctor soon" track', prompt)
        self.assertIn("never describe the result as reassuring or low risk", prompt)

        # No HbA1c: an HbA1c under 5.7% would set a Not flagged track by the lab value, which names itself in the
        # prompt (test_nhanes_repairs).
        args = ("diabetes", 0, 0.01, [], {"age": 50}, 0.5, 0.5)
        with mock.patch.object(screening, "generate_text", side_effect=lambda prompt: prompt):
            clear = screening.generate_ai_guidance(*args, track=triage("diabetes", args[4], 0.01, 0.5))
            self.assertEqual(clear, screening.generate_ai_guidance(*args))   # not urgent: the prompt is unchanged
            self.assertIn("Screening result: Lower screening risk\nModel screening score:", clear)
            self.assertIn("Requirements:\n- Explain", clear)
            values = {"age": 15}   # flagged by the model, no HbA1c to clear it: a child needs a doctor
            doctor = screening.generate_ai_guidance("diabetes", 1, 0.9, [], values, 0.5, 0.5, True,
                                                    track=triage("diabetes", values, 0.9, 0.5))
        self.assertIn('Screening result: Elevated screening risk\nAHEAD track: "Urgent". The patient\'s entered age '
                      "is 15", doctor)
        self.assertNotIn("See a doctor soon", doctor)

    # ------------------------------------------------------------------ Settings
    def test_care_team_access_prints_doctor_and_plan_names_literally(self):
        odd = "Dr [Click to verify](https://evil.example) **Omar**"
        create_user("odd_doc@test.local", "test-password-123", "Doctor", odd)
        grant_doctor(self.patient["id"], "odd_doc@test.local")
        doctor = authenticate("odd_doc@test.local", "test-password-123", "Doctor")
        save_plan(doctor["id"], self.patient["id"], {
            "name": "Metformin *500*", "dosage": "1 tablet", "instructions": "", "times": ["08:00"], "days": range(7),
            "start_date": local_today().isoformat(), "end_date": local_today().isoformat(), "notes": ""})
        at = self.page(self.patient, "settings")
        preferences = next(t for t in at.tabs if t.label == "Preferences")
        lines = [m.value for m in preferences.markdown]
        self.assertIn(f"{markdown_literal(odd)} · {markdown_literal('odd_doc@test.local')}", lines)
        self.assertIn(f"{markdown_literal(odd)} will no longer see your records or manage your medication.", lines)
        self.assertFalse([line for line in lines if "](" in line or "**" in line])
        (warning,) = [w.value for w in preferences.warning]
        self.assertIn(markdown_literal("Metformin *500*"), warning)


if __name__ == "__main__":
    unittest.main()
