"""The patient's My plan page (nav key 'medications'): the calm empty state with and without a linked doctor, the hero
and target cards of an active plan, a paused plan, today's doses with Take/Skip, the 30-day history, re-check due,
drafts never reaching the patient, an urgent result for the plan's condition shown first, an earlier dose reading
missed once the next is due, escaping of doctor names and notes, and doctors never getting the page. Also the tracker's
Check-in starting from the last one, the Goals tab naming the plan's goals, the Screenings line pointing to My plan, and
the home doses card. Every
check runs on a temporary database with a fixed Dubai clock, so the doses' due times never depend on the hour."""
import html
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

import ahead.pages.medications as medications
import ahead.pages.wellness as wellness
import ahead.platform_data as platform_data
import ahead.screening as screening
from ahead.components import markdown_literal
from ahead.platform_data import (
    approve_prevention_plan, grant_doctor, init_platform, notifications, plan_status, prevention_plan_status,
    prevention_progress, save_lifestyle, save_plan, save_prevention_draft, scheduled_doses, set_challenge_day,
    set_goals, get_goals,
)
from ahead.resources import best_model, decision_threshold, feature_summary, features, icon_uri
from ahead.storage import connection, init_db, load_screenings, save_screening
from ahead.wellness import evaluate

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DOCTOR = "Dr. Layla Hassan"
ODD = "Dr. <b>Zaid</b> *Omar* [verify](https://evil.example)"   # HTML and Markdown must print literally
LOG = {"age": 50, "height_cm": 164, "weight_kg": 84, "steps": 5000, "activity_level": "Some movement",
       "exercise_days": 2, "exercise_minutes": 30, "sleep_hours": 6, "sleep_quality": 3, "water_litres": 1.5,
       "fruit_veg": 2, "sugary_food": 1, "fast_food": 0, "stress": 5, "sedentary_hours": 7}   # BMI 31.2


class Clock:
    """The Dubai wall clock the platform reads: stamps, today and the due check of doses."""
    def __init__(self, text):
        self.set(text)

    def set(self, text):
        self.moment = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("Asia/Dubai"))

    def stamp(self):
        return self.moment.astimezone(timezone.utc).isoformat()


class MyPlanFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["AHEAD_DB_PATH"] = self.temp.name + "/my_plan.db"
        os.environ.pop("AHEAD_ENABLE_DEMO_ACCOUNTS", None)
        self.clock = Clock("2026-11-10 12:00")
        for patcher in (mock.patch.multiple(platform_data, now=self.clock.stamp, local_now=lambda: self.clock.moment,
                                            local_today=lambda: self.clock.moment.date()),
                        mock.patch.object(medications, "local_today", lambda: self.clock.moment.date()),
                        mock.patch.object(screening, "gemini_available", lambda: False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        init_db(); init_platform()
        # Inserted directly: sign-in is not under test here, and each password hash takes most of a second.
        with connection() as db:
            self.users = {email: {"id": db.execute("INSERT INTO users(email,password_hash,role,name) VALUES(?,?,?,?)",
                                                   (email, "-", role, name)).lastrowid,
                                  "email": email, "role": role, "name": name, "institution": "", "preferences": {}}
                          for email, role, name in [("p@test.local", "Patient", "Amy Lee"),
                                                    ("d@test.local", "Doctor", DOCTOR),
                                                    ("odd@test.local", "Doctor", ODD)]}
        self.patient, self.doctor, self.odd = (self.users[e] for e in ("p@test.local", "d@test.local", "odd@test.local"))
        self.p = self.patient["id"]

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop("AHEAD_DB_PATH", None)

    # ------------------------------------------------------------------ helpers
    def page(self, user=None, nav="medications", **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {"authenticated": True, "user": user or self.patient, "nav": nav, **state}.items():
            at.session_state[key] = value
        return self.ok(at.run())

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    @staticmethod
    def blocks(node):
        """The html blocks under *node*, leaving out the page's stylesheets (redesign.css names every class)."""
        return [h.proto.body for h in node.get("html") if not h.proto.body.lstrip().startswith("<style")]

    def bodies(self, node):
        return [html.unescape(body) for body in self.blocks(node)]

    def html(self, node, marker):
        """The one html block under *node* containing *marker*, entities decoded."""
        found = [b for b in self.bodies(node) if marker in b]
        self.assertEqual(len(found), 1, found)
        return found[0]

    def raw(self, node):
        return "\n".join(self.blocks(node))

    def order(self, at, *nodes):
        """The positions of *nodes* in the page's top-to-bottom order."""
        def walk(node):
            yield node
            for child in getattr(node, "children", {}).values():
                yield from walk(child)
        seen = [id(n) for n in walk(at.main)]
        return [seen.index(id(n)) for n in nodes]

    def link(self, doctor=None):
        grant_doctor(self.p, (doctor or self.doctor)["email"])

    def log(self, **changes):
        values = {**LOG, **changes}
        save_lifestyle(self.p, values, evaluate(values, {}))

    def plan(self, **changes):
        screening_id = save_screening(self.p, "diabetes", "Model", .55, 1, .5,
                                      {"smoking_history": "current", "bmi": 31.2, "age": 50})
        return {"disease": "diabetes", "screening_id": screening_id, "note": "Small steps count.",
                "recheck_date": "2027-02-02",
                "targets": [{"key": "weight", "goal": 79}, {"key": "activity", "goal": 150},
                            {"key": "fruit_veg", "goal": 5}, {"key": "sleep", "goal": [7, 9]}, {"key": "smoking"}],
                **changes}

    def approve(self, doctor=None, **changes):
        doctor = doctor or self.doctor
        if not platform_data.allowed(doctor["id"], self.p):
            self.link(doctor)
        if not platform_data.lifestyle_history(self.p):
            self.log()
        return approve_prevention_plan(doctor["id"], self.p, self.plan(**changes))

    def medicine(self, times=("08:00", "20:00"), **changes):
        if not platform_data.allowed(self.doctor["id"], self.p):
            self.link()
        return save_plan(self.doctor["id"], self.p, {
            "name": "Metformin", "dosage": "500 mg", "instructions": "With food", "times": list(times),
            "days": range(7), "start_date": "2026-11-09", "end_date": "2026-11-30", "notes": "", **changes})


class EmptyStateTests(MyPlanFixture):
    def test_without_a_linked_doctor_the_share_step_opens_care_team_access(self):
        at = self.page()
        header = self.html(at, 'class="clinical-header"')
        self.assertIn("<h1>My plan</h1>", header)
        notice = self.html(at, 'class="plan-notice calm"')
        self.assertIn("No prevention plan yet", notice)
        self.assertIn("Plans come from a doctor you have linked. When a screening comes back at risk", notice)
        self.assertFalse([b for b in self.bodies(at) if "plan-hero" in b or "dose-main" in b])
        self.assertEqual(len(at.expander), 0)   # no medicines prescribed: no medicines and no history
        self.assertIn("No doctor can see your results yet. Add your doctor under Care team access in Settings.",
                      [c.value for c in at.caption])
        at.button(key="plan_share").click()
        self.ok(at.run())
        self.assertEqual(at.session_state["nav"], "settings")

    def test_with_a_linked_doctor_it_names_them_and_still_shows_medicines(self):
        self.link(); self.link(self.odd)
        pid = self.medicine()
        at = self.page()
        self.assertIn("No prevention plan yet", self.html(at, 'class="plan-notice calm"'))
        self.assertNotIn("plan_share", [b.key for b in at.button])
        self.assertIn(f"Your linked doctors, {markdown_literal(ODD + ' and ' + DOCTOR)}, can see your saved screenings "
                      "and can set up a plan with you.", [c.value for c in at.caption])
        self.assertFalse([b for b in self.bodies(at) if "plan-hero" in b])
        self.assertIn("Metformin · 500 mg", self.html(at, 'class="dose-main"><strong>Metformin · 500 mg</strong><span>08:00'))
        self.assertIn(f"taken_{pid}_08:00", [b.key for b in at.button])
        self.assertEqual([e.label for e in at.expander], ["Medication plans and 30-day history"])

    def test_a_draft_never_reaches_the_patient(self):
        self.link(); self.log()
        save_prevention_draft(self.doctor["id"], self.p, self.plan(note="Draft only"))
        at = self.page()
        self.assertIn("No prevention plan yet", self.html(at, 'class="plan-notice calm"'))
        text = "\n".join(self.bodies(at))
        for hidden in ("plan-hero", "Your plan with", "Draft only"):
            self.assertNotIn(hidden, text)
        self.assertEqual(notifications(self.p), [])


class ActivePlanTests(MyPlanFixture):
    def setUp(self):
        super().setUp()
        self.clock.set("2026-11-09 12:00"); self.log()                    # the starting point: 84 kg, BMI 31.2
        self.clock.set("2026-11-10 09:00"); self.plan_id = self.approve()
        self.clock.set("2026-11-11 18:00"); set_challenge_day(self.p, "2026-11-11", 160)
        self.clock.set("2026-11-12 12:00"); self.log(weight_kg=83, fruit_veg=3, sleep_hours=7.5)

    def test_header_hero_and_target_cards(self):
        at = self.page()
        header = self.html(at, 'class="clinical-header"')
        self.assertIn(f"<h1>Your plan with {DOCTOR}</h1>", header)
        self.assertIn("Agreed on 10 Nov 2026 after your diabetes screening. Your doctor chose these targets from "
                      "standard health advice.", header)
        hero = self.html(at, 'class="plan-hero"')
        self.assertIn('style="--share:50%" role="img" aria-label="2 of 4 targets on track"', hero)
        self.assertIn("<strong>2/4</strong><small>on track</small>", hero)
        self.assertIn("Two of your four targets are on track", hero)
        self.assertIn("Keep going: give fruit and vegetables and sleep a little more attention this week.", hero)
        self.assertIn("Re-check in<strong>12 weeks</strong>", hero)
        self.assertNotIn("Doses this week", hero)   # no medicines prescribed
        self.assertNotIn("Paused", hero)
        self.assertFalse([b for b in self.bodies(at) if "track-panel" in b])   # no urgent result for diabetes

        progress = prevention_progress(self.p, platform_data.active_plan(self.p))
        self.assertEqual([t["status"] for t in progress["targets"]],
                         ["on_track", "on_track", "behind", "behind", "at_recheck"])
        for item in progress["targets"]:
            with self.subTest(target=item["key"]):
                if item["key"] == "smoking":
                    card = at.get_by_key("plan_smoking_summary_tile")
                    self.assertEqual(card.get("button"), [])
                    body = self.html(card, "summary-inner")
                    self.assertIn("2 Feb <small>re-check</small>", body)
                else:
                    card = at.get_by_key(f"plan_{item['key']}_summary_card")
                    body = self.html(card, "summary-inner")
                    self.assertIn(f"{item['current_text']} <small>{item['of_goal']}</small>", body)
                    self.assertIn(f'aria-valuenow="{round(item["share"] * 100)}"', body)
                    (button,) = card.get("button")
                    self.assertEqual(button.label, f"{item['action']} →")
                self.assertIn(f'<span class="plan-status" style="--pill:{item["tone"]}">{item["status_text"]}</span>', body)
                self.assertIn(f'<p class="summary-note">{item["note"]}</p>', body)
                self.assertIn(f'<span class="summary-label">{item["label"]}</span>', body)
        weight = self.html(at.get_by_key("plan_weight_summary_card"), "summary-inner")
        self.assertIn("83 <small>/ 79 kg</small>", weight)
        self.assertIn("Down 1 kg since 10 Nov", weight)
        self.assertIn("On track", weight)
        self.assertIn("160 <small>/ 150 min</small>", self.html(at.get_by_key("plan_activity_summary_card"), "summary-inner"))

        note = self.html(at, 'class="plan-note"')
        self.assertIn(f"<p>“Small steps count.”</p><span>— {DOCTOR}</span>", note)
        self.assertIn("<strong>Re-check on 2 Feb 2027</strong><span>We'll remind you to screen again.</span>",
                      self.html(at.get_by_key("plan_recheck"), "plan-recheck"))

    def test_each_card_goes_where_the_target_is_logged(self):
        for key, page in (("weight", "wellness"), ("activity", "wellness"), ("fruit_veg", "wellness"), ("sleep", "wellness")):
            with self.subTest(target=key):
                at = self.page()
                at.button(key=f"plan_{key}").click()
                self.assertEqual(self.ok(at.run()).session_state["nav"], page)
                hints = [i.value for i in at.info if i.value.startswith("From My plan")]
                # Once, saying which field to change.
                self.assertEqual(hints, [f"From My plan: {wellness.TRACKER_HINTS[key]}. The Check-in starts from your "
                                         "last one, so change only what is new."])
                self.assertEqual([i.value for i in self.ok(at.run()).info if i.value.startswith("From My plan")], [])

    def test_the_new_plan_notification_opens_my_plan(self):
        (note,) = [n for n in notifications(self.p) if n["title"] == "New prevention plan"]
        at = self.page(nav="notifications")
        at.button(key=f"open_{note['id']}").click()
        self.ok(at.run())
        self.assertEqual(at.session_state["nav"], "medications")
        self.assertIn(f"Your plan with {DOCTOR}", self.html(at, 'class="clinical-header"'))

    def test_a_paused_plan_says_so_and_keeps_its_progress(self):
        prevention_plan_status(self.doctor["id"], self.plan_id, "paused")
        at = self.page()
        notice = self.html(at, 'class="plan-notice"')
        self.assertIn("Paused by your doctor", notice)
        self.assertIn(f"{DOCTOR} paused this plan. Your targets and progress stay here and your tracker goals stay as "
                      "they are, until your doctor resumes it.", notice)
        hero = self.html(at, 'class="plan-hero"')
        self.assertIn('<span class="plan-hero-badge">Paused</span>', hero)
        self.assertIn("<strong>2/4</strong>", hero)
        self.assertIn("Screen again on or after this date.", self.html(at.get_by_key("plan_recheck"), "plan-recheck"))

    def test_re_check_due(self):
        self.clock.set("2027-02-02 08:00")
        at = self.page()
        hero = self.html(at, 'class="plan-hero"')
        self.assertIn("Re-check<strong>Due</strong>", hero)
        self.assertNotIn("Re-check in", hero)
        card = at.get_by_key("plan_recheck_due")
        self.assertIn("<strong>Re-check due</strong><span>Your re-check date (2 Feb 2027) has come. Screen again so you "
                      "and your doctor can see how things have changed.</span>", self.html(card, "plan-recheck"))
        self.assertIn(("Time to screen again", "screenings"),
                      [(n["title"], n["action"]) for n in notifications(self.p)])
        at.button(key="plan_screen_again").click()
        self.ok(at.run())
        self.assertEqual((at.session_state["nav"], at.session_state["active_screening"]), ("screenings", "diabetes"))


class MedicineTests(MyPlanFixture):
    def setUp(self):
        super().setUp()
        self.clock.set("2026-11-09 07:00"); self.pid = self.medicine()
        self.clock.set("2026-11-10 12:00")

    def doses(self):
        return [(d["time"], d["status"]) for d in scheduled_doses(self.p, self.clock.moment.date(), self.clock.moment.date())]

    def test_take_and_skip_follow_the_dose_rules(self):
        at = self.page()
        due = at.get_by_key(f"dose_block_{self.pid}_0800")
        self.assertIn('<span class="plan-status" style="--pill:#C98F1E">Due</span>', self.bodies(due))
        self.assertEqual([(b.key, b.label) for b in due.get("button")],
                         [(f"taken_{self.pid}_08:00", "Take"), (f"skipped_{self.pid}_08:00", "Skip")])
        later = at.get_by_key(f"dose_block_{self.pid}_2000")
        self.assertIn('<span class="plan-status" style="--pill:#5B8FA3">Due at 20:00</span>', self.bodies(later))
        self.assertEqual(later.get("button"), [])   # not due yet: mark_dose would refuse it
        self.assertEqual(at.warning, [])

        at.button(key=f"taken_{self.pid}_08:00").click()
        self.ok(at.run())
        self.assertEqual([t.value for t in at.toast], ["Metformin at 08:00 recorded as taken."])
        self.assertEqual(self.doses(), [("08:00", "taken"), ("20:00", "upcoming")])
        due = at.get_by_key(f"dose_block_{self.pid}_0800")
        self.assertIn('<span class="plan-status" style="--pill:#2E9477">Taken</span>', self.bodies(due))
        self.assertEqual([b.label for b in due.get("button")], ["Mark skipped"])

        at.button(key=f"skipped_{self.pid}_08:00").click()
        self.ok(at.run())
        self.assertEqual(self.doses(), [("08:00", "skipped"), ("20:00", "upcoming")])
        self.assertEqual([b.label for b in at.get_by_key(f"dose_block_{self.pid}_0800").get("button")], ["Mark taken"])

    def test_an_earlier_dose_reads_missed_once_the_next_one_is_due(self):
        self.clock.set("2026-11-10 21:00")
        at = self.page()
        early = at.get_by_key(f"dose_block_{self.pid}_0800")
        self.assertIn('<span class="plan-status" style="--pill:#C98F1E">Missed</span>', self.bodies(early))
        self.assertEqual([(b.label, b.proto.type) for b in early.get("button")],
                         [("Taken", "secondary"), ("Skipped", "secondary")])   # record wording: nothing invites a dose
        self.assertEqual([c.value for c in early.get("caption")],
                         ["Record it only if you already took it. If you missed it, do not take it together with your "
                          "20:00 dose; ask your clinician if you are unsure."])
        self.assertEqual(early.get("warning"), [])
        late = at.get_by_key(f"dose_block_{self.pid}_2000")
        self.assertIn('<span class="plan-status" style="--pill:#C98F1E">Due</span>', self.bodies(late))
        self.assertEqual([(b.label, b.proto.type) for b in late.get("button")], [("Take", "primary"), ("Skip", "secondary")])
        self.assertEqual(late.get("caption"), [])
        # The advice not to double up comes before the doses.
        (advice,) = [i for i in at.info if "do not double up" in i.value]
        first, block = self.order(at, advice, early)
        self.assertLess(first, block)

        at.button(key=f"taken_{self.pid}_08:00").click()   # recording it still works
        self.ok(at.run())
        self.assertEqual(self.doses(), [("08:00", "taken"), ("20:00", "missed")])

    def test_a_refused_dose_shows_the_reason_under_it(self):
        at = self.page()
        with mock.patch.object(medications, "mark_dose", side_effect=ValueError("This dose is not due yet.")):
            at.button(key=f"taken_{self.pid}_08:00").click()
            self.ok(at.run())
        block = at.get_by_key(f"dose_block_{self.pid}_0800")
        self.assertEqual([e.value for e in block.get("error")], ["This dose is not due yet."])
        self.assertEqual(at.toast, [])

    def test_a_dose_due_before_its_plan_stopped_keeps_the_warning(self):
        self.clock.set("2026-11-10 12:30"); plan_status(self.doctor["id"], self.pid, "stopped")
        at = self.page()
        block = at.get_by_key(f"dose_block_{self.pid}_0800")
        self.assertIn('<span class="plan-status" style="--pill:#C98F1E">Not logged</span>', self.bodies(block))
        self.assertEqual([b.label for b in block.get("button")], ["Taken", "Skipped"])   # record wording, not "Take"
        self.assertEqual([w.value for w in block.get("warning")],
                         ["This plan was stopped after this dose was due. Record it only if you already took it; do not "
                          "take it now."])
        with self.assertRaises(KeyError):
            at.get_by_key(f"dose_block_{self.pid}_2000")   # after the stop: never due

    def test_history_expander_and_doses_this_week(self):
        self.approve()
        at = self.page()
        (history,) = at.expander
        self.assertEqual(history.label, "Medication plans and 30-day history")
        self.assertIn("**Metformin · 500 mg** — active", [m.value for m in history.markdown])
        self.assertIn("Times: 08:00, 20:00 · Days: Mon, Tue, Wed, Thu, Fri, Sat, Sun", [m.value for m in history.markdown])
        self.assertIn("2026-11-09 to 2026-11-30 · With food", [c.value for c in history.caption])
        (metric,) = history.metric
        self.assertEqual((metric.label, metric.value), ("Last 30 days adherence", "0%"))
        self.assertEqual(len(history.dataframe), 1)
        # Due in the last 7 days: both doses on the 9th and today's 08:00 (20:00 is still to come).
        self.assertIn("Doses this week<strong>0 of 3</strong>", self.html(at, 'class="plan-hero"'))
        at.button(key=f"taken_{self.pid}_08:00").click()
        self.assertIn("Doses this week<strong>1 of 3</strong>", self.html(self.ok(at.run()), 'class="plan-hero"'))
        self.assertIn(("Medication due", "medications"), [(n["title"], n["action"]) for n in notifications(self.p)])


class UrgentResultTests(MyPlanFixture):
    def test_an_urgent_result_for_the_plans_condition_comes_first(self):
        self.clock.set("2026-11-06 10:00")
        urgent = save_screening(self.p, "diabetes", "Model", .7, 1, .5,
                                {"HbA1c_level": 8.1, "blood_glucose_level": 260, "age": 50})
        self.clock.set("2026-11-10 09:00")
        self.approve(screening_id=urgent, targets=[{"key": "activity", "goal": 150}])
        at = self.page()
        panel = self.html(at, 'class="track-panel urgent"')
        self.assertIn("See a doctor soon", panel)
        self.assertIn("Please see a doctor soon so they can confirm it with tests.", panel)
        self.assertEqual(panel.count("<li>"), 1)   # the first reason, as on the home banner
        moment = platform_data.to_local(next(r for r in load_screenings(self.p) if r["id"] == urgent)["created_at"])
        self.assertIn(f"From your diabetes screening on {moment.day} {moment:%b %Y}. This plan goes alongside that "
                      "visit, not instead of it.", panel)
        bodies = self.bodies(at)
        self.assertLess(bodies.index(panel), next(i for i, b in enumerate(bodies) if 'class="plan-hero"' in b))
        self.assertIn("<strong>Re-check on 2 Feb 2027</strong><span>This is your plan's re-check, not the next step: see "
                      "a doctor about the result above first.</span>", self.html(at.get_by_key("plan_recheck"), "plan-recheck"))

        # A later diabetes screening that re-entered both values below the cut-offs clears it.
        save_screening(self.p, "diabetes", "Model", .6, 1, .5, {"HbA1c_level": 5.9, "blood_glucose_level": 120, "age": 50})
        at = self.page()
        self.assertFalse([b for b in self.bodies(at) if "track-panel" in b])
        self.assertIn("We'll remind you to screen again.", self.html(at.get_by_key("plan_recheck"), "plan-recheck"))

    def test_an_urgent_result_for_another_condition_is_left_to_the_home_banner(self):
        save_screening(self.p, "kidney", "Model", .7, 1, .5, {"GFR": 40, "Age": 50})
        self.approve(targets=[{"key": "activity", "goal": 150}])
        at = self.page()
        self.assertFalse([b for b in self.bodies(at) if "track-panel" in b])


class CheckInTests(MyPlanFixture):
    LABELS = {"Age": "age", "Height (cm)": "height_cm", "Weight (kg)": "weight_kg", "Daily steps": "steps",
              "Exercise days per week": "exercise_days", "Minutes per exercise day": "exercise_minutes",
              "Hours of sleep": "sleep_hours", "Water (litres/day)": "water_litres",
              "Fruit/vegetable servings per day": "fruit_veg", "Sugary foods/drinks per day": "sugary_food",
              "Fast food meals per day": "fast_food", "Hours sitting per day": "sedentary_hours",
              "Sleep quality (1–5)": "sleep_quality", "Stress (0–10)": "stress", "Daily activity level": "activity_level"}

    def form(self, at):
        return {self.LABELS[w.label]: w.value for w in [*at.number_input, *at.slider, *at.selectbox] if w.label in self.LABELS}

    def test_the_check_in_starts_from_the_last_one(self):
        at = self.page(nav="wellness")
        self.assertEqual(self.form(at), {**wellness.CHECK_IN_DEFAULTS, "activity_level": "Mostly sitting"})
        self.assertNotIn("Filled in from your last check-in. Change what is new, then save.", [c.value for c in at.caption])

        # The finding: a plan at 84 kg, then a check-in that changes only the hours of sleep.
        self.clock.set("2026-11-09 12:00"); self.log(activity_level="Active", sleep_quality=4)
        self.clock.set("2026-11-10 09:00"); self.approve(targets=[{"key": "weight", "goal": 79}, {"key": "sleep", "goal": [7, 9]}])
        self.clock.set("2026-11-11 08:00")
        at = self.page(nav="wellness")
        self.assertEqual(self.form(at), LOG | {"activity_level": "Active", "sleep_quality": 4})
        self.assertIn("Filled in from your last check-in. Change what is new, then save.", [c.value for c in at.caption])
        next(n for n in at.number_input if n.label == "Hours of sleep").set_value(7.5)
        next(b for b in at.button if b.label == "Save lifestyle check-in").click()
        self.ok(at.run())
        latest = platform_data.lifestyle_history(self.p)[0]["values"]
        self.assertEqual(latest, LOG | {"activity_level": "Active", "sleep_quality": 4, "sleep_hours": 7.5})
        weight = next(t for t in prevention_progress(self.p, platform_data.active_plan(self.p))["targets"]
                      if t["key"] == "weight")
        self.assertEqual(weight["current"], 84)   # not the form's old 75 kg default, which read as "Down 9 kg"

    def test_a_stored_value_is_kept_inside_the_fields_bounds_and_type(self):
        start = wellness.check_in_start
        self.assertEqual((start({"weight_kg": 500}, "weight_kg", 25, 400), start({"weight_kg": 10}, "weight_kg", 25, 400)),
                         (400.0, 25.0))
        self.assertIsInstance(start({"weight_kg": 10}, "weight_kg", 25, 400), float)
        self.assertEqual(start({"height_cm": 164.6}, "height_cm", 90, 250), 165)
        self.assertIsInstance(start({"height_cm": 164.6}, "height_cm", 90, 250), int)
        for values in ({}, {"age": None}, {"age": "x"}, {"age": float("nan")}):
            with self.subTest(values=values):
                self.assertEqual(start(values, "age", 1, 120), 30)


class EscapingTests(MyPlanFixture):
    def test_doctor_names_and_notes_print_literally(self):
        note = "<img src=x onerror=alert(1)> **Walk** daily\nSecond line"
        self.clock.set("2026-11-10 09:00"); first = self.approve(self.odd, note=note)
        at = self.page()
        raw = self.raw(at)
        self.assertNotIn("<b>Zaid</b>", raw)
        self.assertNotIn("<img src=x", raw)
        self.assertIn(f"<h1>Your plan with {html.escape(ODD)}</h1>", raw)
        self.assertIn(f"“{html.escape(note).replace(chr(10), '<br>')}”</p><span>— {html.escape(ODD)}</span>", raw)

        # A second doctor's plan: the chooser names both plans, its labels escaped as Markdown.
        self.clock.set("2026-11-10 10:00"); self.approve(self.doctor, disease="heart", screening_id=None,
                                                         targets=[{"key": "activity", "goal": 150}])
        prevention_plan_status(self.odd["id"], first, "paused")
        at = self.page()
        (chooser,) = at.segmented_control
        self.assertEqual(chooser.options, [markdown_literal(f"{DOCTOR} · heart disease"),
                                           markdown_literal(f"{ODD} · diabetes · paused")])
        self.assertIn(f"<h1>Your plan with {DOCTOR}</h1>", self.raw(at))
        self.assertIn("Agreed on 10 Nov 2026 as your heart disease prevention plan.", self.raw(at))
        chooser.set_value(first)
        raw = self.raw(self.ok(at.run()))
        self.assertIn(f"<h1>Your plan with {html.escape(ODD)}</h1>", raw)
        self.assertIn(f"{html.escape(ODD)} paused this plan.", raw)
        self.assertNotIn("<b>Zaid</b>", raw)

        at = self.page(nav="wellness")
        info = next(i.value for i in at.info if "prevention plan" in i.value)
        self.assertIn(markdown_literal(DOCTOR), info)
        self.assertNotIn("<b>", info)


class DoctorTests(MyPlanFixture):
    def test_doctors_never_get_the_page(self):
        self.link()
        at = self.page(self.doctor)
        self.assertEqual(at.session_state["nav"], "overview")
        self.assertNotIn("nav_medications", [b.key for b in at.sidebar.button])
        self.assertFalse([b for b in self.bodies(at) if "plan-hero" in b or "No prevention plan yet" in b])

        def script():
            from ahead.pages.medications import render_medications
            render_medications()

        direct = AppTest.from_function(script, default_timeout=60)
        direct.session_state["user"] = self.doctor
        self.assertEqual([e.value for e in self.ok(direct.run()).error], ["Doctor access required."])

    def test_patients_see_my_plan_in_the_sidebar(self):
        at = self.page(nav="overview")
        self.assertEqual(at.sidebar.button(key="nav_medications").label, "My plan")
        labels = [b.label for b in at.sidebar.button]
        self.assertEqual(labels[labels.index("My plan") + 1], "Health & Lifestyle Tracker")
        styles = "\n".join(h.proto.body for h in at.get("html") if h.proto.body.lstrip().startswith("<style"))
        self.assertIn(f'.st-key-nav_medications .stButton > button::before {{ background-image:url("{icon_uri("sidebar/plan.svg")}"); }}',
                      styles)   # the checklist icon from the mockup
        self.assertEqual(at.button(key="dashboard_doses").label, "My plan →")


class TrackerGoalsTests(MyPlanFixture):
    def goals_note(self, at):
        notes = [i.value for i in at.info if "prevention plan" in i.value]
        return notes[0] if notes else None

    def test_goals_tab_names_the_goals_the_plan_set_and_its_doctor(self):
        set_goals(self.p, {"steps": 9000, "water": 2.5})
        self.approve()
        self.assertEqual(get_goals(self.p), {"steps": 9000, "water": 2.5, "weight": 79.0, "exercise": 150, "fruit_veg": 5,
                                             "sleep": 8.0})
        at = self.page(nav="wellness")
        self.assertEqual(self.goals_note(at),
                         f"Set by your prevention plan with {markdown_literal(DOCTOR)}: target weight, weekly exercise "
                         "minutes, sleep target and fruit/vegetable servings. The sleep target is the middle of the "
                         "plan's 7–9 h range.\n\nYou can change these goals here. My plan still shows your progress "
                         "against the goals your doctor set.")
        hints = {n.label: n.help for n in at.number_input}
        for label in ("Target weight (kg)", "Weekly exercise minutes", "Sleep target (h)", "Fruit/vegetable servings target"):
            self.assertEqual(hints[label], f"Set by your prevention plan with {markdown_literal(DOCTOR)}.")
        for label in ("Daily step goal", "Daily water target (L)"):
            self.assertFalse(hints[label])

        # The patient changes two goals: only the two still at the plan's values are named.
        weight = next(n for n in at.number_input if n.label == "Target weight (kg)")
        sleep = next(n for n in at.number_input if n.label == "Sleep target (h)")
        weight.set_value(81.0); sleep.set_value(7.5)
        next(b for b in at.button if b.label == "Save goals").click()
        self.ok(at.run())
        self.assertEqual([t.value for t in at.toast], ["Goals saved."])
        self.assertIn(f"plan with {markdown_literal(DOCTOR)}: weekly exercise minutes and fruit/vegetable servings.\n\n",
                      self.goals_note(at))
        self.assertNotIn("middle", self.goals_note(at))

    def test_a_paused_plan_is_marked_and_a_stopped_one_is_not_named(self):
        plan_id = self.approve(targets=[{"key": "activity", "goal": 200}])
        prevention_plan_status(self.doctor["id"], plan_id, "paused")
        self.assertTrue(self.goals_note(self.page(nav="wellness")).startswith(
            f"Set by your prevention plan with {markdown_literal(DOCTOR)} (paused): weekly exercise minutes.\n\n"))
        prevention_plan_status(self.doctor["id"], plan_id, "stopped")
        self.assertIsNone(self.goals_note(self.page(nav="wellness")))


def typical(disease):
    stats = feature_summary(disease)
    return {f: stats[f]["default"] if stats[f]["kind"] == "category" else stats[f]["mode"] if stats[f]["binary"]
            else stats[f]["median"] for f in features(disease)}


def scored(disease, probability, saved=True, **values):
    """A stored Screenings result (as the page keeps it in last_results), flagged when probability >= threshold."""
    model = best_model(disease)
    threshold = decision_threshold(disease, model)
    return {disease: {"values": typical(disease) | values, "probability": probability, "saved": saved,
                      "prediction": int(probability >= threshold), "model": model, "threshold": threshold,
                      "missing": []}}


class ScreeningPointerTests(MyPlanFixture):
    SHARE = "Share with your doctor"

    def screen(self, result):
        return self.page(nav="screenings", active_screening=next(iter(result)), last_results=result)

    def captions(self, at):
        return [c.value for c in at.caption]

    def test_an_at_risk_result_with_an_active_plan_points_to_my_plan(self):
        plan_id = self.approve()
        name = markdown_literal(DOCTOR)
        at = self.screen(scored("diabetes", 0.9))
        self.assertIn('class="track-panel at-risk"', self.raw(at))
        self.assertIn(f"You already follow a diabetes prevention plan with {name}, who can see this result in AHEAD. "
                      "Keep going with it on My plan.", self.captions(at))
        self.assertFalse([c for c in self.captions(at) if "Your linked doctor" in c])
        self.assertNotIn(self.SHARE, [b.label for b in at.button])
        at.button(key="diabetes_share_plan").click()
        self.assertEqual(self.ok(at.run()).session_state["nav"], "medications")

        # History off: the line says the doctor cannot see this result.
        at = self.screen(scored("diabetes", 0.9, saved=False))
        self.assertIn(f"You already follow a diabetes prevention plan with {name}. This result was not saved because "
                      "Keep screening history is off in Settings, so they can't see it in AHEAD. Download the PDF to "
                      "show them.", self.captions(at))

        # Another condition, an urgent result, or a paused plan: the usual share step.
        linked = f"Your linked doctor, {name}, can see this result in AHEAD."
        for result in (scored("heart", 0.99), scored("diabetes", 0.9, HbA1c_level=7.1)):
            at = self.screen(result)
            self.assertIn(linked, self.captions(at))
            self.assertNotIn("diabetes_share_plan", [b.key for b in at.button])
        prevention_plan_status(self.doctor["id"], plan_id, "paused")
        at = self.screen(scored("diabetes", 0.9))
        self.assertIn(linked, self.captions(at))
        self.assertFalse([c for c in self.captions(at) if c.startswith("You already follow")])


if __name__ == "__main__":
    unittest.main()
