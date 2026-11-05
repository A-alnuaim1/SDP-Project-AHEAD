"""Path to lower risk (ahead/risk_plan.py): lever rules, gates (the urgent rules in ahead/triage.py), scores
identical to the Screenings page, and the search."""
import math
import os
import tempfile
import time
import unittest
import warnings
from unittest import mock

import numpy as np
from streamlit.testing.v1 import AppTest

from ahead.config import OPTIONAL_INPUTS
from ahead.resources import (
    best_model, decision_threshold, feature_summary, features, load_dataset, load_pipeline, missing_inputs_profile,
)
from ahead.risk_plan import (
    GATE_STATUSES, GATES, LEVERS, MIN_DROP, SHARED_TARGETS, _batch_scores, allowed_targets, check_change,
    fixed_inputs, gates, lever_effects, lever_for, levers, score, shared_plan, suggest_path, what_if,
)
from ahead.screening import _model_frame
from ahead.storage import init_db

DISEASES = ("diabetes", "heart", "kidney")

# The lever design, written out again independently of LEVERS so a slip in either is caught. The blood glucose
# rule starts at 140 mg/dL (random, not fasting, glucose); the rest is the starting design.
BMI = {"dir": "down", "active": lambda v: v >= 25, "limit": lambda v: max(24.9, v * 0.9)}
HBA1C = {"dir": "down", "active": lambda v: v >= 5.7, "limit": lambda v: max(5.7, v - 1.0)}
NUMERIC = {
    ("diabetes", "bmi"): BMI,
    ("diabetes", "HbA1c_level"): HBA1C,
    ("diabetes", "blood_glucose_level"): {"dir": "down", "active": lambda v: v >= 140, "limit": lambda v: max(140, v - 40)},
    ("heart", "BMI"): BMI,
    ("kidney", "BMI"): BMI,
    ("kidney", "AlcoholConsumption"): {"dir": "down", "active": lambda v: True, "limit": lambda v: v * 0.5},
    ("kidney", "PhysicalActivity"): {"dir": "up", "active": lambda v: True, "limit": lambda v: v + 3},
    ("kidney", "DietQuality"): {"dir": "up", "active": lambda v: True, "limit": lambda v: v + 2},
    ("kidney", "SleepQuality"): {"dir": "up", "active": lambda v: True, "limit": lambda v: v + 1.5},
    ("kidney", "SystolicBP"): {"dir": "down", "active": lambda v: v >= 130, "limit": lambda v: max(120, v - 20)},
    ("kidney", "DiastolicBP"): {"dir": "down", "active": lambda v: v > 80, "limit": lambda v: max(80, v - 10)},
    ("kidney", "FastingBloodSugar"): {"dir": "down", "active": lambda v: v >= 100, "limit": lambda v: max(100, v - 30)},
    ("kidney", "HbA1c"): HBA1C,
}
SWITCHES = {
    ("diabetes", "smoking_history"): ({"current"}, "former"),
    ("heart", "PhysicalActivities"): ({"No"}, "Yes"),
    ("heart", "SmokerStatus"): ({"Current smoker - now smokes every day", "Current smoker - now smokes some days"},
                                "Former smoker"),
    ("heart", "AlcoholDrinkers"): ({"Yes"}, "No"),
    ("kidney", "Smoking"): ({1}, 0),
}
OWNERS = {("diabetes", "HbA1c_level"): "doctor", ("diabetes", "blood_glucose_level"): "doctor",
          ("kidney", "SystolicBP"): "doctor", ("kidney", "DiastolicBP"): "doctor",
          ("kidney", "FastingBloodSugar"): "doctor", ("kidney", "HbA1c"): "doctor"}
NEVER_CHANGED = {
    "diabetes": {"gender", "age", "hypertension", "heart_disease"},
    "heart": {"Sex", "AgeCategory", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "HadStroke",
              "HadAsthma", "HadKidneyDisease", "HadArthritis", "HadDiabetes"},
    "kidney": {"Age", "Gender", "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension", "FamilyHistoryDiabetes",
               "SerumCreatinine", "BUNLevels", "GFR", "ProteinInUrine", "ACR", "HemoglobinLevels", "Edema"},
}
# The gate cut-offs (the urgent rules in ahead/triage.py), written out again: (feature, value that trips, value just
# on the other side, status).
GATE_CASES = {
    "diabetes": [("age", 17.9, 18.0, "not_for_children"), ("HbA1c_level", 6.5, 6.4, "diagnostic_range"),
                 ("blood_glucose_level", 200, 199, "diagnostic_range")],
    "kidney": [("Age", 17, 18, "not_for_children"), ("GFR", 59.9, 60.0, "kidney_markers_abnormal"),
               ("ACR", 30.0, 29.9, "kidney_markers_abnormal"), ("HbA1c", 6.5, 6.4, "diagnostic_range"),
               ("FastingBloodSugar", 126.0, 125.9, "diagnostic_range"),
               ("SystolicBP", 180, 179, "severe_blood_pressure"), ("DiastolicBP", 120, 119, "severe_blood_pressure")],
}
# A heart record no lever applies to, flagged by the model (score about 99%).
HEART_STUCK = {"Sex": "Male", "GeneralHealth": "Poor", "PhysicalHealthDays": 30, "MentalHealthDays": 30,
               "PhysicalActivities": "Yes", "SleepHours": 8, "HadStroke": "Yes", "HadAsthma": "Yes",
               "HadKidneyDisease": "Yes", "HadArthritis": "Yes", "HadDiabetes": "Yes", "SmokerStatus": "Never smoked",
               "AgeCategory": "Age 80 or older", "BMI": 23.0, "AlcoholDrinkers": "No"}


def plain(row) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def sleep_ok(current, new) -> bool:
    if current < 7:
        return current < new <= min(7, current + 2)
    if current > 9:
        return max(9, current - 2) <= new < current
    return False


class RiskPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message=".*serialized model.*")
        cls.rows, cls.gated = {}, {}
        for disease in DISEASES:
            data = load_dataset(disease)
            # Nearly every flagged diabetes and kidney row trips a gate, so those two search the whole file.
            sample = data.sample(4000 if disease == "heart" else len(data), random_state=11)[features(disease)]
            proba = load_pipeline(disease, best_model(disease)).predict_proba(sample)[:, 1]
            cls.rows[disease], cls.gated[disease] = [], []
            for _, row in sample[proba >= decision_threshold(disease)].iterrows():
                values = plain(row)
                bucket = cls.gated[disease] if gates(disease, values) else cls.rows[disease]
                if len(bucket) < (6 if bucket is cls.gated[disease] else 12):
                    bucket.append(values)
                if len(cls.rows[disease]) == 12 and (len(cls.gated[disease]) == 6 or not GATES[disease]):
                    break
        # Blank optional inputs, as the form allows.
        cls.rows["diabetes"] += [{**row, "HbA1c_level": None} for row in cls.rows["diabetes"][:3]]
        cls.rows["diabetes"] += [{**row, "blood_glucose_level": None} for row in cls.rows["diabetes"][3:5]]
        cls.rows["kidney"] += [{**row, "SystolicBP": None, "DiastolicBP": None} for row in cls.rows["kidney"][:3]]
        cls.rows["kidney"] += [{**row, "FastingBloodSugar": None, "HbA1c": None} for row in cls.rows["kidney"][3:5]]

    # ------------------------------------------------------------------ lever table
    def test_lever_table_is_the_reviewed_design(self):
        for disease in DISEASES:
            stats = feature_summary(disease)
            table = {(disease, lever.feature) for lever in LEVERS[disease]}
            expected = {key for key in list(NUMERIC) + list(SWITCHES) + [("heart", "SleepHours")] if key[0] == disease}
            self.assertEqual(table, expected, disease)
            self.assertFalse(NEVER_CHANGED[disease] & {f for _, f in table}, disease)
            self.assertTrue(NEVER_CHANGED[disease] <= set(fixed_inputs(disease)), disease)
            for lever in LEVERS[disease]:
                self.assertIn(lever.feature, features(disease))
                self.assertEqual(lever.owner, OWNERS.get((disease, lever.feature), "patient"), lever.feature)
                self.assertIn(lever.goal, {None, "weight", "exercise", "sleep"})
                info = stats[lever.feature]
                if (disease, lever.feature) in SWITCHES:
                    starts, end = SWITCHES[(disease, lever.feature)]
                    self.assertEqual(set(lever.switch_from), starts)
                    self.assertEqual(lever.switch_to, end)
                    if info["kind"] == "category":   # every spelling must be one the model was trained on
                        self.assertTrue(set(lever.switch_from) | {lever.switch_to} <= set(info["options"]), lever)
                    else:
                        self.assertTrue(info["binary"])
        self.assertEqual(lever_for("diabetes", "bmi").goal, "weight")
        self.assertEqual(lever_for("heart", "PhysicalActivities").goal, "exercise")
        self.assertEqual(lever_for("heart", "SleepHours").goal, "sleep")

    def test_allowed_targets_follow_the_rules_across_the_whole_range(self):
        for disease in DISEASES:
            stats = feature_summary(disease)
            for lever in LEVERS[disease]:
                info = stats[lever.feature]
                if lever.kind == "switch":
                    for option in info.get("options") or [0, 1]:
                        targets = allowed_targets(lever, option, info)
                        starts, end = SWITCHES[(disease, lever.feature)]
                        self.assertEqual(targets, [(end, 1.0)] if option in starts else [], (lever.feature, option))
                    continue
                for current in np.linspace(info["min"] - 5, info["max"] + 5, 157):
                    current = round(float(current), 3)
                    targets = allowed_targets(lever, current, info)
                    for value, share in targets:
                        self.assertTrue(info["min"] - 1e-9 <= value <= info["max"] + 1e-9, (lever.feature, current, value))
                        self.assertTrue(0 < share <= 1 + 1e-9)
                        if lever.feature == "SleepHours":
                            self.assertTrue(sleep_ok(current, value), (current, value))
                            continue
                        rule = NUMERIC[(disease, lever.feature)]
                        self.assertTrue(rule["active"](current), (lever.feature, current))
                        if rule["dir"] == "down":
                            self.assertTrue(rule["limit"](current) - 1e-9 <= value < current, (lever.feature, current, value))
                        else:
                            self.assertTrue(current < value <= rule["limit"](current) + 1e-9, (lever.feature, current, value))
                    if lever.feature != "SleepHours" and not NUMERIC[(disease, lever.feature)]["active"](current):
                        self.assertEqual(targets, [], (lever.feature, current))
                self.assertEqual(allowed_targets(lever, None, info), [])
                self.assertEqual(allowed_targets(lever, float("nan"), info), [])

    def test_glucose_rule_reads_the_value_as_random_glucose(self):
        lever, info = lever_for("diabetes", "blood_glucose_level"), feature_summary("diabetes")["blood_glucose_level"]
        for normal in (100, 126, 130, 139, 140):   # normal for a non-fasting sample: no target
            self.assertEqual(allowed_targets(lever, normal, info), [], normal)
        self.assertEqual([value for value, _ in allowed_targets(lever, 160, info)], [155, 150, 145, 140])
        self.assertEqual(allowed_targets(lever, 199, info)[-1][0], 159)

    # ------------------------------------------------------------------ gates
    def test_gates_offer_no_plan_at_their_cut_offs(self):
        for disease, cases in GATE_CASES.items():
            self.assertEqual([(g.feature, g.status) for g in GATES[disease]], [(f, s) for f, _, _, s in cases])
            values = self.rows[disease][0]
            for feature, trips, clear, status in cases:
                with self.subTest(disease=disease, feature=feature):
                    gated = {**values, feature: trips}
                    self.assertEqual([g["status"] for g in gates(disease, gated)], [status])
                    for owners in (("patient",), ("patient", "doctor")):
                        plan = suggest_path(disease, gated, owners=owners)
                        self.check_plan(disease, gated, plan, owners)
                        self.assertEqual((plan["status"], plan["changes"], plan["crosses"]), (status, [], False))
                        self.assertEqual(plan["gates"][0]["feature"], feature)
                    self.assertEqual(gates(disease, {**values, feature: clear}), [])
                    if feature in OPTIONAL_INPUTS.get(disease, []):   # a blank value never trips a gate
                        self.assertEqual(gates(disease, {**values, feature: None}), [])
        self.assertEqual(GATES["heart"], ())
        self.assertEqual(gates("heart", self.rows["heart"][0]), [])

    def test_gates_come_before_the_score_check(self):
        """A value in a diagnostic range gets no plan even when the model's score is low."""
        values = {"gender": "Female", "age": 30.0, "hypertension": 0, "heart_disease": 0, "smoking_history": "never",
                  "bmi": 22.0, "HbA1c_level": 6.5, "blood_glucose_level": 90}
        self.assertFalse(score("diabetes", values)["flagged"])
        self.assertEqual(suggest_path("diabetes", values)["status"], "diagnostic_range")
        self.assertEqual(suggest_path("diabetes", {**values, "HbA1c_level": 5.0})["status"], "already_below")

    def test_reviewed_examples_are_gated(self):
        child = {"gender": "Female", "age": 14.0, "hypertension": 0, "heart_disease": 0, "smoking_history": "never",
                 "bmi": 29.08, "HbA1c_level": 6.0, "blood_glucose_level": 160}
        self.assertEqual(suggest_path("diabetes", child, owners=("patient", "doctor"))["status"], "not_for_children")
        kidney = load_dataset("kidney")[features("kidney")]
        for index in (299, 1071, 1191):   # eGFR in the 15-30 range
            plan = suggest_path("kidney", plain(kidney.loc[index]), owners=("patient", "doctor"))
            self.assertEqual((plan["status"], plan["changes"]), ("kidney_markers_abnormal", []), index)
        for disease, rows in self.gated.items():
            for values in rows:
                self.assertIn(suggest_path(disease, values)["status"], GATE_STATUSES)

    # ------------------------------------------------------------------ scores and thresholds
    def test_score_is_the_screening_pipeline_path(self):
        for disease in DISEASES:
            model = best_model(disease)
            pipeline = load_pipeline(disease, model)
            for values in self.rows[disease]:
                result = score(disease, values)
                expected = float(pipeline.predict_proba(_model_frame(values, features(disease)))[0][1])
                self.assertEqual(result["score"], expected)
                missing = [f for f in OPTIONAL_INPUTS.get(disease, []) if f in values and blank(values[f])]
                usual = decision_threshold(disease, model)
                profile = missing_inputs_profile(disease, model, missing) if missing else None
                self.assertEqual(result["threshold"], min(profile["threshold"], usual) if profile else usual)
                self.assertEqual(result["flagged"], expected >= result["threshold"])
                self.assertEqual(result["missing"], missing)
                # The search scores many rows at once; each must equal the page's single-row score.
                batch = _batch_scores(pipeline, features(disease), values, [{}, {}])
                self.assertAlmostEqual(float(batch[1]), expected, places=12)

    def test_blank_inputs_lower_the_threshold_like_the_page(self):
        values = self.rows["diabetes"][0]
        self.assertEqual(score("diabetes", {**values, "HbA1c_level": None})["threshold"],
                         missing_inputs_profile("diabetes", "GradientBoosting", ["HbA1c_level"])["threshold"])
        self.assertLess(score("diabetes", {**values, "HbA1c_level": None, "blood_glucose_level": None})["threshold"], 0.5)
        other = score("diabetes", values, "LogisticRegression")
        self.assertEqual(other["threshold"], decision_threshold("diabetes", "LogisticRegression"))
        self.assertEqual(other["model"], "LogisticRegression")

    def test_blank_required_kidney_test_gives_no_score_and_no_plan(self):
        values = {**self.rows["kidney"][0], "SerumCreatinine": None}
        self.assertTrue(score("kidney", values)["no_score"])
        plan = suggest_path("kidney", values, owners=("patient", "doctor"))
        self.assertEqual((plan["status"], plan["changes"], plan["crosses"]), ("no_score", [], False))
        with self.assertRaises(ValueError):   # the form never leaves a non-optional input empty
            score("heart", {**self.rows["heart"][0], "BMI": None})

    def test_score_matches_the_screenings_page(self):
        """Fill and submit the real screening form and compare with score() for the values it saved."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        previous = os.environ.get("AHEAD_DB_PATH")
        os.environ["AHEAD_DB_PATH"] = os.path.join(temp.name, "page.sqlite3")
        self.addCleanup(lambda: os.environ.__setitem__("AHEAD_DB_PATH", previous) if previous
                        else os.environ.pop("AHEAD_DB_PATH", None))
        init_db()

        def page():
            import streamlit as st
            from ahead.screening import render_predictor
            render_predictor(st.session_state.disease)

        cases = [("diabetes", {**self.rows["diabetes"][0], "HbA1c_level": None}),
                 ("heart", self.rows["heart"][0]),
                 ("kidney", {**self.rows["kidney"][0], "SystolicBP": None, "DiastolicBP": None})]
        for disease, values in cases:
            with self.subTest(disease=disease):
                at = AppTest.from_function(page, default_timeout=120)
                for key, value in {"user": {"id": None, "name": "Test Patient", "role": "Patient"},
                                   "prefs": {"keep_history": False}, "last_results": {}, "screening_history": [],
                                   "disease": disease}.items():
                    at.session_state[key] = value
                at.run()
                entered = {}
                for feature, value in values.items():
                    widgets = at.number_input if any(w.key == f"{disease}_{feature}" for w in at.number_input) \
                        else at.selectbox
                    widget = widgets(key=f"{disease}_{feature}")
                    if isinstance(widget.value, int) and isinstance(value, float) and value.is_integer():
                        value = int(value)   # whole-number widgets (age, day counts) hold ints
                    widget.set_value(value)
                    entered[feature] = value
                next(b for b in at.button if b.label == "Generate Screening Result").click().run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                shown = at.session_state["last_results"][disease]
                self.assertEqual(shown["values"], entered)
                mine = score(disease, entered)
                self.assertEqual(mine["score"], shown["probability"])
                self.assertEqual(mine["threshold"], shown["threshold"])
                self.assertEqual(mine["missing"], shown["missing"])
                self.assertEqual(mine["flagged"], bool(shown["prediction"]))

    # ------------------------------------------------------------------ the search
    def check_plan(self, disease, values, plan, owners, max_changes=3):
        allowed = {lever.feature for lever in levers(disease, owners)}
        stats = feature_summary(disease)
        changed = dict(values)
        for change in plan["changes"]:
            feature, old, new = change["feature"], change["from"], change["to"]
            self.assertIn(feature, allowed)
            self.assertNotIn(feature, NEVER_CHANGED[disease])
            self.assertEqual(old, values[feature])
            self.assertFalse(blank(old), "a blank input was filled in")
            if (disease, feature) in SWITCHES:
                starts, end = SWITCHES[(disease, feature)]
                self.assertIn(old, starts)
                self.assertEqual(new, end)
            else:
                info = stats[feature]
                self.assertTrue(info["min"] - 1e-9 <= new <= info["max"] + 1e-9, (feature, new))
                if feature == "SleepHours":
                    self.assertTrue(sleep_ok(old, new), (old, new))
                else:
                    rule = NUMERIC[(disease, feature)]
                    self.assertTrue(rule["active"](old), (feature, old))
                    if rule["dir"] == "down":
                        self.assertTrue(rule["limit"](old) - 1e-9 <= new < old, (feature, old, new))
                    else:
                        self.assertTrue(old < new <= rule["limit"](old) + 1e-9, (feature, old, new))
            changed[feature] = new
        for feature, value in values.items():
            if feature not in {c["feature"] for c in plan["changes"]}:
                self.assertTrue(blank(value) and blank(changed[feature]) or changed[feature] == value)
        self.assertLessEqual(len(plan["changes"]), max_changes)
        base = plan["base_score"]
        if plan["changes"]:
            self.assertEqual(plan["gates"], [])
            self.assertEqual(plan["new_score"], score(disease, changed)["score"])
            for change in plan["changes"]:   # every change lowers the score on its own, not only in combination
                alone = what_if(disease, values, {change["feature"]: change["to"]})["score"]
                self.assertLess(alone, base, change)
        target = plan["threshold"] - plan["margin"] if plan["threshold"] is not None else None
        if plan["status"] == "crosses":
            self.assertTrue(plan["crosses"])
            self.assertLess(plan["new_score"], target)
            for change in plan["changes"]:   # smallest plan: dropping any change no longer crosses
                rest = {c["feature"]: c["to"] for c in plan["changes"] if c is not change}
                self.assertGreaterEqual(what_if(disease, values, rest)["score"], target)
        elif plan["status"] == "best_reduction":
            self.assertFalse(plan["crosses"])
            self.assertLessEqual(plan["new_score"], base - MIN_DROP + 1e-12)
            for change in plan["changes"]:   # every change in a best-reduction path helps
                rest = {c["feature"]: c["to"] for c in plan["changes"] if c is not change}
                self.assertGreater(what_if(disease, values, rest)["score"], plan["new_score"])
        else:
            self.assertEqual(plan["changes"], [])
            self.assertEqual(plan["new_score"], base)
            steps = [s for effect in lever_effects(disease, values, owners=owners)["levers"] for _, s in effect["steps"]]
            if plan["status"] == "no_path":   # no allowed change lowers the score on its own
                self.assertTrue(steps and all(s >= base - 1e-9 for s in steps))
            elif plan["status"] == "negligible":   # something lowers it, but by less than MIN_DROP
                self.assertTrue(0 < plan["largest_drop"] < MIN_DROP)
                self.assertTrue(any(s < base - 1e-9 for s in steps))
                self.assertTrue(all(s > base - MIN_DROP for s in steps))
            elif plan["status"] in GATE_STATUSES:
                self.assertEqual(plan["status"], plan["gates"][0]["status"])
                self.assertEqual(plan["levers"], [])
            elif plan["status"] == "no_levers":
                self.assertEqual((plan["levers"], steps), ([], []))

    def test_paths_respect_every_rule(self):
        statuses = set()
        for disease in DISEASES:
            for values in self.rows[disease]:
                for owners in (("patient",), ("patient", "doctor")):
                    with self.subTest(disease=disease, owners=owners):
                        plan = suggest_path(disease, values, owners=owners)
                        self.check_plan(disease, values, plan, owners)
                        statuses.add(plan["status"])
                plan = suggest_path(disease, values, owners=("patient", "doctor"), max_changes=1)
                self.check_plan(disease, values, plan, ("patient", "doctor"), max_changes=1)
        self.assertTrue({"crosses", "best_reduction", "negligible"} <= statuses, statuses)

    def test_min_drop_separates_best_reductions_from_noise(self):
        cases = [(disease, values) for disease in DISEASES for values in self.rows[disease]
                 if suggest_path(disease, values)["status"] == "negligible"]
        self.assertTrue(cases)
        for disease, values in cases[:4]:
            plan = suggest_path(disease, values)
            with mock.patch("ahead.risk_plan.MIN_DROP", 0.0):   # without a minimum, the same search offers it
                loose = suggest_path(disease, values)
            self.assertEqual(loose["status"], "best_reduction")
            self.assertTrue(0 < loose["score_drop"] <= plan["largest_drop"] + 1e-12)
            self.assertLess(plan["largest_drop"] - loose["score_drop"], 0.001 + 1e-12)

    def test_large_searches_still_follow_the_rules(self):
        """Past MAX_BATCH_ROWS the search goes size by size and shrinks only subsets that cross at full strength."""
        cases = [(disease, values, suggest_path(disease, values, owners=("patient", "doctor")))
                 for disease in ("diabetes", "kidney") for values in self.rows[disease][:4]]
        with mock.patch("ahead.risk_plan.MAX_BATCH_ROWS", 30):
            for disease, values, usual in cases:
                plan = suggest_path(disease, values, owners=("patient", "doctor"))
                self.check_plan(disease, values, plan, ("patient", "doctor"))
                if plan["crosses"]:   # the full search tries a superset, so it never needs more changes
                    self.assertTrue(usual["crosses"])
                    self.assertLessEqual(len(usual["changes"]), len(plan["changes"]))

    def test_blank_inputs_are_never_filled(self):
        values = {**self.rows["diabetes"][0], "HbA1c_level": None, "blood_glucose_level": None}
        plan = suggest_path("diabetes", values, owners=("patient", "doctor"))
        self.assertFalse({"HbA1c_level", "blood_glucose_level"} & {c["feature"] for c in plan["changes"]})
        self.assertFalse({"HbA1c_level", "blood_glucose_level"} & {lv["feature"] for lv in plan["levers"]})
        with self.assertRaisesRegex(ValueError, "never fills in"):
            what_if("diabetes", values, {"HbA1c_level": 6.0})
        kidney = {**self.rows["kidney"][0], "SystolicBP": None, "FastingBloodSugar": float("nan")}
        plan = suggest_path("kidney", kidney, owners=("patient", "doctor"))
        self.assertFalse({"SystolicBP", "FastingBloodSugar"} & {c["feature"] for c in plan["changes"]})

    def test_category_transitions_only_as_allowed(self):
        base = self.rows["diabetes"][0]
        for status in ("never", "No Info", "ever", "not current", "former"):
            plan = suggest_path("diabetes", {**base, "smoking_history": status}, owners=("patient", "doctor"))
            self.assertNotIn("smoking_history", {lv["feature"] for lv in plan["levers"]}, status)
        self.assertIn("smoking_history",
                      {lv["feature"] for lv in suggest_path("diabetes", {**base, "smoking_history": "current"})["levers"]})
        heart = self.rows["heart"][0]
        for status in ("Never smoked", "Former smoker"):
            self.assertNotIn("SmokerStatus", {lv["feature"] for lv in suggest_path(
                "heart", {**heart, "SmokerStatus": status})["levers"]}, status)
        for active in ("Yes",):
            self.assertNotIn("PhysicalActivities", {lv["feature"] for lv in suggest_path(
                "heart", {**heart, "PhysicalActivities": active})["levers"]})
        self.assertNotIn("Smoking", {lv["feature"] for lv in suggest_path(
            "kidney", {**self.rows["kidney"][0], "Smoking": 0})["levers"]})

    def test_what_if_rejects_changes_that_break_the_rules(self):
        diabetes = {**self.rows["diabetes"][0], "bmi": 32.0, "smoking_history": "never", "HbA1c_level": 6.6,
                    "blood_glucose_level": 160}
        bad = [({"bmi": 33.0}, "outside the allowed"),             # wrong direction
               ({"bmi": 28.0}, "outside the allowed"),             # more than 10% lower
               ({"smoking_history": "former"}, "only changes"),     # never-smoker
               ({"age": 30}, "not something"),                     # fixed input
               ({"HbA1c_level": 5.5}, "outside the allowed"),       # below the 5.7 floor
               ({"blood_glucose_level": 110}, "outside the allowed"),   # more than 40 mg/dL
               ({"blood_glucose_level": 135}, "outside the allowed")]   # below the 140 floor
        for changes, message in bad:
            with self.assertRaisesRegex(ValueError, message):
                what_if("diabetes", diabetes, changes)
        with self.assertRaisesRegex(ValueError, "does not apply"):
            what_if("diabetes", {**diabetes, "bmi": 24.0}, {"bmi": 23.0})
        with self.assertRaisesRegex(ValueError, "does not apply"):
            what_if("diabetes", {**diabetes, "blood_glucose_level": 130}, {"blood_glucose_level": 120})
        current = {**diabetes, "smoking_history": "current"}
        with self.assertRaisesRegex(ValueError, "only changes"):
            what_if("diabetes", current, {"smoking_history": "never"})
        heart = {**self.rows["heart"][0], "SleepHours": 5}
        with self.assertRaisesRegex(ValueError, "outside the allowed"):
            what_if("heart", heart, {"SleepHours": 8})
        with self.assertRaisesRegex(ValueError, "does not apply"):
            what_if("heart", {**heart, "SleepHours": 8}, {"SleepHours": 7})
        kidney = {**self.rows["kidney"][0], "DiastolicBP": 80, "SystolicBP": 125, "PhysicalActivity": 9.0}
        with self.assertRaisesRegex(ValueError, "does not apply"):
            what_if("kidney", kidney, {"DiastolicBP": 79})
        with self.assertRaisesRegex(ValueError, "does not apply"):
            what_if("kidney", kidney, {"SystolicBP": 120})
        with self.assertRaisesRegex(ValueError, "outside the allowed"):   # past the data maximum of about 10
            what_if("kidney", kidney, {"PhysicalActivity": 10.5})
        # Allowed changes go through, any value between the current one and the limit (for sliders).
        result = what_if("diabetes", current, {"bmi": 29.3, "smoking_history": "former", "HbA1c_level": 6.05})
        self.assertEqual(result["values"]["smoking_history"], "former")
        self.assertEqual(result["score"], score("diabetes", result["values"])["score"])
        self.assertEqual(what_if("diabetes", current, {"bmi": "29.3"})["score"],
                         what_if("diabetes", current, {"bmi": 29.3})["score"])
        self.assertEqual(result["threshold"], score("diabetes", current)["threshold"])

    def test_deterministic(self):
        for disease in DISEASES:
            for values in self.rows[disease][:4]:
                first = suggest_path(disease, values, owners=("patient", "doctor"))
                again = suggest_path(disease, dict(reversed(list(values.items()))), owners=("patient", "doctor"))
                self.assertEqual(first, again)

    def test_no_plan_cases(self):
        self.assertTrue(score("heart", HEART_STUCK)["flagged"])
        plan = suggest_path("heart", HEART_STUCK)
        self.assertEqual((plan["status"], plan["changes"], plan["crosses"]), ("no_levers", [], False))
        self.assertEqual(plan["new_score"], plan["base_score"])
        # Rules apply, but in this model more sleep and drinking both go with a LOWER score, so cutting sleep
        # from 11 h toward 9 h and stopping alcohol each raise it: nothing lowers the score on its own.
        stuck = {**HEART_STUCK, "SleepHours": 11, "AlcoholDrinkers": "Yes"}
        plan = suggest_path("heart", stuck)
        self.assertEqual({lv["feature"] for lv in plan["levers"]}, {"SleepHours", "AlcoholDrinkers"})
        self.assertEqual((plan["status"], plan["changes"]), ("no_path", []))
        self.check_plan("heart", stuck, plan, ("patient",))
        low = {**self.rows["diabetes"][0], "HbA1c_level": 4.0, "blood_glucose_level": 80, "hypertension": 0,
               "heart_disease": 0, "age": 25.0, "bmi": 22.0}
        self.assertEqual(suggest_path("diabetes", low)["status"], "already_below")
        with self.assertRaises(ValueError):
            suggest_path("heart", stuck, owners=("nurse",))

    def test_lever_effects_cover_each_applicable_lever(self):
        values = self.rows["kidney"][0]
        effects = lever_effects("kidney", values)
        stats = feature_summary("kidney")
        expected = [lever.feature for lever in LEVERS["kidney"]
                    if allowed_targets(lever, values[lever.feature], stats[lever.feature])]
        self.assertEqual([row["feature"] for row in effects["levers"]], expected)
        for row in effects["levers"]:
            levels = [value for value, _ in allowed_targets(lever_for("kidney", row["feature"]), values[row["feature"]],
                                                            stats[row["feature"]])]
            self.assertEqual([value for value, _ in row["steps"]], levels)
            for value, step_score in row["steps"]:
                self.assertAlmostEqual(step_score, what_if("kidney", values, {row["feature"]: value})["score"], places=6)
            self.assertEqual((row["smallest_score"], row["full_score"]), (row["steps"][0][1], row["steps"][-1][1]))

    # ------------------------------------------------------------------ shared plan
    def test_shared_plan_maps_each_target_onto_each_model(self):
        diabetes = {**self.rows["diabetes"][0], "smoking_history": "current", "bmi": 32.0, "HbA1c_level": 6.2,
                    "blood_glucose_level": 160}
        heart = {**self.rows["heart"][0], "SmokerStatus": "Current smoker - now smokes some days",
                 "PhysicalActivities": "No", "AlcoholDrinkers": "Yes", "BMI": 32.0}
        # HbA1c and fasting blood sugar stay under the kidney gates (6.5% and 126 mg/dL), so every target maps.
        kidney = {**self.rows["kidney"][0], "Smoking": 1, "BMI": 32.0, "HbA1c": 6.4, "FastingBloodSugar": 120.0,
                  "PhysicalActivity": 2.0, "AlcoholConsumption": 10.0}
        latest = {"diabetes": diabetes, "heart": heart, "kidney": kidney}
        plans = shared_plan({"bmi": 30.0, "quit_smoking": True, "active": True, "hba1c": 6.0, "alcohol": 1.0}, latest)
        mapped = {d: {a["feature"]: a["to"] for a in plans[d]["applied"]} for d in plans}
        self.assertEqual(mapped["diabetes"], {"bmi": 30.0, "smoking_history": "former", "HbA1c_level": 6.0})
        self.assertEqual(mapped["heart"], {"BMI": 30.0, "SmokerStatus": "Former smoker", "PhysicalActivities": "Yes",
                                           "AlcoholDrinkers": "No"})
        self.assertEqual(mapped["kidney"], {"BMI": 30.0, "Smoking": 0, "HbA1c": 6.0, "PhysicalActivity": 5.0,
                                            "AlcoholConsumption": 5.0})
        statuses = {a["feature"]: a["status"] for a in plans["kidney"]["applied"]}
        self.assertEqual((statuses["PhysicalActivity"], statuses["AlcoholConsumption"]), ("approximate", "limited"))
        # Random glucose (diabetes) and fasting blood sugar (kidney) are never mapped onto each other.
        self.assertNotIn("blood_glucose_level", mapped["diabetes"])
        self.assertNotIn("FastingBloodSugar", mapped["kidney"])
        self.assertFalse(any(f in ("blood_glucose_level", "FastingBloodSugar")
                             for targets in SHARED_TARGETS.values() for f in targets.values()))
        for disease, values in latest.items():
            self.assertEqual(plans[disease]["gates"], [])
            self.assertEqual(plans[disease]["before"], score(disease, values)["score"])
            self.assertEqual(plans[disease]["after"], what_if(disease, values, mapped[disease])["score"])
            for entry in plans[disease]["applied"]:
                alone = what_if(disease, values, {entry["feature"]: entry["to"]})["score"]
                self.assertEqual(entry["score_change_alone"], alone - plans[disease]["before"])
        # A target past a lever's limit is applied up to the limit; a blank input is never filled in.
        limited = shared_plan({"bmi": 25.0, "hba1c": 6.0}, {"diabetes": {**diabetes, "HbA1c_level": None}})["diabetes"]
        self.assertEqual([(a["feature"], a["to"], a["status"]) for a in limited["applied"]], [("bmi", 28.8, "limited")])
        self.assertEqual(limited["skipped"][0]["feature"], "HbA1c_level")
        partial = shared_plan({"alcohol": 0.5}, {"heart": heart, "kidney": kidney})
        self.assertEqual(partial["heart"]["applied"], [])
        self.assertEqual(partial["kidney"]["applied"][0]["to"], 5.0)
        # A condition that trips a gate gets no changes, as suggest_path gives it no plan.
        gated = shared_plan({"bmi": 30.0, "hba1c": 6.0}, {"diabetes": {**diabetes, "HbA1c_level": 6.8}})["diabetes"]
        self.assertEqual((gated["applied"], gated["after"], gated["gates"][0]["status"]),
                         ([], gated["before"], "diagnostic_range"))
        self.assertEqual({s["feature"] for s in gated["skipped"]}, {"bmi", "HbA1c_level"})
        with self.assertRaises(ValueError):
            shared_plan({"glucose": 100}, latest)

    def test_shared_plan_handles_unrounded_and_out_of_range_values(self):
        """Dataset rows and CSV uploads keep full float precision; the form accepts values past the data range."""
        targets = {"bmi": 20.0, "alcohol": 1.0, "active": True, "hba1c": 5.0, "quit_smoking": True}
        data = load_dataset("kidney")[features("kidney")].head(80)
        for index, row in data.iterrows():
            # Clear of every kidney gate, so every target maps: HbA1c and fasting blood sugar move under their cut-offs
            # but keep their unrounded decimals.
            values = {**plain(row), "GFR": 90.0, "ACR": 10.0, "HbA1c": 5 + float(row["HbA1c"]) % 1,
                      "FastingBloodSugar": 70 + float(row["FastingBloodSugar"]) % 50}
            plan = shared_plan(targets, {"kidney": values})["kidney"]   # raised ValueError before the fix
            self.assertTrue(plan["applied"], index)
            for entry in plan["applied"]:
                check_change("kidney", values, entry["feature"], entry["to"])
        stats = feature_summary("kidney")
        values = {**self.rows["kidney"][0], "AlcoholConsumption": 10.0000007, "BMI": 31.0694}
        plan = shared_plan({"alcohol": 1.0, "bmi": 20.0}, {"kidney": values})["kidney"]
        self.assertEqual({a["feature"]: (a["to"], a["status"]) for a in plan["applied"]},
                         {"AlcoholConsumption": (5.1, "limited"), "BMI": (28.0, "limited")})
        for bmi, expected in ((44.0, 39.6), (46.0, None)):   # the kidney data stops at a BMI of about 40
            plan = shared_plan({"bmi": 32.0}, {"kidney": {**self.rows["kidney"][0], "BMI": bmi}})["kidney"]
            if expected is None:
                self.assertEqual(plan["applied"], [])
                self.assertIn("training-data range", plan["skipped"][0]["reason"])
            else:
                self.assertEqual([(a["to"], a["status"]) for a in plan["applied"]], [(expected, "limited")])
                self.assertLessEqual(expected, stats["BMI"]["max"])
        # An HbA1c past the data's maximum (about 10%) is always at or above the 6.5% gate, so no plan is offered.
        plan = shared_plan({"hba1c": 6.0}, {"kidney": {**self.rows["kidney"][0], "HbA1c": 11.2}})["kidney"]
        self.assertEqual(plan["applied"], [])
        self.assertEqual(plan["gates"][0]["status"], "diagnostic_range")
        self.assertIn("no plan is offered", plan["skipped"][0]["reason"])

    # ------------------------------------------------------------------ runtime
    def test_runtime_budget(self):
        for disease in DISEASES:
            suggest_path(disease, self.rows[disease][0], owners=("patient", "doctor"))   # warm caches
            for values in self.rows[disease][:5]:
                start = time.perf_counter()
                suggest_path(disease, values, owners=("patient", "doctor"))
                self.assertLess(time.perf_counter() - start, 1.0, disease)


if __name__ == "__main__":
    unittest.main()
