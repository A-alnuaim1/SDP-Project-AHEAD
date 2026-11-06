"""Screening tracks (ahead/triage.py): every urgent rule at its cut-off, blanks, children, no score, one set of words per
audience, the shared panel and pills (ahead/components.py) and their CSS, and the heart history gates."""
import json
import math
import re
import unittest
from pathlib import Path
from unittest import mock

from ahead.components import TRACK_CLASS, track_panel, track_panel_html, track_pill_html
from ahead.config import form_fields
from ahead.theme import DARK, LIGHT
from ahead.triage import (
    ADULTS_ONLY, AUDIENCES, HISTORY, LABELS, RULES, TRACKS, is_urgent_saved, open_urgent, reported_history, rule_table,
    track_text, triage, triage_saved,
)

# The urgent rules, written out again independently of RULES so a slip in either is caught:
# (feature, direction, cut-off, value as the form shows it with its unit).
VALUE_RULES = {
    "diabetes": [("HbA1c_level", ">=", 6.5, "7.1%"), ("blood_glucose_level", ">=", 200, "212 mg/dL")],
    "heart": [],
    "kidney": [("egfr", "<", 60, "52.3"), ("uacr", ">=", 30, "45 mg/g"), ("systolic_bp", ">=", 180, "182 mmHg"),
               ("diastolic_bp", ">=", 120, "121 mmHg")],
}
AGE = {"diabetes": "age", "kidney": "age"}
SHOWN = {"7.1%": 7.1, "212 mg/dL": 212, "52.3": 52.3, "45 mg/g": 45, "130 mg/dL": 130, "182 mmHg": 182,
         "121 mmHg": 121}
NOT_ENTERED = (None, "", "   ", "abc", "7,1", float("nan"), "nan", float("inf"), "-inf", True, False, [], {})
EXPECTED_LABELS = {
    "urgent": {"patient": ("See a doctor soon", "Urgent"), "doctor": ("Urgent", "Urgent")},
    "at_risk": {"patient": ("At risk", "At risk"), "doctor": ("At risk", "At risk")},
    "not_flagged": {"patient": ("Not flagged", "Not flagged"), "doctor": ("Not flagged", "Not flagged")},
    "no_score": {"patient": ("No score", "No score"), "doctor": ("No score", "No score")},
}
FLAGGED, CLEAR = (0.8, 0.5), (0.2, 0.5)   # (score, threshold)
PATIENT_CONFIRM = re.compile(r"\bA doctor needs to confirm .*with tests\b", re.IGNORECASE)
DOCTOR_CONFIRM = re.compile(r"\ba doctor needs to confirm .*with tests\b")
DIAGNOSIS_CLAIM = re.compile(r"\b(you have|you've got|the patient has|diagnosed with|confirms? (that )?you)\b", re.I)
CSS = Path(__file__).resolve().parents[1] / "ahead" / "redesign.css"


def trips(disease, feature, value, flagged=False):
    score, threshold = FLAGGED if flagged else CLEAR
    return triage(disease, {feature: value}, score, threshold)


def normal_track(disease, feature, value, flagged):
    """The track of one value below its urgent cut-off, written out again independently of triage's lab ranges: an
    HbA1c sets it (5.7-6.4% at risk, under 5.7% not flagged); one kidney test alone leaves it to the model."""
    if disease == "diabetes" and feature == "HbA1c_level" and float(value) < 6.5:
        return "at_risk" if float(value) >= 5.7 else "not_flagged"
    return "at_risk" if flagged else "not_flagged"


def all_results():
    """A spread of results across every track, audience-independent."""
    found = [triage("diabetes", {"age": 40, "HbA1c_level": 7.1}, *FLAGGED),
             triage("diabetes", {"age": 40, "HbA1c_level": 7.1, "blood_glucose_level": 250}, *CLEAR),
             triage("diabetes", {"age": 14, "HbA1c_level": 5.0}, *FLAGGED),
             triage("diabetes", {"age": 14, "HbA1c_level": 6.9}, *FLAGGED),
             triage("diabetes", {"age": 40}, *FLAGGED),
             triage("diabetes", {"age": 40}, *CLEAR),
             triage("kidney", {"age": 60, "egfr": 45, "uacr": 40, "systolic_bp": 182, "diastolic_bp": 92}, *FLAGGED),
             triage("kidney", {"age": 60, "uacr": 45}),
             triage("kidney", {"age": 60}),
             # a measured lab value sets the track
             triage("diabetes", {"age": 40, "HbA1c_level": 6.0}, *CLEAR),
             triage("diabetes", {"age": 40, "HbA1c_level": 6.0}, *FLAGGED),
             triage("diabetes", {"age": 40, "HbA1c_level": 5.2}, *FLAGGED),
             triage("diabetes", {"age": 40, "HbA1c_level": 5.2}),
             triage("kidney", {"age": 60, "egfr": 85, "uacr": 10}, *FLAGGED),
             triage("kidney", {"age": 60, "gender": "Male", "creatinine": 0.9, "uacr": 10}, *CLEAR),
             # an older saved kidney screening's keys
             triage("kidney", {"Age": 60, "GFR": 45, "ACR": 40, "SystolicBP": 182, "DiastolicBP": 92}, *FLAGGED),
             triage("heart", {"BMI": 31}, *FLAGGED),
             triage("heart", {"BMI": 31}, *CLEAR)]
    return found


def luminance(colour):
    def channel(value):
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def rgb(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    return [int(hex_colour[i:i + 2], 16) for i in (0, 2, 4)]


def mix(a, share, b):
    """CSS color-mix(in srgb, a share%, b)."""
    return [x * share + y * (1 - share) for x, y in zip(a, b)]


class TriageRuleTests(unittest.TestCase):
    def test_rule_table_is_the_reviewed_design(self):
        for disease, expected in VALUE_RULES.items():
            rules = [rule for rule in RULES[disease] if rule.kind != ADULTS_ONLY]
            self.assertEqual([(r.feature, "<" if r.below is not None else ">=", r.cut_off) for r in rules],
                             [(f, op, cut) for f, op, cut, _ in expected], disease)
            for rule in RULES[disease]:
                self.assertIn(rule.feature, form_fields(disease), rule)
                self.assertTrue(rule.note, rule)   # every cut-off is documented for the clinician review
            ages = [rule for rule in RULES[disease] if rule.kind == ADULTS_ONLY]
            self.assertEqual([(r.feature, r.below) for r in ages], [(AGE[disease], 18)] if disease in AGE else [])
            self.assertEqual(len(rule_table(disease)), len(RULES[disease]))
        self.assertEqual(RULES["heart"], ())
        with self.assertRaises(ValueError):
            triage("lungs", {}, *FLAGGED)

    def test_every_rule_at_and_around_its_cut_off(self):
        for disease, expected in VALUE_RULES.items():
            for feature, op, cut, _ in expected:
                for value, urgent in ((cut, op == ">="), (cut + 0.01, op == ">="), (cut + 1, op == ">="),
                                      (cut - 0.01, op == "<"), (cut - 1, op == "<")):
                    for flagged in (False, True):
                        with self.subTest(disease=disease, feature=feature, value=value, flagged=flagged):
                            result = trips(disease, feature, value, flagged)
                            normal = normal_track(disease, feature, value, flagged)
                            self.assertEqual(result["track"], "urgent" if urgent else normal)
                            self.assertEqual(result["urgent"], urgent)
                            self.assertEqual([r["feature"] for r in result["reasons"]], [feature] if urgent else [])
                            self.assertEqual(result["flagged"], flagged)
                            # Text from an uploaded CSV row counts the same as a number.
                            self.assertEqual(trips(disease, feature, f" {value!r} ", flagged)["track"], result["track"])

    def test_blank_and_text_values_never_trip(self):
        for disease, expected in VALUE_RULES.items():
            for feature, *_ in expected:
                for value in NOT_ENTERED:
                    with self.subTest(disease=disease, feature=feature, value=value):
                        self.assertEqual(trips(disease, feature, value)["track"], "not_flagged")
                        self.assertEqual(trips(disease, feature, value, flagged=True)["track"], "at_risk")
                self.assertEqual(triage(disease, {}, *CLEAR)["track"], "not_flagged")   # an absent key too
        for value in NOT_ENTERED:   # a blank age is never a child
            self.assertEqual(triage("diabetes", {"age": value}, *FLAGGED)["track"], "at_risk")
        self.assertEqual(triage("kidney", None)["track"], "no_score")

    def test_not_flagged_but_urgent_says_so_plainly(self):
        result = triage("diabetes", {"age": 52, "HbA1c_level": 7.1, "blood_glucose_level": 120}, 0.1, 0.5)
        self.assertEqual((result["track"], result["flagged"]), ("urgent", False))
        self.assertTrue(result["text"]["patient"]["summary"].startswith("The model did not flag this result, but"))
        self.assertTrue(result["text"]["doctor"]["summary"].startswith("The model did not flag this result, but"))
        reason = result["reasons"][0]
        self.assertEqual((reason["value_text"], reason["cut_off_text"]), ("7.1%", "at or above 6.5%"))
        self.assertEqual(reason["patient"], "Your HbA1c of 7.1% is at or above 6.5%, a level doctors use when "
                                            "diagnosing diabetes. A doctor needs to confirm this with tests.")
        self.assertEqual(reason["doctor"], "The patient's HbA1c of 7.1% is at or above 6.5%, a level used when "
                                           "diagnosing diabetes; a doctor needs to confirm it with tests.")
        flagged = triage("diabetes", {"HbA1c_level": 7.1}, *FLAGGED)
        self.assertTrue(flagged["text"]["patient"]["summary"].startswith("The model flagged this result, and"))
        # A score exactly at the threshold is flagged, as on the Screenings page.
        self.assertTrue(triage("heart", {}, 0.6, 0.6)["flagged"])
        self.assertFalse(triage("heart", {}, 0.5999, 0.6)["flagged"])
        with self.assertRaises(ValueError):   # a score is meaningless without the threshold the page used
            triage("heart", {}, 0.7)

    def test_children(self):
        for disease, feature in AGE.items():
            with self.subTest(disease=disease):
                flagged = triage(disease, {feature: 14}, *FLAGGED)
                self.assertEqual((flagged["track"], flagged["child"]), ("urgent", True))
                self.assertEqual([r["kind"] for r in flagged["reasons"]], [ADULTS_ONLY])
                self.assertIn("written for adults", flagged["reasons"][0]["patient"])
                self.assertIn("written for adults", flagged["text"]["patient"]["summary"])
                self.assertIn("under 18", flagged["text"]["doctor"]["summary"])
                # An unflagged child stays not flagged, and a child with no score has no score.
                unflagged = triage(disease, {feature: 14}, *CLEAR)
                self.assertEqual((unflagged["track"], unflagged["child"], unflagged["reasons"]),
                                 ("not_flagged", True, []))
                self.assertEqual(triage(disease, {feature: "14"})["track"], "no_score")
                self.assertEqual(triage(disease, {feature: 17.99}, *FLAGGED)["track"], "urgent")
                self.assertEqual(triage(disease, {feature: 18}, *FLAGGED)["track"], "at_risk")
        # A flagged child with a value in range lists the value first, then the age.
        both = triage("diabetes", {"age": 15, "HbA1c_level": 6.8}, *FLAGGED)
        self.assertEqual([r["feature"] for r in both["reasons"]], ["HbA1c_level", "age"])
        # An unflagged child with a value in range is urgent for the value only.
        value_only = triage("kidney", {"age": 15, "egfr": 40}, *CLEAR)
        self.assertEqual([r["feature"] for r in value_only["reasons"]], ["egfr"])

    def test_no_score_with_and_without_a_rule(self):
        tripped = triage("kidney", {"age": 66, "egfr": 48, "uacr": 40, "creatinine": None})
        self.assertEqual((tripped["track"], tripped["flagged"]), ("urgent", None))
        self.assertEqual([r["feature"] for r in tripped["reasons"]], ["egfr", "uacr"])
        self.assertTrue(tripped["text"]["patient"]["summary"].startswith("AHEAD could not give a score for this "
                                                                         "result, but some values you entered are"))
        self.assertTrue(tripped["text"]["doctor"]["summary"].startswith("There is no model score for this result, "
                                                                        "but entered values are"))
        quiet = triage("kidney", {"age": 66, "egfr": 90, "uacr": None})
        self.assertEqual((quiet["track"], quiet["flagged"], quiet["reasons"]), ("no_score", None, []))
        self.assertEqual(quiet["tone"], "muted")

    def test_heart_has_no_value_rules(self):
        extreme = {"BMI": 60, "SleepHours": 1, "AgeCategory": "Age 80 or older", "HadKidneyDisease": "Yes",
                   "PhysicalHealthDays": 30}
        self.assertEqual(triage("heart", extreme, *FLAGGED)["track"], "at_risk")
        self.assertEqual(triage("heart", extreme, *CLEAR)["track"], "not_flagged")
        self.assertEqual(triage("heart", extreme)["track"], "no_score")
        self.assertEqual(rule_table("heart"), [])

    def test_blood_pressure_is_one_reason(self):
        cases = [({"systolic_bp": 182, "diastolic_bp": 95}, "182/95 mmHg", "blood pressure", "systolic_bp"),
                 ({"systolic_bp": 150, "diastolic_bp": 122}, "150/122 mmHg", "blood pressure", "diastolic_bp"),
                 ({"systolic_bp": 185, "diastolic_bp": 125}, "185/125 mmHg", "blood pressure", "systolic_bp"),
                 ({"systolic_bp": 182, "diastolic_bp": None}, "182 mmHg", "systolic blood pressure", "systolic_bp"),
                 ({"systolic_bp": "", "diastolic_bp": "121"}, "121 mmHg", "diastolic blood pressure", "diastolic_bp"),
                 # an older saved kidney screening reads the same
                 ({"SystolicBP": 150, "DiastolicBP": 122}, "150/122 mmHg", "blood pressure", "diastolic_bp")]
        for values, shown, name, feature in cases:
            with self.subTest(values=values):
                result = triage("kidney", values, *CLEAR)
                self.assertEqual(len(result["reasons"]), 1)
                reason = result["reasons"][0]
                self.assertEqual((reason["value_text"], reason["feature"], reason["label"]),
                                 (shown, feature, "Blood pressure"))
                self.assertTrue(reason["patient"].startswith(f"Your {name} of {shown} is in the severely high range"))
                self.assertTrue(reason["doctor"].startswith(f"The patient's {name} of {shown} is in the severely "
                                                            "high range"))
                self.assertIn("emergency care", reason["patient"])
        self.assertEqual(triage("kidney", {"systolic_bp": 179, "diastolic_bp": 119}, *CLEAR)["track"], "not_flagged")

    def test_blood_pressure_is_listed_first(self):
        # The only reason with emergency-care advice leads, because the patient home banner and the doctor home show
        # just the first reason; the other value reasons keep table order and the age reason stays last.
        values = {"age": 15, "egfr": 45, "uacr": 40, "systolic_bp": 196, "diastolic_bp": 124}
        for score in (FLAGGED, CLEAR, ()):
            with self.subTest(score=score):
                result = triage("kidney", values, *score)
                kinds = ["blood_pressure", "kidney", "kidney"] + ([ADULTS_ONLY] if score == FLAGGED else [])
                self.assertEqual([reason["kind"] for reason in result["reasons"]], kinds)
                self.assertEqual([reason["feature"] for reason in result["reasons"]][1:3], ["egfr", "uacr"])
                first = result["reasons"][0]
                self.assertEqual(first["value_text"], "196/124 mmHg")
                self.assertIn("get emergency care now", first["patient"])
                self.assertIn("symptoms that need emergency care", first["doctor"])
        only_diastolic = triage("kidney", {"egfr": 45, "systolic_bp": 150, "diastolic_bp": 122}, *CLEAR)
        self.assertEqual([r["feature"] for r in only_diastolic["reasons"]], ["diastolic_bp", "egfr"])

    def test_values_are_formatted_like_the_forms(self):
        for disease, expected in VALUE_RULES.items():
            for feature, op, cut, shown in expected:
                with self.subTest(feature=feature):
                    reason = trips(disease, feature, SHOWN[shown])["reasons"][0]
                    self.assertEqual(reason["value_text"], shown)
                    self.assertEqual(reason["value"], SHOWN[shown])
                    self.assertEqual(reason["cut_off"], cut)
                    self.assertIn(shown, reason["patient"])
                    self.assertIn(shown, reason["doctor"])
        self.assertEqual(trips("kidney", "egfr", "52.30")["reasons"][0]["value_text"], "52.3")
        self.assertEqual(trips("kidney", "egfr", 41.256)["reasons"][0]["value_text"], "41.26")
        self.assertEqual(trips("diabetes", "blood_glucose_level", 212.0)["reasons"][0]["value_text"], "212 mg/dL")
        self.assertEqual(trips("diabetes", "blood_glucose_level", "250.50")["reasons"][0]["value_text"], "250.5 mg/dL")
        # ACR prints its unit, because labs also report it in mg/mmol (30 mg/g is about 3 mg/mmol).
        acr = trips("kidney", "uacr", 45)["reasons"][0]
        self.assertEqual(acr["cut_off_text"], "at or above 30 mg/g")
        self.assertTrue(acr["patient"].startswith("Your urine ACR of 45 mg/g is at or above 30 mg/g, "))
        self.assertTrue(acr["doctor"].startswith("The patient's urine ACR of 45 mg/g is at or above 30 mg/g, "))

    def test_a_value_just_below_a_cut_off_never_prints_as_the_cut_off(self):
        # Clinic uploads keep full precision: 59.996 rounds to '60' at two decimals, which would read 'eGFR of 60 is
        # below 60', so the value is printed in full instead.
        for value, shown in ((59.996, "59.996"), (59.9951, "59.9951"), ("59.999", "59.999"), (59.9999999, "59.9999999"),
                             (59.994, "59.99"), (59.99, "59.99")):
            with self.subTest(value=value):
                reason = trips("kidney", "egfr", value)["reasons"][0]
                self.assertEqual(reason["value_text"], shown)
                self.assertIn(f"Your eGFR of {shown} is below 60,", reason["patient"])
                self.assertIn(f"The patient's eGFR of {shown} is below 60,", reason["doctor"])
        for disease, feature in AGE.items():   # the age a flagged child entered, too
            reason = triage(disease, {feature: 17.996}, *FLAGGED)["reasons"][-1]
            self.assertEqual((reason["kind"], reason["value_text"]), (ADULTS_ONLY, "17.996 years"))
            self.assertTrue(reason["patient"].startswith("You entered an age of 17.996."), reason["patient"])
            self.assertTrue(reason["doctor"].startswith("The patient's entered age is 17.996;"), reason["doctor"])
        # Every 'below' rule, just under its cut-off; 'at or above' values never round down past theirs.
        for disease, rules in RULES.items():
            for rule in rules:
                with self.subTest(feature=rule.feature):
                    value = rule.cut_off - 0.001 if rule.below is not None else rule.cut_off + 0.001
                    reason = triage(disease, {rule.feature: value}, *FLAGGED)["reasons"]
                    shown = float(re.match(r"[\d.]+", reason[0]["value_text"])[0])
                    self.assertTrue(shown < rule.cut_off if rule.below is not None else shown >= rule.cut_off, shown)


class TriageWordingTests(unittest.TestCase):
    def test_labels_are_identical_on_every_surface(self):
        self.assertEqual(LABELS, EXPECTED_LABELS)
        self.assertEqual(set(TRACKS), set(LABELS))
        for result in all_results():
            for audience in AUDIENCES:
                label, short = EXPECTED_LABELS[result["track"]][audience]
                text = track_text(result, audience)
                self.assertEqual((text["label"], text["short"]), (label, short))
                self.assertEqual(track_text(result["track"], audience)["label"], label)
                pill = track_pill_html(result, audience)
                self.assertIn(f">{short}</span>", pill)
                self.assertEqual(pill, track_pill_html(result["track"], audience))
                self.assertIn(f">{label}</span>", track_pill_html(result, audience, short=False))
                panel = track_panel_html(result, audience)
                if result["track"] != "no_score":
                    self.assertIn(label, panel)
        tracks = {result["track"] for result in all_results()}
        self.assertEqual(tracks, set(TRACKS))
        with self.assertRaises(ValueError):
            track_text("urgent", "nurse")
        with self.assertRaises(ValueError):
            track_text("maybe")

    def test_wording_rules(self):
        for result in all_results():
            for reason in result["reasons"]:
                self.assertRegex(reason["patient"], PATIENT_CONFIRM)
                self.assertRegex(reason["doctor"], DOCTOR_CONFIRM)
                self.assertRegex(reason["patient"], r"\b(You|Your)\b")
                self.assertTrue(reason["doctor"].startswith("The patient's"), reason["doctor"])
            for audience in AUDIENCES:
                summary = result["text"][audience]["summary"]
                copy = [summary] + [reason[audience] for reason in result["reasons"]]
                for text in copy:
                    self.assertNotRegex(text, DIAGNOSIS_CLAIM)
                    if audience == "doctor":
                        self.assertNotRegex(text, r"(?i)\byou(r)?\b")
                if result["track"] in ("urgent", "at_risk"):
                    self.assertIn("not a diagnosis", summary)
            if result["track"] == "at_risk":
                self.assertIn("prevention plan", result["text"]["patient"]["summary"])
                self.assertIn("Share this result with your doctor", result["text"]["patient"]["summary"])
        for track in TRACKS:
            for audience in AUDIENCES:
                self.assertNotRegex(track_text(track, audience)["summary"], DIAGNOSIS_CLAIM)


class TrackComponentTests(unittest.TestCase):
    def test_panel_escapes_reasons_and_note(self):
        result = triage("kidney", {"egfr": 40, "uacr": 45}, *FLAGGED)
        result["reasons"][0] = {**result["reasons"][0], "patient": '<script>alert("x")</script> & <b>bold</b>'}
        with mock.patch("ahead.components.st.html") as html:
            self.assertTrue(track_panel(result, "patient", note="<img src=x onerror=alert(1)>"))
        body = html.call_args.args[0]
        self.assertNotIn("<script>", body)
        self.assertNotIn("<b>bold", body)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;", body)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", body)
        self.assertIn('<ul class="track-panel-reasons">', body)
        self.assertEqual(body.count("<li>"), 2)
        self.assertIn('class="track-panel urgent"', body)
        # The doctor panel shows the doctor wording.
        doctor = track_panel_html(triage("kidney", {"egfr": 40}, *FLAGGED), "doctor")
        self.assertIn("The patient&#x27;s eGFR of 40 is below 60", doctor)
        self.assertNotIn("Your eGFR", doctor)

    def test_panel_per_track(self):
        at_risk = track_panel_html(triage("heart", {}, *FLAGGED))
        self.assertIn('class="track-panel at-risk"', at_risk)
        self.assertNotIn("<ul", at_risk)
        clear = track_panel_html(triage("heart", {}, *CLEAR))
        self.assertIn('class="track-panel clear"', clear)
        self.assertNotIn("<ul", clear)
        self.assertEqual(clear.count("<p"), 1)   # one short line
        with mock.patch("ahead.components.st.html") as html:
            self.assertFalse(track_panel(triage("kidney", {"egfr": 90})))   # no score, rule or lab range: nothing
        html.assert_not_called()
        self.assertEqual(track_panel_html(triage("kidney", {"egfr": 90})), "")
        self.assertIn('class="track-panel urgent"', track_panel_html(triage("kidney", {"egfr": 40})))

    def test_pills(self):
        self.assertEqual(track_pill_html("urgent", "patient"),
                         '<span class="track-pill urgent" title="See a doctor soon">Urgent</span>')
        self.assertEqual(track_pill_html("at_risk", "doctor"),
                         '<span class="track-pill at-risk" title="At risk">At risk</span>')
        self.assertEqual(track_pill_html("not_flagged"),
                         '<span class="track-pill clear" title="Not flagged">Not flagged</span>')
        self.assertEqual(track_pill_html("no_score"), '<span class="track-pill none" title="No score">No score</span>')
        self.assertEqual(set(TRACK_CLASS), set(TRACKS))

    def test_css_uses_tokens_and_reads_in_both_modes(self):
        css = CSS.read_text(encoding="utf-8")
        rules = re.findall(r"^\.track-(?:panel|pill)[^{]*\{[^}]*\}", css, re.M)
        self.assertTrue(rules)
        for rule in rules:
            self.assertNotRegex(rule, r"#[0-9A-Fa-f]{3,8}\b|rgba?\(", rule)   # tokens only
        for name in ("urgent", "at-risk", "clear"):
            self.assertRegex(css, rf"(?m)^\.track-pill\.{name} \{{--pill:var\(--ahead-", name)
        pill = re.search(r"^\.track-pill \{([^}]*)\}", css, re.M).group(1)
        panel = re.search(r"^\.track-panel \{([^}]*)\}", css, re.M).group(1)
        title = re.search(r"^\.track-panel-title \{([^}]*)\}", css, re.M).group(1)
        pill_bg = int(re.search(r"background:color-mix\(in srgb,var\(--pill\) (\d+)%,var\(--ahead-card\)\)", pill)[1])
        pill_fg = int(re.search(r";color:color-mix\(in srgb,var\(--pill\) (\d+)%,var\(--ahead-title\)\)", pill)[1])
        panel_bg = int(re.search(r"background:color-mix\(in srgb,var\(--tone\) (\d+)%,var\(--ahead-card\)\)", panel)[1])
        title_fg = int(re.search(r"color:color-mix\(in srgb,var\(--tone\) (\d+)%,var\(--ahead-title\)\)", title)[1])
        for mode, tokens in (("light", LIGHT), ("dark", DARK)):
            card, title_colour, text = rgb(tokens["card"]), rgb(tokens["title"]), rgb(tokens["text"])
            for tone in ("danger", "warning", "success", "muted"):
                with self.subTest(mode=mode, tone=tone):
                    colour = rgb(tokens[tone])
                    pill_back, panel_back = mix(colour, pill_bg / 100, card), mix(colour, panel_bg / 100, card)
                    self.assertGreaterEqual(contrast(mix(colour, pill_fg / 100, title_colour), pill_back), 4.5)
                    self.assertGreaterEqual(contrast(mix(colour, title_fg / 100, title_colour), panel_back), 4.5)
                    self.assertGreaterEqual(contrast(text, panel_back), 4.5)


class SavedAndSharedTests(unittest.TestCase):
    def test_saved_rows(self):
        self.assertTrue(is_urgent_saved("diabetes", json.dumps({"age": 50, "HbA1c_level": 6.5}), 0.1, 0.5))
        self.assertTrue(is_urgent_saved("diabetes", {"HbA1c_level": "6.7"}, 0.1, 0.5))
        self.assertFalse(is_urgent_saved("diabetes", '{"HbA1c_level": null}', 0.9, 0.5))
        self.assertEqual(triage_saved("diabetes", '{"HbA1c_level": null}', 0.9, 0.5)["track"], "at_risk")
        for broken in ("", "{}", "not json", "[1, 2]", None, b'{"GFR": 40'):
            self.assertEqual(triage_saved("kidney", broken, 0.2, 0.5)["track"], "not_flagged", broken)
        # A record not scored yet (or half saved) still shows a value in range, under the current or the older key.
        for key in ("egfr", "GFR"):
            self.assertEqual(triage_saved("kidney", f'{{"{key}": "48"}}', None, None)["track"], "urgent")
            self.assertEqual(triage_saved("kidney", f'{{"{key}": "48"}}', 0.9, None)["flagged"], None)
            self.assertEqual(triage_saved("kidney", f'{{"{key}": 90}}', None, 0.5)["track"], "no_score")
        self.assertTrue(is_urgent_saved("diabetes", '{"age": 12}', 0.7, 0.5))
        self.assertFalse(is_urgent_saved("diabetes", '{"age": 12}', 0.3, 0.5))
        self.assertTrue(math.isclose(triage_saved("heart", "{}", "0.75", "0.6")["score"], 0.75))


class HeartHistoryTests(unittest.TestCase):
    """A past heart attack, angina or stroke (HISTORY_GATES) is urgent whatever the score; the heart model compares
    answers with people who reported a heart attack, so it is not meant for them."""

    def test_each_past_event_is_urgent_whatever_the_score(self):
        for feature, patient in (("HadHeartAttack", "a heart attack"), ("HadAngina", "angina"),
                                 ("HadStroke", "a stroke")):
            for answer in ("Yes", "yes", " YES ", 1, "1"):
                for score in (FLAGGED, CLEAR, (None, None)):
                    with self.subTest(feature=feature, answer=answer, score=score):
                        result = triage("heart", {"BMI": 24, feature: answer}, *score)
                        self.assertEqual((result["track"], result["tone"]), ("urgent", "danger"))
                        reason, = result["reasons"]
                        self.assertEqual((reason["kind"], reason["features"]), (HISTORY, [feature]))
                        self.assertEqual(reason["patient"],
                                         f"You told us you have had {patient}. AHEAD's heart screening is for people "
                                         "who have not; please see your doctor about your heart health.")
                        self.assertTrue(reason["doctor"].startswith("The patient reports "), reason["doctor"])
                        self.assertIn("established cardiovascular disease", reason["doctor"])
                        self.assertNotRegex(reason["doctor"], r"(?i)\byou(r)?\b")
                        self.assertIn("score does not apply", result["text"]["patient"]["summary"])
                        self.assertIn("score does not apply", result["text"]["doctor"]["summary"])
                        for text in (result["text"]["patient"]["summary"], result["text"]["doctor"]["summary"]):
                            self.assertIn("not a diagnosis", text)
                            self.assertNotRegex(text, DIAGNOSIS_CLAIM)

    def test_several_events_make_one_reason(self):
        result = triage("heart", {"HadHeartAttack": "Yes", "HadAngina": "No", "HadStroke": "Yes"}, *CLEAR)
        reason, = result["reasons"]
        self.assertEqual(reason["features"], ["HadHeartAttack", "HadStroke"])
        self.assertTrue(reason["patient"].startswith("You told us you have had a heart attack and a stroke."))
        self.assertTrue(reason["doctor"].startswith("The patient reports a previous myocardial infarction and a "
                                                    "previous stroke"))
        every = triage("heart", {"HadHeartAttack": "Yes", "HadAngina": "Yes", "HadStroke": "Yes"}, *CLEAR)
        self.assertIn("a heart attack, angina and a stroke.", every["reasons"][0]["patient"])

    def test_no_or_blank_answers_and_older_records_leave_it_to_the_model(self):
        no = {"HadHeartAttack": "No", "HadAngina": "No", "HadStroke": "No"}
        blank = {"HadHeartAttack": None, "HadAngina": "", "HadStroke": "No"}
        legacy = {"BMI": 31, "HadStroke": "No", "AgeCategory": "Age 65 to 69"}   # saved before the new questions
        for values in (no, blank, legacy, {"HadAngina": "maybe"}):
            with self.subTest(values=values):
                self.assertEqual(triage("heart", values, *FLAGGED)["track"], "at_risk")
                self.assertEqual(triage("heart", values, *CLEAR)["track"], "not_flagged")
                self.assertEqual(triage("heart", values)["track"], "no_score")
        self.assertTrue(is_urgent_saved("heart", json.dumps(legacy | {"HadStroke": "Yes"}), 0.02, 0.08))
        self.assertFalse(is_urgent_saved("heart", json.dumps(legacy), 0.02, 0.08))
        # Only heart has these questions.
        self.assertEqual(triage("diabetes", {"age": 50, "HadHeartAttack": "Yes"}, *CLEAR)["track"], "not_flagged")
        self.assertEqual(reported_history("heart", '{"HadAngina": "Yes"}')[0].feature, "HadAngina")

    def test_a_later_screening_that_answers_no_clears_it(self):
        older = {"disease": "heart", "values_json": json.dumps({"HadHeartAttack": "Yes", "HadAngina": "No",
                                                                "HadStroke": "No"}), "score": 0.2, "threshold": 0.08}
        newer = {"disease": "heart", "values_json": json.dumps({"HadHeartAttack": "No", "HadAngina": "No",
                                                                "HadStroke": "No"}), "score": 0.02, "threshold": 0.08}
        legacy = {"disease": "heart", "values_json": json.dumps({"HadStroke": "No"}), "score": 0.02, "threshold": 0.08}
        self.assertEqual(open_urgent([newer, older]), [])
        # A newer screening saved before the question existed did not re-check it, so the older result stands.
        (row, result), = open_urgent([legacy, older])
        self.assertIs(row, older)
        self.assertEqual(result["reasons"][0]["features"], ["HadHeartAttack"])

    def test_gate_questions_are_on_the_form_and_never_model_inputs(self):
        from ahead.clinical_data import model_frame
        from ahead.config import FEATURE_GROUPS, GATE_ONLY_FIELDS, OPTIONAL_INPUTS
        from ahead.resources import features
        self.assertEqual(FEATURE_GROUPS["heart"]["Medical History"][:3], ["HadHeartAttack", "HadAngina", "HadStroke"])
        self.assertEqual(set(OPTIONAL_INPUTS["heart"]), set(GATE_ONLY_FIELDS["heart"]))
        for field in GATE_ONLY_FIELDS["heart"]:
            self.assertIn(field, form_fields("heart"))
            self.assertNotIn(field, features("heart"))
        self.assertIn("HadStroke", features("heart"))
        frame = model_frame("heart", [{"HadHeartAttack": "Yes", "HadAngina": "Yes", "BMI": 30}], features("heart"))
        self.assertEqual(list(frame.columns), features("heart"))


if __name__ == "__main__":
    unittest.main()
