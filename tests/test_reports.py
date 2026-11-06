"""The screening PDF: wording per audience, value formatting, page fit and hostile text."""

import inspect
import unittest
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

from pypdf import PdfReader

from ahead.config import DISEASES, OPTIONAL_INPUTS, form_fields
from ahead.reports import OTHER_SCRIPT, patient_report
from ahead.resources import feature_summary
from ahead.triage import track_text, triage

METRICS = {"recall_disease": 0.82, "precision_disease": 0.31}
DOCTOR_EXTRAS = {"reviewer": "Dr. Sarah Hassan", "reviewed_at": datetime(2026, 11, 4, 13, 52, tzinfo=ZoneInfo("Asia/Dubai")),
                 "source": "clinic_export.csv, row 12", "factors": ["Higher BMI", "Smoking history"]}
HOSTILE = "<b>x</b> & <script>"
KIDNEY_FACTORS = ["Family history of kidney disease", "Family history of hypertension", "Reduced eGFR",
                  "Higher entered blood pressure", "Reported edema"]
CLINIC_NOTE = ("Seen in clinic with fatigue and nocturia over three months. Creatinine trending up compared with the "
               "spring panel; eGFR now in the 50s. Discussed hydration, NSAID avoidance and blood pressure control. "
               "Repeat renal panel and urine ACR in four to six weeks.")
URGENT_DIABETES = {"HbA1c_level": 7.1, "blood_glucose_level": 212}
# An older saved kidney screening (config.LEGACY_FIELDS): the 22 inputs of the retired kidney form, blood pressure left
# blank. It still prints, and its eGFR and ACR are read by today's rules; its HbA1c and fasting blood sugar are below
# the cut-offs that still apply to these older screenings (triage.LEGACY_RULES).
LEGACY_KIDNEY = {"Age": 54, "Gender": 1, "BMI": 27.65, "Smoking": 0, "AlcoholConsumption": 9.95,
                 "PhysicalActivity": 5.07, "DietQuality": 4.99, "SleepQuality": 6.93, "FamilyHistoryKidneyDisease": 0,
                 "FamilyHistoryHypertension": 0, "FamilyHistoryDiabetes": 0, "SystolicBP": None, "DiastolicBP": 89,
                 "FastingBloodSugar": 104.5, "HbA1c": 5.9, "SerumCreatinine": 2.73, "BUNLevels": 28.08, "GFR": 52.3,
                 "ProteinInUrine": 2.51, "ACR": 150.2, "HemoglobinLevels": 13.91, "Edema": 0}
LEGACY_OPTIONAL = ["SystolicBP", "DiastolicBP", "FastingBloodSugar", "HbA1c", "BUNLevels", "ProteinInUrine",
                   "HemoglobinLevels"]   # the 7 values the retired kidney form let a patient leave blank
# Every current kidney rule except age tripped: eGFR, ACR and blood pressure (one reason).
URGENT_KIDNEY = {"GFR": 52.3, "ACR": 41, "SystolicBP": 182, "DiastolicBP": 95}
KIDNEY_CAVEAT = ("Specificity with the same inputs left blank: 9.8%. It flagged almost everyone without chronic kidney "
                 "disease too, so a flag here tells you very little. Only 64 people in that test data did not have "
                 "chronic kidney disease, so the figures for people without it are a rough guide.")


def typical_values(disease):
    """Every form field at its training median or mode, with one input left blank (kidney: the kidney tests too,
    as most kidney screenings leave them blank, so the model sorts the result)."""
    values, stats = {}, feature_summary(disease)
    for feature in form_fields(disease):
        info = stats[feature]
        if info["kind"] == "category":
            values[feature] = info["default"]
        elif info["binary"]:
            values[feature] = info["mode"]
        else:
            values[feature] = round(info["median"]) if info["integer"] else round(info["median"], 2)
    values[(OPTIONAL_INPUTS.get(disease) or ["SleepHours"])[0]] = None
    if disease == "kidney":
        values.update(dict.fromkeys(["creatinine", "egfr", "uacr"]))
    return values


def pages(pdf):
    return [" ".join(page.extract_text().split()) for page in PdfReader(BytesIO(pdf)).pages]


def flat(text):
    return " ".join(text.split())


class ReportTests(unittest.TestCase):
    def report(self, disease="diabetes", audience="doctor", score=0.72, threshold=0.5, **extra):
        kwargs = {"status": "Reviewed", "note": "Discussed with the patient."} if audience == "doctor" else {}
        kwargs |= DOCTOR_EXTRAS if audience == "doctor" else {"name": "Alice Mansour"}
        return pages(patient_report(disease, "P-00042", extra.pop("values", typical_values(disease)), score, threshold,
                                    "GradientBoosting", METRICS, **(kwargs | {"audience": audience} | extra)))

    def test_each_condition_and_audience_fits_one_page(self):
        # With the Next step band both pages now print (typical kidney values are urgent: ACR, HbA1c and fasting
        # blood sugar are in a diagnostic range).
        for disease in DISEASES:
            for audience, score in (("doctor", 0.72), ("patient", 0.08)):
                with self.subTest(disease=disease, audience=audience):
                    result = triage(disease, typical_values(disease), score, 0.5)
                    text = self.report(disease, audience, score, track=result)
                    self.assertEqual(len(text), 1)
                    page = text[0]
                    self.assertIn(f"NEXT STEP {track_text(result, audience)['label']}", page)
                    title = "Clinical screening summary" if audience == "doctor" else "Your screening summary"
                    self.assertIn(title, page)
                    self.assertIn(DISEASES[disease]["label"].upper(), page)
                    # The panel's headline is the track; the line under it keeps the model's own flag.
                    relation = "at or above" if score >= 0.5 else "below"
                    self.assertIn(f"RESULT · {DISEASES[disease]['label'].upper()} {track_text(result, audience)['label']} "
                                  f"The model's score is {relation} the classification threshold.", page)
                    self.assertIn("Not provided", page)
                    self.assertIn("Gradient boosting model", page)
                    self.assertIn("Educational screening estimate - not a diagnosis. Generated by AHEAD.", page)
                    self.assertIn("Page 1 of 1", page)
                    if audience == "doctor":
                        self.assertNotIn("Not flagged", page)
                        self.assertIn("Reviewed on 4 Nov 2026, 13:52", page)
                        self.assertIn("Source clinic_export.csv, row 12", page)   # short IDs copy out intact
                        self.assertIn("Points to review", page)
                    else:
                        for absent in ("Review status", "Doctor review", "Reviewing doctor", "Pending review", "Doctor note"):
                            self.assertNotIn(absent, page)
                        self.assertIn("Prepared for Alice Mansour", page)
                        self.assertIn("Questions to ask your doctor", page)

    def test_all_kidney_inputs_fit_one_page(self):
        # The current 14-field form, and an older saved screening with the retired form's 22 inputs.
        self.assertEqual(len(typical_values("kidney")), 14)
        self.assertEqual(len(LEGACY_KIDNEY), 22)
        for values in (typical_values("kidney"), LEGACY_KIDNEY):
            for audience in ("doctor", "patient"):
                for track in (None, triage("kidney", values, 0.72, 0.5)):
                    with self.subTest(inputs=len(values), audience=audience, band=bool(track)):
                        self.assertEqual(len(self.report("kidney", audience, values=values, track=track)), 1)

    def test_all_kidney_inputs_with_typical_extras_fit_one_page(self):
        # The Screenings page copy with the optional labs left blank, the real caveat and five factors, and a doctor
        # copy with a three-line note: both used to spill one or two lines onto a near-empty second page. These
        # pages are full, so with a Next step band (at least 33pt) they take two pages: see
        # test_a_band_too_long_for_one_page_moves_sections_rather_than_clipping.
        blanked = LEGACY_KIDNEY | dict.fromkeys(LEGACY_OPTIONAL)
        patient = self.report("kidney", "patient", 0.38, 0.30, values=blanked, usual_threshold=0.5,
                              metric_scope=" with the same inputs left blank", metric_caveat=KIDNEY_CAVEAT,
                              factors=KIDNEY_FACTORS)
        self.assertEqual(len(patient), 1)
        self.assertIn("matter most, and should I have the blank values measured?", patient[0])   # five questions at most
        doctor = self.report("kidney", values=LEGACY_KIDNEY, note=CLINIC_NOTE, factors=KIDNEY_FACTORS)
        self.assertEqual(len(doctor), 1)
        self.assertIn("four to six weeks.", doctor[0])

    def test_values_read_like_the_form(self):
        page = self.report("kidney", values={"Age": 61.0, "Gender": 0, "BMI": 27.456, "Smoking": 1, "Edema": 0,
                                             "SerumCreatinine": 1.234, "BUNLevels": float("nan"), "HbA1c": "N/A",
                                             "ProteinInUrine": 0.0049, "AlcoholConsumption": 10.04, "GFR": -0.0})[0]
        for shown in ("Age 61", "Gender Male", "Body mass index (BMI) 27.5", "Smoking Yes", "Presence of edema No",
                      "Serum creatinine 1.23", "Blood urea nitrogen (BUN) Not provided", "HbA1c (%) Not provided",
                      "Protein in urine 0.0049", "Alcohol consumption 10 ", "Estimated GFR 0 "):
            self.assertIn(shown, page)
        page = self.report(values={"gender": "Female", "smoking_history": "No Info", "hypertension": 1,
                                   "HbA1c_level": 6.5, "bmi": 30.0})[0]
        self.assertIn("Smoking history Not recorded", page)   # the dataset's own non-answer, not a bold value
        self.assertNotIn("Not provided", page)   # nor a blank: no blank pill or footnote
        self.assertIn("History of hypertension Yes", page)
        self.assertIn("HbA1c level (%) 6.5 ", page)   # as entered, like the Clinical Dashboard: no padding zeros
        self.assertIn("Body mass index (BMI) 30 ", page)
        # Free text that reaches the table is capped instead of overflowing the page.
        self.assertEqual(len(self.report(values={"smoking_history": "word " * 300, "gender": "x" * 3000})), 1)

    def test_values_an_urgent_rule_reads_keep_the_band_digits(self):
        # At one decimal, eGFR 59.96 printed as 60 under a band saying it is below 60, and urine ACR 29.996 as 30.
        values = typical_values("kidney") | {"systolic_bp": 130, "uacr": 12, "egfr": 59.96}
        result, text = self.tracked("kidney", "patient", values=values)
        self.assertEqual([reason["feature"] for reason in result["reasons"]], ["egfr"])
        self.assertIn("Your eGFR of 59.96 is below 60", text[0])
        self.assertIn("eGFR 59.96 ", text[0])
        page = self.report("kidney", values=values | {"egfr": 59.996, "uacr": 29.996, "bmi": 27.456})[0]
        for shown in ("eGFR 59.996 ", "Urine ACR (mg/g) 29.996 ", "Body mass index (BMI) 27.5 "):
            self.assertIn(shown, page)
        # An older saved screening's eGFR keeps its digits too, under its own label.
        values = LEGACY_KIDNEY | {"SystolicBP": 130, "HbA1c": 6.2, "ACR": 12, "GFR": 59.96, "FastingBloodSugar": 125.96}
        result, text = self.tracked("kidney", "patient", values=values)
        self.assertEqual([reason["feature"] for reason in result["reasons"]], ["egfr"])
        self.assertIn("Your eGFR of 59.96 is below 60", text[0])
        for shown in ("Estimated GFR 59.96 ", "HbA1c (%) 6.2 "):
            self.assertIn(shown, text[0])
        # More digits only where two would still land on the other side of the cut-off; other values as before.
        page = self.report("kidney", values=values | {"GFR": 59.996, "BMI": 27.456})[0]
        for shown in ("Estimated GFR 59.996 ", "Body mass index (BMI) 27.5 "):
            self.assertIn(shown, page)

    def test_caveat_and_lowered_threshold_are_reported(self):
        page = self.report(threshold=0.41, usual_threshold=0.5, metric_scope=" with the same inputs left blank",
                           metric_caveat="Specificity with the same inputs left blank: 88.1%.")[0]
        self.assertIn("lowered from 50.0% because some inputs were left blank", page)
        self.assertIn("Specificity with the same inputs left blank: 88.1%.", page)
        self.assertIn("Measured on the held-out test set, with the same inputs left blank.", page)
        self.assertNotIn("lowered from", self.report(threshold=0.5, usual_threshold=0.5)[0])

    def test_score_that_rounds_to_the_threshold_keeps_its_side(self):
        page = self.report(score=0.4996, threshold=0.5)[0]
        self.assertIn("Not flagged", page)
        self.assertIn("49.96%", page)
        self.assertIn("Threshold 50.00%", page)
        self.assertNotIn("50.0%", page.replace("50.00%", ""))
        page = self.report(score=0.0, threshold=0.00001)[0]
        self.assertIn("MODEL SCORE 0.00%", page)
        self.assertNotIn("-0.01%", page)

    def test_two_page_report_keeps_the_reading_notes_with_the_result(self):
        note = ("Seen in clinic with fatigue and nocturia over three months. " * 50)[:2500]
        text = self.report("kidney", values=typical_values("kidney"), note=note)
        self.assertEqual(len(text), 2)
        self.assertIn("How to read this result", text[0])
        self.assertTrue(text[1].split("(continued)")[1].strip())   # page 2 carries content, not a stray line
        # Nine factors with every extra: only the tighter section gaps keep the notes on the result's page.
        text = self.report("kidney", values=typical_values("kidney"), threshold=0.41, usual_threshold=0.5,
                           metric_caveat="Specificity with the same inputs left blank: 41.2%.",
                           factors=KIDNEY_FACTORS + ["Smoking", "Higher BMI", "Family history of diabetes", "Higher ACR"])
        self.assertEqual(len(text), 2)
        self.assertIn("Not a rule-out.", text[0])

    def test_audience_is_checked(self):
        page = self.report(audience="Patient", note="Internal note.")[0]   # the role constant's spelling
        self.assertIn("Your screening summary", page)
        self.assertNotIn("Internal note.", page)
        for wrong in ("clinician", None):
            with self.assertRaises(ValueError):
                self.report(audience=wrong)

    def test_session_screening_can_leave_out_the_review(self):
        page = pages(patient_report("heart", "Session screening", typical_values("heart"), 0.7, 0.5,
                                    "LogisticRegression", METRICS, None))[0]
        self.assertNotIn("Review status", page)
        self.assertNotIn("Pending review", page)
        self.assertIn("Screening type Session screening RESULT", page)   # not presented as a patient ID
        self.assertNotIn("Patient ID", page)

    def test_other_scripts_never_print_as_boxes(self):
        arabic = "أحمد المنصوري"
        pdf = patient_report("diabetes", "Session screening", typical_values("diabetes"), 0.3, 0.5, "XGBoost", METRICS,
                             audience="patient", name=arabic + " 😀")
        page = pages(pdf)[0]
        self.assertNotIn("Prepared for", page)   # dropped rather than printed half-marked
        self.assertNotIn(b"ZapfDingbats", pdf)   # the font ReportLab draws missing glyphs with
        page = self.report(note=f"Prefers {arabic}. BP ≥ 140 😀", reviewer="د. سارة", source="ملف.csv, row 3",
                           factors="Higher BMI")[0]
        self.assertIn(f"Prefers {OTHER_SCRIPT}. BP ≥ 140", page)
        self.assertIn(f"Reviewing doctor {OTHER_SCRIPT} ", page)
        self.assertIn(f"Source {OTHER_SCRIPT}.csv, row 3", page)
        self.assertIn("Risk factors noted Higher BMI How to read", page)   # one string is one factor
        self.assertIn("Prepared for Zoë Lukasiewicz-Sahin", self.report(audience="patient", name="Zoë Łukasiewicz-Şahin")[0])

    def test_hostile_text_is_printed_literally_and_long_notes_continue(self):
        note = (HOSTILE + " " + "Long clinical note text that keeps going. " * 70)[:2500]
        text = pages(patient_report("diabetes", HOSTILE, typical_values("diabetes"), 0.6, 0.5, "XGBoost", METRICS,
                                    "Reviewed", note, reviewer=HOSTILE, source=HOSTILE, factors=[HOSTILE]))
        self.assertEqual(len(text), 2)
        self.assertIn(HOSTILE, text[0])
        self.assertIn("keeps going", " ".join(text))
        for number, page in enumerate(text, 1):
            self.assertIn(f"Page {number} of {len(text)}", page)
        patient = self.report(audience="patient", name=HOSTILE, note=note)
        self.assertIn(f"Prepared for {HOSTILE}", patient[0])
        self.assertNotIn("keeps going", " ".join(patient))   # a doctor note never reaches the patient's copy
        unbroken = pages(patient_report("diabetes", "P" * 300, typical_values("diabetes"), 0.6, 0.5, "XGBoost", METRICS,
                                        note="x" * 2500))
        self.assertTrue(unbroken)

    def test_calls_without_the_new_extras_still_work(self):
        values = typical_values("diabetes")
        # The Clinical Dashboard call: positional status and note, no audience.
        page = pages(patient_report("diabetes", "P-7", values, 0.36, 0.5, "RandomForest", METRICS, "Excluded", "",
                                    metric_source="validation"))[0]
        for shown in ("Clinical screening summary", "Patient ID P-7", "Excluded", "Not flagged", "Random forest model",
                      "Measured on the validation set."):
            self.assertIn(shown, page)
        for absent in ("Reviewing doctor", "Source", "Doctor note", "NEXT STEP"):
            self.assertNotIn(absent, page)
        track = inspect.signature(patient_report).parameters["track"]   # added last, keyword-only, off by default
        self.assertEqual((track.kind, track.default), (track.KEYWORD_ONLY, None))
        # A model without stored figures gets no performance section rather than a misleading 0.0%.
        page = pages(patient_report("diabetes", "P-7", values, 0.36, 0.5, "RandomForest", {}, "Reviewed", "Seen."))[0]
        self.assertNotIn("Model performance", page)
        self.assertIn("Doctor note Seen.", page)
        # The Screenings page call.
        page = pages(patient_report("heart", "Session screening", typical_values("heart"), 0.7, 0.6, "LogisticRegression",
                                    METRICS, metric_source="test", metric_scope="", usual_threshold=0.6,
                                    metric_caveat=""))[0]
        self.assertIn("Pending review", page)
        self.assertIn("Flagged", page)

    # ------------------------------------------------------------------ the Next step band (ahead/triage.py)
    def tracked(self, disease="diabetes", audience="patient", score=0.72, values=None, **extra):
        """A report with the track triage() gives this result, and that track."""
        values = typical_values(disease) if values is None else values
        result = triage(disease, values, score, 0.5)
        return result, self.report(disease, audience, score, 0.5, values=values, track=result, **extra)

    def test_band_shows_the_track_in_each_audience_words(self):
        labels = {("urgent", "patient"): "See a doctor soon", ("urgent", "doctor"): "Urgent",
                  ("at_risk", "patient"): "At risk", ("at_risk", "doctor"): "At risk",
                  ("not_flagged", "patient"): "Not flagged", ("not_flagged", "doctor"): "Not flagged"}
        cases = {"urgent": (typical_values("diabetes") | URGENT_DIABETES, 0.72),
                 "at_risk": (typical_values("diabetes"), 0.72), "not_flagged": (typical_values("diabetes"), 0.08)}
        for (track, audience), label in labels.items():
            with self.subTest(track=track, audience=audience):
                values, score = cases[track]
                result, text = self.tracked("diabetes", audience, score, values)
                self.assertEqual(result["track"], track)
                self.assertEqual(len(text), 1)
                page, words = text[0], track_text(result, audience)
                other = "doctor" if audience == "patient" else "patient"
                self.assertEqual(words["label"], label)   # the same words as the pages' pills and panels
                # Right under the result panel, before the first section.
                self.assertLess(page.index("MODEL SCORE"), page.index("NEXT STEP"))
                self.assertLess(page.index("NEXT STEP"), page.index("Entered values"))
                if audience == "doctor":
                    self.assertNotIn("See a doctor soon", page)
                if track != "urgent":   # one strip: the label and summary as one paragraph, no reasons
                    self.assertIn(f"NEXT STEP {label}. {flat(words['summary'])} Entered values", page)
                    continue
                self.assertIn(f"NEXT STEP {label} {flat(words['summary'])}", page)
                self.assertEqual(len(result["reasons"]), 2)
                for reason in result["reasons"]:
                    self.assertIn(flat(reason[audience]), page)
                    self.assertNotIn(flat(reason[other]), page)
                self.assertIn("confirm this with tests" if audience == "patient" else "confirm it with tests",
                              page.split("NEXT STEP")[1].split("Entered values")[0])

    def test_band_says_when_the_model_did_not_flag_an_urgent_result(self):
        result, text = self.tracked("diabetes", "patient", 0.2, typical_values("diabetes") | {"HbA1c_level": 6.8})
        page = text[0]
        self.assertEqual(result["track"], "urgent")
        # The panel (headed by the track, with the model's own flag under it), then the band.
        self.assertLess(page.index("See a doctor soon The model's score is below the classification threshold."),
                        page.index("NEXT STEP See a doctor soon"))
        self.assertIn("NEXT STEP See a doctor soon The model did not flag this result, but", page)
        self.assertIn(flat(result["reasons"][0]["patient"]), page)
        # A flagged child: urgent because the rules are written for adults.
        result, text = self.tracked("diabetes", "doctor", 0.7, typical_values("diabetes") | {"age": 15})
        self.assertEqual([reason["kind"] for reason in result["reasons"]], ["adults_only"])
        self.assertIn(flat(result["reasons"][0]["doctor"]), text[0])

    def test_band_text_is_printed_literally(self):
        values = typical_values("diabetes") | URGENT_DIABETES
        result = triage("diabetes", values, 0.72, 0.5)
        for audience in ("patient", "doctor"):
            result["reasons"][0] |= {audience: f"{HOSTILE} reason"}
            result["text"][audience] |= {"summary": f"{HOSTILE} summary"}
        for audience in ("patient", "doctor"):
            with self.subTest(audience=audience):
                page = self.report("diabetes", audience, values=values, track=result)[0]
                self.assertIn(f"{HOSTILE} summary", page)
                self.assertIn(f"{HOSTILE} reason", page)

    def test_track_must_be_sorted_for_this_result(self):
        values = typical_values("diabetes")
        wrong = {"not flagged under a flagged panel": triage("diabetes", values, 0.2, 0.5),
                 "another condition": triage("heart", typical_values("heart"), 0.72, 0.5),
                 "no score": triage("diabetes", values | URGENT_DIABETES),
                 "not a triage result": "urgent", "unknown track": {"track": "soon", "disease": "diabetes"}}
        for case, track in wrong.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.report("diabetes", values=values, score=0.72, threshold=0.5, track=track)

    def test_every_band_fits_one_page_with_short_input_lists(self):
        for disease in ("diabetes", "heart"):
            for audience in ("patient", "doctor"):
                for score, extra in ((0.72, {}), (0.08, {}), (0.72, URGENT_DIABETES if disease == "diabetes" else {})):
                    with self.subTest(disease=disease, audience=audience, score=score, urgent=bool(extra)):
                        result, text = self.tracked(disease, audience, score, typical_values(disease) | extra,
                                                    factors=KIDNEY_FACTORS[:3])
                        self.assertEqual(len(text), 1)
                        self.assertIn("NEXT STEP", text[0])
                        for reason in result["reasons"]:
                            self.assertIn(flat(reason[audience]), text[0])

    def test_band_shrinks_before_a_kidney_copy_takes_a_second_page(self):
        # An older saved kidney screening (all 22 inputs) with two urgent reasons, eGFR and ACR: the band keeps the most
        # words that still fit one page. Each reason in words, then one short line per value reason (label, value,
        # cut-off), then the values on one line. The summary always says a doctor confirms them with tests; blood
        # pressure keeps its words in every form.
        kidney = LEGACY_KIDNEY
        pending = {"status": "", "note": "", "reviewer": "", "reviewed_at": None,
                   "factors": ["Higher entered blood pressure", "Higher entered urine ACR"]}
        cases = {   # the Screenings copy (patient) and the Clinical Dashboard copy (doctor): (values, extras, form)
            ("patient", "every value entered"): (kidney | {"SystolicBP": 130}, {}, "words"),
            ("patient", "one value blank"): (kidney, {}, "words"),
            ("patient", "three factors"): (kidney, {"factors": KIDNEY_FACTORS[:3]}, "lines"),
            ("patient", "blood pressure too"): (kidney | URGENT_KIDNEY, {}, "words"),
            ("doctor", "pending record"): (kidney, pending, "words"),
            ("doctor", "pending, blood pressure too"): (kidney | {"SystolicBP": 185}, pending, "words"),
            ("doctor", "reviewed with a note"): (kidney, {}, "chips"),
        }
        for (audience, case), (values, extra, form) in cases.items():
            with self.subTest(audience=audience, case=case):
                result, text = self.tracked("kidney", audience, values=values, **extra)
                self.assertEqual(len(text), 1)
                words, band = track_text(result, audience), text[0].split("NEXT STEP")[1].split("Entered values")[0]
                self.assertIn("confirm it with tests", words["summary"])
                short = [reason for reason in result["reasons"] if reason["kind"] != "blood_pressure"]
                self.assertEqual([reason["feature"] for reason in short], ["egfr", "uacr"])
                for reason in result["reasons"]:   # words where the form keeps them
                    self.assertEqual(flat(reason[audience]) in band, form == "words" or reason not in short)
                if form == "words":
                    self.assertIn(f"{words['label']} {flat(words['summary'])}", band)
                    continue
                self.assertIn(f"{words['label']}. {flat(words['summary'])}", band)   # one strip, as for at risk
                if form == "lines":
                    for reason in short:
                        self.assertIn(f"{reason['label']} {reason['value_text']}, {reason['cut_off_text']}", band)
                else:
                    self.assertIn(" · ".join(f"{reason['label']} {reason['value_text']}" for reason in short), band)
                    self.assertNotIn("at or above", band)
                if "blood pressure" in case:
                    emergency = "get emergency care now" if audience == "patient" else "symptoms that need emergency care"
                    self.assertIn(emergency, band)

    def test_at_risk_strip_leaves_room_for_a_review_note(self):
        # A heart record with a three-line note used to take two pages once its at-risk band was added. Since the heart
        # form asks about a past heart attack and angina (17 inputs), five long factors on top of that continue on a
        # second page; the band stays on the first and the note stays whole.
        for factors, most in (((), 1), (KIDNEY_FACTORS, 2)):
            with self.subTest(factors=len(factors)):
                result, text = self.tracked("heart", "doctor", values=typical_values("heart"), note=CLINIC_NOTE,
                                            factors=factors)
                self.assertEqual(result["track"], "at_risk")
                self.assertLessEqual(len(text), most)
                self.assertIn(f"NEXT STEP At risk. {flat(track_text(result, 'doctor')['summary'])}", text[0])
                self.assertTrue(any("Seen in clinic" in page and "four to six weeks." in page for page in text))

    def test_urgent_patient_copy_asks_about_a_visit_soon(self):
        # Under "See a doctor soon" the questions are for that visit: no "next appointment", "how often" or check-ups.
        routine = ("next appointment", "How often should I be checked", "Keep up your regular check-ups")
        for values, score, first in (
                (URGENT_DIABETES, 0.06, "Some of my values are in a range doctors use when diagnosing a condition. "
                                        "Which tests will confirm it?"),
                ({"HbA1c_level": 6.5}, 0.72, "One of my values is in a range doctors use when diagnosing a "
                                             "condition. Which tests will confirm it?"),
                ({"age": 15}, 0.72, "AHEAD's screening rules are written for adults. Which tests would confirm this "
                                    "result for me?")):
            with self.subTest(values=values, score=score):
                result, text = self.tracked("diabetes", "patient", score, typical_values("diabetes") | values)
                self.assertEqual(result["track"], "urgent")
                questions = text[0].split("Questions to ask your doctor")[1]
                self.assertIn("Please bring it to a doctor soon and ask: " + first, questions)
                self.assertNotIn("My result was flagged", questions)
                for absent in routine:
                    self.assertNotIn(absent, text[0])
        # Any other track, or no band, keeps the routine questions.
        for track in (triage("diabetes", typical_values("diabetes"), 0.72, 0.5), None):
            with self.subTest(track=track and track["track"]):
                page = self.report("diabetes", "patient", 0.72, track=track)[0]
                for shown in routine + ("My result was flagged. What should we look at first?",):
                    self.assertIn(shown, page)
        # The doctor copy's points to review are unchanged.
        self.assertIn("Confirm findings with appropriate testing",
                      self.tracked("diabetes", "doctor", values=typical_values("diabetes") | URGENT_DIABETES)[1][0])

    def test_heart_copy_is_not_a_future_risk_and_a_past_event_asks_about_heart_care(self):
        values = typical_values("heart") | {"HadHeartAttack": "Yes", "HadAngina": "No"}
        result, text = self.tracked("heart", "patient", 0.03, values)
        self.assertEqual((result["track"], len(text)), ("urgent", 1))
        page = text[0]
        self.assertIn("Not a 10-year risk.", page)
        self.assertNotIn("Not a probability.", page)
        self.assertIn("You told us you have had a heart attack.", page)
        self.assertIn("Please bring it to a doctor soon and ask: I have had a heart attack, angina or a stroke. Which "
                      "heart check-ups and care do I need?", page)
        self.assertNotIn("Some of my values are", page)
        doctor = self.tracked("heart", "doctor", 0.03, values)[1][0]
        self.assertIn("Not a 10-year risk.", doctor)
        self.assertIn("The patient reports a previous myocardial infarction", doctor)
        # Other conditions keep their reading notes; an upload without the gate answers says they were not checked.
        self.assertIn("Not a probability.", self.report("diabetes", "patient")[0])
        blank = pages(patient_report("heart", "P-1", typical_values("heart") | {"HadAngina": None}, 0.03, 0.08,
                                     "LogisticRegression", METRICS))[0]
        self.assertIn("These history questions are not model inputs", blank)
        self.assertNotIn("filled these inputs", blank)

    def test_a_band_too_long_for_one_page_moves_sections_rather_than_clipping(self):
        # Urgent reasons, a long note and five factors cannot share one page with 22 inputs: the page is full without
        # a band, and even the smallest band is taller than the room left. The band stays whole, in words, under the
        # result with the inputs, and every later section moves to page 2 once and in full.
        for audience in ("doctor", "patient"):
            for values, count in ((LEGACY_KIDNEY | URGENT_KIDNEY, 3), (LEGACY_KIDNEY, 2)):
                with self.subTest(audience=audience, reasons=count):
                    result, text = self.tracked("kidney", audience, values=values, note=CLINIC_NOTE,
                                                factors=KIDNEY_FACTORS)
                    self.assertEqual(len(result["reasons"]), count)
                    self.assertEqual(len(text), 2)
                    first, both = text[0], " ".join(text)
                    band = first.split("NEXT STEP")[1].split("Entered values")[0]
                    for reason in result["reasons"]:
                        self.assertIn(flat(reason[audience]), band)
                    self.assertIn("Presence of edema", first)   # the last input
                    headings = ["Entered values", "How to read this result", "Model performance"] + (
                        ["Risk factors noted", "Doctor note", "Points to review"] if audience == "doctor" else
                        ["Factors from your answers", "Questions to ask your doctor"])
                    for heading in headings:
                        self.assertEqual(both.count(heading), 1, heading)
                    self.assertEqual("four to six weeks." in both, audience == "doctor")
                    self.assertGreater(len(text[1].split("(continued)")[1].split()), 60)   # not a stray last page
                    self.assertIn("Page 2 of 2", text[1])

    def test_missing_score_is_refused(self):
        with self.assertRaises(ValueError):
            patient_report("diabetes", "P-1", {}, None, 0.5, "XGBoost", METRICS)


if __name__ == "__main__":
    unittest.main()
