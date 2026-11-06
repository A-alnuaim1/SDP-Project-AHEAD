"""
ahead/triage.py
===============
The SORT step: every screening result gets one track, and every surface (patient and doctor pages, tables,
reports) shows it in the same words.

  urgent       "See a doctor soon" (doctors read "Urgent"): an entered value is already in a range doctors use when
               diagnosing a condition, whatever the model's score. A flagged result for someone under 18 is urgent
               too, because the screening rules are written for adults. On the heart screening, a Yes to a past heart
               attack, angina or stroke (HISTORY_GATES) is urgent whatever the score: the model compares answers with
               people who reported a heart attack, so it is not meant for people who have had one.
  at_risk      "At risk": the model flagged the result (score >= the threshold the page used), or a measured lab value
               is in an at-risk range (an HbA1c of 5.7-6.4%, prediabetes), and no urgent rule tripped. A prevention
               plan agreed with a doctor can help.
  not_flagged  "Not flagged": below the threshold and no urgent rule, or measured labs in the usual range (an HbA1c
               under 5.7%; an eGFR of 60 or more with a urine ACR under 30 mg/g), whatever the model's score.
  no_score     there is no model score (an unscored record) and neither a rule nor a lab value set the track.

A measured lab value counts for more than the model's estimate: the diabetes and kidney models read only answers
(config.MODEL_INPUTS), and were trained to estimate exactly these lab results (an HbA1c of 5.7% or more; an eGFR below
60 or an ACR of 30 mg/g or more), so a lab entered on the form sets the track (LAB_RANGES) and the model decides only
when it is missing. A kidney screening with creatinine but no eGFR gets eGFR from creatinine (egfr_from_creatinine,
CKD-EPI 2021). Older saved kidney screenings keep their earlier keys (config.LEGACY_FIELDS); prepared_values() reads
them under the current names, so they sort the same way.

The cut-offs in RULES are prototype values for a clinician to confirm, kept in one table so they can be read and
changed in one place.

Pure Python (no Streamlit), so pages, reports and scripts share it. Values may arrive as numbers or as text (uploaded
CSV rows); a blank, non-numeric or non-finite value counts as not entered and never trips a rule. Nothing here is a
diagnosis: every urgent reason says a doctor needs to confirm it with tests. missing_tests() names the test a doctor
asks for when a value an urgent rule reads was left blank, in the same words on every doctor surface.
"""

import json
import math
from dataclasses import dataclass

from ahead.config import LEGACY_FIELDS

TRACKS = ("urgent", "at_risk", "not_flagged", "no_score")   # most pressing first, for sorting lists
AUDIENCES = ("patient", "doctor")
ADULTS_ONLY = "adults_only"   # the rule kind that only counts for a flagged result
HISTORY = "history"           # the reason kind for a past event the person reports (HISTORY_GATES)


# =============================================================================
# RULES
# =============================================================================

@dataclass(frozen=True)
class Rule:
    """
    One urgent cut-off for one model input. A rule trips when the entered value is < below or >= at_or_above.

    kind: 'diabetes' or 'kidney' (a range doctors use when diagnosing that condition), 'blood_pressure' (the
          severely high range) or ADULTS_ONLY (age; trips only for a flagged result).
    name: the value as a sentence names it ("Your {name} of 7.1% ..."); label: the same for chips and table cells.
    """
    feature: str
    kind: str
    name: str
    label: str
    below: float | None = None
    at_or_above: float | None = None
    unit: str = ""
    article: str = "a"
    note: str = ""   # what the cut-off is, for the clinician review

    @property
    def cut_off(self) -> float:
        return self.below if self.below is not None else self.at_or_above

    def trips(self, value) -> bool:
        number = entered(value)
        if number is None:
            return False
        return ((self.below is not None and number < self.below)
                or (self.at_or_above is not None and number >= self.at_or_above))

    def cut_off_text(self) -> str:
        """The cut-off as the form shows the value: '6.5%', '200 mg/dL', '60'."""
        return with_unit(self.cut_off, self.unit)

    def text(self) -> str:
        """The rule in words for tables and reports: 'HbA1c 6.5% or more', 'eGFR below 60'."""
        if self.below is not None:
            return f"{self.name} below {self.cut_off_text()}"
        return f"{self.name} {self.cut_off_text()} or more"


_CONFIRM = "A repeat test confirms it."
_ADULT_NOTE = ("Only for a result the model flagged: the screening rules are written for adults, and a child's values "
               "are read against age and sex percentiles, so a flagged child needs a doctor.")

# Prototype cut-offs for a clinician to confirm (not guideline citations), in table order. triage() orders its
# reasons differently: blood pressure first, then the other value reasons in table order, then the age reason.
RULES = {
    "diabetes": (
        Rule("age", ADULTS_ONLY, "age", "Age", below=18, unit="years", article="an", note=_ADULT_NOTE),
        Rule("HbA1c_level", "diabetes", "HbA1c", "HbA1c", at_or_above=6.5, unit="%", article="an",
             note=f"The usual diabetes diagnostic cut-off. {_CONFIRM}"),
        Rule("blood_glucose_level", "diabetes", "blood glucose", "Blood glucose", at_or_above=200, unit="mg/dL",
             note="The dataset does not say the sample was fasting, so it is read as random (non-fasting) glucose; "
                  "200 mg/dL is the usual random-glucose diagnostic cut-off. A doctor confirms it with a fasting "
                  "glucose or HbA1c test."),
    ),
    "heart": (),   # every heart age category is 18 or older, and no heart input is a lab or vital-sign value
    "kidney": (
        Rule("age", ADULTS_ONLY, "age", "Age", below=18, unit="years", article="an", note=_ADULT_NOTE),
        Rule("egfr", "kidney", "eGFR", "eGFR", below=60, article="an",
             note="mL/min/1.73 m², the one unit eGFR is reported in, so the form and the reasons leave it out. Worked "
                  "out from serum creatinine (CKD-EPI 2021) when only creatinine was entered. The chronic kidney "
                  "disease range when it lasts three months, so a doctor repeats the test promptly to exclude a "
                  "short-term change (AKI) and again at 3 months to confirm chronicity."),
        Rule("uacr", "kidney", "urine ACR", "Urine ACR", at_or_above=30, unit="mg/g",
             note="Raised albumin in the urine, the chronic kidney disease range when it lasts three months, so a "
                  "doctor repeats the test promptly to exclude a short-term change and again at 3 months to confirm "
                  "chronicity. Many labs report mg/mmol instead (30 mg/g is about 3 mg/mmol), so the unit is printed "
                  "with the value."),
        Rule("systolic_bp", "blood_pressure", "systolic blood pressure", "Blood pressure", at_or_above=180,
             unit="mmHg", note="Severely high blood pressure; with symptoms it needs emergency care. Shown together "
                               "with the diastolic value as one reason."),
        Rule("diastolic_bp", "blood_pressure", "diastolic blood pressure", "Blood pressure", at_or_above=120,
             unit="mmHg", note="Severely high blood pressure; with symptoms it needs emergency care. Shown together "
                               "with the systolic value as one reason."),
    ),
}

# Rules for keys only older saved screenings hold (config.LEGACY_FIELDS keys with no current twin): the kidney form
# before the switch asked for HbA1c and fasting blood sugar, and a diabetes-range value there was urgent. triage() reads
# them as well as RULES, so those screenings sort the same way; the current form never has these keys.
LEGACY_RULES = {
    "kidney": (
        Rule("HbA1c", "diabetes", "HbA1c", "HbA1c", at_or_above=6.5, unit="%", article="an",
             note=f"Older kidney screenings only. The usual diabetes diagnostic cut-off. {_CONFIRM}"),
        Rule("FastingBloodSugar", "diabetes", "fasting blood sugar", "Fasting blood sugar", at_or_above=126,
             unit="mg/dL", note=f"Older kidney screenings only. The usual fasting diabetes diagnostic cut-off. "
                                f"{_CONFIRM}"),
    ),
}

_RANGE = {   # what an urgent value's range means, per rule kind and audience
    "diabetes": {"patient": "a level doctors use when diagnosing diabetes",
                 "doctor": "a level used when diagnosing diabetes"},
    "kidney": {"patient": "a level doctors use when diagnosing chronic kidney disease",
               "doctor": "a level used when diagnosing chronic kidney disease"},
}


@dataclass(frozen=True)
class HistoryGate:
    """A past event that, answered Yes, puts the result on the urgent track whatever the model's score."""
    feature: str
    patient: str   # the event in the patient's words ("a heart attack")
    doctor: str    # the event in clinical words ("a previous myocardial infarction")


# The heart model compares answers with BRFSS 2022 respondents who reported a past heart attack, so its score means
# nothing useful for someone who has had one, or angina or a stroke: they need their doctor, not a screening. A blank
# answer (a heart screening saved before these questions, an upload without the columns) is not checked.
HISTORY_GATES = {
    "diabetes": (),
    "heart": (HistoryGate("HadHeartAttack", "a heart attack", "a previous myocardial infarction"),
              HistoryGate("HadAngina", "angina", "known angina or coronary heart disease"),
              HistoryGate("HadStroke", "a stroke", "a previous stroke")),
    "kidney": (),
}
_ANSWERS = {"yes": True, "y": True, "true": True, "1": True, "1.0": True,
            "no": False, "n": False, "false": False, "0": False, "0.0": False}


def answer(value) -> bool | None:
    """A Yes/No answer as True/False, or None when it is blank or unreadable."""
    return None if value is None else _ANSWERS.get(str(value).strip().lower())


def reported_history(disease: str, values) -> list:
    """The HISTORY_GATES a screening (a dict or saved JSON text) answered Yes, in table order."""
    values = _saved_values(values)
    return [gate for gate in HISTORY_GATES.get(disease, ()) if answer(values.get(gate.feature))]


def rule_table(disease: str) -> list:
    """The urgent rules as rows: the reviewed rule design, checked against RULES by the tests."""
    return [{"feature": rule.feature, "label": rule.label, "kind": rule.kind, "rule": rule.text(),
             "flagged_only": rule.kind == ADULTS_ONLY, "note": rule.note} for rule in _rules(disease)]


def _rules(disease: str) -> tuple:
    if disease not in RULES:
        raise ValueError(f"Unknown condition '{disease}'; expected one of {', '.join(RULES)}.")
    return RULES[disease]


# =============================================================================
# VALUES
# =============================================================================

def entered(value) -> float | None:
    """The value as a number, or None when it was not entered (blank, text, NaN, infinite or a yes/no flag)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def egfr_from_creatinine(creatinine, age, gender) -> float | None:
    """
    eGFR (mL/min/1.73 m²) from serum creatinine (mg/dL, IDMS-standardised) with the 2021 race-free CKD-EPI equation,
    rounded to 0.1, or None without an adult age, a readable sex and a positive creatinine. Adapted from
    estimate_egfr_from_creatinine in the teammate's AHEAD_Combined_Project/ahead/kidney_indicators.py. An estimate
    for adults; the laboratory's own eGFR, when entered, is used instead.
    """
    creatinine, age = entered(creatinine), entered(age)
    female = {"female": True, "f": True, "1": True, "1.0": True,
              "male": False, "m": False, "0": False, "0.0": False}.get(str(gender).strip().lower())
    if creatinine is None or creatinine <= 0 or age is None or not 18 <= age <= 120 or female is None:
        return None
    kappa, alpha = (0.7, -0.241) if female else (0.9, -0.302)
    ratio = creatinine / kappa
    egfr = 142 * min(ratio, 1.0) ** alpha * max(ratio, 1.0) ** -1.2 * 0.9938 ** age * (1.012 if female else 1.0)
    return round(egfr, 1)


def _prepared(disease: str, values) -> tuple:
    """(values under the current names, eGFR worked out from creatinine or None); see prepared_values."""
    values = dict(_saved_values(values))
    for old, new in LEGACY_FIELDS.get(disease, {}).items():
        if new and old in values and new not in values:
            values[new] = values[old]
    derived = None
    if disease == "kidney" and entered(values.get("egfr")) is None:
        derived = egfr_from_creatinine(values.get("creatinine"), values.get("age"), values.get("gender"))
        if derived is not None:
            values["egfr"] = derived
    return values, derived


def prepared_values(disease: str, values) -> dict:
    """
    A screening's values (a dict or saved JSON text) as the rules read them: an older saved screening's keys
    (config.LEGACY_FIELDS) under the current names, and for kidney an eGFR worked out from creatinine when eGFR is
    blank (egfr_from_creatinine). Other modules use it to read a value the same way the track does.
    """
    return _prepared(disease, values)[0]


def number_text(number: float) -> str:
    """A number as the forms and the doctor's record view show it: at most two decimals, no trailing zeros."""
    return f"{round(number, 2) + 0.0:.2f}".rstrip("0").rstrip(".")   # + 0.0 turns -0.0 into 0.0


def with_unit(number: float, unit: str = "") -> str:
    """'7.1%', '182 mg/dL', '45 mg/g', '52.3' (eGFR has no unit on the form)."""
    return _unit(number_text(number), unit)


def _unit(text: str, unit: str) -> str:
    if not unit:
        return text
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def _shown(rule: Rule, number: float) -> str:
    """
    The value as a reason prints it: number_text, except that a value under a 'below' cut-off which two decimals
    would round up onto it is printed in full (59.996, not 60), so a reason never reads 'eGFR of 60 is below 60'.
    An 'at or above' value cannot round below its cut-off, since every cut-off has at most two decimals.
    """
    text = number_text(number)
    if rule.below is not None and number < rule.below <= float(text):
        return repr(number)   # the shortest text that reads back as the same number, so it stays below the cut-off
    return text


# =============================================================================
# REASONS
# =============================================================================

_PATIENT_CONFIRM = "A doctor needs to confirm this with tests."
_DOCTOR_CONFIRM = "a doctor needs to confirm it with tests"


# A kidney value counts as chronic kidney disease only when it lasts three months, but a new low eGFR or high ACR is
# repeated soon first, to rule out a short-term change; waiting 3 months is never the first step.
_CONFIRM_TEXT = {
    "kidney": {"patient": "A doctor needs to confirm this with tests, repeating this test soon to rule out a short-term "
                          "change and again after about 3 months.",
               "doctor": "a doctor needs to confirm it with tests: repeat promptly to exclude AKI; chronicity is "
                         "confirmed at 3 months"},
}


def _value_reason(rule: Rule, number: float, creatinine=None) -> dict:
    """creatinine: the entered creatinine when this eGFR was worked out from it."""
    relation = "below" if rule.below is not None else "at or above"
    value, cut = _unit(_shown(rule, number), rule.unit), rule.cut_off_text()
    confirm = _CONFIRM_TEXT.get(rule.kind, {"patient": _PATIENT_CONFIRM, "doctor": _DOCTOR_CONFIRM})
    patient_from = doctor_from = ""
    if creatinine is not None:
        patient_from = f" (worked out from your creatinine of {with_unit(creatinine, 'mg/dL')})"
        doctor_from = f" (CKD-EPI 2021 from a creatinine of {with_unit(creatinine, 'mg/dL')})"
    return {
        "feature": rule.feature, "features": [rule.feature], "kind": rule.kind, "label": rule.label,
        "value": number, "cut_off": rule.cut_off, "value_text": value, "cut_off_text": f"{relation} {cut}",
        "patient": f"Your {rule.name} of {value}{patient_from} is {relation} {cut}, {_RANGE[rule.kind]['patient']}. "
                   f"{confirm['patient']}",
        "doctor": f"The patient's {rule.name} of {value}{doctor_from} is {relation} {cut}, "
                  f"{_RANGE[rule.kind]['doctor']}; {confirm['doctor']}.",
    }


def _pressure_reason(rules: tuple, tripped: list, values: dict) -> dict:
    """One reason for blood pressure, with both readings when both were entered (150/122 mmHg)."""
    by_feature = {rule.feature: rule for rule in rules if rule.kind == "blood_pressure"}
    systolic, diastolic = entered(values.get("systolic_bp")), entered(values.get("diastolic_bp"))
    first = tripped[0]   # the systolic rule when both trip (table order)
    if systolic is not None and diastolic is not None:
        name, value = "blood pressure", f"{number_text(systolic)}/{number_text(diastolic)} mmHg"
        sys_cut, dia_cut = by_feature["systolic_bp"].cut_off_text(), by_feature["diastolic_bp"].cut_off_text()
        patient_cut = f"{sys_cut} or more on top, or {dia_cut} or more on the bottom"
        doctor_cut = f"systolic {sys_cut} or more, or diastolic {dia_cut} or more"
        features = ["systolic_bp", "diastolic_bp"]
    else:
        name = first.name
        value = with_unit(entered(values.get(first.feature)), first.unit)
        patient_cut = doctor_cut = f"{first.cut_off_text()} or more"
        features = [first.feature]
    return {
        "feature": first.feature, "features": features, "kind": "blood_pressure", "label": first.label,
        "value": entered(values.get(first.feature)), "cut_off": first.cut_off, "value_text": value,
        "cut_off_text": doctor_cut,
        "patient": f"Your {name} of {value} is in the severely high range ({patient_cut}). {_PATIENT_CONFIRM} "
                   "If you also have chest pain, shortness of breath, a severe headache or trouble seeing or "
                   "speaking, get emergency care now.",
        "doctor": f"The patient's {name} of {value} is in the severely high range ({doctor_cut}); "
                  f"{_DOCTOR_CONFIRM} and ask about symptoms that need emergency care.",
    }


def _adult_reason(rule: Rule, number: float) -> dict:
    age = _shown(rule, number)
    return {
        "feature": rule.feature, "features": [rule.feature], "kind": ADULTS_ONLY, "label": rule.label,
        "value": number, "cut_off": rule.cut_off, "value_text": f"{age} years",
        "cut_off_text": f"below {rule.cut_off_text()}",
        "patient": f"You entered an age of {age}. AHEAD's screening rules are written for adults, so a doctor needs "
                   "to confirm this flagged result with tests.",
        "doctor": f"The patient's entered age is {age}; AHEAD's screening rules are written for adults, so a doctor "
                  "needs to confirm this flagged result with tests.",
    }


def _joined(words: list) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _history_reason(gates: list) -> dict:
    """One reason for every past event answered Yes ('a heart attack and a stroke')."""
    features = [gate.feature for gate in gates]
    return {
        "feature": features[0], "features": features, "kind": HISTORY, "label": "Heart history",
        "value": None, "cut_off": None, "value_text": _joined([gate.patient for gate in gates]), "cut_off_text": "",
        "patient": f"You told us you have had {_joined([gate.patient for gate in gates])}. AHEAD's heart screening is "
                   "for people who have not; please see your doctor about your heart health.",
        "doctor": f"The patient reports {_joined([gate.doctor for gate in gates])} (established cardiovascular "
                  "disease). AHEAD's heart screening is not intended for these patients, so the model score does not "
                  "apply; the patient should see a doctor for a cardiovascular review.",
    }


# =============================================================================
# LAB RANGES (a measured value sets a non-urgent track; the model decides only without one)
# =============================================================================

PREDIABETES = (5.7, 6.5)        # HbA1c %: at or above the first and below the second (6.5% is an urgent rule)
KIDNEY_USUAL = (60, 30)         # eGFR at or above, and urine ACR (mg/g) below: the usual range for both

# The lab ranges as rows for the clinician review, beside rule_table.
LAB_RANGES = {
    "diabetes": ("HbA1c 5.7% to 6.4% (the prediabetes range): At risk, whatever the model's score.",
                 "HbA1c under 5.7%: Not flagged, whatever the model's score."),
    "heart": (),
    "kidney": ("eGFR 60 or more and urine ACR under 30 mg/g, both known: Not flagged, whatever the model's score. "
               "Test yearly with diabetes or high blood pressure.",),
}


def _under(number: float, cut: float, unit: str = "") -> str:
    """A value below *cut* as text that never rounds onto it (5.699 prints in full, not as 5.7)."""
    text = number_text(number)
    return _unit(repr(number) if number < cut <= float(text) else text, unit)


def _model_words(flagged, track: str, values: str = "values") -> dict:
    """What the model's estimate did, for a track a lab value set: nothing to add without a score. values: 'value'
    for one measured value, 'values' for more."""
    if flagged is None:
        return {"patient": "", "doctor": ""}
    if track == "at_risk":
        if flagged:
            return {"patient": " The model's estimate agrees.", "doctor": "; the model score also flagged it"}
        return {"patient": " The model's estimate did not flag this result, but a measured value counts for more "
                           "than the estimate.",
                "doctor": "; the measured value takes precedence over the model score, which was below the threshold"}
    if flagged:
        return {"patient": " The model's estimate flagged this result, but a measured value counts for more than the "
                           "estimate, so it is not flagged.",
                "doctor": f"; the measured {values} take{'s' if values == 'value' else ''} precedence over the model "
                          "score, which was at or above the threshold"}
    return {"patient": " The model's estimate did not flag it either.", "doctor": "; the model score was below the "
                                                                              "threshold"}


def _lab_finding(disease: str, values: dict, flagged, derived) -> dict | None:
    """
    The non-urgent track a measured lab value sets, or None when the labs leave it to the model:
      diabetes  HbA1c 5.7-6.4% -> at_risk; under 5.7% -> not_flagged (6.5% or more is an urgent rule)
      kidney    eGFR >= 60 and urine ACR < 30 mg/g, both known -> not_flagged (either past its cut-off is urgent)
    Returns {'track', 'features', 'patient', 'doctor'}; the two texts are the track's summary.
    """
    if disease == "diabetes":
        hba1c = entered(values.get("HbA1c_level"))
        if hba1c is None or hba1c >= PREDIABETES[1]:
            return None
        if hba1c >= PREDIABETES[0]:
            value, words = _under(hba1c, PREDIABETES[1], "%"), _model_words(flagged, "at_risk")
            return {"track": "at_risk", "features": ["HbA1c_level"],
                    "patient": f"Your HbA1c of {value} is in the prediabetes range (5.7% to 6.4%).{words['patient']} "
                               "This is not a diagnosis, but a prevention plan agreed with your doctor can help. Share "
                               "this result with your doctor to get started.",
                    "doctor": f"HbA1c of {value} is in the prediabetes range (5.7–6.4%){words['doctor']}. No entered "
                              "value reached an urgent cut-off. This is not a diagnosis, but a prevention plan agreed "
                              "with the patient can help."}
        value, words = _under(hba1c, PREDIABETES[0], "%"), _model_words(flagged, "not_flagged", "value")
        return {"track": "not_flagged", "features": ["HbA1c_level"],
                "patient": f"Your HbA1c of {value} is below the prediabetes range (under 5.7%).{words['patient']} "
                           "Keep up your routine check-ups; this does not rule out disease.",
                "doctor": f"Measured HbA1c of {value} is below 5.7%{words['doctor']}. This does not rule out "
                          "disease."}
    if disease == "kidney":
        egfr, acr = entered(values.get("egfr")), entered(values.get("uacr"))
        if egfr is None or acr is None or egfr < KIDNEY_USUAL[0] or acr >= KIDNEY_USUAL[1]:
            return None
        gfr_text, acr_text = number_text(egfr), _under(acr, KIDNEY_USUAL[1], "mg/g")
        creatinine = entered(values.get("creatinine")) if derived is not None else None
        patient_from = (f" (worked out from your creatinine of {with_unit(creatinine, 'mg/dL')})"
                        if creatinine is not None else "")
        doctor_from = f" (CKD-EPI 2021 from creatinine)" if creatinine is not None else ""
        words = _model_words(flagged, "not_flagged")
        return {"track": "not_flagged", "features": ["egfr", "uacr"],
                "patient": f"Your eGFR of {gfr_text}{patient_from} and urine ACR of {acr_text} are both in the usual "
                           f"range (eGFR 60 or more, ACR under 30 mg/g).{words['patient']} With diabetes or high blood "
                           "pressure, these tests are usually repeated once a year. This does not rule out disease.",
                "doctor": f"eGFR {gfr_text}{doctor_from} and urine ACR {acr_text} are in the usual range (eGFR 60 or "
                          f"more, ACR under 30 mg/g){words['doctor']}. Repeat both tests yearly when diabetes or "
                          "hypertension is present. This does not rule out disease."}
    return None


# =============================================================================
# WORDS PER TRACK
# =============================================================================

LABELS = {   # (label, short label) per audience; the only place these words are written
    "urgent": {"patient": ("See a doctor soon", "Urgent"), "doctor": ("Urgent", "Urgent")},
    "at_risk": {"patient": ("At risk", "At risk"), "doctor": ("At risk", "At risk")},
    "not_flagged": {"patient": ("Not flagged", "Not flagged"), "doctor": ("Not flagged", "Not flagged")},
    "no_score": {"patient": ("No score", "No score"), "doctor": ("No score", "No score")},
}
TONES = {"urgent": "danger", "at_risk": "warning", "not_flagged": "success", "no_score": "muted"}

# Summaries that do not depend on the result's details; the urgent one here is for a bare track (no result).
_SUMMARIES = {
    "urgent": {"patient": "Please see a doctor soon so they can confirm this result with tests. This is not a "
                          "diagnosis.",
               "doctor": "The patient should see a doctor soon to confirm this result with tests; this is not a "
                         "diagnosis."},
    "at_risk": {"patient": "The model flagged this result. This is not a diagnosis, but a prevention plan agreed with "
                           "your doctor can help. Share this result with your doctor to get started.",
                "doctor": "The model flagged this result and no entered value reached an urgent cut-off. This is not "
                          "a diagnosis, but a prevention plan agreed with the patient can help."},
    "not_flagged": {"patient": "The model did not flag this result. Keep up your routine check-ups; this does not "
                               "rule out disease.",
                    "doctor": "The model did not flag this result and no entered value reached an urgent cut-off. "
                              "This does not rule out disease."},
    "no_score": {"patient": "AHEAD could not give a score for this result.",
                 "doctor": "There is no model score for this result."},
}


def _urgent_summary(audience: str, flagged, values_tripped: int) -> str:
    """The urgent summary; with no value reason the result is urgent only because a flagged patient is a child."""
    patient = audience == "patient"
    if flagged is None:
        lead = "AHEAD could not give a score for this result, but" if patient else \
            "There is no model score for this result, but"
    elif flagged:
        lead = "The model flagged this result, and"
    else:
        lead = "The model did not flag this result, but"
    if values_tripped:
        if patient:
            what = "a value you entered is" if values_tripped == 1 else "some values you entered are"
        else:
            what = "an entered value is" if values_tripped == 1 else "entered values are"
        clause = f"{what} already in a range doctors use when diagnosing a condition"
    else:   # only a flagged child
        clause = ("AHEAD's screening rules are written for adults" if patient
                  else "the patient is under 18; AHEAD's screening rules are written for adults")
    if patient:
        return f"{lead} {clause}. Please see a doctor soon so they can confirm it with tests. This is not a diagnosis."
    return f"{lead} {clause}. The patient should see a doctor soon to confirm it with tests; this is not a diagnosis."


# The urgent summary for a reported past heart attack, angina or stroke: the model's score does not apply, whatever it is.
_HISTORY_SUMMARY = {
    "patient": "The model's score does not apply to people who have had a heart attack, angina or a stroke, so please "
               "talk to your doctor rather than rely on it. This result is not a diagnosis.",
    "doctor": "The heart screening is not intended for established cardiovascular disease, so the model score does not "
              "apply. The patient should see a doctor soon for a cardiovascular review; this is not a diagnosis.",
}


def _texts(track: str, flagged, reasons: list, lab: dict | None = None) -> dict:
    out = {}
    for audience in AUDIENCES:
        label, short = LABELS[track][audience]
        if track == "urgent" and any(reason["kind"] == HISTORY for reason in reasons):
            summary = _HISTORY_SUMMARY[audience]
        elif track == "urgent":
            tripped = sum(reason["kind"] != ADULTS_ONLY for reason in reasons)
            summary = _urgent_summary(audience, flagged, tripped)
        elif lab:
            summary = lab[audience]
        else:
            summary = _SUMMARIES[track][audience]
        out[audience] = {"label": label, "short": short, "summary": summary}
    return out


def track_text(result, audience: str = "patient") -> dict:
    """
    {'label', 'short', 'summary'} for a triage() result, or for a bare track name (its labels and a summary that
    does not depend on the result's details). Labels are the same on every surface for an audience.
    """
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}.")
    if isinstance(result, dict):
        return dict(result["text"][audience])
    if result not in LABELS:
        raise ValueError(f"Unknown track '{result}'; expected one of {', '.join(TRACKS)}.")
    label, short = LABELS[result][audience]
    return {"label": label, "short": short, "summary": _SUMMARIES[result][audience]}


# =============================================================================
# TRIAGE
# =============================================================================

def triage(disease: str, values: dict, score=None, threshold=None) -> dict:
    """
    Sort one screening result into a track.

    values: the screening's entered values ({field: value}, the form's keys or an older saved screening's); numbers
            or text, blanks allowed. Read through prepared_values (current names, eGFR from creatinine).
    score, threshold: the model's score and the threshold the page used (both, or neither when there is no score).

    Returns a plain dict:
      track      'urgent' | 'at_risk' | 'not_flagged' | 'no_score'
      flagged    score >= threshold (the model's own flag, whatever a lab value says), or None without a score
      lab        the lab finding that set an at_risk or not_flagged track ({'track', 'features', 'patient',
                 'doctor'}), or None when the rules or the model set it
      urgent     track == 'urgent'
      reasons    why it is urgent, most time-critical first: a reported past heart attack, angina or stroke (one
                 reason, kind HISTORY, value and cut_off None), blood pressure (both readings as one reason), the other
                 value reasons in table order, then the age reason; each
                 {feature, features, kind, label, value, cut_off, value_text, cut_off_text, patient, doctor}
      child      an age under 18 was entered (whether or not that made the result urgent)
      tone       'danger' | 'warning' | 'success' | 'muted'
      text       {'patient': {label, short, summary}, 'doctor': {...}}
    plus disease, score and threshold.
    """
    rules = _rules(disease)
    values, derived = _prepared(disease, values or {})
    score, threshold = entered(score), entered(threshold)
    if score is not None and threshold is None:
        raise ValueError("A score needs the threshold the page used.")
    flagged = None if score is None or threshold is None else score >= threshold

    tripped = [rule for rule in rules + LEGACY_RULES.get(disease, ())
               if rule.kind != ADULTS_ONLY and rule.trips(values.get(rule.feature))]
    pressure = [rule for rule in tripped if rule.kind == "blood_pressure"]
    history = reported_history(disease, values)
    # A reported past heart attack, angina or stroke first (heart has no value rules), then blood pressure: the only
    # value that can need emergency care today, and the patient home banner and the doctor home show only the first
    # reason.
    reasons = [_history_reason(history)] if history else []
    reasons += [_pressure_reason(rules, pressure, values)] if pressure else []
    reasons += [_value_reason(rule, entered(values.get(rule.feature)),
                              entered(values.get("creatinine")) if rule.feature == "egfr" and derived is not None
                              else None)
                for rule in tripped if rule.kind != "blood_pressure"]
    lab = None if reasons else _lab_finding(disease, values, flagged, derived)
    age_rule = next((rule for rule in rules if rule.kind == ADULTS_ONLY), None)
    child = age_rule is not None and age_rule.trips(values.get(age_rule.feature))
    # A child is urgent when the result is flagged: by the model, unless a lab value cleared it, or by a lab range.
    if child and (lab["track"] == "at_risk" if lab else flagged):
        reasons.append(_adult_reason(age_rule, entered(values.get(age_rule.feature))))
        lab = None

    if reasons:
        track = "urgent"
    elif lab:
        track = lab["track"]
    elif flagged is None:
        track = "no_score"
    else:
        track = "at_risk" if flagged else "not_flagged"
    return {"disease": disease, "track": track, "flagged": flagged, "urgent": track == "urgent", "score": score,
            "threshold": threshold, "reasons": reasons, "child": child, "lab": lab, "tone": TONES[track],
            "text": _texts(track, flagged, reasons, lab)}


def _saved_values(values_json) -> dict:
    if isinstance(values_json, dict):
        return values_json
    if isinstance(values_json, (bytes, str)) and values_json.strip():
        try:
            loaded = json.loads(values_json)
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def triage_saved(disease: str, values_json, score=None, threshold=None) -> dict:
    """
    triage() for a stored screening or clinic record: values_json as saved (JSON text, or an already decoded dict),
    and the saved score and threshold. Unreadable JSON counts as no values; a score without its threshold (or the
    reverse) counts as no score, so a half-saved row never raises in a history list.
    """
    if entered(score) is None or entered(threshold) is None:
        score = threshold = None
    return triage(disease, _saved_values(values_json), score, threshold)


def is_urgent_saved(disease: str, values_json, score=None, threshold=None) -> bool:
    """True when a stored screening or record is on the urgent track (see triage_saved)."""
    return triage_saved(disease, values_json, score, threshold)["urgent"]


def entered_features(values_json, disease: str | None = None) -> set:
    """The inputs a saved screening has a value for (unreadable JSON counts as none); with *disease*, under the
    current names and with an eGFR worked out from creatinine (prepared_values), as the rules read them, and with each
    HISTORY_GATES question answered Yes or No."""
    values = prepared_values(disease, values_json) if disease else _saved_values(values_json)
    gates = {gate.feature for gate in HISTORY_GATES.get(disease, ())}
    return {feature for feature, value in values.items()
            if entered(value) is not None or (feature in gates and answer(value) is not None)}


def open_urgent(rows) -> list:
    """
    One patient's urgent results that still stand, at most one per condition, newest first, as (row, triage result).
    rows are saved screenings, newest first. A later screening of the same condition clears an urgent reason only
    when it re-entered every value behind it (below the cut-off, or it would be urgent itself); a value left blank was
    not re-checked, so the reason stands. Each result keeps only the reasons no later screening re-checked.
    """
    checked, found = {}, {}
    for row in rows:
        disease = row["disease"]
        if disease in found:
            continue
        result = triage_saved(disease, row["values_json"], row["score"], row["threshold"])
        later = checked.setdefault(disease, set())
        reasons = [reason for reason in result["reasons"] if not set(reason["features"]) <= later]
        if reasons:
            found[disease] = (row, dict(result, reasons=reasons))
        else:
            later |= entered_features(row["values_json"], disease)
    return list(found.values())


# =============================================================================
# TESTS TO ASK FOR
# =============================================================================

@dataclass(frozen=True)
class LabTest:
    """
    A test to ask for when a screening left blank a value the track reads (an urgent rule or a lab range): without
    it AHEAD could not check that value, so the model's estimate sorted the result on its own.

    asks:   the values whose blank asks for this test.
    covers: the blank values the test also stands for (asks included), so related blanks become one action: an HbA1c
            test stands for a blank blood glucose too, a creatinine blood test gives eGFR, and one blood pressure
            reading covers both numbers.
    action: what the doctor does, as a clause ("ask for an HbA1c test to confirm").
    name:   what to call the values when every one in covers is blank ("blood pressure"); otherwise each blank value
            is named by its rule ("HbA1c and blood glucose", "diastolic blood pressure").
    """
    asks: tuple
    covers: tuple
    action: str
    name: str = ""


# In the order a doctor reads them. Heart has no urgent rule. A kidney eGFR worked out from creatinine counts as
# entered (prepared_values), so only a screening with neither asks for the blood test.
LAB_TESTS = {
    "diabetes": (
        LabTest(("HbA1c_level",), ("HbA1c_level", "blood_glucose_level"), "ask for an HbA1c test to confirm"),
    ),
    "heart": (),
    "kidney": (
        LabTest(("egfr",), ("egfr", "creatinine"), "ask for a creatinine (eGFR) blood test", name="creatinine and eGFR"),
        LabTest(("uacr",), ("uacr",), "ask for a urine ACR test"),
        LabTest(("systolic_bp", "diastolic_bp"), ("systolic_bp", "diastolic_bp"), "check the patient's blood pressure",
                name="blood pressure"),
    ),
}


def missing_tests(disease: str, values, later=()) -> list:
    """
    The tests to ask for because a screening left blank a value the track reads, in LAB_TESTS order, each once.
    The doctor home's rows, the doctor's record view and the Prevention Plan strip all show these words.

    values: the saved inputs, as a dict or as saved JSON text, read through prepared_values. A value counts as left
    blank only when the inputs include it and entered() reads it as not entered, so an older record saved without its
    inputs asks for nothing.
    later:  the values a newer screening of the same condition entered (entered_later); a value entered there counts
            as entered, so the ask clears once the patient has had the test and screened again.
    An entered age under 18 asks for nothing: these tests come from adult cut-offs, and a flagged child's urgent
    reason already says a doctor needs to confirm the result with tests.

    Returns [{'features', 'label', 'action', 'line', 'doctor'}]; for a blank HbA1c:
      features  ['HbA1c_level']
      label     'HbA1c'
      action    'ask for an HbA1c test to confirm'
      line      'HbA1c not entered · ask for an HbA1c test to confirm'   (rows and the strip)
      doctor    'HbA1c not entered. Ask for an HbA1c test to confirm.'   (the record view)
    """
    names = {rule.feature: rule.name for rule in _rules(disease) if rule.kind != ADULTS_ONLY} | {
        "creatinine": "creatinine"}
    values = prepared_values(disease, values)
    age = next((rule for rule in _rules(disease) if rule.kind == ADULTS_ONLY), None)
    if age is not None and age.trips(values.get(age.feature)):
        return []
    later = set(later)
    read = set(names) | {feature for test in LAB_TESTS[disease] for feature in test.covers}
    blank = {feature for feature in read
             if feature in values and feature not in later and entered(values[feature]) is None}
    found = []
    for test in LAB_TESTS[disease]:
        if not blank & set(test.asks):
            continue
        covered = [feature for feature in test.covers if feature in blank]
        blank -= set(covered)
        label = test.name if test.name and len(covered) == len(test.covers) else \
            " and ".join(names[feature] for feature in covered)
        if label.split(" ", 1)[0].islower():   # 'blood pressure' -> 'Blood pressure'; eGFR and HbA1c keep their case
            label = label[:1].upper() + label[1:]
        found.append({"features": covered, "label": label, "action": test.action,
                      "line": f"{label} not entered · {test.action}",
                      "doctor": f"{label} not entered. {test.action[:1].upper()}{test.action[1:]}."})
    return found


def entered_later(rows) -> dict:
    """
    For saved screenings newest first: {row id: the values a newer screening by the same patient for the same
    condition entered}, the `later` argument of missing_tests. Rows without an owner_id count as one patient's.
    """
    seen, found = {}, {}
    for row in rows:
        mine = seen.setdefault((row.get("owner_id"), row["disease"]), set())
        found[row["id"]] = set(mine)
        mine |= entered_features(row["values_json"], row["disease"])
    return found
