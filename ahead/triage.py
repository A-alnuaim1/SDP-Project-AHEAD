"""
ahead/triage.py
===============
The SORT step: every screening result gets one track, and every surface (patient and doctor pages, tables,
reports) shows it in the same words.

  urgent       "See a doctor soon" (doctors read "Urgent"): an entered value is already in a range doctors use when
               diagnosing a condition, whatever the model's score. A flagged result for someone under 18 is urgent
               too, because the screening rules are written for adults.
  at_risk      "At risk": the model flagged the result (score >= the threshold the page used) and no urgent rule
               tripped. A prevention plan agreed with a doctor can help.
  not_flagged  "Not flagged": below the threshold and no urgent rule.
  no_score     the page could not give a score (a required kidney test left blank) and no urgent rule tripped.

The cut-offs in RULES are prototype values for a clinician to confirm, kept in one table so they can be read and
changed in one place; ahead/risk_plan.py takes its gates from here. The offline study behind them
(docs/risk_plan_evaluation.md) found that nearly every flagged diabetes and kidney test row already has the condition,
with labs already in these ranges, so AHEAD sorts results into tracks instead of offering model-chosen plans.

Pure Python (no Streamlit), so pages, reports and scripts share it. Values may arrive as numbers or as text (uploaded
CSV rows); a blank, non-numeric or non-finite value counts as not entered and never trips a rule. Nothing here is a
diagnosis: every urgent reason says a doctor needs to confirm it with tests. missing_tests() names the test a doctor
asks for when a value an urgent rule reads was left blank, in the same words on every doctor surface.
"""

import json
import math
from dataclasses import dataclass

TRACKS = ("urgent", "at_risk", "not_flagged", "no_score")   # most pressing first, for sorting lists
AUDIENCES = ("patient", "doctor")
ADULTS_ONLY = "adults_only"   # the rule kind that only counts for a flagged result


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

# Prototype cut-offs for a clinician to confirm (not guideline citations), in table order. Age comes first so
# ahead/risk_plan.py keeps reporting a child as its first gate. triage() orders its reasons differently: blood
# pressure first, then the other value reasons in table order, then the age reason.
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
        Rule("Age", ADULTS_ONLY, "age", "Age", below=18, unit="years", article="an", note=_ADULT_NOTE),
        Rule("GFR", "kidney", "eGFR", "eGFR", below=60, article="an",
             note="mL/min/1.73 m², the one unit eGFR is reported in, so the form and the reasons leave it out. The "
                  "chronic kidney disease range when it lasts three months, so a doctor repeats the test."),
        Rule("ACR", "kidney", "urine ACR", "Urine ACR", at_or_above=30, unit="mg/g",
             note="Raised albumin in the urine, the chronic kidney disease range when it lasts three months, so a "
                  "doctor repeats the test. Many labs report mg/mmol instead (30 mg/g is about 3 mg/mmol), so the "
                  "unit is printed with the value."),
        Rule("HbA1c", "diabetes", "HbA1c", "HbA1c", at_or_above=6.5, unit="%", article="an",
             note=f"The usual diabetes diagnostic cut-off, the same as on the diabetes form. {_CONFIRM}"),
        Rule("FastingBloodSugar", "diabetes", "fasting blood sugar", "Fasting blood sugar", at_or_above=126,
             unit="mg/dL", note=f"The usual fasting diabetes diagnostic cut-off. {_CONFIRM}"),
        Rule("SystolicBP", "blood_pressure", "systolic blood pressure", "Blood pressure", at_or_above=180,
             unit="mmHg", note="Severely high blood pressure; with symptoms it needs emergency care. Shown together "
                               "with the diastolic value as one reason."),
        Rule("DiastolicBP", "blood_pressure", "diastolic blood pressure", "Blood pressure", at_or_above=120,
             unit="mmHg", note="Severely high blood pressure; with symptoms it needs emergency care. Shown together "
                               "with the systolic value as one reason."),
    ),
}

_RANGE = {   # what an urgent value's range means, per rule kind and audience
    "diabetes": {"patient": "a level doctors use when diagnosing diabetes",
                 "doctor": "a level used when diagnosing diabetes"},
    "kidney": {"patient": "a level doctors use when diagnosing chronic kidney disease",
               "doctor": "a level used when diagnosing chronic kidney disease"},
}


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


def _value_reason(rule: Rule, number: float) -> dict:
    relation = "below" if rule.below is not None else "at or above"
    value, cut = _unit(_shown(rule, number), rule.unit), rule.cut_off_text()
    return {
        "feature": rule.feature, "features": [rule.feature], "kind": rule.kind, "label": rule.label,
        "value": number, "cut_off": rule.cut_off, "value_text": value, "cut_off_text": f"{relation} {cut}",
        "patient": f"Your {rule.name} of {value} is {relation} {cut}, {_RANGE[rule.kind]['patient']}. "
                   f"{_PATIENT_CONFIRM}",
        "doctor": f"The patient's {rule.name} of {value} is {relation} {cut}, {_RANGE[rule.kind]['doctor']}; "
                  f"{_DOCTOR_CONFIRM}.",
    }


def _pressure_reason(rules: tuple, tripped: list, values: dict) -> dict:
    """One reason for blood pressure, with both readings when both were entered (150/122 mmHg)."""
    by_feature = {rule.feature: rule for rule in rules if rule.kind == "blood_pressure"}
    systolic, diastolic = entered(values.get("SystolicBP")), entered(values.get("DiastolicBP"))
    first = tripped[0]   # the systolic rule when both trip (table order)
    if systolic is not None and diastolic is not None:
        name, value = "blood pressure", f"{number_text(systolic)}/{number_text(diastolic)} mmHg"
        sys_cut, dia_cut = by_feature["SystolicBP"].cut_off_text(), by_feature["DiastolicBP"].cut_off_text()
        patient_cut = f"{sys_cut} or more on top, or {dia_cut} or more on the bottom"
        doctor_cut = f"systolic {sys_cut} or more, or diastolic {dia_cut} or more"
        features = ["SystolicBP", "DiastolicBP"]
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


def _texts(track: str, flagged, reasons: list) -> dict:
    out = {}
    for audience in AUDIENCES:
        label, short = LABELS[track][audience]
        if track == "urgent":
            tripped = sum(reason["kind"] != ADULTS_ONLY for reason in reasons)
            summary = _urgent_summary(audience, flagged, tripped)
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

    values: the entered model inputs ({feature: value}); numbers or text, blanks allowed.
    score, threshold: the model's score and the threshold the page used (both, or neither when there is no score).

    Returns a plain dict:
      track      'urgent' | 'at_risk' | 'not_flagged' | 'no_score'
      flagged    score >= threshold, or None without a score
      urgent     track == 'urgent'
      reasons    why it is urgent, most time-critical first: blood pressure (both readings as one reason), the other
                 value reasons in table order, then the age reason; each
                 {feature, features, kind, label, value, cut_off, value_text, cut_off_text, patient, doctor}
      child      an age under 18 was entered (whether or not that made the result urgent)
      tone       'danger' | 'warning' | 'success' | 'muted'
      text       {'patient': {label, short, summary}, 'doctor': {...}}
    plus disease, score and threshold.
    """
    rules = _rules(disease)
    values = values or {}
    score, threshold = entered(score), entered(threshold)
    if score is not None and threshold is None:
        raise ValueError("A score needs the threshold the page used.")
    flagged = None if score is None or threshold is None else score >= threshold

    tripped = [rule for rule in rules if rule.kind != ADULTS_ONLY and rule.trips(values.get(rule.feature))]
    pressure = [rule for rule in tripped if rule.kind == "blood_pressure"]
    # Blood pressure first: the only reason that can need emergency care today, and the patient home banner and the
    # doctor home show only the first reason.
    reasons = [_pressure_reason(rules, pressure, values)] if pressure else []
    reasons += [_value_reason(rule, entered(values.get(rule.feature))) for rule in tripped
                if rule.kind != "blood_pressure"]
    age_rule = next((rule for rule in rules if rule.kind == ADULTS_ONLY), None)
    child = age_rule is not None and age_rule.trips(values.get(age_rule.feature))
    if child and flagged:
        reasons.append(_adult_reason(age_rule, entered(values.get(age_rule.feature))))

    if reasons:
        track = "urgent"
    elif flagged is None:
        track = "no_score"
    else:
        track = "at_risk" if flagged else "not_flagged"
    return {"disease": disease, "track": track, "flagged": flagged, "urgent": track == "urgent", "score": score,
            "threshold": threshold, "reasons": reasons, "child": child, "tone": TONES[track],
            "text": _texts(track, flagged, reasons)}


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


def entered_features(values_json) -> set:
    """The inputs a saved screening has a value for (unreadable JSON counts as none)."""
    values = _saved_values(values_json)
    return {feature for feature, value in values.items() if entered(value) is not None}


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
            later |= entered_features(row["values_json"])
    return list(found.values())


# =============================================================================
# TESTS TO ASK FOR
# =============================================================================

@dataclass(frozen=True)
class LabTest:
    """
    A test to ask for when a screening left blank a value an urgent rule reads: the model used a typical value instead
    and the urgent check could not use it.

    asks:   the values whose blank asks for this test.
    covers: the blank values the test also stands for (asks included), so related blanks become one action: an HbA1c
            test stands for a blank blood glucose too, and one blood pressure reading for both numbers.
    action: what the doctor does, as a clause ("ask for an HbA1c test").
    name:   what to call the values when every one in covers is blank ("blood pressure"); otherwise each blank value
            is named by its rule ("HbA1c and blood glucose", "diastolic blood pressure").
    """
    asks: tuple
    covers: tuple
    action: str
    name: str = ""


# In the order a doctor reads them; every urgent-rule value except the age rules and the kidney tests below. eGFR and
# urine ACR are not here: with serum creatinine they are the tests a kidney screening needs for a score
# (config.REQUIRED_FOR_SCORE), so a screening without them gets no score and is not saved. Heart has no urgent rule.
LAB_TESTS = {
    "diabetes": (
        LabTest(("HbA1c_level",), ("HbA1c_level", "blood_glucose_level"), "ask for an HbA1c test"),
        LabTest(("blood_glucose_level",), ("blood_glucose_level",), "ask for a blood glucose test"),
    ),
    "heart": (),
    "kidney": (
        LabTest(("HbA1c",), ("HbA1c", "FastingBloodSugar"), "ask for an HbA1c test"),
        LabTest(("FastingBloodSugar",), ("FastingBloodSugar",), "ask for a fasting blood sugar test"),
        LabTest(("SystolicBP", "DiastolicBP"), ("SystolicBP", "DiastolicBP"), "check the patient's blood pressure",
                name="blood pressure"),
    ),
}


def missing_tests(disease: str, values, later=()) -> list:
    """
    The tests to ask for because a screening left blank a value an urgent rule reads, in LAB_TESTS order, each once.
    The doctor home's at-risk rows, the doctor's record view and the Prevention Plan strip all show these words.

    values: the saved inputs, as a dict or as saved JSON text. A value counts as left blank only when the inputs
    include it and entered() reads it as not entered, so an older record saved without its inputs asks for nothing.
    later:  the values a newer screening of the same condition entered (entered_later); a value entered there counts
            as entered, so the ask clears once the patient has had the test and screened again.
    An entered age under 18 asks for nothing: these tests come from adult cut-offs, and a flagged child's urgent
    reason already says a doctor needs to confirm the result with tests.

    Returns [{'features', 'label', 'action', 'line', 'doctor'}]; for a blank HbA1c:
      features  ['HbA1c_level']
      label     'HbA1c'
      action    'ask for an HbA1c test'
      line      'HbA1c not entered · ask for an HbA1c test'                                  (rows and the strip)
      doctor    'HbA1c not entered: the model used a typical value. Ask for an HbA1c test.'   (the record view)
    """
    rules = {rule.feature: rule for rule in _rules(disease) if rule.kind != ADULTS_ONLY}
    values = _saved_values(values)
    age = next((rule for rule in _rules(disease) if rule.kind == ADULTS_ONLY), None)
    if age is not None and age.trips(values.get(age.feature)):
        return []
    later = set(later)
    blank = {feature for feature in rules
             if feature in values and feature not in later and entered(values[feature]) is None}
    found = []
    for test in LAB_TESTS[disease]:
        if not blank & set(test.asks):
            continue
        covered = [feature for feature in test.covers if feature in blank]
        blank -= set(covered)
        label = test.name if test.name and len(covered) == len(test.covers) else \
            " and ".join(rules[feature].name for feature in covered)
        label = label[:1].upper() + label[1:]
        typical = "a typical value" if len(covered) == 1 else "typical values"
        found.append({"features": covered, "label": label, "action": test.action,
                      "line": f"{label} not entered · {test.action}",
                      "doctor": f"{label} not entered: the model used {typical}. "
                                f"{test.action[:1].upper()}{test.action[1:]}."})
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
        mine |= entered_features(row["values_json"])
    return found
