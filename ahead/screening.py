"""
ahead/screening.py
==================
The screening flow: input form, prediction, gauge, the result's track
(ahead/triage.py), guidance, AHEAD Insight (Gemini) and model interpretation.
Patients read it about themselves; doctor accounts get the same flow worded
about the entered values and the patient.
"""

import math
import re
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ahead.clinical_data import OPTIONAL_BLANK, input_bounds, shown_value
from ahead.components import (
    current_user, disclaimer, go_to, markdown_literal, model_label, model_selector, page_header, pref, record_screening,
    result_card, section_heading, track_panel,
)
from ahead.config import (
    DISEASE_GROUP_DESCRIPTIONS, DISEASES, FEATURE_GROUPS, GENDER_CODED_FIELDS, GROUP_DESCRIPTIONS, OPTIONAL_INPUTS,
    REQUIRED_FOR_SCORE, pretty_label,
)
from ahead.resources import (
    best_model, decision_threshold, feature_summary, features, final_metrics, gemini_available, generate_text,
    load_pipeline, missing_inputs_profile, validation_results,
)
from ahead.pages.settings import open_care_team
from ahead.platform_data import doctors_for, prevention_plans
from ahead.reports import patient_report
from ahead.theme import show_chart, style_figure, tokens
from ahead.triage import track_text, triage


# =============================================================================
# INPUT COMPONENTS
# =============================================================================

def _age_category_sort_key(value: str):
    match = re.search(r"\d+", value)
    return int(match.group()) if match else 999


def _blank(value) -> bool:
    """An optional input the user left empty (stored as None; NaN only inside model frames)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _num(value):
    return None if _blank(value) else float(value)


def _model_frame(values: dict, columns=None) -> pd.DataFrame:
    """Blank optional inputs become NaN so each pipeline's median imputer fills them."""
    return pd.DataFrame([{k: (np.nan if _blank(v) else v) for k, v in values.items()}], columns=columns)


def render_feature_input(stats: dict, feature: str, key: str, optional: bool = False, needed: bool = False):
    """
    One form widget for *feature*, built from the cached dataset statistics. Optional measurements
    start empty instead of at the training median, so nobody submits a typical value by accident.
    """
    info = stats[feature]
    label = pretty_label(feature)

    if info["kind"] == "category":
        options = info["options"]
        if feature == "AgeCategory":
            options = sorted(options, key=_age_category_sort_key)
        else:
            options = sorted(options, key=str.lower)
        default_index = options.index(info["default"]) if info["default"] in options else 0
        return st.selectbox(label, options, index=default_index, key=key)

    if info["binary"] and feature in GENDER_CODED_FIELDS:
        names = ("Male", "Female")
        return st.selectbox(label, [0, 1], index=info["mode"] or 0,
                            format_func=lambda x: names[x], key=key)

    if info["binary"]:   # any other 0/1-coded column is a No/Yes choice
        names = ("No", "Yes")
        return st.selectbox(label, [0, 1], index=info["mode"] or 0,
                            format_func=lambda x: names[x], key=key)

    median = info["median"]
    # Same widened range the clinical upload accepts, so both sides score the same values.
    low, high = input_bounds(feature, info)

    blank = {"value": None, "placeholder": "Needed for a score" if needed else "Optional"} \
        if optional else {}

    if feature.lower() == "age" or info["integer"]:
        return st.number_input(
            label, min_value=low, max_value=high,
            **({"value": int(round(median))} | blank), step=1, key=key,
        )
    return st.number_input(
        label, min_value=low, max_value=high, **({"value": round(median, 2)} | blank),
        step=0.1, key=key,
    )


def individual_sensitivity(pipeline, values: dict, stats: dict) -> list:
    """Illustrative change when one entered value is replaced by the training median/mode."""
    baseline = float(pipeline.predict_proba(_model_frame(values))[0, 1])
    comparisons = []
    for feature, actual in values.items():
        if _blank(actual):      # left blank: the model already used the typical value
            continue
        info = stats[feature]
        reference = info["default"] if info["kind"] == "category" else (
            info["mode"] if info["binary"] else info["median"])
        if reference is None or reference == actual:
            continue
        changed = dict(values)
        changed[feature] = reference
        difference = baseline - float(pipeline.predict_proba(_model_frame(changed))[0, 1])
        comparisons.append((feature, actual, reference, difference))
    return sorted(comparisons, key=lambda row: abs(row[3]), reverse=True)[:5]


# =============================================================================
# VISUALS
# =============================================================================

def risk_gauge(probability: float, threshold: float, title: str) -> go.Figure:
    t = tokens()
    pct, thr = probability * 100, threshold * 100
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            number={"suffix": "%", "font": {"size": 38, "color": t["title"]}},
            title={"text": title, "font": {"size": 14, "color": t["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": t["muted"], "tickfont": {"color": t["muted"]}},
                "bar": {"color": t["accent"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, thr], "color": t["success_soft"]},
                    {"range": [thr, 100], "color": t["danger_soft"]},
                ],
                "threshold": {"line": {"color": t["danger"], "width": 4}, "value": thr},
            },
        )
    )
    figure.update_layout(
        height=275, margin=dict(l=25, r=25, t=55, b=15), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"]),
    )
    return figure


def _column_label(name: str, inputs: list) -> str:
    """A preprocessed column as people read it: the input's label, plus the category of a one-hot column
    ('AgeCategory_Age 80 or older' -> 'Age Category: Age 80 or older'). *inputs* are longest first."""
    base = next((f for f in inputs if name == f or name.startswith(f + "_")), name)
    return pretty_label(base) + (f": {name[len(base) + 1:]}" if name != base else "")


def model_feature_importance(model, title: str, top_n: int = 12):
    """Bar chart of global importances; returns (figure, axis_label) or (None, reason)."""
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    try:
        names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None, "Feature names are not available for this pipeline."
    inputs = sorted(map(str, getattr(preprocessor, "feature_names_in_", [])), key=len, reverse=True)
    names = [_column_label(name, inputs) for name in names]

    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
        axis_label = "Relative model importance"
    elif hasattr(classifier, "coef_"):
        importance = np.abs(np.asarray(classifier.coef_)[0])
        axis_label = "Absolute coefficient magnitude"
    else:
        return None, (
            "This model type (for example an RBF-kernel SVM) does not expose per-feature "
            "importances, so no interpretation chart is available."
        )

    if len(names) != len(importance):
        return None, "Feature names and importances do not align for this model."

    frame = (
        pd.DataFrame({"Feature": names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )
    figure = px.bar(frame, x="Importance", y="Feature", orientation="h", title=title)
    figure.update_traces(marker_color=tokens()["accent"])
    style_figure(figure, height=410)
    figure.update_layout(xaxis_title=axis_label, yaxis_title="")
    return figure, axis_label


# =============================================================================
# RULE-BASED GUIDANCE
# =============================================================================

def _is_yes(value) -> bool:
    return str(value).strip().lower() in {"yes", "1", "true"}


def _smoker(value) -> bool:
    """Exact-match categories that indicate any smoking history (never a substring test)."""
    v = str(value).strip().lower()
    if v in {"never", "never smoked", "no info", "", "nan"}:
        return False
    return v.startswith("current") or v.startswith("former") or v == "ever" or v == "not current"


def identify_risk_factors(disease: str, values: dict) -> list:
    factors = []
    g = values.get

    if disease == "diabetes":
        if float(g("bmi", 0)) >= 30:
            factors.append("Higher BMI")
        if int(g("hypertension", 0)) == 1:
            factors.append("History of hypertension")
        if int(g("heart_disease", 0)) == 1:
            factors.append("History of heart disease")
        hba1c, glucose = _num(g("HbA1c_level")), _num(g("blood_glucose_level"))
        if hba1c is not None and hba1c >= 5.7:
            factors.append("Higher entered HbA1c")
        if glucose is not None and glucose >= 100:
            factors.append("Higher entered glucose")
        if _smoker(g("smoking_history", "")):
            factors.append("Smoking history")

    elif disease == "heart":
        if float(g("BMI", 0)) >= 30:
            factors.append("Higher BMI")
        if str(g("PhysicalActivities", "")).strip().lower() == "no":
            factors.append("No regular physical activity")
        if _smoker(g("SmokerStatus", "")):
            factors.append("Smoking history")
        if _is_yes(g("HadStroke", "")):
            factors.append("Previous stroke")
        diabetes = str(g("HadDiabetes", "")).strip().lower()
        if diabetes.startswith("yes"):
            factors.append("History of diabetes" + (" (gestational)" if "pregnancy" in diabetes else ""))
        elif "pre-diabetes" in diabetes or "borderline" in diabetes:
            factors.append("Pre-diabetes / borderline diabetes")
        if _is_yes(g("HadKidneyDisease", "")):
            factors.append("History of kidney disease")

    elif disease == "kidney":
        if int(g("Smoking", 0)) == 1:
            factors.append("Smoking")
        if float(g("BMI", 0)) >= 30:
            factors.append("Higher BMI")
        if int(g("FamilyHistoryKidneyDisease", 0)) == 1:
            factors.append("Family history of kidney disease")
        if int(g("FamilyHistoryHypertension", 0)) == 1:
            factors.append("Family history of hypertension")
        if int(g("FamilyHistoryDiabetes", 0)) == 1:
            factors.append("Family history of diabetes")
        if int(g("Edema", 0)) == 1:
            factors.append("Reported edema")
        systolic, diastolic = _num(g("SystolicBP")), _num(g("DiastolicBP"))
        if (systolic is not None and systolic >= 130) or (diastolic is not None and diastolic >= 80):
            factors.append("Higher entered blood pressure")
        gfr, acr = _num(g("GFR")), _num(g("ACR"))
        if gfr is not None and gfr < 60:
            factors.append("Lower entered GFR")
        if acr is not None and acr >= 30:
            factors.append("Higher entered urine ACR")

    return factors


NEXT_STEPS = {
    "diabetes": {
        1: [
            ("Arrange a medical review",
             "Book an appointment with a primary-care doctor or diabetes clinic to review this result "
             "together with your symptoms, medical history and previous tests."),
            ("Ask about confirmatory blood testing",
             "Discuss whether HbA1c, fasting plasma glucose or other appropriate diabetes testing should "
             "be repeated or confirmed clinically."),
            ("Review blood pressure and cardiovascular risk",
             "Diabetes risk commonly overlaps with blood-pressure, cholesterol and cardiovascular risk, "
             "so ask whether these should also be checked."),
            ("Make a structured lifestyle plan",
             "If your clinician confirms prediabetes or increased risk, work on sustainable nutrition, "
             "activity and weight goals rather than extreme diets or rapid weight loss."),
        ],
        0: [
            ("Continue routine screening",
             "A lower screening result does not rule out diabetes. Continue blood-sugar screening when "
             "recommended based on your age, medical history and other risk factors."),
            ("Maintain metabolic health",
             "Continue regular physical activity, balanced meals, adequate sleep and realistic long-term "
             "weight management."),
        ],
    },
    "heart": {
        1: [
            ("Arrange a cardiovascular risk review",
             "Book a primary-care appointment and discuss whether a cardiology assessment is appropriate, "
             "especially if you have previous stroke, diabetes, kidney disease or a significant smoking history."),
            ("Check major cardiovascular risk factors",
             "Ask for review of blood pressure, cholesterol or lipid levels, blood sugar and your overall "
             "cardiovascular risk."),
            ("Address smoking if applicable",
             "If you currently smoke, ask your healthcare professional about evidence-based "
             "smoking-cessation support."),
            ("Build physical activity gradually and safely",
             "If you do not have medical restrictions, work toward regular physical activity. If exercise "
             "causes chest discomfort, unusual breathlessness or dizziness, stop and seek medical advice."),
        ],
        0: [
            ("Keep monitoring cardiovascular health",
             "A lower screening result does not eliminate future heart risk. Continue routine checks of "
             "blood pressure, cholesterol and blood sugar when recommended."),
            ("Protect modifiable risk factors",
             "Avoid smoking, stay physically active, maintain healthy sleep and follow a balanced eating pattern."),
        ],
    },
    "kidney": {
        1: [
            ("Arrange a kidney-health review",
             "Book an appointment with a primary-care doctor. Depending on your laboratory results and "
             "medical history, they may recommend referral to a nephrologist."),
            ("Ask about kidney-function testing",
             "Discuss repeat kidney assessment including serum creatinine/eGFR and a urine "
             "albumin-to-creatinine ratio (uACR), especially if previous results were abnormal."),
            ("Review blood pressure and diabetes control",
             "High blood pressure and diabetes can affect kidney health. Ask whether your blood pressure, "
             "HbA1c or glucose needs additional monitoring."),
            ("Review medicines and supplements",
             "Tell your clinician or pharmacist about prescription medicines, over-the-counter painkillers "
             "and supplements. Do not stop prescribed medicine on your own."),
        ],
        0: [
            ("Continue routine kidney monitoring",
             "If you have diabetes, hypertension or a strong family history of kidney disease, ask how often "
             "kidney-function and urine testing should be performed."),
            ("Protect long-term kidney health",
             "Maintain blood-pressure and blood-sugar control, avoid smoking and discuss regular use of "
             "over-the-counter painkillers with a healthcare professional."),
        ],
    },
}

# What a doctor reads instead: points to weigh, never specific orders.
DOCTOR_NEXT_STEPS = {
    "diabetes": {
        1: [
            ("Review the result in clinical context",
             "Interpret this screening estimate alongside the patient's symptoms, history and previous results. "
             "It is not a diagnosis."),
            ("Consider confirmatory assessment",
             "Confirmatory glycaemic assessment, such as HbA1c or fasting plasma glucose, may be considered "
             "as clinically indicated."),
            ("Review related cardiometabolic risk",
             "Diabetes risk often overlaps with blood-pressure, lipid and cardiovascular risk; consider whether "
             "these have been assessed recently."),
            ("Discuss lifestyle support where appropriate",
             "If increased risk is confirmed, sustainable nutrition, activity and weight support can be "
             "discussed with the patient."),
        ],
        0: [
            ("Continue routine screening",
             "A lower screening result does not rule out diabetes. Routine screening can continue according to "
             "the patient's age, history and other risk factors."),
            ("Review individual values",
             "An entered value may still merit attention when the overall result is lower; see Key factors."),
        ],
    },
    "heart": {
        1: [
            ("Review cardiovascular risk in context",
             "Interpret this estimate alongside the patient's history, particularly any previous stroke, diabetes, "
             "kidney disease or significant smoking history."),
            ("Consider further cardiovascular assessment",
             "Review of blood pressure, lipids, glycaemic status and overall cardiovascular risk, or specialist "
             "assessment, may be considered as clinically indicated."),
            ("Review smoking status",
             "If the patient currently smokes, evidence-based cessation support can be discussed."),
            ("Check for exertional symptoms",
             "Before encouraging more activity, consider whether chest discomfort, unusual breathlessness or "
             "dizziness on exertion has been reported."),
        ],
        0: [
            ("Continue routine cardiovascular monitoring",
             "A lower screening result does not eliminate future risk. Routine checks of blood pressure, lipids "
             "and glycaemic status can continue as recommended."),
            ("Review modifiable risk factors",
             "Smoking, physical activity, sleep and diet remain relevant to long-term cardiovascular risk."),
        ],
    },
    "kidney": {
        1: [
            ("Review kidney health in context",
             "Interpret this estimate alongside the patient's laboratory history and comorbidities; nephrology "
             "referral may be considered where clinically indicated."),
            ("Consider repeat kidney assessment",
             "Repeat serum creatinine/eGFR and urine albumin-to-creatinine ratio (uACR) may be considered to "
             "confirm or exclude a persistent abnormality."),
            ("Review blood pressure and glycaemic control",
             "Hypertension and diabetes commonly affect kidney health; consider whether current monitoring "
             "is adequate."),
            ("Review medicines",
             "Prescribed and over-the-counter medicines, including regular painkillers and supplements, may be "
             "relevant to kidney health."),
        ],
        0: [
            ("Continue routine kidney monitoring",
             "Where diabetes, hypertension or a family history of kidney disease is present, routine kidney-function "
             "and urine testing can continue as recommended."),
            ("Review long-term kidney risk",
             "Blood-pressure and glycaemic control, smoking and regular painkiller use remain relevant to "
             "long-term kidney health."),
        ],
    },
}

URGENT_MESSAGES = {
    "heart": (
        "<strong>Seek emergency medical care immediately</strong> for new or severe chest pressure or pain, "
        "shortness of breath, fainting, or pain/discomfort spreading to the arm, back, neck or jaw. "
        "Do not rely on the AHEAD result during an emergency."
    ),
    "kidney": (
        "<strong>Seek urgent medical care</strong> if you develop severe shortness of breath, confusion, "
        "very little or no urine, rapidly worsening swelling, or another severe new symptom. "
        "This screening tool is not designed for emergencies."
    ),
    "diabetes": (
        "<strong>Seek urgent medical care</strong> for severe illness, confusion, fainting, persistent vomiting, "
        "severe dehydration or other rapidly worsening symptoms. Do not use this screening result to decide "
        "whether emergency care is needed."
    ),
}

DOCTOR_URGENT_MESSAGES = {
    "heart": (
        "<strong>Red flags:</strong> new or severe chest pressure or pain, shortness of breath, syncope, or pain "
        "spreading to the arm, back, neck or jaw need emergency assessment. The AHEAD result should not guide "
        "urgent decisions."
    ),
    "kidney": (
        "<strong>Red flags:</strong> severe shortness of breath, confusion, little or no urine output, rapidly "
        "worsening swelling or another severe new symptom need urgent assessment. This screening tool is not "
        "designed for emergencies."
    ),
    "diabetes": (
        "<strong>Red flags:</strong> severe illness, confusion, syncope, persistent vomiting or severe dehydration "
        "need urgent assessment. The screening result should not be used to decide whether emergency care is needed."
    ),
}


def get_next_steps(disease: str, prediction: int, doctor: bool = False, track: dict | None = None) -> list:
    """
    The steps for a result. An urgent *track* (a triage() result) reads the flagged steps, which open with a medical
    review, even when the model did not flag it: routine screening would contradict "See a doctor soon".
    """
    return (DOCTOR_NEXT_STEPS if doctor else NEXT_STEPS)[disease][1 if track and track["urgent"] else prediction]


def _urgent(disease: str, doctor: bool) -> None:
    st.html(f'<div class="notice urgent">{(DOCTOR_URGENT_MESSAGES if doctor else URGENT_MESSAGES)[disease]}</div>')


def _action_cards(steps: list) -> None:
    for index, (title, text) in enumerate(steps, start=1):
        st.html(
            f"""
<div class="action-card">
    <span class="action-number">{index}</span>
    <span class="action-title">{escape(title)}</span>
    <div class="action-text">{escape(text)}</div>
</div>
"""
        )


# =============================================================================
# AHEAD INSIGHT (Gemini)
# =============================================================================

def _insight_track(prediction: int, track: dict | None, doctor: bool) -> tuple:
    """
    (result label, track line, extra requirement) for the prompt. An urgent track names its reasons, so AHEAD Insight
    is never told only "Lower screening risk" for a result the page marks "See a doctor soon".
    """
    result_label = "Elevated screening risk" if prediction == 1 else "Lower screening risk"
    if not (track and track["urgent"]):
        return result_label, "", ""
    label = track_text(track, "doctor" if doctor else "patient")["label"]
    if prediction != 1:   # only an entered value can make a result the model did not flag urgent
        result_label = ("Not flagged by the model, but entered values are in a range doctors use when diagnosing a "
                        "condition")
    reasons = " ".join(reason["doctor"] for reason in track["reasons"])   # third person suits both prompts
    return (result_label, f'AHEAD track: "{label}". {reasons}\n',
            f'- AHEAD has put this result on its "{label}" track for the reasons above. Say so plainly, say that a '
            "doctor needs to confirm it with tests, and never describe the result as reassuring or low risk.\n")


def generate_ai_guidance(disease, prediction, probability, risk_factors, user_values, threshold, usual_threshold,
                         doctor: bool = False, track: dict | None = None) -> str:
    result_label, track_line, track_rule = _insight_track(prediction, track, doctor)
    cut_off = f"{threshold * 100:g}%" + (
        f" (usual {usual_threshold * 100:g}%, lowered because some lab values were not provided; "
        "a score above this cut-off is flagged even if the number looks low)" if threshold < usual_threshold else "")
    friendly_values = {pretty_label(k): v for k, v in user_values.items() if not _blank(v)}
    not_provided = [pretty_label(k) for k, v in user_values.items() if _blank(v)]
    if doctor:
        return generate_text(f"""
You are AHEAD Insight, the interpretation layer inside AHEAD, a university educational early-awareness
screening prototype. It is not a diagnostic medical device and must never present a screening result as a
confirmed diagnosis.

A clinician entered a patient's values and the disease-specific machine-learning model has already produced
the screening result. Your role is NOT to make a new prediction and NOT to repeat AHEAD's standard points to
review. Help the clinician read the existing result in the context of the entered values.

Condition screened: {DISEASES[disease]['long_label']}
Screening result: {result_label}
{track_line}Model screening score: {probability * 100:.1f}%
Classification threshold: {cut_off}
Relevant factors identified by AHEAD: {risk_factors if risk_factors else 'No major rule-based factors highlighted'}
Entered values: {friendly_values}
Not entered (the model used typical training-data values instead): {', '.join(not_provided) or 'none'}

Write a concise, professional interpretation for a clinician.

Use exactly these headings:
### Summary of the Result
### Notable Entered Values
### Points to Consider

Requirements:
{track_rule}- Refer to "the patient" in the third person; never address the reader as the patient.
- Make clear that the result is an educational screening estimate, not a diagnosis, and that the score is not a calibrated probability.
- In "Notable Entered Values", identify 2 to 4 entered values or history items that are relevant, without claiming that any single input caused the prediction.
- If an individual measurement may deserve attention even when the overall result is lower, say so.
- In "Points to Consider", give 2 to 4 brief considerations. Do not prescribe specific tests, referrals or treatment; say "as clinically indicated" where relevant.
- Do not recommend starting, stopping or changing medication.
- Do not invent symptoms, diagnoses, laboratory results or risk factors that were not entered, and do not guess values that were not entered.
- Keep the answer brief enough to read inside a screening application.
""")
    prompt = f"""
You are AHEAD Insight, the personalized interpretation layer inside AHEAD.

AHEAD is a university educational early-awareness screening prototype.
It is not a diagnostic medical device and must never present a screening
result as a confirmed diagnosis.

The disease-specific machine-learning model has already produced the screening
result. Your role is NOT to make a new prediction and NOT to repeat AHEAD's
standard Recommended Next Steps. Your role is to help the user understand the
existing result in the context of the information they entered.

Condition screened: {DISEASES[disease]['long_label']}
Screening result: {result_label}
{track_line}Model screening score: {probability * 100:.1f}%
Classification threshold: {cut_off}
Relevant factors identified by AHEAD: {risk_factors if risk_factors else 'No major rule-based factors highlighted'}
Patient-entered information: {friendly_values}
Not provided by the patient (the model used typical training-data values instead): {', '.join(not_provided) or 'none'}

Write a concise, calm, professional and patient-friendly interpretation.

Use exactly these headings:
### Understanding Your Result
### What Stands Out From Your Inputs
### Questions to Discuss With a Healthcare Professional

Requirements:
{track_rule}- Explain the screening result in simple language and make clear that it is an educational screening estimate, not a diagnosis.
- Explain the model score without describing it as the user's true medical probability.
- Relate the explanation to the user's actual entered information.
- In "What Stands Out From Your Inputs", identify 2 to 4 relevant entered values or history items that are worth understanding or discussing.
- Do not claim that any single input caused the model prediction.
- If an individual clinical measurement may deserve professional interpretation even when the overall screening result is lower, clearly explain that distinction.
- In "Questions to Discuss With a Healthcare Professional", generate 2 to 4 specific questions based on the screening result and entered information.
- Do NOT create another "Recommended Next Steps" section and do not repeat generic lifestyle advice already covered elsewhere in AHEAD.
- Do not diagnose the user or say they definitely have or definitely do not have the condition.
- Do not recommend starting, stopping, or changing prescription medication.
- Do not invent symptoms, diagnoses, laboratory results, or risk factors that were not provided.
- Do not comment on or guess values the patient did not provide. If any were not provided, say briefly that adding them from a recent test would make the screening more reliable.
- Keep the answer brief enough to read comfortably inside a screening application.
"""
    return generate_text(prompt)


# =============================================================================
# MISSING LAB RESULTS
# =============================================================================

LAB_ADVICE = {
    "diabetes": "An HbA1c or fasting blood glucose test gives a much more reliable picture; "
                "repeat this screening with your results when you have them.",
    "kidney": "Add these values from a recent reading or test report when you have them for a more precise estimate.",
}
DOCTOR_LAB_ADVICE = {
    "diabetes": "An HbA1c or fasting plasma glucose result gives a much more reliable picture; repeat the screening "
                "with it when it is available.",
    "kidney": "Adding these values from a recent reading or laboratory report gives a more precise estimate.",
}

# Shown when a value in REQUIRED_FOR_SCORE is blank.
NO_SCORE_REASON = {
    "kidney": "Kidney health is checked with a blood test for creatinine (which also gives eGFR) and a urine test "
              "for albumin (ACR). AHEAD asks for all three before it gives a kidney estimate.",
}

NO_SCORE_STEPS = {
    "kidney": [
        ("Ask about kidney function tests",
         "A blood test for creatinine and eGFR and a urine test for albumin (ACR) are routine. They are especially "
         "recommended if you have diabetes, high blood pressure or a family history of kidney disease."),
        ("Repeat this screening with your results",
         "Enter creatinine, eGFR and urine ACR from your test report to get a kidney estimate."),
        ("Keep an eye on blood pressure",
         "High blood pressure is a common cause of kidney problems. A home or pharmacy reading is a simple start."),
    ],
}
DOCTOR_NO_SCORE_STEPS = {
    "kidney": [
        ("Consider kidney-function assessment",
         "Serum creatinine with eGFR and a urine albumin-to-creatinine ratio (ACR) are the usual assessment, "
         "particularly with diabetes, hypertension or a family history of kidney disease."),
        ("Repeat the screening with the results",
         "Enter serum creatinine, eGFR and urine ACR from the laboratory report to obtain a kidney estimate."),
        ("Review blood pressure",
         "Hypertension is a common contributor to kidney disease; a recent reading helps put the result in context."),
    ],
}

# Form notes that do not assume the reader is the person being screened (config's notes say "you").
DOCTOR_GROUP_DESCRIPTIONS = {
    "Blood Pressure": "Optional. Enter the most recent reading if one is available.",
    "Clinical Measurements": "Optional. Recent laboratory values; leave blank any that are not available.",
}
DOCTOR_DISEASE_GROUP_DESCRIPTIONS = {
    "kidney": {
        "Clinical Measurements": "From a recent blood or urine test. Serum creatinine, eGFR and urine ACR are needed "
                                 "for a kidney estimate; leave blank any others that are not available. ACR is in mg/g "
                                 "(a result in mg/mmol times 8.84 is mg/g).",
    },
}
DOCTOR_SUBTITLES = {
    "diabetes": "Enter the lifestyle and medical-history values, and lab values where available, to generate an "
                "early-awareness diabetes screening estimate.",
    "kidney": "Enter the lifestyle and family-history values and the kidney test results to generate a chronic "
              "kidney disease screening estimate.",
}


def _group_note(disease: str, section: str, doctor: bool) -> str:
    doctor_note = (DOCTOR_DISEASE_GROUP_DESCRIPTIONS.get(disease, {}).get(section)
                   or DOCTOR_GROUP_DESCRIPTIONS.get(section)) if doctor else None
    return (doctor_note or DISEASE_GROUP_DESCRIPTIONS.get(disease, {}).get(section)
            or GROUP_DESCRIPTIONS.get(section, "Other values used by the model."))


def _short(feature: str) -> str:
    """Label without its unit, for running text."""
    return pretty_label(feature).split(" (")[0]


def _listed(names: list) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _pct(share: float) -> str:
    """Whole percent, without rounding 99.6% up to a perfect-looking 100%."""
    return f"{share:.1%}" if 0.995 <= share < 1 else f"{share:.0%}"


def low_specificity_caveat(disease: str, profile: dict) -> str:
    """Extra warning when, with these inputs blank, the model cleared few people without the condition."""
    label, test, negatives = DISEASES[disease]["long_label"], profile["test"], profile.get("test_negatives")
    text = ""
    if test["specificity"] < 0.15:
        text += (f" It flagged {'everyone' if test['specificity'] == 0 else 'almost everyone'} without {label} too, "
                 "so a flag here tells you very little.")
    if negatives and negatives < 100:
        text += (f" Only {negatives} people in that test data did not have {label}, so the figures for people "
                 "without it are a rough guide.")
    return text


def _left_blank(names: list, doctor: bool) -> str:
    """'You left X blank' for patients; 'X was left blank' for doctors."""
    if not doctor:
        return f"You left {_listed(names)} blank"
    return f"{_listed(names)} {'was' if len(names) == 1 else 'were'} left blank"


def missing_note(disease: str, result: dict, doctor: bool = False) -> str:
    """The caveat shown above a score that was produced with optional measurements left blank."""
    label = DISEASES[disease]["long_label"]
    paragraphs = [f"{_left_blank([_short(f) for f in result['missing']], doctor)}, so the model used typical "
                  "values from its training data in their place."]
    usual = decision_threshold(disease, result["model"])
    if result["threshold"] < usual:
        paragraphs[0] += (f" To make up for that, AHEAD flags a result from {result['threshold'] * 100:g}% instead "
                          f"of its usual {usual * 100:g}%.")
    profile = result.get("profile")
    if profile:
        test, full = profile["test"], profile["full"]
        measured = (f"On held-out test data with the same values blank, the model caught {_pct(test['recall'])} of "
                    f"people with {label} and correctly cleared {_pct(test['specificity'])} of people without it")
        if full:
            measured += f" (with every value entered: {_pct(full['recall'])} and {_pct(full['specificity'])})"
        measured += "."
        measured += low_specificity_caveat(disease, profile)
        paragraphs.append(measured)
    paragraphs.append("A lower score does not rule the condition out. "
                      + (DOCTOR_LAB_ADVICE if doctor else LAB_ADVICE).get(disease, ""))
    return "\n\n".join(paragraphs)


def lab_results_needed(disease: str, result: dict, doctor: bool = False) -> None:
    """Shown instead of a score when a measurement the model cannot work without was left blank."""
    needed = [_short(f) for f in REQUIRED_FOR_SCORE[disease] if f in result["missing"]]
    section_heading("Lab results needed", f"AHEAD can't give a {DISEASES[disease]['label'].lower()} estimate without them.")
    source = "the entered values" if doctor else "your answers"
    st.warning(
        f"{_left_blank(needed, doctor)}. {NO_SCORE_REASON[disease]} No score is shown and nothing was saved; "
        f"the factors below come only from {source}."
    )
    # Draws nothing unless an entered value is already in a diagnostic range.
    _track(triage(disease, result["values"]), doctor, f"{disease}_share", "no_score")
    factors = identify_risk_factors(disease, result["values"])
    if factors:
        st.html("".join(f'<span class="factor-chip">{escape(f)}</span>' for f in factors))
    else:
        st.info(f"No rule-based risk factors were highlighted from {source}.")
    if doctor:
        section_heading("Points to review", f"Considerations before a {DISEASES[disease]['label'].lower()} estimate can be given.")
        _action_cards(DOCTOR_NO_SCORE_STEPS[disease])
    else:
        section_heading("What you can do next", "")
        _action_cards(NO_SCORE_STEPS[disease])
        st.link_button("Find a nearby clinic (map search)", "https://www.google.com/maps/search/hospital+near+me/")
    _urgent(disease, doctor)


# =============================================================================
# TRACK (ahead/triage.py sorts every result; the share line is reused on the patient home)
# =============================================================================

# Why a doctor cannot see a result in AHEAD: (reason, what to do with a linked doctor, what to do without one).
NOT_SAVED = {
    "history_off": ("This result was not saved because Keep screening history is off in Settings",
                    "Download the PDF to show them.", "Download the PDF to show your doctor."),
    "no_score": ("AHEAD saves only results with a score", "Tell them about the values above.",
                 "Tell your doctor about the values above."),
}


def share_with_doctor(key: str, not_saved: str = "") -> None:
    """
    Who can see a patient's urgent or at-risk result. Linked doctors see saved results, so they are named; with none
    linked, a button opens Settings > Care team access. *not_saved* (a NOT_SAVED key) says why this result was not
    saved, so no doctor can see it in AHEAD.
    """
    owner = (st.session_state.get("user") or {}).get("id")
    names = sorted((d["name"] for d in doctors_for(owner)), key=str.lower) if owner else []
    who = f"your linked doctor{'s' if len(names) > 1 else ''}, {markdown_literal(_listed(names))}," if names else ""
    if not_saved:
        why, tell_them, tell_doctor = NOT_SAVED[not_saved]
        st.caption(f"{why}, so {who} can't see it in AHEAD. {tell_them}" if names else f"{why}. {tell_doctor}")
    elif names:
        st.caption(f"{who[:1].upper()}{who[1:]} can see this result in AHEAD.")
    else:
        with st.container(key=f"{key}_row", horizontal=True, vertical_alignment="center"):
            if st.button("Share with your doctor", key=key, type="primary"):
                open_care_team()
            st.caption("No doctor can see your results yet. Add your doctor under Care team access in Settings.")


def plan_pointer(key: str, disease: str, not_saved: str = "") -> bool:
    """
    For a patient's at-risk result: when they already follow an active prevention plan for this condition, one line
    pointing to My plan (with who can see the result) instead of the share step. Returns True when shown.
    """
    owner = (st.session_state.get("user") or {}).get("id")
    plan = next((p for p in prevention_plans(owner) if p["status"] == "active" and p["disease"] == disease),
                None) if owner else None
    if not plan:
        return False
    name = markdown_literal(plan["doctor_name"])
    line = f"You already follow a {DISEASES[disease]['long_label']} prevention plan with {name}"
    if not_saved:
        why, tell_them, _ = NOT_SAVED[not_saved]
        line += f". {why}, so they can't see it in AHEAD. {tell_them}"
    else:
        line += ", who can see this result in AHEAD. Keep going with it on My plan."
    with st.container(key=f"{key}_plan_row", horizontal=True, vertical_alignment="center"):
        if st.button("Open My plan", key=f"{key}_plan", type="primary"):
            go_to("medications")
        st.caption(line)
    return True


def _track(result: dict, doctor: bool, key: str, not_saved: str = "", disease: str = "") -> None:
    """The result's track under the result card; a patient on the urgent or at-risk track also reads who can see it,
    or, at risk with an active prevention plan for this condition already, the line pointing to My plan."""
    track_panel(result, "doctor" if doctor else "patient")
    if not doctor and result["track"] in ("urgent", "at_risk"):
        if not (result["track"] == "at_risk" and plan_pointer(key, disease, not_saved)):
            share_with_doctor(key, not_saved)


# =============================================================================
# PREDICTOR PAGE
# =============================================================================

def render_predictor(disease: str) -> bool:
    """Render the guided screening for *disease*. Returns True when a result is on screen."""
    info = DISEASES[disease]
    disease_label = info["long_label"]
    doctor = current_user()["is_doctor"]   # a doctor enters someone else's values: third-person wording

    page_header("AHEAD Clinical Screening", info["page_title"],
                DOCTOR_SUBTITLES.get(disease, info["page_subtitle"]) if doctor else info["page_subtitle"])

    if doctor:
        section_heading("Prediction model", "Choose the model that scores the entered values.")
    else:
        st.markdown("#### Prediction model")
    pipeline, selected_name, threshold = model_selector(disease, key=f"{disease}_model_selector",
                                                        blank_labs=bool(OPTIONAL_INPUTS.get(disease)))
    if pipeline is None:
        st.error("No trained model is available. Run `python train_model.py` first.")
        return False

    stats = feature_summary(disease)
    feature_list = features(disease)

    # Every model feature must appear in the form; anything missing from FEATURE_GROUPS
    # would otherwise be silently imputed by the pipeline.
    groups = dict(FEATURE_GROUPS[disease])
    grouped = {f for section in groups.values() for f in section}
    leftover = [f for f in feature_list if f not in grouped]
    if leftover:
        groups["Additional Information"] = leftover

    # ------------------------------------------------------------------ form
    with st.form(f"{disease}_form"):
        values = {}
        for number, (section, section_features) in enumerate(groups.items(), start=1):
            st.html(
                f"""
<div class="form-section-title">
    <div class="form-section-number">{number}</div>
    <div><h3>{escape(section)}</h3><p>{escape(_group_note(disease, section, doctor))}</p></div>
</div>
"""
            )
            active = [f for f in section_features if f in feature_list]
            needed = [f for f in REQUIRED_FOR_SCORE.get(disease, []) if f in active]
            columns = st.columns(3)
            for index, feature in enumerate(active):
                with columns[index % 3]:
                    values[feature] = render_feature_input(
                        stats, feature, f"{disease}_{feature}",
                        optional=feature in OPTIONAL_INPUTS.get(disease, []), needed=feature in needed,
                    )
        submitted = st.form_submit_button("Generate Screening Result", width="stretch", type="primary")

    # ------------------------------------------------------------------ compute / restore
    result = None
    if submitted:
        missing = [f for f in OPTIONAL_INPUTS.get(disease, []) if f in values and _blank(values[f])]
        if any(f in missing for f in REQUIRED_FOR_SCORE.get(disease, [])):
            result = {"values": values, "no_score": True, "missing": missing, "model": selected_name}
        else:
            probability = float(pipeline.predict_proba(_model_frame(values, feature_list))[0][1])
            profile = missing_inputs_profile(disease, selected_name, missing) if missing else None
            # A blank input may lower the threshold (scripts/missing_inputs.py), never raise it.
            used_threshold = min(profile["threshold"], threshold) if profile else threshold
            prediction = int(probability >= used_threshold)
            result = {
                "values": values, "probability": probability, "prediction": prediction,
                "model": selected_name, "threshold": used_threshold, "missing": missing, "profile": profile,
                "saved": pref("keep_history"),   # record_screening saves only then, so only then can a doctor see it
            }
            record_screening(disease, selected_name, probability, prediction, used_threshold, values)
        st.session_state.last_results[disease] = result
    elif disease in st.session_state.last_results:
        result = st.session_state.last_results[disease]
        st.info(f"Showing {'the' if doctor else 'your'} most recent result for this screening. "
                "Submit the form again to update it.")

    if result is None:
        disclaimer()
        return False

    if result.get("no_score"):
        lab_results_needed(disease, result, doctor)
        disclaimer()
        return True

    values, probability, prediction = result["values"], result["probability"], result["prediction"]
    model_name, threshold = result["model"], result["threshold"]
    track = triage(disease, values, probability, threshold)   # the threshold this result was flagged against
    input_frame = pd.DataFrame([{k: ("Not provided" if _blank(v) else v) for k, v in values.items()}],
                               columns=feature_list)
    # Everything below describes the model that produced *this* result, which may differ
    # from the one currently chosen in the selector when a stored result is being shown.
    result_pipeline = pipeline if model_name == selected_name else (load_pipeline(disease, model_name) or pipeline)
    if model_name != selected_name:
        named = model_label if doctor else str
        st.caption(f"This result was produced by {named(model_name)}. Submit the form to screen with {named(selected_name)}.")

    # ------------------------------------------------------------------ result
    section_heading("Screening result" if doctor else "Screening Result",
                    "Risk estimate generated from the information entered above.")
    if result.get("missing"):
        st.warning(missing_note(disease, result, doctor))
    left, right = st.columns(2)
    with left:
        # Screenings here are not attached to a record, so the card speaks about the entered values.
        result_card(prediction, probability, disease_label, model_name, threshold,
                    audience="doctor" if doctor else "patient", subject="The entered values")
    with right:
        from ahead.visuals import score_ring
        score_ring(probability * 100, "Model screening score",
                   status="Flagged for review" if prediction else "Not flagged",
                   color="#C75D5D" if prediction else "#2E9477",
                   note=f"Classification threshold: {threshold * 100:g}%. This score is not a calibrated clinical probability.")
    _track(track, doctor, f"{disease}_share", "" if result.get("saved", True) else "history_off", disease)
    profile = result.get("profile")
    if profile:
        # Figures for this model with the same inputs blank, not the full-input ones.
        report_metrics = {"precision_disease": profile["test"]["precision"], "recall_disease": profile["test"]["recall"]}
        source, scope = "test", " with the same inputs left blank"
    else:
        report_metrics = final_metrics(disease) if model_name == best_model(disease) else next(
            (m for m in validation_results(disease) if m["model"] == model_name), {})
        source = "test" if model_name == best_model(disease) else "validation"
        scope = " with every value entered" if result.get("missing") else ""
    st.info(
        f"{model_label(model_name) if doctor else model_name} {source}-set precision{scope}: {report_metrics.get('precision_disease', 0):.1%}; "
        f"recall: {report_metrics.get('recall_disease', 0):.1%}. "
        "Precision describes how often flagged examples were positive in that dataset; "
        "recall describes how many positive examples the model flagged. "
        "Performance for an individual or a different population can differ."
    )
    st.download_button("Download screening PDF", patient_report(
        disease, "Session screening", values, probability, threshold, model_name, report_metrics,
        metric_source=source, metric_scope=scope, usual_threshold=decision_threshold(disease, model_name),
        metric_caveat=(f"Specificity with the same inputs left blank: {profile['test']['specificity'] * 100:.1f}%."
                       + low_specificity_caveat(disease, profile)) if profile else "",
        # A session screening has no saved record, so there is no review status to print.
        status=None, audience="doctor" if doctor else "patient", name="" if doctor else current_user()["name"],
        factors=identify_risk_factors(disease, values), track=track),
        file_name=f"ahead_{disease}_screening.pdf", mime="application/pdf")

    # No History tab for doctors: their saved screenings are of different people, so one trend line would mix them.
    labels = ["Points to review" if doctor else "Recommendations", "Key factors", "AHEAD Insight", "Model information"]
    recommendations_tab, factors_tab, insight_tab, model_tab, *history_tab = st.tabs(labels + ([] if doctor else ["History"]))
    with recommendations_tab:
        # ------------------------------------------------------------------ next steps
        if doctor:
            section_heading("Points to review", "Considerations for this result. Further assessment as clinically indicated.")
        else:
            section_heading("Recommended Next Steps", "Practical actions to consider after this screening result.")
        _action_cards(get_next_steps(disease, prediction, doctor, track))
        if not doctor:
            st.link_button("Find a nearby clinic (map search)", "https://www.google.com/maps/search/hospital+near+me/")

    with factors_tab:
        # ------------------------------------------------------------------ risk factors
        risk_factors = identify_risk_factors(disease, values)
        section_heading("Factors identified from the entered values" if doctor else "Factors Identified From Your Inputs",
                        "Entered factors that may be relevant to the screening result.")
        if risk_factors:
            st.html("".join(f'<span class="factor-chip">{escape(f)}</span>' for f in risk_factors))
        else:
            st.info(
                "No major rule-based risk factors were highlighted from the entered values. "
                "The machine-learning model may still use combinations of features in its prediction."
            )

        _urgent(disease, doctor)

    with insight_tab:
        # ------------------------------------------------------------------ AHEAD insight
        section_heading("AHEAD Insight", "Explanation of this result for clinical review." if doctor
                        else "Personalized explanation of your screening.")
        insight_key = f"{model_name}|{probability:.4f}|{sorted(values.items())}"
        insights = st.session_state.setdefault("insights", {})
        stored = insights.get(disease)
        if not gemini_available():
            st.info(
                "AHEAD Insight is not configured on this deployment (no Gemini API key). "
                + ("The screening result and the points to review are complete without it." if doctor else
                   "Your screening result and the standard AHEAD recommendations above are complete without it.")
            )
        elif stored and stored["key"] == insight_key:
            st.markdown(stored["text"])
            st.caption("Generated by Google Gemini from the entered values. Educational text, not clinical guidance."
                       if doctor else
                       "Generated by Google Gemini from the values you entered. Educational text, not medical advice.")
        else:
            st.caption(
                "AHEAD Insight sends the entered values (no name or contact details) to Google Gemini to write a "
                "short interpretation of this result for clinical review." if doctor else
                "AHEAD Insight sends the values you entered above (no name or contact details) to Google Gemini "
                "to write a plain-language explanation of this result."
            )
            if st.button("Generate AHEAD Insight", key=f"{disease}_insight", type="primary"):
                with st.spinner("Preparing the insight…" if doctor else "Preparing your personalized insight…"):
                    try:
                        text = generate_ai_guidance(disease, prediction, probability, risk_factors, values,
                                                    threshold, decision_threshold(disease, model_name), doctor,
                                                    track=track)
                    except Exception:
                        text = ""
                if text:
                    insights[disease] = {"key": insight_key, "text": text}
                    st.rerun()
                st.warning(
                    "AHEAD Insight is temporarily unavailable. "
                    + ("The screening result and the points to review are still available — you can try again."
                       if doctor else "Your screening result and the standard "
                       "AHEAD recommendations above are still available — you can try again.")
                )

    with model_tab:
        # ------------------------------------------------------------------ interpretation
        section_heading("Model interpretation" if doctor else "Model Interpretation",
                        "Features with the strongest overall influence on this model.")
        figure, note = model_feature_importance(result_pipeline, f"{info['label']} — Leading Model Features")
        if figure is not None:
            show_chart(figure)
            if model_name == "LogisticRegression":
                st.caption(
                    "For Logistic Regression this chart uses absolute coefficient magnitude. "
                    "It represents model influence, not medical causation."
                )
        else:
            st.info(note)

        caveat = ("For each row below, one entered value was replaced with a typical training-data value. "
                  "The score difference is illustrative, not a cause, diagnosis, or medical recommendation.")
        if doctor:
            section_heading("Model input sensitivity for these values", caveat)
        else:
            st.subheader("Your result: model input sensitivity")
            st.caption(caveat)
        try:
            local = individual_sensitivity(result_pipeline, values, stats)
            if doctor:  # read like the Clinical Dashboard: no rows whose entry equals the typical value once rounded
                local = [(f, shown_value(f, v, stats[f]), shown_value(f, r, stats[f]), d) for f, v, r, d in local]
                local = [row for row in local if row[1] != row[2]]
            if local:
                axis = 'Score change (points)' if doctor else 'Score difference (points)'  # doctors: as in the table below
                sensitivity_frame = pd.DataFrame([{'Input':pretty_label(f),axis:delta*100} for f,_,_,delta in local])
                sensitivity_figure = px.bar(sensitivity_frame,x=axis,y='Input',orientation='h',title='Illustrative input sensitivity')
                # Rose and blue-grey keep at least 3:1 contrast on both the light and the dark card.
                sensitivity_figure.update_traces(marker_color=['#C9706E' if v>=0 else '#5B8FA3' for v in sensitivity_frame[axis]])
                show_chart(style_figure(sensitivity_figure,300))
                st.dataframe(pd.DataFrame([{"Input": pretty_label(f), "Entered": v, "Typical value": reference,
                                            "Score change (points)": round(delta * 100, 1) + 0.0} if doctor else
                                           {"Input": pretty_label(f), "Your value": str(v), "Comparison value": str(reference),
                                            "Score change (percentage points)": round(delta * 100, 1)}
                                           for f, v, reference, delta in local]), hide_index=True, width="stretch")
            elif doctor:
                st.info("The entered values match the typical training values, so there is no input comparison to show.")
            else:
                st.info("No input comparisons were available.")
        except (ValueError, TypeError):
            st.info("Individual model comparison is unavailable for this result.")

        with st.expander("View entered information"):
            if doctor:  # Male/Female, No/Yes and rounded numbers, as in the Clinical Dashboard value table
                optional = OPTIONAL_INPUTS.get(disease, [])
                st.dataframe(pd.DataFrame({"Input": [pretty_label(f) for f in feature_list],
                                           "Value": [shown_value(f, values.get(f), stats[f],
                                                                 OPTIONAL_BLANK if f in optional else "Not provided")
                                                     for f in feature_list]}),
                             hide_index=True, width="stretch")
            else:
                summary = input_frame.T.rename(columns={0: "Entered Value"})
                summary.index = [pretty_label(f) for f in summary.index]
                summary["Entered Value"] = summary["Entered Value"].astype(str)  # mixed types → Arrow-safe
                st.dataframe(summary, width="stretch")

    if doctor:
        st.caption("No score history is shown for doctor accounts: screenings run here are not tied to one patient, "
                   "so a trend would mix different people. A linked patient's saved assessments and score history "
                   "are in Patient Intelligence.")
    else:
        with history_tab[0]:
            from ahead.storage import load_screenings
            from ahead.visuals import trend
            prior = [r for r in load_screenings(st.session_state.user['id']) if r['disease']==disease and r['model']==model_name]
            if len(prior)>1:
                frame=pd.DataFrame([{'Date':r['created_at'],'Model score (%)':r['score']*100} for r in reversed(prior)])
                trend(frame,'Date','Model score (%)','Saved results · same disease and model')
                st.caption('Only saved results for this condition and model are compared. Scores are not diagnoses.')
            else:
                st.info('Save two assessments for this condition and model to see a history chart.')

    disclaimer()
    return True
