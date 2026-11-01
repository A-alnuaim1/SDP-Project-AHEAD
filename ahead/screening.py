"""
ahead/screening.py
==================
The patient-facing screening flow: input form, prediction, gauge, guidance,
AHEAD Insight (Gemini) and model interpretation.
"""

import math
import re
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ahead.components import (
    disclaimer, model_selector, page_header, record_screening, result_card, section_heading,
)
from ahead.config import (
    DISEASE_GROUP_DESCRIPTIONS, DISEASES, FEATURE_GROUPS, GENDER_CODED_FIELDS, GROUP_DESCRIPTIONS, OPTIONAL_INPUTS,
    REQUIRED_FOR_SCORE, pretty_label,
)
from ahead.resources import (
    best_model, decision_threshold, feature_summary, features, final_metrics, gemini_available, generate_text,
    load_pipeline, missing_inputs_profile, validation_results,
)
from ahead.reports import patient_report
from ahead.theme import show_chart, style_figure, tokens


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

    minimum, maximum, median = info["min"], info["max"], info["median"]
    # Allow realistic values slightly outside the training range instead of hard-capping
    # at the dataset extremes (e.g. an 85-year-old when the dataset stops at 80).
    span = max(maximum - minimum, 1.0)
    low = minimum - 0.25 * span if minimum < 0 else max(0.0, minimum - 0.25 * span)
    high = maximum + 0.25 * span

    blank = {"value": None, "placeholder": "Needed for a score" if needed else "Optional"} \
        if optional else {}

    if feature.lower() == "age" or info["integer"]:
        return st.number_input(
            label, min_value=int(np.floor(low)), max_value=int(np.ceil(high)),
            **({"value": int(round(median))} | blank), step=1, key=key,
        )
    return st.number_input(
        label, min_value=round(low, 2), max_value=round(high, 2), **({"value": round(median, 2)} | blank),
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


def model_feature_importance(model, title: str, top_n: int = 12):
    """Bar chart of global importances; returns (figure, axis_label) or (None, reason)."""
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    try:
        names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None, "Feature names are not available for this pipeline."

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


def get_next_steps(disease: str, prediction: int) -> list:
    return NEXT_STEPS[disease][prediction]


# =============================================================================
# AHEAD INSIGHT (Gemini)
# =============================================================================

def generate_ai_guidance(disease, prediction, probability, risk_factors, user_values, threshold, usual_threshold) -> str:
    result_label = "Elevated screening risk" if prediction == 1 else "Lower screening risk"
    cut_off = f"{threshold * 100:g}%" + (
        f" (usual {usual_threshold * 100:g}%, lowered because some lab values were not provided; "
        "a score above this cut-off is flagged even if the number looks low)" if threshold < usual_threshold else "")
    friendly_values = {pretty_label(k): v for k, v in user_values.items() if not _blank(v)}
    not_provided = [pretty_label(k) for k, v in user_values.items() if _blank(v)]
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
Model screening score: {probability * 100:.1f}%
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
- Explain the screening result in simple language and make clear that it is an educational screening estimate, not a diagnosis.
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


def missing_note(disease: str, result: dict) -> str:
    """The caveat shown above a score that was produced with optional measurements left blank."""
    label = DISEASES[disease]["long_label"]
    paragraphs = [f"You left {_listed([_short(f) for f in result['missing']])} blank, so the model used typical "
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
    paragraphs.append("A lower score does not rule the condition out. " + LAB_ADVICE.get(disease, ""))
    return "\n\n".join(paragraphs)


def lab_results_needed(disease: str, result: dict) -> None:
    """Shown instead of a score when a measurement the model cannot work without was left blank."""
    needed = [_short(f) for f in REQUIRED_FOR_SCORE[disease] if f in result["missing"]]
    section_heading("Lab results needed", f"AHEAD can't give a {DISEASES[disease]['label'].lower()} estimate without them.")
    st.warning(
        f"You left {_listed(needed)} blank. {NO_SCORE_REASON[disease]} No score is shown and nothing was saved; "
        "the factors below come only from your answers."
    )
    factors = identify_risk_factors(disease, result["values"])
    if factors:
        st.html("".join(f'<span class="factor-chip">{escape(f)}</span>' for f in factors))
    else:
        st.info("No rule-based risk factors were highlighted from your answers.")
    section_heading("What you can do next", "")
    for index, (title, text) in enumerate(NO_SCORE_STEPS[disease], start=1):
        st.html(
            f"""
<div class="action-card">
    <span class="action-number">{index}</span>
    <span class="action-title">{escape(title)}</span>
    <div class="action-text">{escape(text)}</div>
</div>
"""
        )
    st.link_button("Find a nearby clinic (map search)", "https://www.google.com/maps/search/hospital+near+me/")
    st.html(f'<div class="notice urgent">{URGENT_MESSAGES[disease]}</div>')


# =============================================================================
# PREDICTOR PAGE
# =============================================================================

def render_predictor(disease: str) -> bool:
    """Render the guided screening for *disease*. Returns True when a result is on screen."""
    info = DISEASES[disease]
    disease_label = info["long_label"]

    page_header("AHEAD Clinical Screening", info["page_title"], info["page_subtitle"])

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
    <div><h3>{escape(section)}</h3><p>{escape(DISEASE_GROUP_DESCRIPTIONS.get(disease, {}).get(section)
        or GROUP_DESCRIPTIONS.get(section, "Other values used by the model."))}</p></div>
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
            }
            record_screening(disease, selected_name, probability, prediction, used_threshold, values)
        st.session_state.last_results[disease] = result
    elif disease in st.session_state.last_results:
        result = st.session_state.last_results[disease]
        st.info("Showing your most recent result for this screening. Submit the form again to update it.")

    if result is None:
        disclaimer()
        return False

    if result.get("no_score"):
        lab_results_needed(disease, result)
        disclaimer()
        return True

    values, probability, prediction = result["values"], result["probability"], result["prediction"]
    model_name, threshold = result["model"], result["threshold"]
    input_frame = pd.DataFrame([{k: ("Not provided" if _blank(v) else v) for k, v in values.items()}],
                               columns=feature_list)
    # Everything below describes the model that produced *this* result, which may differ
    # from the one currently chosen in the selector when a stored result is being shown.
    result_pipeline = pipeline if model_name == selected_name else (load_pipeline(disease, model_name) or pipeline)
    if model_name != selected_name:
        st.caption(f"This result was produced by {model_name}. Submit the form to screen with {selected_name}.")

    # ------------------------------------------------------------------ result
    section_heading("Screening Result", "Risk estimate generated from the information entered above.")
    if result.get("missing"):
        st.warning(missing_note(disease, result))
    left, right = st.columns(2)
    with left:
        result_card(prediction, probability, disease_label, model_name, threshold)
    with right:
        from ahead.visuals import score_ring
        score_ring(probability * 100, "Model screening score",
                   status="Flagged for review" if prediction else "Not flagged",
                   color="#C75D5D" if prediction else "#2E9477",
                   note=f"Classification threshold: {threshold * 100:g}%. This score is not a calibrated clinical probability.")
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
        f"{model_name} {source}-set precision{scope}: {report_metrics.get('precision_disease', 0):.1%}; "
        f"recall: {report_metrics.get('recall_disease', 0):.1%}. "
        "Precision describes how often flagged examples were positive in that dataset; "
        "recall describes how many positive examples the model flagged. "
        "Performance for an individual or a different population can differ."
    )
    st.download_button("Download screening PDF", patient_report(
        disease, "Session screening", values, probability, threshold, model_name, report_metrics,
        metric_source=source, metric_scope=scope, usual_threshold=decision_threshold(disease, model_name),
        metric_caveat=(f"Specificity with the same inputs left blank: {profile['test']['specificity'] * 100:.1f}%."
                       + low_specificity_caveat(disease, profile)) if profile else ""),
        file_name=f"ahead_{disease}_screening.pdf", mime="application/pdf")

    recommendations_tab, factors_tab, insight_tab, model_tab, history_tab = st.tabs(["Recommendations", "Key factors", "AHEAD Insight", "Model information", "History"])
    with recommendations_tab:
        # ------------------------------------------------------------------ next steps
        section_heading("Recommended Next Steps", "Practical actions to consider after this screening result.")
        steps = get_next_steps(disease, prediction)
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
        st.link_button("Find a nearby clinic (map search)", "https://www.google.com/maps/search/hospital+near+me/")

    with factors_tab:
        # ------------------------------------------------------------------ risk factors
        risk_factors = identify_risk_factors(disease, values)
        section_heading("Factors Identified From Your Inputs", "Entered factors that may be relevant to the screening result.")
        if risk_factors:
            st.html("".join(f'<span class="factor-chip">{escape(f)}</span>' for f in risk_factors))
        else:
            st.info(
                "No major rule-based risk factors were highlighted from the entered values. "
                "The machine-learning model may still use combinations of features in its prediction."
            )

        st.html(f'<div class="notice urgent">{URGENT_MESSAGES[disease]}</div>')

    with insight_tab:
        # ------------------------------------------------------------------ AHEAD insight
        section_heading("AHEAD Insight", "Personalized explanation of your screening.")
        insight_key = f"{model_name}|{probability:.4f}|{sorted(values.items())}"
        insights = st.session_state.setdefault("insights", {})
        stored = insights.get(disease)
        if not gemini_available():
            st.info(
                "AHEAD Insight is not configured on this deployment (no Gemini API key). "
                "Your screening result and the standard AHEAD recommendations above are complete without it."
            )
        elif stored and stored["key"] == insight_key:
            st.markdown(stored["text"])
            st.caption("Generated by Google Gemini from the values you entered. Educational text, not medical advice.")
        else:
            st.caption(
                "AHEAD Insight sends the values you entered above (no name or contact details) to Google Gemini "
                "to write a plain-language explanation of this result."
            )
            if st.button("Generate AHEAD Insight", key=f"{disease}_insight", type="primary"):
                with st.spinner("Preparing your personalized insight…"):
                    try:
                        text = generate_ai_guidance(disease, prediction, probability, risk_factors, values,
                                                    threshold, decision_threshold(disease, model_name))
                    except Exception:
                        text = ""
                if text:
                    insights[disease] = {"key": insight_key, "text": text}
                    st.rerun()
                st.warning(
                    "AHEAD Insight is temporarily unavailable. Your screening result and the standard "
                    "AHEAD recommendations above are still available — you can try again."
                )

    with model_tab:
        # ------------------------------------------------------------------ interpretation
        section_heading("Model Interpretation", "Features with the strongest overall influence on this model.")
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

        st.subheader("Your result: model input sensitivity")
        st.caption("For each row below, one entered value was replaced with a typical training-data value. "
                   "The score difference is illustrative, not a cause, diagnosis, or medical recommendation.")
        try:
            local = individual_sensitivity(result_pipeline, values, stats)
            if local:
                sensitivity_frame = pd.DataFrame([{'Input':pretty_label(f),'Score difference (points)':delta*100} for f,_,_,delta in local])
                sensitivity_figure = px.bar(sensitivity_frame,x='Score difference (points)',y='Input',orientation='h',title='Illustrative input sensitivity')
                sensitivity_figure.update_traces(marker_color=['#14556B' if v>=0 else '#5B8FA3' for v in sensitivity_frame['Score difference (points)']])
                show_chart(style_figure(sensitivity_figure,300))
                st.dataframe(pd.DataFrame([{"Input": pretty_label(f), "Your value": str(v),
                                            "Comparison value": str(reference),
                                            "Score change (percentage points)": round(delta * 100, 1)}
                                           for f, v, reference, delta in local]), hide_index=True, width="stretch")
            else:
                st.info("No input comparisons were available.")
        except (ValueError, TypeError):
            st.info("Individual model comparison is unavailable for this result.")

        with st.expander("View entered information"):
            summary = input_frame.T.rename(columns={0: "Entered Value"})
            summary.index = [pretty_label(f) for f in summary.index]
            summary["Entered Value"] = summary["Entered Value"].astype(str)  # mixed types → Arrow-safe
            st.dataframe(summary, width="stretch")

    with history_tab:
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
