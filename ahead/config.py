"""
ahead/config.py
===============
Static configuration: paths, the disease registry, form layout and labels,
navigation and demo accounts.  Everything that describes *what* the app
screens for lives here so the rest of the code never repeats it.
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
ASSETS_DIR = BASE / "assets"
ICONS_DIR = ASSETS_DIR / "icons"

GEMINI_MODEL = "gemini-3.5-flash-lite"
# Light model: speech-to-text for voice questions, and the assistant's fallback when the main model is
# overloaded or times out, so voice and chat stay responsive.
GEMINI_FAST_MODEL = "gemini-3.5-flash-lite"


# =============================================================================
# DISEASE REGISTRY — single source of truth for names, copy and datasets
# =============================================================================

DISEASES = {
    "diabetes": {
        "label": "Diabetes",
        "long_label": "diabetes",
        "card_title": "Diabetes Screening",
        "page_title": "Diabetes Risk Assessment",
        "page_subtitle": (
            "Answer a few questions about yourself and your health. The model estimates whether your HbA1c is likely "
            "to be in the prediabetes range or higher. If you have a recent HbA1c result, add it: it sets your result "
            "instead of the model's estimate. A blood glucose result counts only when it is 200 mg/dL or more. This "
            "screening is for people who have not been told by a doctor that they have diabetes."
        ),
        "card_copy": "Estimates whether HbA1c is likely to be in the prediabetes range or higher; an HbA1c result is "
                     "optional and sets the result. This screening is for people who have not been told by a doctor "
                     "that they have diabetes.",
        "icon": "conditions/diabetes.svg",
        "colour": "#1F8FA8",
        "dataset": "nhanes_diabetes.csv",
        "target": "hba1c_57_or_higher",
    },
    "heart": {
        "label": "Cardiovascular",
        "long_label": "heart disease",
        "card_title": "Cardiovascular Screening",
        "page_title": "Cardiovascular Risk Assessment",
        "page_subtitle": (
            "Answer the lifestyle, general-health and medical-history questions. The model compares your answers "
            "with those of adults in a CDC health survey who said they had had a heart attack. It is not your chance "
            "of having a heart attack in the next 10 years. This screening is for people who have not had a heart "
            "attack, angina or a stroke."
        ),
        "card_copy": "Compares lifestyle and health answers with adults in a CDC survey who reported a heart attack; "
                     "not a 10-year risk.",
        "icon": "conditions/heart.svg",
        "colour": "#D2565C",
        "dataset": "heart.csv",
        "target": "HadHeartAttack",
        # What the score means. The label is a past heart attack (BRFSS 2022) and the model is trained without
        # re-weighting, so its score is close to the share of similar respondents who reported one (docs/
        # data_sources.md, Calibration); it is not a future risk.
        "score_note": {
            "patient": "The score is the model's estimate of the share of people in a CDC health survey with answers "
                       "like yours who said they had had a heart attack. It is not your chance of having a heart "
                       "attack in the next 10 years.",
            "doctor": "The score estimates the share of BRFSS 2022 respondents with similar answers who reported a "
                      "past myocardial infarction (cross-sectional); it is not a 10-year cardiovascular risk.",
            "short": "It is not a 10-year risk of a heart attack.",
        },
    },
    "kidney": {
        "label": "Kidney Health",
        "long_label": "chronic kidney disease",
        "card_title": "Kidney Health Screening",
        "page_title": "Kidney Health Assessment",
        "page_subtitle": (
            "Answer a few lifestyle and health questions. The model estimates whether a kidney blood or urine test "
            "is likely to be outside the usual range. Kidney test results are optional. An eGFR below 60 (entered, or "
            "worked out from your creatinine) or a urine ACR of 30 mg/g or more is flagged for a doctor; with both "
            "tests in the usual range, the result is not flagged."
        ),
        "card_copy": "Lifestyle and health questions; kidney test results (creatinine, eGFR, urine ACR) are optional.",
        "icon": "conditions/kidney.svg",
        "colour": "#2E9477",
        "dataset": "nhanes_kidney.csv",
        "target": "kidney_marker",
    },
}


# =============================================================================
# FORM LAYOUT
# =============================================================================

FEATURE_GROUPS = {
    "diabetes": {
        "Lifestyle & Patient Profile": ["gender", "age", "smoking_history", "bmi"],
        "Medical History": ["hypertension", "heart_disease"],
        "Clinical Measurements": ["HbA1c_level", "blood_glucose_level"],
    },
    "heart": {
        "Lifestyle & Patient Profile": [
            "Sex", "AgeCategory", "PhysicalActivities", "SleepHours",
            "SmokerStatus", "AlcoholDrinkers", "BMI",
        ],
        "General Health": ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays"],
        "Medical History": [
            "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadKidneyDisease", "HadArthritis",
            "HadDiabetes",
        ],
    },
    "kidney": {
        "Lifestyle & Patient Profile": ["age", "gender", "bmi", "smoking_history", "physically_active"],
        "Medical History": ["hypertension", "diabetes", "heart_disease", "family_kidney"],
        "Blood Pressure": ["systolic_bp", "diastolic_bp"],
        "Clinical Measurements": ["creatinine", "egfr", "uacr"],
    },
}


def form_fields(disease: str) -> list:
    """Every field the condition's form asks for, in form order (saved screenings and uploads use these keys)."""
    return [field for section in FEATURE_GROUPS[disease].values() for field in section]


# Heart form questions that only sort the result (ahead/triage.HISTORY_GATES, with HadStroke, which is also a model
# input) and never reach the model: a Yes puts the result on the urgent track.
GATE_ONLY_FIELDS = {"heart": ("HadHeartAttack", "HadAngina")}

# Fields a person may leave blank: lab tests and readings they may not have to hand, and (kidney) a family history
# they may not know. For diabetes and kidney none of them is a model input (see MODEL_INPUTS): ahead/triage.py reads
# the labs with fixed rules, so a blank one is simply not checked. Heart's gate questions are always answered on the
# form; they are optional only so heart screenings and uploads from before these questions still score.
OPTIONAL_INPUTS = {
    "diabetes": ["HbA1c_level", "blood_glucose_level"],
    "heart": list(GATE_ONLY_FIELDS["heart"]),
    "kidney": ["family_kidney"] + FEATURE_GROUPS["kidney"]["Blood Pressure"]
              + FEATURE_GROUPS["kidney"]["Clinical Measurements"],
}

# The model inputs built from the form's answers (ahead/clinical_data.model_inputs), named as in the training files
# from scripts/build_nhanes_training.py. Labs are never model inputs for these two: they define the training labels
# (HbA1c 5.7% or more; eGFR below 60 or urine ACR 30 mg/g or more), so ahead/triage.py handles them with rules.
# A condition missing here (heart) gives its form values to the model as they are. The diabetes form still asks
# about coronary heart disease for the doctor, but it is not a diabetes model input: among people never told they
# have diabetes it went with a slightly lower HbA1c (they are tested more), which must not read as protective.
MODEL_INPUTS = {
    "diabetes": ("gender", "age", "bmi", "smoked_100", "hypertension"),
    "kidney": ("gender", "age", "bmi", "smoked_100", "physically_active", "hypertension", "diabetes", "heart_disease"),
}

# Each condition's age field (the clinical upload can work it out from a date of birth).
AGE_FIELD = {"diabetes": "age", "heart": "AgeCategory", "kidney": "age"}

# Form fields that are not columns of the training file, described the way resources.feature_summary describes a
# column. 'bounds' are the values the form and the upload accept (inclusive); 'accepted' also takes older answers
# that saved screenings and uploads may hold but the form no longer offers.
_YES_NO = {"kind": "number", "binary": True, "mode": 0, "integer": True, "min": 0.0, "max": 1.0, "median": 0.0}
SMOKING_OPTIONS = ["never", "former", "current"]
FORM_FIELD_SPECS = {
    "diabetes": {
        "smoking_history": {"kind": "category", "options": SMOKING_OPTIONS, "default": "never",
                            "accepted": SMOKING_OPTIONS + ["ever", "not current", "No Info"]},
        # Same accepted ranges as before the switch, so a value past them (an HbA1c of 11%, a glucose of 420) still
        # needs a doctor to check it in the Clinical Dashboard.
        "HbA1c_level": {"kind": "number", "binary": False, "mode": None, "integer": False, "min": 3.5, "max": 9.0,
                        "median": 5.6, "bounds": (2.12, 10.38)},
        "blood_glucose_level": {"kind": "number", "binary": False, "mode": None, "integer": True, "min": 80.0,
                                "max": 300.0, "median": 100.0, "bounds": (25, 355)},
    },
    "kidney": {
        "smoking_history": {"kind": "category", "options": SMOKING_OPTIONS, "default": "never",
                            "accepted": SMOKING_OPTIONS},
        "family_kidney": dict(_YES_NO),
        "systolic_bp": {"kind": "number", "binary": False, "mode": None, "integer": True, "min": 90.0, "max": 179.0,
                        "median": 120.0, "bounds": (68, 202)},
        "diastolic_bp": {"kind": "number", "binary": False, "mode": None, "integer": True, "min": 60.0, "max": 119.0,
                         "median": 80.0, "bounds": (45, 134)},
        "creatinine": {"kind": "number", "binary": False, "mode": None, "integer": False, "min": 0.4, "max": 5.0,
                       "median": 0.9, "bounds": (0.2, 15.0)},
        "egfr": {"kind": "number", "binary": False, "mode": None, "integer": False, "min": 15.0, "max": 120.0,
                 "median": 90.0, "bounds": (1.0, 200.0)},
        "uacr": {"kind": "number", "binary": False, "mode": None, "integer": False, "min": 0.0, "max": 300.0,
                 "median": 8.0, "bounds": (0.0, 5000.0)},
    },
}

# The kidney form before the switch to NHANES (old key -> current key). Screenings and uploads saved then keep these
# keys: they still display, sort into a track and print, but the current model never scores them (their inputs are
# not its inputs). Keys with no current twin (diet and sleep scores, BUN ...) only display.
LEGACY_FIELDS = {
    "kidney": {
        "Age": "age", "Gender": "gender", "BMI": "bmi", "GFR": "egfr", "ACR": "uacr", "SerumCreatinine": "creatinine",
        "SystolicBP": "systolic_bp", "DiastolicBP": "diastolic_bp", "FamilyHistoryKidneyDisease": "family_kidney",
        "Smoking": None, "AlcoholConsumption": None, "PhysicalActivity": None, "DietQuality": None,
        "SleepQuality": None, "FamilyHistoryHypertension": None, "FamilyHistoryDiabetes": None, "Edema": None,
        "FastingBloodSugar": None, "HbA1c": None, "BUNLevels": None, "ProteinInUrine": None, "HemoglobinLevels": None,
    },
}


def is_legacy(disease: str, values) -> bool:
    """True for saved values from a form that has since changed (any key in LEGACY_FIELDS)."""
    return isinstance(values, dict) and any(key in LEGACY_FIELDS.get(disease, {}) for key in values)


GROUP_DESCRIPTIONS = {
    "Lifestyle & Patient Profile": "Demographic information and everyday lifestyle factors.",
    "General Health": "Overall physical and mental health indicators.",
    "Medical History": "Previous conditions relevant to this assessment.",
    "Family & Medical History": "Family history and current medical indicators.",
    "Blood Pressure": "Optional. A recent home or pharmacy reading is fine.",
    "Clinical Measurements": "Optional. Values from a recent blood test; leave any you don't have blank.",
}

# Per-condition wording where a shared section means something different.
DISEASE_GROUP_DESCRIPTIONS = {
    "diabetes": {
        "Clinical Measurements": "Optional. Values from a recent blood test; leave any you don't have blank. They are "
                                 "not part of the model's estimate: AHEAD checks them against the usual ranges.",
    },
    "heart": {
        "Medical History": "Conditions a doctor has told you about. This screening is for people who have not had a "
                           "heart attack, angina or a stroke.",
    },
    "kidney": {
        "Medical History": "Conditions a doctor has told you about, and your family history.",
        "Clinical Measurements": "Optional. From a recent blood or urine test; leave blank any you don't have. If you "
                                 "have creatinine but no eGFR, AHEAD works eGFR out from it. Enter ACR in mg/g (a "
                                 "result in mg/mmol times 8.84 is mg/g).",
    },
}

FIELD_LABELS = {
    # diabetes
    "gender": "Gender",
    "age": "Age",
    "hypertension": "History of Hypertension",
    "heart_disease": "History of Heart Disease",
    "smoking_history": "Smoking History",
    "bmi": "Body Mass Index (BMI)",
    "HbA1c_level": "HbA1c Level (%)",
    "blood_glucose_level": "Blood Glucose Level (mg/dL, any time of day)",
    # heart
    "Sex": "Sex",
    "GeneralHealth": "General Health",
    "PhysicalHealthDays": "Poor Physical Health Days",
    "MentalHealthDays": "Poor Mental Health Days",
    "PhysicalActivities": "Regular Physical Activity",
    "SleepHours": "Average Sleep Hours",
    "HadHeartAttack": "History of Heart Attack",
    "HadAngina": "History of Angina or Coronary Heart Disease",
    "HadStroke": "History of Stroke",
    "HadAsthma": "History of Asthma",
    "HadKidneyDisease": "History of Kidney Disease",
    "HadArthritis": "History of Arthritis",
    "HadDiabetes": "History of Diabetes",
    "SmokerStatus": "Smoking Status",
    "AgeCategory": "Age Category",
    "BMI": "Body Mass Index (BMI)",
    "AlcoholDrinkers": "Alcohol Consumption Status",
    # kidney (diabetes's age, gender, bmi, smoking_history, hypertension and heart_disease are shared)
    "physically_active": "Physically Active",
    "diabetes": "History of Diabetes",
    "family_kidney": "Family History of Kidney Disease",
    "systolic_bp": "Systolic Blood Pressure (mmHg)",
    "diastolic_bp": "Diastolic Blood Pressure (mmHg)",
    "creatinine": "Serum Creatinine (mg/dL)",
    "egfr": "eGFR",
    "uacr": "Urine ACR (mg/g)",
    # model inputs worked out from the answers
    "smoked_100": "Has Smoked (100+ Cigarettes)",
    # the kidney form before the switch (LEGACY_FIELDS), for saved screenings
    "Age": "Age",
    "Gender": "Gender",
    "Smoking": "Smoking",
    "AlcoholConsumption": "Alcohol Consumption",
    "PhysicalActivity": "Physical Activity",
    "DietQuality": "Diet Quality Score",
    "SleepQuality": "Sleep Quality Score",
    "FamilyHistoryKidneyDisease": "Family History of Kidney Disease",
    "FamilyHistoryHypertension": "Family History of Hypertension",
    "FamilyHistoryDiabetes": "Family History of Diabetes",
    "SystolicBP": "Systolic Blood Pressure (mmHg)",
    "DiastolicBP": "Diastolic Blood Pressure (mmHg)",
    "FastingBloodSugar": "Fasting Blood Sugar (mg/dL)",
    "HbA1c": "HbA1c (%)",
    "SerumCreatinine": "Serum Creatinine",
    "BUNLevels": "Blood Urea Nitrogen (BUN)",
    "GFR": "Estimated GFR",
    "ProteinInUrine": "Protein in Urine",
    "ACR": "Albumin-to-Creatinine Ratio (ACR)",
    "HemoglobinLevels": "Hemoglobin Level",
    "Edema": "Presence of Edema",
}

# The questions the patient's form asks (doctors read the short labels above). NHANES and BRFSS asked "Has a doctor
# ever told you ...", so the model's answers mean the same thing.
FORM_QUESTIONS = {
    "diabetes": {
        "hypertension": "Has a doctor told you that you have high blood pressure?",
        "heart_disease": "Has a doctor told you that you have coronary heart disease?",
    },
    "heart": {
        "HadHeartAttack": "Has a doctor ever told you that you had a heart attack?",
        "HadAngina": "Has a doctor ever told you that you have angina or coronary heart disease?",
        "HadStroke": "Has a doctor ever told you that you had a stroke?",
    },
    "kidney": {
        "physically_active": "Do you do moderate or vigorous sport, fitness or recreation in a usual week?",
        "hypertension": "Has a doctor told you that you have high blood pressure?",
        "diabetes": "Has a doctor told you that you have diabetes?",
        "heart_disease": "Has a doctor told you that you have coronary heart disease?",
        "family_kidney": "Does a parent, brother or sister have kidney disease?",
    },
}

# 0/1 columns rendered as Male/Female (the old kidney form's encoding); the YES_NO_FIELDS read No/Yes
GENDER_CODED_FIELDS = {"Gender"}
YES_NO_FIELDS = {"hypertension", "heart_disease", "diabetes", "physically_active", "family_kidney", "smoked_100",
                 "Smoking", "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension", "FamilyHistoryDiabetes", "Edema"}


def pretty_label(feature: str) -> str:
    return FIELD_LABELS.get(feature, feature.replace("_", " ").title())


# Models are stored by class name; screens and reports show them as words.
MODEL_NAMES = {"GradientBoosting": "Gradient boosting", "RandomForest": "Random forest",
               "LogisticRegression": "Logistic regression", "SVM": "Support vector machine",
               "DecisionTree": "Decision tree", "XGBoost": "XGBoost"}


def model_label(name: str) -> str:
    """A stored model class name as people read it ('GradientBoosting' -> 'Gradient boosting')."""
    return MODEL_NAMES.get(name, name)


# =============================================================================
# NAVIGATION
# =============================================================================

NAV_ITEMS = [
    # key,        label,               icon file,       doctor-only
    # Kept in sidebar-group order (Workspace, My health, Care & insights, Support & account)
    # so every group heading appears once.
    ("overview",  "Overview",          "home.svg",      False),
    ("screenings", "Screenings",       "screenings.svg", False),
    # The patient's prevention plan and medicines; the key stays 'medications' so older links and notifications work.
    ("medications", "My plan",         "plan.svg", False),
    ("wellness", "Health & Lifestyle Tracker", "wellness.svg", False),
    ("food", "Food Nutrition Analyzer", "food.svg", False),
    ("analytics", "Data & Analytics",  "analytics.svg", False),
    ("clinical",  "Clinical Dashboard", "clinical.svg", True),
    ("intelligence", "Advanced Analytics & Patient Intelligence", "intelligence.svg", True),
    ("notifications", "Notifications", "notifications.svg", False),
    ("assistant", "AI Assistant",      "assistant.svg", False),
    ("settings",  "Settings",          "settings.svg",  False),
]



# =============================================================================
# ROLE NAMES
# =============================================================================

ROLE_PATIENT = "Patient"
ROLE_DOCTOR = "Doctor"

DISCLAIMER_TEXT = (
    "AHEAD is an educational and research screening prototype. It does not "
    "diagnose disease and should not be used to start, stop or change treatment. "
    "A qualified healthcare professional should interpret symptoms and clinical "
    "test results."
)
