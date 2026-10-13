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

GEMINI_MODEL = "gemini-3.8-flash"


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
            "Complete the lifestyle, medical-history and clinical sections to "
            "generate an early-awareness diabetes screening estimate."
        ),
        "card_copy": "Lifestyle, medical history, HbA1c and blood-glucose information.",
        "icon": "conditions/diabetes.svg",
        "colour": "#1F8FA8",
        "dataset": "diabetes.csv",
        "target": "diabetes",
    },
    "heart": {
        "label": "Cardiovascular",
        "long_label": "heart disease",
        "card_title": "Cardiovascular Screening",
        "page_title": "Cardiovascular Risk Assessment",
        "page_subtitle": (
            "Complete the lifestyle, general-health and medical-history sections "
            "to generate a cardiovascular screening estimate."
        ),
        "card_copy": "Lifestyle, general health and cardiovascular-history indicators.",
        "icon": "conditions/heart.svg",
        "colour": "#D2565C",
        "dataset": "heart.csv",
        "target": "HadHeartAttack",
    },
    "kidney": {
        "label": "Kidney Health",
        "long_label": "chronic kidney disease",
        "card_title": "Kidney Health Screening",
        "page_title": "Kidney Health Assessment",
        "page_subtitle": (
            "Complete the lifestyle, family-history and clinical-laboratory sections "
            "to generate a chronic kidney disease screening estimate."
        ),
        "card_copy": "Lifestyle, family history and kidney-related clinical measurements.",
        "icon": "conditions/kidney.svg",
        "colour": "#2E9477",
        "dataset": "kidney_disease.csv",
        "target": "Diagnosis",
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
            "HadStroke", "HadAsthma", "HadKidneyDisease", "HadArthritis", "HadDiabetes",
        ],
    },
    "kidney": {
        "Lifestyle & Patient Profile": [
            "Age", "Gender", "BMI", "Smoking", "AlcoholConsumption",
            "PhysicalActivity", "DietQuality", "SleepQuality",
        ],
        "Family & Medical History": [
            "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension",
            "FamilyHistoryDiabetes", "Edema",
        ],
        "Clinical Measurements": [
            "SystolicBP", "DiastolicBP", "FastingBloodSugar", "HbA1c",
            "SerumCreatinine", "BUNLevels", "GFR", "ProteinInUrine", "ACR",
            "HemoglobinLevels",
        ],
    },
}

GROUP_DESCRIPTIONS = {
    "Lifestyle & Patient Profile": "Demographic information and everyday lifestyle factors.",
    "General Health": "Overall physical and mental health indicators.",
    "Medical History": "Previous conditions relevant to this assessment.",
    "Family & Medical History": "Family history and current medical indicators.",
    "Clinical Measurements": "Laboratory results and measurable clinical values.",
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
    "blood_glucose_level": "Blood Glucose Level (mg/dL)",
    # heart
    "Sex": "Sex",
    "GeneralHealth": "General Health",
    "PhysicalHealthDays": "Poor Physical Health Days",
    "MentalHealthDays": "Poor Mental Health Days",
    "PhysicalActivities": "Regular Physical Activity",
    "SleepHours": "Average Sleep Hours",
    "HadStroke": "History of Stroke",
    "HadAsthma": "History of Asthma",
    "HadKidneyDisease": "History of Kidney Disease",
    "HadArthritis": "History of Arthritis",
    "HadDiabetes": "History of Diabetes",
    "SmokerStatus": "Smoking Status",
    "AgeCategory": "Age Category",
    "BMI": "Body Mass Index (BMI)",
    "AlcoholDrinkers": "Alcohol Consumption Status",
    # kidney
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

# 0/1 columns rendered as Male/Female (kidney dataset encoding); every other 0/1 column is No/Yes
GENDER_CODED_FIELDS = {"Gender"}


def pretty_label(feature: str) -> str:
    return FIELD_LABELS.get(feature, feature.replace("_", " ").title())


# =============================================================================
# NAVIGATION
# =============================================================================

NAV_ITEMS = [
    # key,        label,               icon file,       doctor-only
    ("overview",  "Overview",          "home.svg",      False),
    ("screenings", "Screenings",       "screenings.svg", False),
    ("wellness", "Health & Lifestyle Tracker", "analytics.svg", False),
    ("food", "Food Nutrition Analyzer", "screenings.svg", False),
    ("medications", "Medication Monitoring", "clinical.svg", False),
    ("notifications", "Notifications", "settings.svg", False),
    ("voice", "Voice Assistant", "assistant.svg", False),
    ("analytics", "Data & Analytics",  "analytics.svg", False),
    ("clinical",  "Clinical Dashboard", "clinical.svg", True),
    ("intelligence", "Advanced Analytics & Patient Intelligence", "analytics.svg", True),
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
