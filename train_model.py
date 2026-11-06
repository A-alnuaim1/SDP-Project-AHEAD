"""
train_model.py
==============
Train and benchmark AHEAD's machine-learning models.

Data (see docs/data_sources.md):
    - Diabetes: data/nhanes_diabetes.csv (CDC NHANES 1999-2023). Label: HbA1c >= 5.7% at the exam
      in adults never told they have diabetes.
    - Kidney: data/nhanes_kidney.csv (CDC NHANES 1999-2023). Label: eGFR < 60 or urine ACR >= 30 mg/g
      at that single visit.
    - Heart: data/heart.csv (CDC BRFSS 2022). Label: ever told they had a heart attack (a past event).
    Build the two NHANES files first with: python scripts/build_nhanes_training.py

For each condition, this script benchmarks Random Forest, Logistic Regression, Gradient Boosting,
XGBoost (if installed), SVM and Decision Tree inside Pipeline(preprocessor -> classifier).

Split:
    - Diabetes and kidney: test = the 2021-2023 NHANES cycle only; validation = stratified 20% of the
      earlier cycles (random_state=42); training = the rest. 'cycle' is never a feature.
    - Heart: stratified 64% train / 16% validation / 20% test (random_state=42), as before.
    The test set is never used for fitting, model selection or thresholds.

Models are trained without class re-weighting, so a score is close to the share of similar people in
the training data who had the outcome (checked on the test set as "calibration_test").

Selection: Logistic Regression is the recommended model for every condition: its score moves one way
per input and its coefficients can be explained, which the team's own study showed matters (tree models
reversed direction for weight and blood sugar). The other families are trained as benchmarks; the gap
between the best family's validation AUC and Logistic Regression's is recorded as "logistic_auc_gap".

Threshold: each family's probability threshold is tuned on the validation set to maximise Youden's J
(sensitivity + specificity - 1), which weighs missed cases and false alarms equally. The selected model's results are reported once on the test set;
the other families' test results are recorded for comparison only.

Run:
    python train_model.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("xgboost not installed - XGBoost will be skipped.")
    print("Install it with: pip install xgboost\n")


# =============================================================================
# Paths
# =============================================================================

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

MODELS_DIR.mkdir(exist_ok=True)
ALL_MODELS_DIR = MODELS_DIR / "all_models"
ALL_MODELS_DIR.mkdir(exist_ok=True)

NHANES_SUMMARY = DATA_DIR / "nhanes" / "build_summary.json"


# =============================================================================
# Features and data sources
# =============================================================================

# Labs are never model inputs for diabetes or kidney: they define the labels and are handled by rules.
DIABETES_FEATURES = [
    "gender",
    "age",
    "bmi",
    "smoked_100",
    "hypertension",
]


HEART_FEATURES = [
    "Sex",
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "PhysicalActivities",
    "SleepHours",
    "HadStroke",
    "HadAsthma",
    "HadKidneyDisease",
    "HadArthritis",
    "HadDiabetes",
    "SmokerStatus",
    "AgeCategory",
    "BMI",
    "AlcoholDrinkers",
]


KIDNEY_FEATURES = [
    "gender",
    "age",
    "bmi",
    "smoked_100",
    "physically_active",
    "hypertension",
    "diabetes",
    "heart_disease",
]


TEST_CYCLE = "2021-2023"

NHANES_SOURCE = (
    "CDC/NCHS National Health and Nutrition Examination Survey (NHANES) 1999-2023 public-use data, "
    "rebuilt from the raw CDC files into data/nhanes/nhanes_1999_2023_research.csv.gz by scripts/fetch_nhanes.py "
    "and scripts/rebuild_nhanes.py (public domain; cite NHANES)."
)

DISEASE_CONFIGS = {
    "diabetes": {
        "csv": "nhanes_diabetes.csv",
        "features": DIABETES_FEATURES,
        "target": "hba1c_57_or_higher",
        "split": "cycle",
        "data_source": NHANES_SOURCE,
        "label_definition": (
            "1 if HbA1c measured at the NHANES exam was 5.7% or higher (prediabetes or diabetes range), "
            "else 0. A single lab value, not a diagnosis."
        ),
        "population": (
            "NHANES adults 18+ with exam data who had never been told they have diabetes (borderline or "
            "unknown answers left out) and had HbA1c measured."
        ),
    },
    "heart": {
        "csv": "heart.csv",
        "features": HEART_FEATURES,
        "target": "HadHeartAttack",
        "split": "random",
        "data_source": (
            "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022, via Kaggle 'Indicators of Heart "
            "Disease (2022 UPDATE)' by Kamil Pytlak (CC0)."
        ),
        "label_definition": (
            "1 if the respondent was ever told they had a heart attack (a past event, not a future one), "
            "else 0."
        ),
        "population": "US adults who answered the BRFSS 2022 telephone survey (complete-answer rows).",
    },
    "kidney": {
        "csv": "nhanes_kidney.csv",
        "features": KIDNEY_FEATURES,
        "target": "kidney_marker",
        "split": "cycle",
        "data_source": NHANES_SOURCE,
        "label_definition": (
            "1 if eGFR (CKD-EPI 2021) was below 60 or urine albumin-to-creatinine ratio was 30 mg/g or "
            "higher at that single visit, else 0. A single-visit kidney marker, not confirmed chronic "
            "kidney disease."
        ),
        "population": "NHANES adults 18+ with both eGFR and urine albumin-to-creatinine ratio measured.",
    },
}


# =============================================================================
# Settings
# =============================================================================

SVM_MAX_TRAIN_SAMPLES = 15000
THRESHOLD_GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)

SELECTION_RULE = (
    "Logistic Regression for every condition, because its score moves one way per answer and each answer's "
    "effect can be explained; the other families are benchmarks. The threshold maximises Youden's J "
    "(sensitivity + specificity - 1) on the validation set."
)


# =============================================================================
# Helper functions
# =============================================================================

def clean_binary_target(series):
    """
    Convert common binary target formats to 0 and 1.
    """

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().unique())

        if unique_values.issubset({0, 1}):
            return series.astype(int)

    cleaned = series.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "disease": 1,
        "no disease": 0,
        "1": 1,
        "0": 0,
    }

    converted = cleaned.map(mapping)

    if converted.isna().any():
        unknown = cleaned[converted.isna()].unique()

        raise ValueError(
            f"Target contains values that could not be converted: {unknown}"
        )

    return converted.astype(int)


def load_training_frame(disease):
    """The rows train_model uses for *disease*: features, target and (for NHANES) cycle."""
    config = DISEASE_CONFIGS[disease]
    csv_path = DATA_DIR / config["csv"]
    if not csv_path.exists():
        hint = " Run python scripts/build_nhanes_training.py first." if config["split"] == "cycle" else ""
        raise FileNotFoundError(f"Dataset not found: {csv_path}.{hint}")

    columns = config["features"] + [config["target"]] + (["cycle"] if config["split"] == "cycle" else [])
    df = pd.read_csv(csv_path)
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{disease}: Missing columns: {missing_columns}")

    df = df[columns].dropna(subset=[config["target"]]).copy()
    df[config["target"]] = clean_binary_target(df[config["target"]])
    return df


def split_data(df, disease):
    """
    (X_train, X_val, X_test, y_train, y_val, y_test).
    NHANES: test = TEST_CYCLE; validation = stratified 20% of the earlier cycles.
    Heart: stratified 20% test, then 20% of the rest for validation.
    """
    config = DISEASE_CONFIGS[disease]
    features, target = config["features"], config["target"]

    if config["split"] == "cycle":
        is_test = df["cycle"] == TEST_CYCLE
        if not is_test.any():
            raise ValueError(f"{disease}: no rows in the test cycle {TEST_CYCLE}")
        earlier = df[~is_test]
        X_test, y_test = df.loc[is_test, features], df.loc[is_test, target]
        X_train, X_val, y_train, y_val = train_test_split(
            earlier[features], earlier[target], test_size=0.20, random_state=42, stratify=earlier[target],
        )
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df[features], df[target], test_size=0.20, random_state=42, stratify=df[target],
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def describe_split(disease, y_train, y_val, y_test):
    config = DISEASE_CONFIGS[disease]
    if config["split"] == "cycle":
        method = (f"Test: NHANES {TEST_CYCLE} cycle only. Validation: stratified 20% of the earlier cycles "
                  "(random_state=42). Training: the rest of the earlier cycles.")
    else:
        method = ("Stratified random split, 64% train / 16% validation / 20% test (random_state=42). "
                  "BRFSS 2022 is a single survey year.")

    def counts(y):
        return {"rows": int(len(y)), "positive": int(y.sum()), "positive_rate": round(float(y.mean()), 4)}

    return {"method": method, "train": counts(y_train), "validation": counts(y_val), "test": counts(y_test)}


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def get_candidate_models():
    """
    Return candidate classifiers. No class re-weighting: with class_weight='balanced' the scores were
    far above the observed rates (heart: about 76% shown where about 17% had the outcome).
    """

    models = {
        # Kept small (a benchmark only): deeper forests made 65-73 MB files, too big for GitHub and Streamlit Cloud.
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1,
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "SVM": SVC(
            probability=True,
            random_state=42,
        ),

        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )

    return models


def subsample_for_svm(
    X_train,
    y_train,
    max_samples=SVM_MAX_TRAIN_SAMPLES,
):
    if len(X_train) <= max_samples:
        return X_train, y_train

    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_samples,
        stratify=y_train,
        random_state=42,
    )

    return X_sub, y_sub


def calculate_metrics(
    y_true,
    y_pred,
    y_proba=None,
):
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    negatives = np.asarray(y_true) == 0

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_disease": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_disease": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float((np.asarray(y_pred)[negatives] == 0).mean()), 4) if negatives.any() else 0.0,
        "f1_disease": round(float(f1_per_class[1]), 4),
        "f1_no_disease": round(float(f1_per_class[0]), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f2_disease": round(float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)), 4),
        "roc_auc": None,
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)

    return metrics


def find_best_threshold(
    y_true,
    y_proba,
):
    """
    Find the probability threshold that maximises Youden's J (sensitivity + specificity - 1) on validation
    data (ties go to the lower threshold). F2 is kept in each row for the analytics chart and reports.
    """

    rows = []
    y_true = np.asarray(y_true)

    for threshold in THRESHOLD_GRID:
        y_pred = (y_proba >= threshold).astype(int)
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        negatives = y_true == 0
        specificity = float((y_pred[negatives] == 0).mean()) if negatives.any() else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "youden": round(recall + specificity - 1, 4),
                "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
            }
        )

    best_row = max(rows, key=lambda row: row["youden"])

    return best_row["threshold"], rows, best_row


def select_model(validation_auc):
    """Logistic Regression (see SELECTION_RULE); the best family is only used when it is missing."""
    if "LogisticRegression" in validation_auc:
        return "LogisticRegression"
    return max(validation_auc, key=validation_auc.get)


def calibration_check(y_true, y_proba, threshold):
    """Mean model score against the observed rate, overall and among the rows the threshold flags."""
    y_true = np.asarray(y_true)
    flagged = y_proba >= threshold
    return {
        "threshold": threshold,
        "rows": int(len(y_true)),
        "mean_predicted": round(float(y_proba.mean()), 4),
        "observed_rate": round(float(y_true.mean()), 4),
        "flagged_share": round(float(flagged.mean()), 4),
        "mean_predicted_flagged": round(float(y_proba[flagged].mean()), 4) if flagged.any() else None,
        "observed_rate_flagged": round(float(y_true[flagged].mean()), 4) if flagged.any() else None,
        "mean_predicted_not_flagged": round(float(y_proba[~flagged].mean()), 4) if (~flagged).any() else None,
        "observed_rate_not_flagged": round(float(y_true[~flagged].mean()), 4) if (~flagged).any() else None,
    }


# =============================================================================
# Core training function
# =============================================================================

def train_and_benchmark(disease):
    config = DISEASE_CONFIGS[disease]
    feature_columns, target_column = config["features"], config["target"]

    print("\n" + "=" * 72)
    print(f"BENCHMARKING MODELS FOR: {disease.upper()}")
    print("=" * 72)

    df = load_training_frame(disease)
    y = df[target_column]
    X = df[feature_columns]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Samples: {len(df):,} | Positive: {(y == 1).sum():,} | Negative: {(y == 0).sum():,}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, disease)
    split = describe_split(disease, y_train, y_val, y_test)
    print(split["method"])
    for part in ("train", "validation", "test"):
        info = split[part]
        print(f"  {part:<10} rows {info['rows']:>7,}  positive {info['positive']:>6,} ({info['positive_rate']:.1%})")
    print()

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    results = []                 # validation metrics at the default 0.5 cut-off
    validation_auc = {}
    model_thresholds = {}        # each family's validation threshold (Youden's J)
    threshold_tables = {}
    test_probabilities = {}      # recorded for reporting only; never used to choose anything

    print(f"{'Model':<20}{'Val AUC':>9}{'Val J*':>9}{'Thr*':>7}{'Test AUC':>10}{'Sens':>8}{'Spec':>8}{'Prec':>8}")
    print("-" * 79)

    for name, classifier in get_candidate_models().items():

        # Each candidate gets its own copy of the preprocessor: sklearn Pipelines
        # do not clone steps, so sharing one ColumnTransformer would let a later
        # fit (e.g. on the SVM subsample) silently overwrite an earlier model's.
        pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("classifier", classifier)])

        if name == "SVM":
            X_fit, y_fit = subsample_for_svm(X_train, y_train)
            if len(X_fit) < len(X_train):
                print(f"(SVM training subsample: {len(X_fit):,})")
        else:
            X_fit, y_fit = X_train, y_train

        pipeline.fit(X_fit, y_fit)

        # Save every benchmark pipeline so the app can let users/doctors compare models.
        model_file_name = name.lower().replace(" ", "_")
        with open(ALL_MODELS_DIR / f"{disease}_{model_file_name}.pkl", "wb") as candidate_file:
            pickle.dump(pipeline, candidate_file)

        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, (y_proba_val >= 0.5).astype(int), y_proba_val)
        metrics["model"] = name
        results.append(metrics)
        validation_auc[name] = metrics["roc_auc"]

        threshold, rows, best_row = find_best_threshold(y_val, y_proba_val)
        model_thresholds[name] = threshold
        threshold_tables[name] = (rows, best_row)

        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        test_probabilities[name] = y_proba_test
        test = calculate_metrics(y_test, (y_proba_test >= threshold).astype(int), y_proba_test)

        print(f"{name:<20}{metrics['roc_auc']:>9.4f}{best_row['youden']:>9.4f}{threshold:>7.2f}"
              f"{test['roc_auc']:>10.4f}{test['recall_disease']:>8.4f}{test['specificity']:>8.4f}"
              f"{test['precision_disease']:>8.4f}")

    print("-" * 79)
    print("* Youden's J and threshold tuned on the validation set; test columns use that threshold.")

    # -------------------------------------------------------------------------
    # Selection and threshold (validation set only)
    # -------------------------------------------------------------------------

    best_name = select_model(validation_auc)
    decision_threshold = model_thresholds[best_name]
    threshold_rows, best_threshold_row = threshold_tables[best_name]
    top = max(validation_auc, key=validation_auc.get)

    print(f"\nBest validation AUC: {top} ({validation_auc[top]:.4f})")
    print(f"Selected model: {best_name} (validation AUC {validation_auc[best_name]:.4f})")
    print(f"Chosen threshold: {decision_threshold} | validation precision {best_threshold_row['precision']} "
          f"| recall {best_threshold_row['recall']} | specificity {best_threshold_row['specificity']}")

    # -------------------------------------------------------------------------
    # Test-set results for every family (reporting only)
    # -------------------------------------------------------------------------

    test_results = []
    for name, proba in test_probabilities.items():
        threshold = model_thresholds[name]
        row = calculate_metrics(y_test, (proba >= threshold).astype(int), proba)
        row.update({"model": name, "threshold": threshold, "flagged_share": round(float((proba >= threshold).mean()), 4)})
        test_results.append(row)

    # -------------------------------------------------------------------------
    # FINAL test-set evaluation of the selected model
    # -------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print(f"FINAL TEST RESULTS: {disease.upper()}")
    print("=" * 72)

    y_proba_test = test_probabilities[best_name]
    y_pred_test = (y_proba_test >= decision_threshold).astype(int)
    final_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
    calibration = calibration_check(y_test, y_proba_test, decision_threshold)

    print(f"\nBest model: {best_name}")
    print(f"Decision threshold: {decision_threshold}")
    for key in ("accuracy", "precision_disease", "recall_disease", "specificity", "f1_disease", "f2_disease",
                "roc_auc"):
        print(f"{key}: {final_metrics[key]}")
    print(f"Calibration (test): mean score {calibration['mean_predicted']} vs observed {calibration['observed_rate']}; "
          f"flagged {calibration['flagged_share']:.1%}: mean score {calibration['mean_predicted_flagged']} vs "
          f"observed {calibration['observed_rate_flagged']}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["No Disease", "Disease"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    with open(ALL_MODELS_DIR / f"{disease}_{best_name.lower()}.pkl", "rb") as file:
        best_pipeline = pickle.load(file)
    model_path = MODELS_DIR / f"{disease}_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(best_pipeline, file)
    print(f"\nSaved model -> {model_path}")

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    return {
        "features": feature_columns,
        "target": target_column,
        "data_source": config["data_source"],
        "label_definition": config["label_definition"],
        "population": config["population"],
        "split": split,
        "n_samples": len(df),
        "positive_cases": int((y == 1).sum()),
        "negative_cases": int((y == 0).sum()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "selection_rule": SELECTION_RULE,
        "best_validation_score": round(float(validation_auc[best_name]), 4),
        "logistic_auc_gap": round(float(max(validation_auc.values()) - validation_auc[best_name]), 4),
        "decision_threshold": decision_threshold,
        "model_thresholds": model_thresholds,
        "validation_model_results": results,
        "threshold_results": threshold_rows,
        "final_test_metrics": final_metrics,
        "test_model_results": test_results,
        "calibration_test": calibration,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    metadata = {}

    for disease in DISEASE_CONFIGS:
        print(f"\nLoading {disease} dataset...")
        metadata[disease] = train_and_benchmark(disease)

    # The partial ADA risk score on the same diabetes test rows (scripts/build_nhanes_training.py).
    if NHANES_SUMMARY.exists():
        summary = json.loads(NHANES_SUMMARY.read_text())
        ada = summary.get("diabetes", {}).get("ada_comparison")
        if ada and ada.get("test_rows") == metadata["diabetes"]["split"]["test"]["rows"]:
            metadata["diabetes"]["ada_comparison"] = ada

    # =========================================================================
    # Save metadata
    # =========================================================================

    metadata_path = MODELS_DIR / "model_meta.json"
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)
    print(f"\nMetadata saved -> {metadata_path}")

    # =========================================================================
    # Final summary
    # =========================================================================

    print("\n" + "=" * 72)
    print("ALL DISEASES BENCHMARKED SUCCESSFULLY")
    print("=" * 72)

    for disease, info in metadata.items():
        test = info["final_test_metrics"]
        print(
            f"{disease.capitalize():<10} best model: {info['best_model']:<20} validation AUC="
            f"{info['best_validation_score']:.4f} threshold={info['decision_threshold']} | test AUC="
            f"{test['roc_auc']:.4f} sens={test['recall_disease']:.4f} spec={test['specificity']:.4f} "
            f"prec={test['precision_disease']:.4f}"
        )
    if "ada_comparison" in metadata["diabetes"]:
        print(f"Diabetes partial ADA score test AUC: {metadata['diabetes']['ada_comparison']['roc_auc']:.4f}")

    print("=" * 72)
