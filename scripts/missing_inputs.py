"""
scripts/missing_inputs.py
=========================
How each saved model performs when a patient leaves optional measurements blank.

Diabetes lab values and most kidney measurements are optional in the screening form. A blank
value is filled by each pipeline's median imputer, which shifts scores: with both diabetes labs
missing, the recommended model flags only about 1 in 20 people with diabetes at its usual
threshold. For every saved model and every pattern of blank inputs the form allows, this script
reproduces the split train_model.py uses and records, on the held-out test split, recall,
specificity, precision and AUC at the threshold the app will apply, plus the same figures with
every value entered for comparison.

Thresholds
- Diabetes: chosen on the validation split by maximising F2 (the recall-weighted score already
  used for the heart threshold), never above the model's usual threshold, ties going to the
  higher value. A blank lab can only make a flag easier to get, not harder.
- Kidney: the usual threshold. About 92% of the kidney dataset has the disease, so F2 would
  simply flag everyone, and the validation split has too few negatives to tune on. The app
  shows the measured figures instead, including when the model flags almost everyone.

Kidney patterns leave out creatinine, eGFR and ACR: the app asks for those before it gives a
kidney estimate (ahead.config.REQUIRED_FOR_SCORE).

Run from the project root after train_model.py:
    python scripts/missing_inputs.py
Output: models/missing_inputs.json
"""

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahead.config import OPTIONAL_INPUTS, REQUIRED_FOR_SCORE  # noqa: E402
from train_model import (  # noqa: E402
    ALL_MODELS_DIR, DATA_DIR, DIABETES_FEATURES, KIDNEY_FEATURES, MODELS_DIR, clean_binary_target,
)
import pandas as pd  # noqa: E402

DATASETS = {"diabetes": ("diabetes.csv", DIABETES_FEATURES), "kidney": ("kidney_disease.csv", KIDNEY_FEATURES)}
TUNED = {"diabetes"}
THRESHOLDS = np.round(np.arange(0.02, 0.96, 0.01), 2)


def evaluate(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    flagged = proba >= threshold
    positives = y_true == 1
    return {
        "recall": round(float(flagged[positives].mean()), 4),
        "specificity": round(float((~flagged[~positives]).mean()), 4),
        "precision": round(float(positives[flagged].mean()), 4) if flagged.any() else 0.0,
        "flagged": round(float(flagged.mean()), 4),
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def pick_threshold(y_val, proba, usual):
    """Highest F2-maximising threshold at or below the usual one."""
    grid = THRESHOLDS[THRESHOLDS <= usual + 1e-9]
    scores = np.array([fbeta_score(y_val, (proba >= t).astype(int), beta=2, zero_division=0) for t in grid])
    return float(grid[np.flatnonzero(np.isclose(scores, scores.max()))[-1]])


def patterns(disease):
    fields = [f for f in OPTIONAL_INPUTS[disease] if f not in REQUIRED_FOR_SCORE.get(disease, [])]
    return [list(c) for r in range(1, len(fields) + 1) for c in combinations(fields, r)]


def main():
    output = {"method": (
        "Held-out test split of train_model.py (20%, random_state=42, stratified), with the listed inputs set to "
        "missing so each pipeline's median imputer fills them. Diabetes thresholds: F2-maximising on the validation "
        "split, capped at the model's usual threshold. Kidney: usual threshold. '' is the result with every value "
        "entered at the usual threshold.")}

    for disease, (csv, feature_list) in DATASETS.items():
        meta = json.loads((MODELS_DIR / "model_meta.json").read_text())[disease]
        df = pd.read_csv(DATA_DIR / csv)
        df = df[feature_list + [meta["target"]]].dropna(subset=[meta["target"]])
        df[meta["target"]] = clean_binary_target(df[meta["target"]])
        X, y = df[feature_list], df[meta["target"]]
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        _, X_val, _, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42,
                                              stratify=y_train_val)
        output[disease] = {"test_counts": {"positive": int(y_test.sum()), "negative": int((y_test == 0).sum())},
                           "models": {}}

        for row in meta["validation_model_results"]:
            name = row["model"]
            path = ALL_MODELS_DIR / f"{disease}_{name.lower().replace(' ', '_')}.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as file:
                pipeline = pickle.load(file)
            usual = float(meta.get("decision_threshold", 0.5)) if name == meta["best_model"] else 0.5
            entries = {"": {"threshold": usual, "test": evaluate(y_test, pipeline.predict_proba(X_test)[:, 1], usual)}}
            for missing in patterns(disease):
                val, test = X_val.copy(), X_test.copy()
                val[missing] = np.nan
                test[missing] = np.nan
                threshold = usual
                if disease in TUNED:
                    threshold = pick_threshold(y_val, pipeline.predict_proba(val)[:, 1], usual)
                entries["+".join(sorted(missing))] = {
                    "threshold": threshold, "test": evaluate(y_test, pipeline.predict_proba(test)[:, 1], threshold)}
            output[disease]["models"][name] = entries
            print(f"{disease:9} {name:20} {len(entries) - 1} patterns, usual threshold {usual:.2f}")

    path = MODELS_DIR / "missing_inputs.json"
    path.write_text(json.dumps(output, separators=(",", ":")))
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
