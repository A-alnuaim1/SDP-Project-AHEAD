"""
scripts/missing_inputs.py
=========================
How each saved model performs when a patient leaves an optional model input blank.

A blank model input is filled by the pipeline's median imputer, which shifts scores. For every saved
model and every pattern of blank inputs the form allows, this script rebuilds the split train_model.py
uses and records, on the held-out test set, recall, specificity, precision and AUC at the threshold the
app will apply, plus the same figures with every value entered for comparison.

Only optional inputs that are also model inputs are measured. Since the switch to NHANES data
(docs/data_sources.md) the diabetes and kidney models take no lab values: HbA1c, glucose, creatinine,
eGFR and urine ACR are checked by rules only, so there is nothing to measure and those conditions get an
empty "models" entry (the app then finds no profile). Heart's optional inputs (the past heart attack and
angina questions) only sort the result and are not model inputs either.

Threshold for a blank pattern: chosen on the validation set by maximising F2, never above the model's
usual threshold, ties going to the higher value. A blank input can only make a flag easier to get.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahead.config import OPTIONAL_INPUTS  # noqa: E402
from train_model import ALL_MODELS_DIR, DISEASE_CONFIGS, MODELS_DIR, load_training_frame, split_data  # noqa: E402

THRESHOLDS = np.round(np.arange(0.01, 0.96, 0.01), 2)


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


def optional_model_inputs(disease, features):
    """Optional form fields that are model inputs."""
    return [f for f in OPTIONAL_INPUTS.get(disease, []) if f in features]


def main():
    output = {"method": (
        "Held-out test set of train_model.py, with the listed optional model inputs set to missing so each "
        "pipeline's median imputer fills them. Thresholds: F2-maximising on the validation set, capped at the "
        "model's usual threshold. '' is the result with every value entered at the usual threshold. A condition "
        "with no optional model inputs has no models listed.")}
    all_meta = json.loads((MODELS_DIR / "model_meta.json").read_text())

    for disease in DISEASE_CONFIGS:
        meta = all_meta[disease]
        fields = optional_model_inputs(disease, meta["features"])
        output[disease] = {"optional_model_inputs": fields, "models": {}}
        if not fields:
            print(f"{disease:9} no optional model inputs: nothing to measure")
            continue

        _, X_val, X_test, _, y_val, y_test = split_data(load_training_frame(disease), disease)
        output[disease]["test_counts"] = {"positive": int(y_test.sum()), "negative": int((y_test == 0).sum())}
        blank_patterns = [list(c) for r in range(1, len(fields) + 1) for c in combinations(fields, r)]

        for row in meta["validation_model_results"]:
            name = row["model"]
            path = ALL_MODELS_DIR / f"{disease}_{name.lower().replace(' ', '_')}.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as file:
                pipeline = pickle.load(file)
            # The app applies the tuned threshold to the recommended model only (ahead.resources.decision_threshold).
            usual = float(meta.get("decision_threshold", 0.5)) if name == meta["best_model"] else 0.5
            entries = {"": {"threshold": usual, "test": evaluate(y_test, pipeline.predict_proba(X_test)[:, 1], usual)}}
            for missing in blank_patterns:
                val, test = X_val.copy(), X_test.copy()
                val[missing] = np.nan
                test[missing] = np.nan
                threshold = pick_threshold(y_val, pipeline.predict_proba(val)[:, 1], usual)
                entries["+".join(sorted(missing))] = {
                    "threshold": threshold, "test": evaluate(y_test, pipeline.predict_proba(test)[:, 1], threshold)}
            output[disease]["models"][name] = entries
            print(f"{disease:9} {name:20} {len(entries) - 1} patterns, usual threshold {usual:.2f}")

    path = MODELS_DIR / "missing_inputs.json"
    path.write_text(json.dumps(output, separators=(",", ":")))
    print(f"Saved {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
