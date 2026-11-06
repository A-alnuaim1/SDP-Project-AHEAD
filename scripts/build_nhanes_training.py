"""
scripts/build_nhanes_training.py
================================
Build the diabetes and kidney training files from the NHANES research file.

Source: data/nhanes/nhanes_1999_2023_research.csv.gz, built from the raw CDC NHANES public-use files by scripts/rebuild_nhanes.py
(1999-2023, adults 18+; run scripts/fetch_nhanes.py first). See docs/data_sources.md.

Outputs (columns are fixed; the app and train_model.py rely on them):
    data/nhanes_diabetes.csv
        adults 18+ never told they have diabetes (borderline or unknown left out), HbA1c measured.
        cycle, gender, age, bmi, smoked_100, hypertension, heart_disease, hba1c_57_or_higher
    data/nhanes_kidney.csv
        adults 18+ with both eGFR and urine albumin-to-creatinine ratio measured.
        cycle, gender, age, bmi, smoked_100, physically_active, hypertension, diabetes, heart_disease,
        kidney_marker
0/1 columns are blank when the participant did not answer. Rows with no exam data at all
(interview-only participants) are dropped first.

It also writes data/nhanes/build_summary.json: row counts and positive rates per split, and the
partial ADA risk score (Bang et al. 2009) on the diabetes test rows, which train_model.py copies
into models/model_meta.json for comparison.

Run from the project root:
    python scripts/build_nhanes_training.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "nhanes" / "nhanes_1999_2023_research.csv.gz"
DIABETES_OUT = ROOT / "data" / "nhanes_diabetes.csv"
KIDNEY_OUT = ROOT / "data" / "nhanes_kidney.csv"
SUMMARY_OUT = ROOT / "data" / "nhanes" / "build_summary.json"

TEST_CYCLE = "2021-2023"   # the held-out test set in train_model.py
EXAM_COLUMNS = ["bmi", "glucose_mg_dl", "hba1c_pct", "cholesterol_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
                "uacr_mg_g", "egfr_calculated"]

DIABETES_COLUMNS = ["cycle", "gender", "age", "bmi", "smoked_100", "hypertension", "heart_disease",
                    "hba1c_57_or_higher"]
KIDNEY_COLUMNS = ["cycle", "gender", "age", "bmi", "smoked_100", "physically_active", "hypertension", "diabetes",
                  "heart_disease", "kidney_marker"]


def yes_no(series: pd.Series, yes: str, no: str) -> pd.Series:
    """Map two answer texts to 1 / 0; anything else (including blank) stays missing."""
    unknown = set(series.dropna().unique()) - {yes, no}
    if unknown:
        raise ValueError(f"Unexpected answers in {series.name}: {sorted(unknown)}")
    return series.map({yes: 1, no: 0}).astype("Int64")


def zero_one(series: pd.Series) -> pd.Series:
    unknown = set(series.dropna().unique()) - {0, 1}
    if unknown:
        raise ValueError(f"Unexpected values in {series.name}: {sorted(unknown)}")
    return series.astype("Int64")


def load_source() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE)
    no_exam = raw[EXAM_COLUMNS].isna().all(axis=1)
    duplicates = raw.duplicated(keep=False)
    print(f"Source rows: {len(raw):,}; dropped {int(no_exam.sum()):,} with no exam data, including all "
          f"{int(duplicates.sum()):,} exact-duplicate rows ({int((duplicates & ~no_exam).sum())} duplicates have exam data)")
    raw = raw[~no_exam & (raw["age"] >= 18)].copy()

    frame = pd.DataFrame({
        "cycle": raw["cycle"],
        "gender": raw["sex"],
        "age": raw["age"].round().astype("Int64"),
        "bmi": raw["bmi"].round(2),
        "smoked_100": yes_no(raw["smoking"], "Ever smoked ≥100 cigarettes", "Never smoked ≥100 cigarettes"),
        "physically_active": yes_no(raw["activity"], "Leisure activity reported", "No leisure activity reported"),
        "hypertension": yes_no(raw["hypertension"], "Previously told: yes", "Previously told: no"),
        "diabetes": zero_one(raw["diabetes_history"]),
        "heart_disease": zero_one(raw["heart_history"]),
        "hba1c": raw["hba1c_pct"],
        "kidney_marker": zero_one(raw["kidney_marker"]),
    })
    if set(frame["gender"].unique()) - {"Female", "Male"}:
        raise ValueError(f"Unexpected sex values: {sorted(frame['gender'].unique())}")
    return frame


def ada_points(frame: pd.DataFrame) -> pd.Series:
    """
    Partial ADA diabetes risk test (Bang et al., Ann Intern Med 2009): age, sex, high blood pressure,
    physical activity and BMI. Family history and gestational diabetes are not in the extract, so the
    maximum here is 9 of the full 11 points. A blank answer scores 0.
    """
    age = pd.cut(frame["age"].astype(float), [-np.inf, 40, 50, 60, np.inf], right=False, labels=[0, 1, 2, 3])
    bmi = pd.cut(frame["bmi"], [-np.inf, 25, 30, 40, np.inf], right=False, labels=[0, 1, 2, 3])
    return (age.astype(float).fillna(0) + bmi.astype(float).fillna(0)
            + (frame["gender"] == "Male").astype(int)
            + frame["hypertension"].fillna(0).astype(int)
            + (frame["physically_active"] == 0).fillna(False).astype(int))


def split_counts(frame: pd.DataFrame, target: str) -> dict:
    counts = {}
    for name, rows in (("train_and_validation", frame[frame["cycle"] != TEST_CYCLE]),
                       ("test", frame[frame["cycle"] == TEST_CYCLE]), ("all", frame)):
        counts[name] = {"rows": int(len(rows)), "positive": int(rows[target].sum()),
                        "positive_rate": round(float(rows[target].mean()), 4)}
        print(f"  {name:21} rows {len(rows):>6,}  positive {int(rows[target].sum()):>6,} "
              f"({100 * rows[target].mean():.1f}%)")
    return counts


def main() -> None:
    frame = load_source()
    summary = {"source": str(SOURCE.relative_to(ROOT)), "test_cycle": TEST_CYCLE}

    diabetes = frame[(frame["diabetes"] == 0) & frame["hba1c"].notna()].copy()
    diabetes["hba1c_57_or_higher"] = (diabetes["hba1c"] >= 5.7).astype(int)
    print(f"\nDiabetes ({DIABETES_OUT.name}): never told diabetes, HbA1c measured")
    summary["diabetes"] = {"splits": split_counts(diabetes, "hba1c_57_or_higher")}
    test = diabetes[diabetes["cycle"] == TEST_CYCLE]
    ada = ada_points(test)
    summary["diabetes"]["ada_comparison"] = {
        "score": "Partial ADA risk test points (Bang et al. 2009): age, sex, high blood pressure, physical "
                 "activity, BMI. No family history or gestational diabetes; blank answers score 0.",
        "test_rows": int(len(test)),
        "roc_auc": round(float(roc_auc_score(test["hba1c_57_or_higher"], ada)), 4),
    }
    print(f"  partial ADA score test AUC (HbA1c >= 5.7%): {summary['diabetes']['ada_comparison']['roc_auc']:.4f}")
    diabetes[DIABETES_COLUMNS].to_csv(DIABETES_OUT, index=False)

    kidney = frame[frame["kidney_marker"].notna()].copy()
    print(f"\nKidney ({KIDNEY_OUT.name}): eGFR and urine ACR measured")
    summary["kidney"] = {"splits": split_counts(kidney, "kidney_marker")}
    kidney[KIDNEY_COLUMNS].to_csv(KIDNEY_OUT, index=False)

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {DIABETES_OUT.relative_to(ROOT)}, {KIDNEY_OUT.relative_to(ROOT)}, {SUMMARY_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
