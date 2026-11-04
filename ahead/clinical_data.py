"""Validate uploaded records before they can reach the screening pipeline."""

import math
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ahead.config import GENDER_CODED_FIELDS


def age_from_dob(value, today=None):
    if value is None or str(value).strip() == "":
        return None
    day = date.fromisoformat(str(value)[:10])
    today = today or datetime.now(ZoneInfo('Asia/Dubai')).date()
    years = today.year - day.year - ((today.month, today.day) < (day.month, day.day))
    if not 0 <= years <= 120:
        raise ValueError("Date of birth must describe an age from 0 to 120.")
    return years


def age_category(age, choices):
    for choice in choices:
        numbers = [int(n) for n in re.findall(r"\d+", str(choice))]
        if len(numbers) == 2 and numbers[0] <= age <= numbers[1]:
            return choice
        if len(numbers) == 1 and age >= numbers[0] and ("older" in str(choice).lower() or "+" in str(choice)):
            return choice
    raise ValueError("Date of birth does not match the model's age categories.")


def dob_age(dob, disease, stats, reference_date=None):
    """The age model input a DOB gives (years, or the matching AgeCategory for heart); None without a DOB."""
    age = age_from_dob(dob, reference_date)
    return age_category(age, stats["AgeCategory"]["options"]) if age is not None and disease == "heart" else age


# Hard limits the widened range never crosses: day counts cover the past 30 days, sleep is hours per day,
# and a BMI or serum creatinine near zero is an entry error, not a patient.
PHYSICAL_LIMITS = {"physicalhealthdays": (0.0, 30.0), "mentalhealthdays": (0.0, 30.0), "sleephours": (1.0, 24.0),
                   "bmi": (10.0, math.inf), "serumcreatinine": (0.2, math.inf)}


def input_bounds(feature, info):
    """Values a model input accepts: the training range widened by a quarter of its span, so realistic values
    just past the dataset extremes (e.g. an 85-year-old when the data stops at 80) can still be scored."""
    span = max(info["max"] - info["min"], 1.0)
    floor, cap = PHYSICAL_LIMITS.get(feature.lower(), (-math.inf, math.inf))
    low = info["min"] - 0.25 * span if info["min"] < 0 else max(0.0, info["min"] - 0.25 * span)
    low, high = max(low, floor), min(info["max"] + 0.25 * span, cap)
    if feature.lower() == "age" or info["integer"]:
        return int(math.floor(low)), int(math.ceil(high))
    return round(low, 2), round(high, 2)


def validate_row(values, disease, required, stats, dob=None, reference_date=None):
    """Return normalized model inputs and actionable issues; never silently score an invalid row."""
    clean, issues = {}, []
    age_key = "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    try:
        derived = dob_age(dob, disease, stats, reference_date)
        if derived is not None:
            supplied, consistent = values.get(age_key), True
            if supplied is not None and not pd.isna(supplied) and str(supplied).strip():
                try:
                    consistent = str(supplied).strip().lower() == str(derived).strip().lower() if disease == 'heart' else int(float(supplied)) == derived
                except (ValueError, TypeError, OverflowError):
                    consistent = False
                if not consistent:
                    issues.append(f'{age_key}: entered age conflicts with DOB; verify both.')
            if consistent:  # a conflicting entered age stays visible next to the DOB until one is corrected
                values[age_key] = derived
    except (ValueError, TypeError):
        issues.append("Invalid date of birth; use YYYY-MM-DD.")
    for feature in required:
        value = values.get(feature)
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            issues.append(f"{feature}: missing; enter a value before scoring.")
            clean[feature] = None
            continue
        info, entered, found = stats[feature], value, len(issues)
        if info["kind"] == "number":
            if isinstance(value, str) and info.get("binary"):
                mapping = {"yes": 1, "true": 1, "y": 1, "no": 0, "false": 0, "n": 0}
                if feature in GENDER_CODED_FIELDS:
                    mapping.update({"male": 0, "m": 0, "female": 1, "f": 1})
                value = mapping.get(value.strip().lower(), value)
            try:
                number = float(value)
            except (ValueError, TypeError):
                number = float("nan")
            low, high = input_bounds(feature, info)
            if not math.isfinite(number):
                issues.append(f"{feature}: enter a number.")
            elif info.get("integer") and not number.is_integer():
                issues.append(f"{feature}: enter a whole number.")
            elif info.get("binary") and number not in (0, 1):
                issues.append(f"{feature}: enter 0 or 1.")
            elif not low <= number <= high:
                issues.append(f"{feature}: {number:g} is too far outside the training range for the model to score; "
                              f"it accepts {low:g} to {high:g}.")
            clean[feature] = int(number) if info["integer"] and number.is_integer() else number
        else:
            options = {str(option).strip().lower(): option for option in info["options"]}
            clean[feature] = options.get(str(value).strip().lower())
            if clean[feature] is None:
                issues.append(f"{feature}: choose a category from the training data.")
        if len(issues) > found:
            # Keep a failing value exactly as entered so the doctor sees and corrects it; never a truncated guess.
            clean[feature] = entered
    return clean, issues


def prepare_upload(frame, disease, required, stats, id_columns, reference_date=None):
    """Read rows without requiring age at upload; all other model columns must exist."""
    age_key = "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    absent = [f for f in required if f not in frame.columns and f != age_key]
    if absent:
        raise ValueError("Missing model columns: " + ", ".join(absent))
    id_column = next((c for c in id_columns if c in frame.columns), None)
    dob_column = next((c for c in ("DOB", "dob", "DateOfBirth", "date_of_birth", "Date of Birth") if c in frame.columns), None)
    output = []
    for index, (_, row) in enumerate(frame.iterrows(), 1):
        dob = row[dob_column] if dob_column else None
        dob = None if dob is None or pd.isna(dob) else str(dob).strip()[:10] or None
        values = {f: row[f] if f in frame.columns else None for f in required}
        values = {f: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for f, v in values.items()}
        clean, issues = validate_row(values, disease, required, stats, dob, reference_date)
        patient = str(row[id_column]).strip() if id_column and pd.notna(row[id_column]) else ""
        output.append({"patient_id": (patient or f"Row {index}")[:100], "values": clean, "dob": dob, "issues": issues})
    return output
