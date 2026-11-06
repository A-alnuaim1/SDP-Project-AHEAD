"""Validate uploaded records before they can reach the screening pipeline."""

import math
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ahead.config import AGE_FIELD, GENDER_CODED_FIELDS, MODEL_INPUTS, OPTIONAL_INPUTS, YES_NO_FIELDS, pretty_label


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
# The doctor's wording for an optional value left blank, as in Patient Intelligence. Optional values are never model
# inputs any more (config.MODEL_INPUTS), so nothing is filled in for them.
OPTIONAL_BLANK = "Not provided"

# smoking_history -> smoked_100 (NHANES: smoked at least 100 cigarettes in life). The diabetes form's older answers
# 'ever' and 'not current' mean the person has smoked; 'No Info', a blank or anything else is unknown (missing).
SMOKED_100 = {"never": 0, "former": 1, "current": 1, "ever": 1, "not current": 1}
_YES = {"1": 1, "yes": 1, "true": 1, "y": 1, "0": 0, "no": 0, "false": 0, "n": 0, "1.0": 1, "0.0": 0}


def _zero_one(value):
    """A Yes/No answer as 1/0, or NaN when it is blank or unreadable."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return _YES.get(str(value).strip().lower(), np.nan)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def model_inputs(disease, values):
    """
    The model's input row from a screening's answers (config.MODEL_INPUTS): smoked_100 from smoking_history, Yes/No
    answers as 1/0, and a blank as NaN for the pipeline's imputer. Lab values never reach the diabetes or kidney model.
    A condition without MODEL_INPUTS (heart) gets its answers as they are.
    """
    values = values or {}
    if disease not in MODEL_INPUTS:
        return dict(values)
    row = {}
    for feature in MODEL_INPUTS[disease]:
        if feature == "smoked_100":
            row[feature] = SMOKED_100.get(str(values.get("smoking_history") or "").strip().lower(), np.nan)
        elif feature == "gender":
            text = str(values.get("gender") or "").strip().lower()
            row[feature] = {"female": "Female", "male": "Male"}.get(text, np.nan)
        elif feature in YES_NO_FIELDS:
            row[feature] = _zero_one(values.get(feature))
        else:
            row[feature] = _number(values.get(feature))
    return row


def model_frame(disease, rows, columns):
    """One model input row per screening's answers, in the pipeline's column order (resources.features)."""
    return pd.DataFrame([model_inputs(disease, values) for values in rows], columns=columns)


def input_bounds(feature, info):
    """Values a model input accepts: the training range widened by a quarter of its span, so realistic values
    just past the dataset extremes (e.g. an 85-year-old when the data stops at 80) can still be scored. A form
    field that is not a training column (config.FORM_FIELD_SPECS) gives its own bounds."""
    if "bounds" in info:
        return info["bounds"]
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
    age_key, optional = AGE_FIELD[disease], set(OPTIONAL_INPUTS.get(disease, ()))
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
            if feature not in optional:   # an optional value may stay blank
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
            if info.get("binary") and number not in (0, 1):  # NaN too; the record editor shows these fields as words
                issues.append(f"{feature}: enter {'Male or Female' if feature in GENDER_CODED_FIELDS else 'Yes or No'}.")
            elif not math.isfinite(number):
                issues.append(f"{feature}: enter a number.")
            elif info.get("integer") and not number.is_integer():
                issues.append(f"{feature}: enter a whole number.")
            elif not low <= number <= high:
                issues.append(f"{feature}: {number:g} is too far outside the training range for the model to score; "
                              f"it accepts {low:g} to {high:g}.")
            clean[feature] = int(number) if info["integer"] and number.is_integer() else number
        else:
            options = {str(option).strip().lower(): option for option in info.get("accepted", info["options"])}
            clean[feature] = options.get(str(value).strip().lower())
            if clean[feature] is None:
                issues.append(f"{feature}: choose a category from the training data.")
        if len(issues) > found:
            # Keep a failing value exactly as entered so the doctor sees and corrects it; never a truncated guess.
            clean[feature] = entered
    return clean, issues


def issue_text(issue):
    """A stored issue as the doctor reads it. Issues keep the 'feature: message.' form (the record editor and the
    start-up repair parse it); on screen the feature gets its label and the full stop goes, so lists join cleanly."""
    feature, sep, message = issue.partition(": ")
    return (f"{pretty_label(feature)}: {message}" if sep and re.fullmatch(r"\w+", feature) else issue).rstrip(". ")


def shown_value(feature, value, info, blank="Not provided"):
    """A model input as the doctor reads it: Male/Female and No/Yes like the patient form, sentence-case
    categories and numbers to two decimals. A value that failed validation is shown exactly as entered;
    a blank reads *blank* (OPTIONAL_BLANK for an optional lab the model filled in)."""
    if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, float) and math.isnan(value)):
        return blank
    if info["kind"] == "category":
        text = str(value).strip()
        known = text.lower() in {str(option).strip().lower() for option in info.get("accepted", info["options"])}
        return text[:1].upper() + text[1:].lower() if known else str(value)
    try:
        number = float(value)
    except (ValueError, TypeError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if (info.get("binary") or feature in GENDER_CODED_FIELDS | YES_NO_FIELDS) and number in (0, 1):
        return (("Male", "Female") if feature in GENDER_CODED_FIELDS else ("No", "Yes"))[int(number)]
    return f"{round(number, 2) + 0.0:.2f}".rstrip("0").rstrip(".")  # + 0.0 turns -0.0 into 0.0


def field_info(stats, feature):
    """The statistics for a field, or plain ones for a key the current form no longer has (an older saved record's
    config.LEGACY_FIELDS), so it still displays: words for 0/1 answers, numbers as entered."""
    return stats.get(feature) or {"kind": "number", "binary": feature in GENDER_CODED_FIELDS | YES_NO_FIELDS,
                                  "integer": False, "options": []}


def prepare_upload(frame, disease, required, stats, id_columns, reference_date=None):
    """Read rows without requiring age at upload; every other column the form asks for must exist, except the optional
    ones (config.OPTIONAL_INPUTS), which read as blank when absent."""
    age_key, optional = AGE_FIELD[disease], set(OPTIONAL_INPUTS.get(disease, ()))
    absent = [f for f in required if f not in frame.columns and f != age_key and f not in optional]
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
