"""
ahead/risk_plan.py
==================
"Path to lower risk", step 1: the engine only, no UI. For a flagged screening it looks for the smallest
realistic set of changes to modifiable inputs that brings the model's score below the threshold the
Screenings page applies. Patients work on habits (weight, smoking, activity, sleep, alcohol, diet);
doctors set lab and vital-sign targets.

Everything here is the model's score for changed inputs. It is not a guarantee, a forecast of what will
happen to the person, or a causal effect: the models learnt associations in their training data, and a
lower score after a change only says how the trained model reacts to those values.

The lever rules in LEVERS are prototype assumptions for a clinician to review. They are not guideline citations;
every category, unit and range below was checked against feature_summary() and the datasets. The gate cut-offs
in GATES are the urgent rules in ahead/triage.py, the one table the Screenings track and this engine share.
The UI does not use this engine: the study it ran (scripts/evaluate_risk_plans.py, docs/risk_plan_evaluation.md)
is why AHEAD sorts results into tracks instead of offering model-chosen plans.
"""

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd

from ahead.config import OPTIONAL_INPUTS, REQUIRED_FOR_SCORE, pretty_label
from ahead.resources import (
    best_model, decision_threshold, feature_summary, features, load_pipeline, missing_inputs_profile,
)
from ahead.screening import _blank, _model_frame
from ahead.triage import ADULTS_ONLY, RULES, Rule

OWNERS = ("patient", "doctor")
# Each numeric lever is tried at these shares of its allowed change; a category lever is one transition.
LEVEL_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
# Above this many level combinations for one plan size, only subsets that cross at full strength are shrunk.
MAX_BATCH_ROWS = 20000
# Best-reduction paths whose scores are within 0.1 point of the lowest count as equally good (fewer changes win).
SAME_SCORE = 0.001
# A best reduction smaller than 1 percentage point is not offered as a plan: it is noise, not a reason to change
# a habit. A prototype value for clinician review.
MIN_DROP = 0.01
# Score differences below this are floating-point noise between batch and single-row scoring, not an effect.
TINY = 1e-9

CAVEAT = ("These are the model's scores for changed values, not a guarantee or a causal effect. The model learnt "
          "patterns in past data; a lower score after a change does not show that making the change would lower "
          "this person's risk.")


# =============================================================================
# LEVER TABLE
# =============================================================================

@dataclass(frozen=True)
class Lever:
    """
    One modifiable model input and the rule a plan must follow when changing it.

    kind: "down" or "up" (one direction only), "band" (toward a healthy band from either side) or
          "switch" (one category transition, switch_from -> switch_to).
    Every numeric target also stays inside the training-data range of the feature, and is rounded to *step*
    toward the current value, so rounding never takes a target past its limit.
    """
    feature: str
    owner: str                          # "patient" (a habit) or "doctor" (a lab or vital-sign target)
    label: str                          # people-facing wording
    kind: str
    active_at: float | None = None      # down levers: only when the current value is >= this (> if active_strict)
    active_strict: bool = False
    max_change: float | None = None     # largest change in one plan, in the feature's own unit
    max_relative: float | None = None   # largest change as a share of the current value
    floor: float | None = None          # down levers never go below this
    ceiling: float | None = None        # up levers never go above this
    band: tuple | None = None           # band levers: (low, high)
    switch_from: tuple = ()
    switch_to: object = None
    step: float = 0.1
    goal: str | None = None             # wellness tracker goal this maps to later ('weight', 'exercise', 'sleep')
    unit: str = ""
    note: str = ""

    def rule(self) -> str:
        """The rule in words, for reports and the clinician review."""
        if self.kind == "switch":
            def word(v):   # 0/1-coded answers read as yes/no
                return f"'{v}'" if isinstance(v, str) else f"{v} ({'yes' if v else 'no'})"
            return " or ".join(word(v) for v in self.switch_from) + f" -> {word(self.switch_to)} only"
        unit = f" {self.unit}" if self.unit else ""
        parts = []
        if self.kind == "band":
            low, high = self.band
            parts.append(f"if below {low:g}{unit}: up toward {low:g}; if above {high:g}{unit}: down toward {high:g}")
        else:
            parts.append(f"{self.kind} only")
        if self.active_at is not None:
            parts.append(f"only if {'>' if self.active_strict else '>='} {self.active_at:g}{unit}")
        if self.max_change is not None:
            parts.append(f"at most {self.max_change:g}{unit} per plan")
        if self.max_relative is not None:
            parts.append(f"at most {self.max_relative:.0%} of the current value")
        if self.floor is not None:
            parts.append(f"never below {self.floor:g}{unit}")
        if self.ceiling is not None:
            parts.append(f"never above {self.ceiling:g}{unit}")
        parts.append("inside the training-data range")
        return "; ".join(parts)


def _bmi(feature: str) -> Lever:
    return Lever(feature, "patient", "Body weight (BMI)", "down", active_at=25.0, max_relative=0.10, floor=24.9,
                 step=0.1, goal="weight", unit="kg/m²")


def _hba1c(feature: str) -> Lever:
    return Lever(feature, "doctor", "HbA1c target", "down", active_at=5.7, max_change=1.0, floor=5.7, step=0.1,
                 unit="%")


# Prototype assumptions for clinician review; one table so the rules can be read and changed in one place.
LEVERS = {
    "diabetes": (
        _bmi("bmi"),
        Lever("smoking_history", "patient", "Stop smoking", "switch", switch_from=("current",), switch_to="former",
              note="'ever', 'not current', 'never' and 'No Info' are left alone."),
        _hba1c("HbA1c_level"),
        Lever("blood_glucose_level", "doctor", "Blood glucose target", "down", active_at=140, max_change=40, floor=140,
              step=1, unit="mg/dL",
              note="The dataset does not say the sample was fasting, so it is read as random (non-fasting) glucose: "
                   "the rule starts at 140 mg/dL, not the fasting 100 cut-off. A clinician confirms which it is."),
    ),
    "heart": (
        _bmi("BMI"),
        Lever("PhysicalActivities", "patient", "Regular physical activity", "switch", switch_from=("No",),
              switch_to="Yes", goal="exercise"),
        Lever("SmokerStatus", "patient", "Stop smoking", "switch",
              switch_from=("Current smoker - now smokes every day", "Current smoker - now smokes some days"),
              switch_to="Former smoker", note="'Never smoked' and 'Former smoker' are left alone."),
        Lever("SleepHours", "patient", "Sleep hours", "band", band=(7, 9), max_change=2, step=1, goal="sleep",
              unit="h", note="Whole hours, as the screening form and the dataset record them."),
        Lever("AlcoholDrinkers", "patient", "Stop drinking alcohol", "switch", switch_from=("Yes",), switch_to="No"),
    ),
    "kidney": (
        _bmi("BMI"),
        Lever("Smoking", "patient", "Stop smoking", "switch", switch_from=(1,), switch_to=0),
        Lever("AlcoholConsumption", "patient", "Alcohol intake", "down", max_relative=0.5, step=0.1,
              note="Dataset scale 0-20; its unit is not documented in AHEAD."),
        Lever("PhysicalActivity", "patient", "Physical activity", "up", max_change=3, step=0.1, goal="exercise",
              note="Dataset scale 0-10; its unit is not documented in AHEAD."),
        Lever("DietQuality", "patient", "Diet quality", "up", max_change=2, step=0.1, note="Score 0-10."),
        Lever("SleepQuality", "patient", "Sleep quality", "up", max_change=1.5, step=0.1,
              note="Score 4-10 in the data; a quality score, not hours, so no tracker goal."),
        Lever("SystolicBP", "doctor", "Systolic blood pressure target", "down", active_at=130, max_change=20,
              floor=120, step=1, unit="mmHg"),
        Lever("DiastolicBP", "doctor", "Diastolic blood pressure target", "down", active_at=80, active_strict=True,
              max_change=10, floor=80, step=1, unit="mmHg"),
        Lever("FastingBloodSugar", "doctor", "Fasting blood sugar target", "down", active_at=100, max_change=30,
              floor=100, step=1, unit="mg/dL"),
        _hba1c("HbA1c"),
    ),
}

# Shared plan across conditions: each shared target and the input it sets in each condition's model.
# - bmi: kg/m² in all three datasets.
# - quit_smoking: each model's own current-smoker -> former/non-smoker encoding (see the switch levers).
# - active: heart 'PhysicalActivities' No -> Yes. The kidney data records activity as a 0-10 number, not
#   yes/no, so 'active' raises it by the kidney lever's full allowed step (+3, capped) and is marked approximate.
# - hba1c: % in the diabetes and kidney datasets; the heart model has no HbA1c input.
# - alcohol: the share of intake to cut (1 = stop). Heart only records drinker yes/no, so only a full stop
#   maps there; kidney intake drops by that share, limited to its lever's 50%.
# Never shared: diabetes blood glucose is a random (non-fasting) value and kidney blood sugar is fasting, so
# neither is mapped onto the other; sleep hours (heart) and the sleep-quality score (kidney) differ in kind;
# blood pressure and diet quality exist only in the kidney model.
SHARED_TARGETS = {
    "bmi": {"diabetes": "bmi", "heart": "BMI", "kidney": "BMI"},
    "quit_smoking": {"diabetes": "smoking_history", "heart": "SmokerStatus", "kidney": "Smoking"},
    "active": {"heart": "PhysicalActivities", "kidney": "PhysicalActivity"},
    "hba1c": {"diabetes": "HbA1c_level", "kidney": "HbA1c"},
    "alcohol": {"heart": "AlcoholDrinkers", "kidney": "AlcoholConsumption"},
}


@dataclass(frozen=True)
class Gate:
    """
    A value at which no plan is offered at all, because a path to a lower screening score is the wrong message:
    the person is a child (the rules are written for adults), or an entered value is already in a range that
    needs a clinician (a diabetes diagnostic cut-off, kidney markers in the chronic kidney disease range, or
    severely high blood pressure). The cut-off itself is the urgent rule in ahead/triage.py (*source*), so the
    Screenings track and the plan engine never disagree. A blank value never trips a gate.
    """
    source: Rule
    status: str                         # the plan status when this gate trips
    reason: str = ""

    @property
    def feature(self) -> str:
        return self.source.feature

    @property
    def below(self) -> float | None:
        return self.source.below

    @property
    def at_or_above(self) -> float | None:
        return self.source.at_or_above

    @property
    def unit(self) -> str:
        return self.source.unit

    def trips(self, value) -> bool:
        return self.source.trips(value)

    def rule(self) -> str:
        label = pretty_label(self.feature)
        unit = f" {self.unit}" if self.unit and self.unit not in label else ""   # "HbA1c (%)" already names it
        if self.below is not None:
            return f"{label} below {self.below:g}{unit}"
        return f"{label} {self.at_or_above:g}{unit} or more"


_ADULTS_ONLY = ("The plan rules are written for adults: a child's BMI is read against age and sex percentiles, and "
                "a high HbA1c or glucose in a child needs a clinician, not a habit plan.")
_DIAGNOSTIC = ("is a common diabetes diagnostic cut-off, so this result needs a clinician to confirm it, not a plan "
               "to lower the screening score.")
_KIDNEY = ("is in the chronic kidney disease range, so a clinician should review the kidney results; a habit plan "
           "would only move the screening score while the markers a plan never changes stay the same.")
_PRESSURE = ("is in the severely high range, so a clinician needs to see it soon, not a plan to lower the screening "
             "score.")
# The plan status each kind of urgent rule sets, and why no plan is offered.
GATE_KINDS = {ADULTS_ONLY: ("not_for_children", None), "diabetes": ("diagnostic_range", _DIAGNOSTIC),
              "kidney": ("kidney_markers_abnormal", _KIDNEY), "blood_pressure": ("severe_blood_pressure", _PRESSURE)}
GATE_STATUSES = tuple(status for status, _ in GATE_KINDS.values())


def _gate(rule: Rule) -> Gate:
    status, why = GATE_KINDS[rule.kind]
    if rule.kind == ADULTS_ONLY:
        return Gate(rule, status, _ADULTS_ONLY)
    cut = f"below {rule.cut_off_text()}" if rule.below is not None else f"of {rule.cut_off_text()} or more"
    return Gate(rule, status, f"{rule.article.capitalize()} {rule.name} {cut} {why}")


# Every urgent rule in ahead/triage.py is a gate, in its table order (the first that trips sets the status). The
# plan engine applies the age rule to any result, flagged or not: a child never gets a habit plan.
GATES = {disease: tuple(_gate(rule) for rule in rules) for disease, rules in RULES.items()}


def gates(disease: str, values: dict) -> list:
    """The gates these values trip, in table order; no plan is offered when any trips."""
    return [{"feature": gate.feature, "value": values.get(gate.feature), "status": gate.status, "rule": gate.rule(),
             "reason": gate.reason}
            for gate in GATES[disease] if gate.trips(values.get(gate.feature))]


def gate_table(disease: str) -> list:
    """The gate cut-offs as rows, for reports and the clinician review."""
    return [{"feature": gate.feature, "input": pretty_label(gate.feature), "status": gate.status, "rule": gate.rule(),
             "reason": gate.reason} for gate in GATES[disease]]


def levers(disease: str, owners=OWNERS) -> list:
    return [lever for lever in LEVERS[disease] if lever.owner in owners]


def lever_for(disease: str, feature: str) -> Lever | None:
    return next((lever for lever in LEVERS[disease] if lever.feature == feature), None)


def fixed_inputs(disease: str) -> list:
    """Model inputs no plan ever changes (age, sex, history, self-rated health, kidney lab markers...)."""
    changeable = {lever.feature for lever in LEVERS[disease]}
    return [f for f in features(disease) if f not in changeable]


def lever_table(disease: str) -> list:
    """The lever rules as rows, for reports and the clinician review."""
    return [{"feature": lever.feature, "input": pretty_label(lever.feature), "owner": lever.owner,
             "label": lever.label, "rule": lever.rule(), "goal": lever.goal, "note": lever.note}
            for lever in LEVERS[disease]]


# =============================================================================
# SCORING (the Screenings page's own path)
# =============================================================================

def blank_optional(disease: str, values: dict) -> list:
    """Optional inputs left blank, in form order (an absent key reaches the model as missing too)."""
    return [f for f in OPTIONAL_INPUTS.get(disease, []) if _blank(values.get(f))]


def score(disease: str, values: dict, model_name: str | None = None) -> dict:
    """
    The score and threshold the Screenings page shows for these values: the same model frame (blanks as NaN,
    columns in model order) and the lowered threshold scripts/missing_inputs.py measured for blank optional
    inputs, never above the usual one. A blank required kidney test gives no score, as on the page.
    """
    name = model_name or best_model(disease)
    optional = OPTIONAL_INPUTS.get(disease, [])
    absent = [f for f in features(disease) if f not in optional and _blank(values.get(f))]
    if absent:
        raise ValueError(f"The screening form always has a value for {', '.join(absent)}.")
    missing = blank_optional(disease, values)
    usual = decision_threshold(disease, name)
    result = {"disease": disease, "model": name, "usual_threshold": usual, "missing": missing}
    if any(f in missing for f in REQUIRED_FOR_SCORE.get(disease, [])):
        return {**result, "no_score": True, "score": None, "threshold": None, "flagged": None}
    pipeline = load_pipeline(disease, name)
    if pipeline is None:
        raise LookupError(f"No saved {name} model for {disease}.")
    probability = float(pipeline.predict_proba(_model_frame(values, features(disease)))[0][1])
    profile = missing_inputs_profile(disease, name, missing) if missing else None
    threshold = min(profile["threshold"], usual) if profile else usual
    return {**result, "no_score": False, "score": probability, "threshold": threshold,
            "flagged": probability >= threshold}


def _batch_scores(pipeline, feature_list: list, values: dict, rows: list) -> np.ndarray:
    """Scores for many changed copies of *values* in one call; each row is a {feature: new value} dict."""
    base = {f: (np.nan if _blank(values.get(f)) else values.get(f)) for f in feature_list}
    columns = {f: [base[f]] * len(rows) for f in feature_list}
    for index, changes in enumerate(rows):
        for feature, value in changes.items():
            columns[feature][index] = value
    return pipeline.predict_proba(pd.DataFrame(columns, columns=feature_list))[:, 1]


# =============================================================================
# LEVER RULES
# =============================================================================

def _matches(value, options) -> bool:
    if _blank(value):
        return False
    for option in options:
        if isinstance(option, str):
            if str(value).strip().lower() == option.lower():
                return True
        else:
            try:
                if float(value) == float(option):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _active(lever: Lever, current: float) -> bool:
    if lever.active_at is None:
        return True
    return current > lever.active_at if lever.active_strict else current >= lever.active_at


def _bound(lever: Lever, current, info: dict) -> float | None:
    """The furthest value the lever allows from *current* (inside the training range), or None if it does not apply."""
    if lever.kind == "switch" or _blank(current):
        return None
    current = float(current)
    if not math.isfinite(current):
        return None
    if lever.kind == "band":
        low, high = lever.band
        if current < low:
            bound = min(low, current + lever.max_change, info["max"])
            return bound if bound > current else None
        if current > high:
            bound = max(high, current - lever.max_change, info["min"])
            return bound if bound < current else None
        return None
    if lever.kind == "down":
        if not _active(lever, current):
            return None
        limits = [info["min"]]
        limits += [lever.floor] if lever.floor is not None else []
        limits += [current - lever.max_change] if lever.max_change is not None else []
        limits += [current * (1 - lever.max_relative)] if lever.max_relative is not None else []
        bound = max(limits)
        return bound if bound < current else None
    limits = [info["max"]]
    limits += [lever.ceiling] if lever.ceiling is not None else []
    limits += [current + lever.max_change] if lever.max_change is not None else []
    bound = min(limits)
    return bound if bound > current else None


def _snap(value: float, step: float, current: float) -> float:
    """Round to the step grid toward the current value, so a rounded target never passes its limit."""
    units = round(value / step, 6)
    return round((math.ceil(units) if current > value else math.floor(units)) * step, 6)


def _typed(value: float, info: dict):
    return int(value) if info.get("integer") and float(value).is_integer() else float(value)


def allowed_targets(lever: Lever, current, info: dict) -> list:
    """
    The values a plan may try for one lever, weakest first, as (value, share of the allowed change).
    Empty when the lever does not apply: a blank input, a value outside its activation rule, or no room left.
    """
    if lever.kind == "switch":
        return [(lever.switch_to, 1.0)] if _matches(current, lever.switch_from) else []
    bound = _bound(lever, current, info)
    if bound is None:
        return []
    current, targets, seen = float(current), [], set()
    span = bound - current
    for fraction in LEVEL_FRACTIONS:
        value = _snap(current + fraction * span, lever.step, current)
        moved = value - current
        if moved * span <= 0 or abs(moved) > abs(span) + 1e-9 or value in seen:
            continue
        if not info["min"] - 1e-9 <= value <= info["max"] + 1e-9:
            continue
        seen.add(value)
        targets.append((_typed(value, info), moved / span))
    return targets


def check_change(disease: str, values: dict, feature: str, new) -> None:
    """Raise ValueError, in words, when changing *feature* to *new* breaks the lever rules."""
    lever = lever_for(disease, feature)
    label = pretty_label(feature)
    if lever is None:
        raise ValueError(f"{label} is not something a plan changes.")
    current = values.get(feature)
    if _blank(current):
        raise ValueError(f"{label} was not entered, and a plan never fills in a blank value.")
    info = feature_summary(disease)[feature]
    if lever.kind == "switch":
        if _matches(new, (current,)):
            return
        if not _matches(current, lever.switch_from) or not _matches(new, (lever.switch_to,)):
            raise ValueError(f"{label}: a plan only changes {lever.rule()}.")
        return
    try:
        number = float(new)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: enter a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label}: enter a number.")
    current = float(current)
    if number == current:
        return
    bound = _bound(lever, current, info)
    if bound is None:
        raise ValueError(f"{label}: the plan rule ({lever.rule()}) does not apply at {current:g}.")
    if (number - current) * (bound - current) <= 0 or abs(number - current) > abs(bound - current) + 1e-9:
        raise ValueError(f"{label}: {number:g} is outside the allowed change from {current:g} "
                         f"(at most to {bound:g}; {lever.rule()}).")
    if not info["min"] - 1e-9 <= number <= info["max"] + 1e-9:
        raise ValueError(f"{label}: {number:g} is outside the training-data range "
                         f"({info['min']:g} to {info['max']:g}).")


def apply_changes(disease: str, values: dict, changes: dict) -> dict:
    """A copy of *values* with *changes* applied after checking each one; category targets use the data's spelling."""
    changed = dict(values)
    for feature, new in changes.items():
        check_change(disease, values, feature, new)
        lever = lever_for(disease, feature)
        if lever.kind == "switch":
            changed[feature] = lever.switch_to if _matches(new, (lever.switch_to,)) else values[feature]
        else:   # a slider may send text or a float; the model gets a number
            changed[feature] = _typed(float(new), feature_summary(disease)[feature])
    return changed


def what_if(disease: str, values: dict, changes: dict, model_name: str | None = None) -> dict:
    """The score() result for *values* with *changes* ({feature: new value}); rejects changes that break the rules."""
    changed = apply_changes(disease, values, changes)
    return {**score(disease, changed, model_name), "values": changed}


# =============================================================================
# PATH SEARCH
# =============================================================================

def _options(disease: str, values: dict, owners) -> list:
    """(lever, [(value, share of the allowed change, strongest allowed level?), ...]) for each lever that applies."""
    stats = feature_summary(disease)
    found = []
    for lever in levers(disease, owners):
        targets = allowed_targets(lever, values.get(lever.feature), stats[lever.feature])
        if targets:
            found.append((lever, [(value, share, n == len(targets) - 1) for n, (value, share) in enumerate(targets)]))
    return found


def _lowering(pipeline, feature_list, values, options) -> list:
    """
    The options cut to the levels that lower the score on their own (a lever left with none is dropped), so every
    change in a plan helps by itself and not only through an interaction in the model.
    """
    rows = [{}] + [{lever.feature: target[0]} for lever, targets in options for target in targets]
    scores = _batch_scores(pipeline, feature_list, values, rows)
    kept, start = [], 1
    for lever, targets in options:
        mine = [target for target, s in zip(targets, scores[start:start + len(targets)]) if s < scores[0] - TINY]
        start += len(targets)
        if mine:
            kept.append((lever, mine))
    return kept


def _combos(options: list, size: int, subsets=None) -> list:
    """Every (subset, level picks) of *size* levers, in a fixed order."""
    subsets = combinations(range(len(options)), size) if subsets is None else subsets
    return [(subset, picks) for subset in subsets
            for picks in product(*[range(len(options[i][1])) for i in subset])]


def _evaluate(pipeline, feature_list, values, options, combos) -> list:
    rows = [{options[i][0].feature: options[i][1][p][0] for i, p in zip(subset, picks)} for subset, picks in combos]
    scores = _batch_scores(pipeline, feature_list, values, rows) if rows else []
    return [{"subset": subset, "picks": picks, "score": float(s),
             "total": sum(options[i][1][p][1] for i, p in zip(subset, picks))}
            for (subset, picks), s in zip(combos, scores)]


def _search(pipeline, feature_list, values, options, target, max_changes, worthwhile):
    """
    (chosen combination, crosses, lowest score found). The chosen one is the smallest crossing plan; without one,
    the best reduction among plans scoring at or below *worthwhile*, or None when no plan gets that low.
    """
    sizes = range(1, min(max_changes, len(options)) + 1)
    every = [combo for size in sizes for combo in _combos(options, size)]
    if len(every) <= MAX_BATCH_ROWS:
        evaluated = _evaluate(pipeline, feature_list, values, options, every)   # one model call for every size
    else:
        evaluated = []
        for size in sizes:
            combos = _combos(options, size)
            if len(combos) > MAX_BATCH_ROWS:
                # Too many level combinations: every subset at full strength, then shrink only those that cross.
                strongest = [(subset, tuple(len(options[i][1]) - 1 for i in subset))
                             for subset in combinations(range(len(options)), size)]
                full = _evaluate(pipeline, feature_list, values, options, strongest)
                evaluated += full
                crossing_subsets = [row["subset"] for row in full if row["score"] < target]
                combos = _combos(options, size, crossing_subsets)[:MAX_BATCH_ROWS]
            evaluated += _evaluate(pipeline, feature_list, values, options, combos)
            if any(row["score"] < target for row in evaluated):
                break
    if not evaluated:
        return None, False, None
    lowest = min(row["score"] for row in evaluated)
    crossing = [row for row in evaluated if row["score"] < target]
    if crossing:
        smallest = min(len(row["subset"]) for row in crossing)
        return min((row for row in crossing if len(row["subset"]) == smallest),
                   key=lambda r: (r["total"], r["score"], r["subset"], r["picks"])), True, lowest
    if lowest > worthwhile:
        return None, False, lowest
    # Scores within SAME_SCORE of the lowest count as equally good, but every candidate must itself be worthwhile.
    near = [row for row in evaluated if row["score"] <= min(lowest + SAME_SCORE, worthwhile)]
    return min(near, key=lambda r: (len(r["subset"]), r["total"], r["score"], r["subset"], r["picks"])), False, lowest


def suggest_path(disease: str, values: dict, model_name: str | None = None, threshold: float | None = None,
                 owners=("patient",), max_changes: int = 3, margin: float = 0.02) -> dict:
    """
    The smallest set of allowed changes that takes the model's score below threshold - margin.

    No plan is offered when a gate in GATES trips (a child, or an entered value already in a range that needs a
    clinician: the urgent rules in ahead/triage.py). Otherwise each numeric lever is tried at 25/50/75/100% of its allowed change and each category
    lever as its one transition. Each of those changes is first scored on its own and only the ones that lower the
    score on their own are kept. Every plan of 1..max_changes of them, at every level, is then scored in one batch
    (size by size, shrinking only subsets that cross at full strength, when there are more than MAX_BATCH_ROWS).
    Among the crossing plans of the smallest size the one with the lowest total share of allowed change wins, then
    the lowest score. Without a crossing plan, 'changes' holds the best reduction (fewest changes among scores
    within SAME_SCORE of the lowest) when it lowers the score by at least MIN_DROP. Deterministic for the same inputs.

    status:
      'crosses'          'changes' takes the score below the target
      'best_reduction'   nothing crosses; 'changes' lowers the score by at least MIN_DROP
      'negligible'       allowed changes lower the score, but by less than MIN_DROP; no plan ('largest_drop' holds
                         the largest drop any plan reached)
      'no_path'          no allowed change lowers the score on its own
      'no_levers'        no lever's rule applies to these values
      'already_below'    the score is already below the target
      'no_score'         a required kidney test is blank, so the Screenings page gives no score
      'not_for_children', 'diagnostic_range', 'kidney_markers_abnormal', 'severe_blood_pressure'
                         the first gate that trips ('gates' lists every one); no plan
    """
    owners = tuple(owners)
    if not owners or any(owner not in OWNERS for owner in owners):
        raise ValueError(f"owners must be drawn from {OWNERS}.")
    if max_changes < 1:
        raise ValueError("max_changes must be at least 1.")
    base = score(disease, values, model_name)
    plan = {"disease": disease, "model": base["model"], "owners": list(owners), "margin": margin,
            "base_score": base["score"], "threshold": base["threshold"], "target": None, "flagged": base["flagged"],
            "missing": base["missing"], "crosses": False, "status": "no_score", "gates": [], "changes": [],
            "new_score": base["score"], "score_drop": 0.0, "largest_drop": None, "levers": [], "note": CAVEAT}
    if base["no_score"]:
        return plan
    cut = base["threshold"] if threshold is None else float(threshold)
    plan["threshold"] = cut
    plan["target"] = target = cut - margin
    plan["gates"] = tripped = gates(disease, values)
    if tripped:
        return {**plan, "status": tripped[0]["status"]}
    options = _options(disease, values, owners)
    plan["levers"] = [{"feature": lever.feature, "label": lever.label, "owner": lever.owner,
                       "from": values.get(lever.feature), "targets": [target[0] for target in targets]}
                      for lever, targets in options]
    if base["score"] < target:
        return {**plan, "status": "already_below", "crosses": True}
    if not options:
        return {**plan, "status": "no_levers"}

    feature_list = features(disease)
    pipeline = load_pipeline(disease, base["model"])
    options = _lowering(pipeline, feature_list, values, options)
    if not options:
        return {**plan, "status": "no_path"}
    chosen, crosses, lowest = _search(pipeline, feature_list, values, options, target, max_changes,
                                      base["score"] - MIN_DROP)
    if chosen is None:
        return {**plan, "status": "negligible", "largest_drop": base["score"] - lowest}

    changes = []
    for i, p in zip(chosen["subset"], chosen["picks"]):
        lever, targets = options[i]
        value, share, strongest = targets[p]
        changes.append({"feature": lever.feature, "label": lever.label, "owner": lever.owner,
                        "from": values.get(lever.feature), "to": value, "share_of_limit": share,
                        "at_limit": strongest, "goal": lever.goal})
    # The reported score comes from the page's own single-row path, not the batch.
    after = what_if(disease, values, {c["feature"]: c["to"] for c in changes}, base["model"])
    crosses = after["score"] < target
    return {**plan, "status": "crosses" if crosses else "best_reduction", "crosses": crosses, "changes": changes,
            "new_score": after["score"], "score_drop": base["score"] - after["score"]}


def lever_effects(disease: str, values: dict, model_name: str | None = None, owners=OWNERS) -> dict:
    """
    For each lever whose rule applies (gates are not checked here), the score after each allowed level on its own,
    weakest first ('steps'), plus the smallest and the full allowed change. A step that RAISES the score moves it
    the wrong way for this person; scores that go both up and down along the steps show the model is not
    monotonic for that input here.
    """
    base = score(disease, values, model_name)
    if base["no_score"]:
        return {"base_score": None, "levers": []}
    options = _options(disease, values, owners)
    rows = [{lever.feature: target[0]} for lever, targets in options for target in targets]
    scores = _batch_scores(load_pipeline(disease, base["model"]), features(disease), values, rows) if rows else []
    found, start = [], 0
    for lever, targets in options:
        steps = [(target[0], float(s)) for target, s in zip(targets, scores[start:start + len(targets)])]
        start += len(targets)
        found.append({"feature": lever.feature, "owner": lever.owner, "from": values.get(lever.feature),
                      "smallest_to": steps[0][0], "smallest_score": steps[0][1],
                      "full_to": steps[-1][0], "full_score": steps[-1][1], "steps": steps})
    return {"base_score": base["score"], "levers": found}


# =============================================================================
# SHARED PLAN ACROSS CONDITIONS
# =============================================================================

def _shared_value(disease: str, key: str, target, values: dict, feature: str):
    """(value to set, status, reason) for one shared target in one condition; value None when it does not apply."""
    lever = lever_for(disease, feature)
    current = values.get(feature)
    if _blank(current):
        return None, "skipped", "not entered; a plan never fills in a blank value"
    info = feature_summary(disease)[feature]
    if lever.kind == "switch":
        if key == "alcohol" and target < 1:
            return None, "skipped", "this model only records whether someone drinks, so only stopping maps here"
        if not _matches(current, lever.switch_from):
            return None, "skipped", f"already '{current}'; the plan only changes {lever.rule()}"
        return lever.switch_to, "applied", ""
    bound = _bound(lever, current, info)
    if bound is None:
        return None, "skipped", f"the plan rule does not apply at {float(current):g} ({lever.rule()})"
    # The strongest level the search itself would use: on the step grid toward the current value and inside the
    # training range, so it always passes check_change (the raw bound can sit just past the limit or the range).
    levels = allowed_targets(lever, current, info)
    if not levels:
        return None, "skipped", "no allowed change stays inside this model's training-data range"
    strongest = levels[-1][0]
    current = float(current)
    if key == "active":
        return strongest, "approximate", ("this model records activity as a number, not yes/no; set to its largest "
                                          "allowed increase")
    desired = current * (1 - target) if key == "alcohol" else float(target)
    if desired >= current:
        return None, "skipped", f"the target is not below the current value {current:g}"
    if desired > info["max"]:
        return None, "skipped", "the target is above this model's training-data range"
    if desired < bound - 1e-9:
        return strongest, "limited", f"limited to the largest allowed change ({lever.rule()})"
    return desired, "applied", ""


def shared_plan(targets: dict, latest: dict, model_names: dict | None = None) -> dict:
    """
    Before/after scores per condition for one plan written in shared terms.

    targets: any of bmi (target kg/m²), quit_smoking (True), active (True), hba1c (target %), alcohol
    (share of intake to cut, 0-1, where 1 means stop). latest: {disease: the latest screening values}.
    Each target is mapped onto each model's own input and encoding (SHARED_TARGETS) and must pass that
    model's lever rules; a target beyond a lever's limit is applied up to the limit and marked 'limited'.
    A condition whose values trip a gate (GATES) gets no changes, as suggest_path offers it no plan.
    """
    unknown = set(targets) - set(SHARED_TARGETS)
    if unknown:
        raise ValueError(f"Unknown shared targets: {', '.join(sorted(unknown))}.")
    alcohol = targets.get("alcohol")
    if alcohol is not None and not 0 < float(alcohol) <= 1:
        raise ValueError("alcohol is the share of intake to cut, above 0 and at most 1.")
    plans = {}
    for disease, values in latest.items():
        if values is None:
            continue
        name = (model_names or {}).get(disease)
        before = score(disease, values, name)
        tripped = [] if before["no_score"] else gates(disease, values)
        applied, skipped, changes = [], [], {}
        for key, target in targets.items():
            if not target:   # None, False or 0: not part of this plan
                continue
            feature = SHARED_TARGETS[key].get(disease)
            if feature is None:
                skipped.append({"target": key, "feature": None, "reason": "this condition's model has no matching input"})
                continue
            if tripped:
                skipped.append({"target": key, "feature": feature,
                                "reason": f"no plan is offered for this condition ({tripped[0]['rule']})"})
                continue
            value, status, reason = _shared_value(disease, key, float(target) if key == "alcohol" else target,
                                                  values, feature)
            if value is None:
                skipped.append({"target": key, "feature": feature, "reason": reason})
                continue
            changes[feature] = value
            applied.append({"target": key, "feature": feature, "from": values.get(feature), "to": value,
                            "status": status, "reason": reason})
        after = what_if(disease, values, changes, name) if changes else {**before, "values": dict(values)}
        if not before["no_score"]:   # each change on its own, so one that RAISES this model's score is visible
            for entry in applied:
                alone = what_if(disease, values, {entry["feature"]: entry["to"]}, name)["score"]
                entry["score_change_alone"] = alone - before["score"]
        plans[disease] = {"model": before["model"], "before": before["score"], "after": after["score"],
                          "threshold": before["threshold"], "flagged_before": before["flagged"],
                          "flagged_after": after["flagged"], "no_score": before["no_score"], "gates": tripped,
                          "applied": applied, "skipped": skipped}
    return plans
