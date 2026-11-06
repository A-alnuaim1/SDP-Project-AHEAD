"""
ahead/prevention.py
===================
The PLAN step: the prevention plan a doctor agrees with a patient after a screening, most often one on the at_risk
track of ahead/triage.py. The doctor starts from standard advice for the condition (STARTING_POINTS), keeps the
targets that fit, adjusts the goals and approves; the patient follows the plan on the My plan page, and its progress
comes from what the patient logs in the lifestyle tracker and the 30x30 activity log.

The model never chooses a target, and nothing here says a plan lowers the model's score or guarantees an outcome:
the study in docs/risk_plan_evaluation.md is why AHEAD offers standard advice a doctor approves instead.

TARGETS is the one catalog of target rules. Its goals, bounds and the status rules below are prototype values for a
clinician to confirm, not guideline citations:

  weight     kg; offered only when the BMI is 25 or more (the latest tracker log's weight and height, else the
             screening's BMI). Default goal: about 6% below the latest logged weight, rounded to the nearest 0.5 kg
             (5-7% is the usual aim).
  activity   minutes a week; default goal 150.
  fruit_veg  servings a day; default goal 5.
  sleep      hours a night; default goal range 7-9.
  smoking    stop smoking; offered only when the condition's screening says current smoker. No daily progress: the
             doctor checks it at the re-check.
  under 18   no target is offered: they are written for adults (an adult BMI cut-off, a weight-loss goal, adult
             activity and sleep guidance), so a child's plan is agreed outside AHEAD. The age is the condition's own
             answer (diabetes age, kidney Age; heart's age categories all start at 18), else the latest tracker log's
             age; the cut-off is ahead/triage.py's age rule.

Goal bounds match the tracker's Goals form, so a goal the plan writes there always fits the form. Every target is
logged in the lifestyle tracker (fruit and vegetables as servings in a check-in; the food analyzer's meals hold no
servings), so each card's action opens the tracker.

Progress (progress()), with "the last 7 days" meaning the 7 local (Dubai) days ending today:
  weight     the latest weight logged since the target started, against the starting weight (the latest weight when
             the target started, or the first one logged after it). A target starts when the plan is approved, or
             when a later edit adds it (its 'since'). On track: at or below the goal, or down by at least half of what
             an even pace from the start to the goal by the re-check date would have lost so far (both to 0.1 kg).
             Behind otherwise. Not logged yet: no weight logged since the target started.
  activity   minutes logged in the 30x30 activity log in the last 7 days; with no entry there, the weekly estimate
             (exercise days x minutes) of the latest tracker check-in in the last 7 days. On track: the goal reached.
  fruit_veg  the average of the last 7 days' tracker check-ins (one per day: that day's latest). On track: the goal
  sleep      reached (sleep: the average inside the goal range).
  Any target with no data in its window is Not logged yet, which never counts as on track.

Pure Python (no Streamlit), so both pages, ahead/platform_data.py and tests share it, and every page shows the same
words through the *_text helpers.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ahead.triage import ADULTS_ONLY, RULES, entered, number_text

ZONE = ZoneInfo("Asia/Dubai")   # the same local zone as ahead/platform_data.py (which imports this module)
WINDOW_DAYS = 7
RECHECK_WEEKS = 12              # the default re-check, in weeks from the day the plan is approved
RECHECK_BOUNDS = (2, 52)        # the weeks a doctor can choose
WEIGHT_CUT = 0.06               # the default weight goal is this share below the latest logged weight
WEIGHT_BMI = 25.0               # a weight target is offered only at or above this BMI
# No target is offered under this age: ahead/triage.py's age rule, the one cut-off for "a child" in AHEAD.
ADULT_AGE = next(rule.below for rule in RULES["diabetes"] if rule.kind == ADULTS_ONLY)
NOTE_LIMIT = 1000               # characters in the doctor's note
STATUSES = ("draft", "active", "paused", "stopped")


# =============================================================================
# CATALOG
# =============================================================================

@dataclass(frozen=True)
class Target:
    """
    One target a plan can hold.

    kind: 'down' (toward a lower goal), 'at_least' (reach the goal), 'range' (stay inside (low, high)) or 'check'
          (no daily progress; the doctor checks it at the re-check).
    unit: printed after a value ('82.4 kg', '110 min', '3 a day', '6.5 h'); per: added after a goal in full
          sentences ('150 min a week'). goal_key: the tracker goal the plan writes (lifestyle_goals), if any.
    page/action: where the patient logs it, and the button that goes there.
    """
    key: str
    label: str            # the doctor's targets table and plan history
    short: str            # the patient's cards and the plan preview
    measure: str          # the small line under the label in the doctor's table
    kind: str
    unit: str = ""
    per: str = ""
    default: object = None
    bounds: tuple = ()    # (min, max) of a goal, the tracker Goals form's bounds
    step: float = 1
    goal_key: str | None = None
    page: str | None = None
    action: str = ""
    why: str = ""
    icon: str = ""        # assets/icons/dashboard/<icon>.svg
    color: str = ""


TARGETS = {
    "weight": Target("weight", "Body weight", "Body weight", "Tracked from the weight log", "down", unit="kg",
                     bounds=(25.0, 400.0), step=0.5, goal_key="weight", page="wellness", action="Log weight",
                     why="About 6% lower (5–7% is the usual aim)", icon="weight", color="#8E7CC3"),
    "activity": Target("activity", "Physical activity", "Activity", "Minutes a week", "at_least", unit="min",
                       per=" a week", default=150, bounds=(30, 2000), step=10, goal_key="exercise", page="wellness",
                       action="Log activity", why="Standard weekly guidance for adults", icon="activity",
                       color="#0E9AA7"),
    "fruit_veg": Target("fruit_veg", "Fruit and vegetables", "Fruit and vegetables", "Servings a day", "at_least",
                        unit="a day", default=5, bounds=(1, 15), step=1, goal_key="fruit_veg", page="wellness",
                        action="Log servings", why="Standard advice: five a day", icon="food", color="#2E9477"),
    "sleep": Target("sleep", "Sleep", "Sleep", "Hours a night", "range", unit="h", per=" a night", default=(7.0, 9.0),
                    bounds=(6.0, 12.0), step=0.5, goal_key="sleep", page="wellness", action="Log sleep",
                    why="The usual adult range", icon="sleep", color="#C98F1E"),
    "smoking": Target("smoking", "Stop smoking", "Stop smoking", "Checked at the re-check", "check",
                      why="Shown only for current smokers", icon="smoking", color="#C9706E"),
}


@dataclass(frozen=True)
class StartingPoint:
    """The standard advice a doctor starts from for one condition: its targets in order, and any wording that
    differs from the catalog's (why)."""
    label: str
    advice: str
    targets: tuple = tuple(TARGETS)
    why: dict = field(default_factory=dict)


STARTING_POINTS = {
    "diabetes": StartingPoint(
        "Diabetes prevention · standard advice",
        "Standard diabetes-prevention advice: lose about 5–7% of body weight when overweight, be active for "
        "150 minutes a week, eat five servings of fruit and vegetables a day, sleep 7–9 hours and stop smoking."),
    "heart": StartingPoint(
        "Heart health · standard advice",
        "Standard heart-health advice: be active for 150 minutes a week, eat five servings of fruit and vegetables "
        "a day, work toward a healthy weight, sleep 7–9 hours and stop smoking."),
    "kidney": StartingPoint(
        "Kidney health · standard advice",
        "Standard kidney-health advice: work toward a healthy weight, be active for 150 minutes a week, eat five "
        "servings of fruit and vegetables a day, sleep 7–9 hours and stop smoking. Salt, protein and potassium "
        "advice stays with the doctor.",
        why={"fruit_veg": "Five a day; review if kidney tests change (potassium)"}),
}

# Why a plan fits the condition's latest result, in the doctor's strip, per triage track. A child's result gets the
# child lines instead: no target is offered under 18, and a flagged child is urgent because of the age alone, so the
# urgent line ("a value is ... in a diagnostic range") may not hold.
FIT = {
    "at_risk": "Flagged by the model; no entered value is in a diagnostic range, so a prevention plan fits.",
    "urgent": "A value is already in a diagnostic range, so a visit comes first. A plan can still go alongside it.",
    "not_flagged": "Not flagged by the model. A plan is optional; standard advice still applies.",
    "no_score": "There is no model score for this result. A plan is optional; standard advice still applies.",
    "child": "The patient is under 18, and AHEAD's targets are written for adults, so agree any plan outside AHEAD.",
    "urgent_child": "This result needs a visit first. The patient is under 18, and AHEAD's targets are written for "
                    "adults, so agree any plan outside AHEAD.",
}
CHILD_REASON = "Not offered: under 18; these targets are written for adults"

STATUS_TEXT = {"draft": "Draft", "active": "Active", "paused": "Paused", "stopped": "Stopped"}
PROGRESS_TEXT = {"on_track": "On track", "behind": "Behind", "not_logged": "Not logged yet",
                 "at_recheck": "Checked at re-check"}
PROGRESS_TONES = {"on_track": "#2E9477", "behind": "#C98F1E", "not_logged": "#6B7C8F", "at_recheck": "#5B8FA3"}
HISTORY_TEXT = {"drafted": "Draft saved", "draft_edited": "Draft updated", "approved": "Approved and sent",
                "edited": "Edited", "paused": "Paused", "active": "Resumed", "stopped": "Stopped",
                "discarded": "Draft discarded"}
_REASONS = {"replaced": "replaced by a newer plan", "access removed": "the doctor's access was removed",
            "another plan approved": "another plan was approved"}

# Each condition's smoking answer, read as one status. Only 'current' offers the stop-smoking target.
SMOKING_TEXT = {"current": "Current smoker", "former": "Former smoker", "ever": "Has smoked", "never": "Never smoked",
                "non_smoker": "Non-smoker", None: "Not recorded"}
_DIABETES_SMOKING = {"current": "current", "former": "former", "not current": "former", "ever": "ever",
                     "never": "never"}   # 'No Info' and anything else: not recorded


def starting_point(disease: str) -> StartingPoint:
    if disease not in STARTING_POINTS:
        raise ValueError(f"Unknown condition '{disease}'; expected one of {', '.join(STARTING_POINTS)}.")
    return STARTING_POINTS[disease]


def fit_text(result, child: bool = False) -> str:
    """
    Why a plan fits a result, for the doctor: result is a triage() result or a bare track name. child: the patient is
    under 18 (child_age() is not None); a triage result's own 'child' (an age under 18 on the screening) counts too.
    """
    if isinstance(result, dict):
        track, child = result.get("track"), child or bool(result.get("child"))
    else:
        track = result
    if child:
        return FIT["urgent_child" if track == "urgent" else "child"]
    return FIT.get(track, FIT["no_score"])


# =============================================================================
# READING THE INPUTS
# =============================================================================

def _local(stamp) -> datetime:
    moment = datetime.fromisoformat(stamp)
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(ZONE)


def _values(log) -> dict:
    """A tracker log's values: rows from platform_data.lifestyle_history carry them under 'values'."""
    return (log.get("values") if isinstance(log.get("values"), dict) else log) or {}


def _latest(logs) -> dict:
    return _values(logs[0]) if logs else {}


def _number(values: dict, key: str, positive=False):
    number = entered(values.get(key))
    return None if number is None or number < 0 or (positive and number == 0) else number


def weekly_minutes(values: dict):
    """A tracker log's weekly exercise estimate (days x minutes a day), or None when either is missing."""
    days, minutes = _number(values, "exercise_days"), _number(values, "exercise_minutes")
    return None if days is None or minutes is None else round(days * minutes)


def body_mass_index(disease: str, values: dict, logs=()) -> tuple:
    """(BMI rounded to 0.1, 'tracker' or 'screening'), or (None, None): the latest tracker log's weight and height
    first, since the weight goal is set from it, else the screening's BMI."""
    log = _latest(logs)
    weight, height = _number(log, "weight_kg", True), _number(log, "height_cm", True)
    if weight and height:
        return round(weight / (height / 100) ** 2, 1), "tracker"
    bmi = _number(values or {}, "bmi" if disease == "diabetes" else "BMI", True)
    return (round(bmi, 1), "screening") if bmi else (None, None)


def smoking_status(disease: str, values: dict):
    """'current', 'former', 'ever', 'never', 'non_smoker' or None (not recorded), from the condition's own answer:
    diabetes smoking_history, heart SmokerStatus, kidney Smoking (1 = smokes)."""
    values = values or {}
    if disease == "diabetes":
        return _DIABETES_SMOKING.get(str(values.get("smoking_history", "")).strip().lower())
    if disease == "heart":
        answer = str(values.get("SmokerStatus", "")).strip().lower()
        if answer.startswith("current smoker"):
            return "current"
        return {"former smoker": "former", "never smoked": "never"}.get(answer)
    if disease == "kidney":
        coded = entered(values.get("Smoking"))
        return {1: "current", 0: "non_smoker"}.get(int(coded)) if coded in (0, 1) else None
    return None


def child_age(disease: str, values: dict, logs=()):
    """The patient's age when it is under 18 (ADULT_AGE), else None: the condition's own age answer (triage's age rule:
    diabetes age, kidney Age; heart's age categories all start at 18), or the latest tracker log's age when the
    screening has none."""
    values = values or {}
    rule = next((rule for rule in RULES.get(disease, ()) if rule.kind == ADULTS_ONLY), None)
    if disease == "heart" and str(values.get("AgeCategory") or "").strip():
        return None
    age = _number(values, rule.feature) if rule else None
    if age is None:
        age = _number(_latest(logs), "age")
    return age if age is not None and age < ADULT_AGE else None


def facts(disease: str, values: dict, logs=()) -> list:
    """Short facts for the doctor's strip: ['BMI 31.2', 'former smoker'], with 'age 12 (under 18)' first for a child;
    a missing one is left out."""
    bmi, _ = body_mass_index(disease, values, logs)
    smoker = smoking_status(disease, values)
    age = child_age(disease, values, logs)
    return [*([f"age {number_text(age)} (under {ADULT_AGE:g})"] if age is not None else []),
            *([f"BMI {bmi:.1f}"] if bmi else []), *([SMOKING_TEXT[smoker].lower()] if smoker else [])]


# =============================================================================
# DISPLAY TEXT
# =============================================================================

def _unit(text: str, unit: str) -> str:
    return f"{text} {unit}" if unit else text


def date_text(day) -> str:
    """'29 Jan 2027'."""
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return f"{day.day} {day:%b %Y}"


def _goal_number(key: str, goal) -> str:
    if TARGETS[key].kind == "range":
        low, high = goal
        return number_text(low) if low == high else f"{number_text(low)}–{number_text(high)}"
    return number_text(goal)


def goal_text(key: str, goal, full: bool = True) -> str:
    """A goal with its unit: '79 kg', '150 min a week' ('150 min' when not full), '5 a day', '7–9 h a night'."""
    target = TARGETS[key]
    if target.kind == "check":
        return "Stop"
    if goal is None:
        return "No goal set"
    return _unit(_goal_number(key, goal), target.unit) + (target.per if full else "")


def value_text(key: str, value) -> str:
    """A measured value with its unit ('84 kg', '60 min', '2 a day', '6 h'), or 'Not logged yet'."""
    if key == "smoking":
        return SMOKING_TEXT.get(value, SMOKING_TEXT[None])
    return "Not logged yet" if value is None else _unit(number_text(value), TARGETS[key].unit)


def summary_line(target: dict) -> str:
    """One stored target as the plan preview reads it: 'Body weight · 84 → 79 kg', 'Activity · 150 min a week'."""
    key, goal, baseline = target["key"], target.get("goal"), target.get("baseline")
    if key == "smoking":
        return "Stop smoking · checked at the re-check"
    if key == "weight" and baseline is not None and goal is not None:
        return f"{TARGETS[key].short} · {number_text(baseline)} → {goal_text(key, goal)}"
    return f"{TARGETS[key].short} · {goal_text(key, goal)}"


def why_text(disease: str, key: str, baseline=None, goal=None) -> str:
    """The short reason in the doctor's table; for weight, the share the goal is below the latest weight."""
    if key == "weight" and baseline and goal is not None:
        if goal >= baseline:
            return "Not below the latest weight (5–7% lower is the usual aim)"
        return f"About {round((baseline - goal) / baseline * 100)}% lower (5–7% is the usual aim)"
    return starting_point(disease).why.get(key, TARGETS[key].why)


def history_text(entry: dict) -> str:
    """A plan history row in words: 'Approved and sent', 'Stopped: replaced by a newer plan'."""
    action = HISTORY_TEXT.get(entry.get("action"), str(entry.get("action", "")).replace("_", " ").capitalize())
    reason = (entry.get("details") or {}).get("reason")
    return f"{action}: {_REASONS.get(reason, reason)}" if reason else action


def _words(count: int) -> str:
    return ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")[count] \
        if 0 <= count <= 10 else str(count)


def _join(names: list) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def recheck_status(recheck_date, today: date) -> dict:
    """{'date', 'date_text', 'days', 'due', 'short', 'text'}: 'Re-check in 12 weeks' (whole weeks from two weeks
    out, days before that) or 'Re-check due' once the date has come."""
    day = date.fromisoformat(recheck_date) if isinstance(recheck_date, str) else recheck_date
    days = (day - today).days
    if days <= 0:
        short = "Due"
    elif days >= 14:
        short = f"{round(days / 7)} weeks"
    else:
        short = f"{days} day{'s' * (days != 1)}"
    return {"date": day, "date_text": date_text(day), "days": days, "due": days <= 0, "short": short,
            "text": "Re-check due" if days <= 0 else f"Re-check in {short}"}


def recheck_date(weeks: int = RECHECK_WEEKS, start: date | None = None) -> date:
    """The re-check date *weeks* after start (default: today in Dubai)."""
    low, high = RECHECK_BOUNDS
    if not low <= int(weeks) <= high:
        raise ValueError(f"Choose a re-check between {low} and {high} weeks.")
    return (start or datetime.now(ZONE).date()) + timedelta(weeks=int(weeks))


# =============================================================================
# TEMPLATE
# =============================================================================

def default_weight_goal(weight) -> float | None:
    """About 6% below the weight, rounded half up to the nearest 0.5 kg."""
    return None if not weight else math.floor(weight * (1 - WEIGHT_CUT) * 2 + 0.5) / 2


def template(disease: str, latest_values: dict | None = None, lifestyle_logs=()) -> list:
    """
    The condition's starting point as suggested targets, in catalog order, for the doctor to adjust.

    latest_values: the condition's latest screening inputs; lifestyle_logs: tracker logs newest first (rows from
    platform_data.lifestyle_history, or bare values dicts). Each target:
      key, label, measure, unit, kind, icon, color, min, max, step
      offered      the catalog allows it for this patient (weight: BMI >= 25; smoking: a current smoker; nothing
                   under 18)
      enabled      ticked by default: offered and a default goal is available
      reason       why it is not offered or not ticked ('' otherwise): 'Not needed: former smoker', CHILD_REASON
      baseline     from the latest tracker log (smoking: the screening's status), or None
      baseline_text, goal (a number, [low, high] for sleep, None for smoking or a weight with nothing logged),
      goal_text, why
    """
    point = starting_point(disease)
    values, logs = latest_values or {}, list(lifestyle_logs or ())
    log = _latest(logs)
    bmi, source = body_mass_index(disease, values, logs)
    smoker = smoking_status(disease, values)
    child = child_age(disease, values, logs) is not None
    rows = []
    for key in point.targets:
        target = TARGETS[key]
        offered, reason = True, ""
        if key == "weight":
            baseline = _number(log, "weight_kg", True)
            goal = default_weight_goal(baseline)
            if bmi is None:
                offered, reason = False, "Not offered: no BMI recorded yet"
            elif bmi < WEIGHT_BMI:
                offered, reason = False, f"Not offered: BMI {bmi:.1f} is under 25"
            elif goal is None:
                reason = "No weight logged yet: enter a goal"
        elif key == "activity":
            baseline, goal = weekly_minutes(log), target.default
        elif key == "fruit_veg":
            baseline, goal = _number(log, "fruit_veg"), target.default
        elif key == "sleep":
            baseline, goal = _number(log, "sleep_hours"), list(target.default)
        else:   # smoking
            baseline, goal = smoker, None
            if smoker != "current":
                offered = False
                reason = (f"Not needed: {SMOKING_TEXT[smoker].lower()}" if smoker
                          else "Not offered: smoking not recorded")
        if child:
            offered, reason = False, CHILD_REASON
            goal = None if key == "weight" else goal   # no weight-loss number for a child, even greyed out
        rows.append({
            "key": key, "label": target.label, "measure": target.measure, "unit": target.unit, "kind": target.kind,
            "icon": target.icon, "color": target.color, "min": target.bounds[0] if target.bounds else None,
            "max": target.bounds[1] if target.bounds else None, "step": target.step,
            "offered": offered, "enabled": offered and (goal is not None or key == "smoking"), "reason": reason,
            "baseline": baseline, "baseline_text": value_text(key, baseline), "goal": goal,
            "goal_text": goal_text(key, goal), "why": why_text(disease, key, baseline, goal),
            **({"bmi": bmi, "bmi_source": source} if key == "weight" else {}),
        })
    return rows


# =============================================================================
# CHECKING A PLAN
# =============================================================================

def _goal(key: str, goal):
    """The goal checked against the catalog bounds, in its stored form; ValueError in the doctor's words."""
    target = TARGETS[key]
    if target.kind == "check":
        return None
    low, high = target.bounds
    bounds = f"between {number_text(low)} and {number_text(high)}{' ' + target.unit if target.unit else ''}"
    if target.kind == "range":
        try:
            first, second = (entered(x) for x in goal)
        except (TypeError, ValueError):
            first = second = None
        if first is None or second is None or not low <= first <= second <= high:
            raise ValueError(f"Enter a {target.label.lower()} range {bounds}, the lower number first.")
        return [round(first, 2), round(second, 2)]
    number = entered(goal)
    if number is None or not low <= number <= high:
        raise ValueError(f"Enter a {target.label.lower()} goal {bounds}.")
    return round(number, 1) if key == "weight" else int(round(number))


def check_targets(targets) -> list:
    """The doctor's chosen targets as stored: [{'key', 'goal'}] in catalog order, each goal inside its bounds."""
    chosen = {}
    for item in targets or ():
        key = item.get("key") if isinstance(item, dict) else None
        if key not in TARGETS:
            raise ValueError(f"Unknown target '{key}'.")
        if key in chosen:
            raise ValueError(f"{TARGETS[key].label} is listed twice.")
        chosen[key] = {"key": key, "goal": _goal(key, item.get("goal"))}
    if not chosen:
        raise ValueError("Choose at least one target.")
    return [chosen[key] for key in TARGETS if key in chosen]


def check_plan(plan: dict, today: date | None = None, recheck_changed: bool = True) -> dict:
    """
    A plan as the doctor's page sends it, checked: {'disease', 'screening_id', 'targets', 'note', 'recheck_date'}.
    The re-check must be after today unless it is an unchanged date on an existing plan (recheck_changed=False).
    """
    disease = plan.get("disease")
    starting_point(disease)
    note = str(plan.get("note") or "").strip()
    if len(note) > NOTE_LIMIT:
        raise ValueError(f"Keep the note under {NOTE_LIMIT} characters.")
    try:
        when = plan["recheck_date"]
        when = when if isinstance(when, date) else date.fromisoformat(str(when))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Choose a re-check date.") from None
    if recheck_changed and when <= (today or datetime.now(ZONE).date()):
        raise ValueError("Choose a re-check date after today.")
    screening = plan.get("screening_id")
    return {"disease": disease, "screening_id": int(screening) if screening is not None else None,
            "targets": check_targets(plan.get("targets")), "note": note, "recheck_date": when.isoformat()}


def tracker_goals(targets) -> dict:
    """The tracker goals a plan's targets set ({'weight': 79.0, 'exercise': 150, 'sleep': 8.0, 'fruit_veg': 5});
    sleep is the middle of its range, since the tracker keeps one sleep goal."""
    goals = {}
    for item in targets:
        target = TARGETS[item["key"]]
        if target.goal_key and item.get("goal") is not None:
            goal = item["goal"]
            goals[target.goal_key] = round(sum(goal) / 2, 2) if target.kind == "range" else goal
    return goals


def goals_from_plan(targets, goals: dict) -> set:
    """The tracker goal keys whose saved value is still the one the plan set (the patient may have changed them)."""
    return {key for key, value in tracker_goals(targets).items()
            if entered(goals.get(key)) is not None and math.isclose(float(goals[key]), float(value))}


# =============================================================================
# PROGRESS
# =============================================================================

def _week(logs, today: date) -> list:
    """The last 7 days' tracker values, one per local day (that day's latest log), newest first."""
    first, seen, out = today - timedelta(days=WINDOW_DAYS - 1), set(), []
    for log in logs:
        stamp = log.get("created_at") if isinstance(log, dict) else None
        day = _local(stamp).date() if stamp else None
        if day is None or not first <= day <= today or day in seen:
            continue
        seen.add(day)
        out.append(_values(log))
    return out


def _average(week: list, key: str):
    numbers = [n for n in (_number(v, key) for v in week) if n is not None]
    return (round(sum(numbers) / len(numbers), 1), len(numbers)) if numbers else (None, 0)


def _share(kind: str, current, goal, start=None) -> float | None:
    """How far along the goal the current value is, 0-1 to three decimals, for a progress bar."""
    if current is None or goal is None:
        return None
    if kind == "down":
        if start is None or start <= goal:
            return 1.0 if current <= goal else 0.0
        share = (start - current) / (start - goal)
    elif kind == "range":
        low, high = goal
        share = 1.0 if low <= current <= high else (current / low if current < low else high / current)
    else:
        share = current / goal if goal else 1.0
    return round(max(0.0, min(1.0, share)), 3)


def _weight(item: dict, plan: dict, logs, started: datetime | None, today: date) -> dict:
    goal, baseline = item.get("goal"), item.get("baseline")
    # A weight target an edit added counts from that edit, not from the plan's approval.
    started = _local(item["since"]) if item.get("since") else started
    since = [(_local(log["created_at"]), _number(_values(log), "weight_kg", True)) for log in logs
             if isinstance(log, dict) and log.get("created_at")
             and (started is None or _local(log["created_at"]) > started)]
    since = sorted((at, w) for at, w in since if w is not None)
    start_day = started.date() if started else today
    if not since:
        note = (f"Started at {value_text('weight', baseline)}; log your weight to see progress" if baseline
                else "Log your weight in the tracker to see progress")
        return {"current": None, "status": "not_logged", "note": note, "share": None}
    current = since[-1][1]
    start, start_day = (baseline, start_day) if baseline else (since[0][1], since[0][0].date())
    change = round(current - start, 1)
    when = f"{start_day.day} {start_day:%b}"
    note = (f"Down {number_text(-change)} kg since {when}" if change < 0 else
            f"Up {number_text(change)} kg since {when}" if change > 0 else f"No change since {when}")
    on_track = goal is not None and current <= goal
    if not on_track and goal is not None and start > goal:
        try:
            total = max(1, (date.fromisoformat(plan["recheck_date"]) - start_day).days)
        except (KeyError, TypeError, ValueError):
            total = RECHECK_WEEKS * 7
        pace = min(1.0, max(0, (today - start_day).days) / total)
        # To 0.1 kg, as weights are shown, so a steady weight in the first days is not already behind.
        on_track = round(start - current, 1) >= round((start - goal) * pace / 2, 1)
    return {"current": current, "status": "on_track" if on_track else "behind", "note": note,
            "share": _share("down", current, goal, start)}


def _started(plan: dict) -> datetime | None:
    stamp = plan.get("started_at")
    return _local(stamp) if stamp else None


def progress(plan: dict, lifestyle_logs=(), challenge_days=None, today: date | None = None) -> dict:
    """
    Progress on a plan's targets (see the module docstring for the rules).

    plan: a stored plan ({'targets': [{'key', 'goal', 'baseline', and 'since' (UTC) when an edit added it}],
    'started_at', 'recheck_date', ...});
    lifestyle_logs: tracker logs newest first, each with 'created_at'; challenge_days: {iso day: minutes}.
    Returns {'targets': [...], 'on_track', 'tracked', 'attention', 'headline', 'message', 'recheck'}; each target
      key, label (the patient's short label), icon, color, page, action, goal, baseline, goal_text,
      current (None when not logged), current_text ('82.4', or '—'), of_goal ('/ 79 kg'), share (0-1 or None),
      status ('on_track' | 'behind' | 'not_logged' | 'at_recheck'), status_text, tone, note, tracked
    tracked counts targets with daily progress (not smoking); attention names the behind and not-logged ones.
    """
    today = today or datetime.now(ZONE).date()
    logs = [log for log in (lifestyle_logs or ()) if isinstance(log, dict)]
    week = _week(logs, today)
    days = challenge_days or {}
    window = [(today - timedelta(days=i)).isoformat() for i in range(WINDOW_DAYS)]
    started = _started(plan)
    items = []
    for item in plan.get("targets") or ():
        key = item["key"]
        target, goal = TARGETS[key], item.get("goal")
        if key == "weight":
            found = _weight(item, plan, logs, started, today)
        elif key == "activity":
            logged = [day for day in window if day in days]
            if logged:
                current, note = sum(int(days[day]) for day in logged), "Last 7 days"
            else:
                current = next((m for m in (weekly_minutes(v) for v in week) if m is not None), None)
                note = "Weekly estimate from this week's check-in" if current is not None else \
                    "Log activity minutes to see progress"
            found = {"current": current, "note": note}
        elif key in ("fruit_veg", "sleep"):
            current, count = _average(week, "fruit_veg" if key == "fruit_veg" else "sleep_hours")
            average = "7-day average" if key == "fruit_veg" else "7-night average"
            found = {"current": current,
                     "note": (average if count > 1 else "From this week's check-in") if count else
                     ("Log servings in a tracker check-in to see progress" if key == "fruit_veg"
                      else "Log your sleep in the tracker to see progress")}
        else:   # smoking: no daily progress
            found = {"current": None, "status": "at_recheck", "note": "Your doctor checks this at your re-check",
                     "share": None}
        current = found["current"]
        if "status" not in found:
            if current is None:
                status = "not_logged"
            elif target.kind == "range":
                status = "on_track" if goal and goal[0] <= current <= goal[1] else "behind"
            else:
                status = "on_track" if goal is not None and current >= goal else "behind"
            found.update(status=status, share=_share(target.kind, current, goal))
        items.append({
            "key": key, "label": target.short, "icon": target.icon, "color": target.color, "page": target.page,
            "action": target.action, "goal": goal, "baseline": item.get("baseline"),
            "goal_text": goal_text(key, goal), "current": current,
            "current_text": "—" if current is None else number_text(current),
            "of_goal": "" if target.kind == "check" else f"/ {goal_text(key, goal, full=False)}",
            "share": found["share"], "status": found["status"], "status_text": PROGRESS_TEXT[found["status"]],
            "tone": PROGRESS_TONES[found["status"]], "note": found["note"], "tracked": target.kind != "check",
        })
    tracked = [i for i in items if i["tracked"]]
    on_track = sum(i["status"] == "on_track" for i in tracked)
    behind = [i["label"].lower() for i in tracked if i["status"] == "behind"]
    missing = [i["label"].lower() for i in tracked if i["status"] == "not_logged"]
    recheck = recheck_status(plan["recheck_date"], today) if plan.get("recheck_date") else None
    return {"targets": items, "on_track": on_track, "tracked": len(tracked), "attention": behind + missing,
            "headline": _headline(on_track, len(tracked), len(missing)), "message": _message(behind, missing, tracked),
            "recheck": recheck}


def _headline(on_track: int, tracked: int, missing: int) -> str:
    """'Two of your four targets are on track'."""
    if not tracked:
        return "Your doctor checks this plan at your re-check"
    if missing == tracked:
        return "Log your first check-in to see your progress"
    if on_track == tracked:
        return "Your target is on track" if tracked == 1 else f"All {_words(tracked)} of your targets are on track"
    if tracked == 1:
        return "Your target is not on track yet"
    if not on_track:
        return f"None of your {_words(tracked)} targets are on track yet"
    return f"{_words(on_track).capitalize()} of your {_words(tracked)} targets {'is' if on_track == 1 else 'are'} " \
           "on track"


def _message(behind: list, missing: list, tracked: list) -> str:
    """The encouraging line under the headline, naming what needs attention."""
    parts = []
    if behind:
        # No verb after the names, so a single plural label ('fruit and vegetables') still reads right.
        parts.append(f"Keep going: give {_join(behind)} a little more attention this week.")
    if missing:
        parts.append(f"Log your {_join(missing)} to see {'its' if len(missing) == 1 else 'their'} progress.")
    if not parts and tracked:
        parts.append("Great work: keep it up until your re-check.")
    return " ".join(parts)
