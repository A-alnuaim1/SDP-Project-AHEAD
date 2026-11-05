"""
scripts/evaluate_risk_plans.py
==============================
Offline evaluation of the "path to lower risk" engine (ahead/risk_plan.py), step 1 of 7.

1. Rebuilds the held-out TEST split train_model.py used for each condition (20% of the rows, random_state=42,
   stratified) and proves it is the same split by recomputing the recorded final test metrics of the
   recommended model (models/model_meta.json). If any metric differs beyond rounding, it stops and reports
   that instead of evaluating on other data.
2. On the test rows the recommended model flags (a fixed random sample, seed 42, of at most --sample rows per
   condition), it counts the rows a gate keeps from getting any plan (children, values already in a diabetes
   diagnostic or chronic kidney disease range, severely high blood pressure: the urgent rules in ahead/triage.py,
   which the Screenings track uses too), searches the other rows for patient-only and patient + doctor paths, and
   reports how often a path exists, how many changes and how much score drop it takes, which levers it uses,
   how often a change moves the score the wrong way or changes direction along its steps, and the runtime.
3. With --demo-db (a COPY of an AHEAD database; it is opened read-only) it adds each demo patient's latest
   saved screening as a worked example.

Run from the project root:
    python scripts/evaluate_risk_plans.py --out <dir> [--sample 1000] [--demo-db <copy.sqlite3>]
Writes summary.json, rows.csv and report.md to <dir>. matplotlib is not in requirements.txt, so no charts.
"""

import argparse
import json
import sqlite3
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.logger import set_log_level

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahead.clinical_data import shown_value  # noqa: E402
from ahead.config import DISEASES, OPTIONAL_INPUTS, REQUIRED_FOR_SCORE, pretty_label  # noqa: E402
from ahead.resources import (  # noqa: E402
    best_model, decision_threshold, feature_summary, features, final_metrics, load_pipeline, meta,
)
from ahead.risk_plan import (  # noqa: E402
    CAVEAT, GATE_STATUSES, GATES, LEVEL_FRACTIONS, LEVERS, MIN_DROP, SHARED_TARGETS, gate_table, lever_effects,
    lever_table, score, suggest_path,
)
from ahead.triage import ADULTS_ONLY, RULES  # noqa: E402
from train_model import DATA_DIR, calculate_metrics, clean_binary_target  # noqa: E402

ORDER = ("diabetes", "heart", "kidney")
SHARED_WORDS = {"bmi": "BMI target", "quit_smoking": "Stop smoking", "active": "Become physically active",
                "hba1c": "HbA1c target (doctor)", "alcohol": "Cut or stop alcohol"}
MODES = {"patient": ("patient",), "both": ("patient", "doctor")}
MODE_NAMES = {"patient": "patient only", "both": "patient + doctor"}
GATE_WORDS = {"not_for_children": "under 18",
              "diagnostic_range": "HbA1c, glucose or fasting blood sugar in a diabetes diagnostic range",
              "kidney_markers_abnormal": "eGFR or ACR in the chronic kidney disease range",
              "severe_blood_pressure": "blood pressure in the severely high range"}
SAME = 1e-9   # score differences smaller than this count as "no effect"


# =============================================================================
# SPLIT
# =============================================================================

def test_split(disease: str):
    """The held-out test rows exactly as train_model.train_and_benchmark builds them."""
    info = meta(disease)
    frame = pd.read_csv(DATA_DIR / DISEASES[disease]["dataset"])
    frame = frame[info["features"] + [info["target"]]].copy().dropna(subset=[info["target"]])
    frame[info["target"]] = clean_binary_target(frame[info["target"]])
    X, y = frame[info["features"]], frame[info["target"]]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return X_test, y_test


def check_split(disease: str, X_test, y_test) -> dict:
    """Recompute the recorded final test metrics; train_model applies a tuned threshold only for heart."""
    name = best_model(disease)
    pipeline = load_pipeline(disease, name)
    threshold = decision_threshold(disease, name)
    proba = pipeline.predict_proba(X_test)[:, 1]
    tuned = bool(meta(disease).get("threshold_results")) and threshold != 0.5
    predicted = (proba >= threshold).astype(int) if tuned else pipeline.predict(X_test)
    recomputed = calculate_metrics(y_test, predicted, proba)
    recorded = final_metrics(disease)
    differences = {key: abs(recomputed[key] - recorded[key]) for key in recorded}
    return {"model": name, "test_rows": int(len(X_test)), "test_positive": int(y_test.sum()),
            "threshold": threshold, "rule": f"score >= {threshold:g}" if tuned else "pipeline.predict (0.5)",
            "recorded": recorded, "recomputed": recomputed, "max_difference": max(differences.values()),
            "matches": set(recorded) == set(recomputed) and max(differences.values()) <= 1e-4 + 1e-12,
            "proba": proba, "predicted": predicted}


# =============================================================================
# EVALUATION
# =============================================================================

def plain(row) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def shown(disease: str, feature: str, value) -> str:
    return shown_value(feature, value, feature_summary(disease)[feature])


def change_text(disease: str, changes: list) -> str:
    return "; ".join(f"{pretty_label(c['feature'])}: {shown(disease, c['feature'], c['from'])} -> "
                     f"{shown(disease, c['feature'], c['to'])}" for c in changes)


def quartiles(values: list) -> dict | None:
    if not values:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": round(float(median), 2), "q1": round(float(q1), 2), "q3": round(float(q3), 2),
            "n": len(values)}


def share(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def evaluate_disease(disease: str, X_test, y_test, proba, sample_size: int, seed: int):
    threshold = decision_threshold(disease, best_model(disease))
    flagged = X_test[proba >= threshold]
    sample = flagged.sample(n=sample_size, random_state=seed) if len(flagged) > sample_size else flagged
    sample = sample.sort_index()
    # Warm the caches (dataset statistics, model) so the timings measure the search, not loading heart.csv.
    feature_summary(disease)
    for _, row in sample.iterrows():
        if suggest_path(disease, plain(row), owners=MODES["both"])["status"] not in GATE_STATUSES:
            break
    rows, effects, borderline = [], [], 0
    for index, row in sample.iterrows():
        values = plain(row)
        base = score(disease, values)
        if not base["flagged"]:   # the page's single-row score disagrees with the batch at the last decimal
            borderline += 1
            continue
        record = {"disease": disease, "dataset_row": int(index), "has_condition": int(y_test.loc[index]),
                  "base_score": base["score"], "threshold": base["threshold"]}
        for mode, owners in MODES.items():
            start = time.perf_counter()
            plan = suggest_path(disease, values, owners=owners)
            elapsed = (time.perf_counter() - start) * 1000
            record["gate"] = plan["status"] if plan["status"] in GATE_STATUSES else ""
            record["gate_inputs"] = ",".join(g["feature"] for g in plan["gates"])
            record |= {f"{mode}_status": plan["status"], f"{mode}_crosses": plan["crosses"],
                       f"{mode}_changes": len(plan["changes"]), f"{mode}_new_score": plan["new_score"],
                       f"{mode}_drop_pp": plan["score_drop"] * 100,
                       f"{mode}_largest_drop_pp": None if plan["largest_drop"] is None else plan["largest_drop"] * 100,
                       f"{mode}_ms": elapsed,
                       f"{mode}_levers_available": ",".join(lv["feature"] for lv in plan["levers"]),
                       f"{mode}_path": change_text(disease, plan["changes"]),
                       f"{mode}_detail": plan["changes"]}
        # How the model reacts to each allowed change on its own, gates aside (this describes the model).
        found = lever_effects(disease, values)
        record["single_step_pp"] = json.dumps({e["feature"]: round((e["smallest_score"] - base["score"]) * 100, 4)
                                               for e in found["levers"]})
        effects += [{**e, "base": base["score"]} for e in found["levers"]]
        rows.append(record)
    if borderline:
        print(f"{disease}: {borderline} sampled rows were not flagged by the single-row score and were skipped")
    return len(flagged), rows, effects


def reverses(effect: dict) -> bool:
    """The score goes both up and down along base -> 25% -> ... -> 100% of the allowed change."""
    path = [effect["base"]] + [s for _, s in effect["steps"]]
    steps = [b - a for a, b in zip(path, path[1:])]
    return any(d > SAME for d in steps) and any(d < -SAME for d in steps)


def summarise(disease: str, n_test: int, n_flagged: int, rows: list, effects: list) -> dict:
    n = len(rows)
    eligible = [r for r in rows if not r["gate"]]
    gated = Counter(r["gate"] for r in rows if r["gate"])
    result = {"test_rows": n_test, "flagged_in_test": n_flagged, "sampled": n,
              "sample_note": (f"random sample of {n:,} of {n_flagged:,} flagged test rows, seed 42" if n < n_flagged
                              else f"all {n:,} flagged test rows"),
              "with_condition": sum(r["has_condition"] for r in rows),
              "gated": dict(gated), "gated_total": sum(gated.values()),
              "gated_with_condition": sum(r["has_condition"] for r in rows if r["gate"]),
              "eligible": len(eligible), "eligible_with_condition": sum(r["has_condition"] for r in eligible),
              "modes": {}, "levers": {}, "direction": {}}
    for mode in MODES:
        crossing = [r for r in eligible if r[f"{mode}_crosses"]]
        missed = [r for r in eligible if not r[f"{mode}_crosses"]]
        reduced = [r for r in missed if r[f"{mode}_status"] == "best_reduction"]
        negligible = [r for r in missed if r[f"{mode}_status"] == "negligible"]
        times = [r[f"{mode}_ms"] for r in eligible]
        by_condition = {}
        for label, name in ((1, "with_condition"), (0, "without_condition")):
            group = [r for r in eligible if r["has_condition"] == label]
            paths = sum(r[f"{mode}_crosses"] for r in group)
            by_condition[name] = {"eligible": len(group), "with_path": paths, "with_path_pct": share(paths, len(group))}
        result["modes"][mode] = {
            "with_path": len(crossing), "with_path_pct_of_eligible": share(len(crossing), len(eligible)),
            "with_path_pct_of_evaluated": share(len(crossing), n),
            "statuses": dict(Counter(r[f"{mode}_status"] for r in rows)),
            "by_condition": by_condition,
            "changes_needed": quartiles([r[f"{mode}_changes"] for r in crossing]),
            "changes_count": {k: sum(r[f"{mode}_changes"] == k for r in crossing) for k in (1, 2, 3)},
            "score_drop_pp_with_path": quartiles([r[f"{mode}_drop_pp"] for r in crossing]),
            # Rows that did not cross: the best reductions on their own, then every such row with 0 where no
            # plan is offered (negligible, no_path, no_levers), so the figure is not read as all non-crossing rows.
            "not_crossing": len(missed),
            "best_reduction": len(reduced),
            "score_drop_pp_best_reduction": quartiles([r[f"{mode}_drop_pp"] for r in reduced]),
            "score_drop_pp_not_crossing": quartiles([r[f"{mode}_drop_pp"] for r in missed]),
            "negligible": len(negligible),
            "largest_drop_pp_negligible": quartiles([r[f"{mode}_largest_drop_pp"] for r in negligible]),
            "runtime_ms": {"median": round(float(np.median(times)), 1),
                           "p95": round(float(np.percentile(times, 95)), 1),
                           "max": round(float(np.max(times)), 1), "rows": len(times)} if times else None,
        }
    for lever in LEVERS[disease]:
        feature = lever.feature
        entry = {"owner": lever.owner, "label": lever.label}
        for mode, owners in MODES.items():
            if lever.owner not in owners:
                continue
            crossing = [r for r in eligible if r[f"{mode}_crosses"]]
            used = [c for r in crossing for c in r[f"{mode}_detail"] if c["feature"] == feature]
            available = sum(feature in r[f"{mode}_levers_available"].split(",") for r in eligible)
            entry[mode] = {"available": available, "available_pct": share(available, len(eligible)),
                           "used_in_paths": len(used), "used_pct_of_paths": share(len(used), len(crossing)),
                           "at_full_limit": sum(c["at_limit"] for c in used),
                           "at_full_limit_pct_of_uses": share(sum(c["at_limit"] for c in used), len(used))}
        result["levers"][feature] = entry
        mine = [e for e in effects if e["feature"] == feature]
        raises = sum(e["smallest_score"] > e["base"] + SAME for e in mine)
        flat = sum(abs(e["smallest_score"] - e["base"]) <= SAME for e in mine)
        several = [e for e in mine if len(e["steps"]) >= 2]
        direction = {
            "rows_where_lever_applies": len(mine),
            # (a) the change moves the score the wrong way (up) for that row
            "wrong_direction": {
                "smallest_step_raises_pct": share(raises, len(mine)),
                "smallest_step_no_effect_pct": share(flat, len(mine)),
                "smallest_step_lowers_pct": share(len(mine) - raises - flat, len(mine)),
                "full_change_raises_pct": share(sum(e["full_score"] > e["base"] + SAME for e in mine), len(mine))},
            # (b) the score changes direction along the 25-100% steps in the same row (true non-monotonicity);
            # a category change is one step, so it cannot be measured there
            "reverses_along_steps": None if lever.kind == "switch" else {
                "rows_with_several_steps": len(several),
                "reverses": sum(reverses(e) for e in several),
                "reverses_pct": share(sum(reverses(e) for e in several), len(several))},
        }
        if lever.kind == "band":
            low, high = lever.band
            for side, group in (("below", [e for e in mine if float(e["from"]) < low]),
                                ("above", [e for e in mine if float(e["from"]) > high])):
                direction[f"{side}_band"] = {"rows": len(group),
                                             "full_change_raises": sum(e["full_score"] > e["base"] + SAME for e in group)}
        result["direction"][feature] = direction
    return result


def linear_directions(disease: str) -> dict | None:
    """For a linear model, each lever's coefficient (log-odds): per standard deviation, or the category gap."""
    pipeline = load_pipeline(disease, best_model(disease))
    classifier = pipeline.steps[-1][1]
    if not hasattr(classifier, "coef_"):
        return None
    names = list(pipeline.steps[0][1].get_feature_names_out())
    weight = dict(zip(names, classifier.coef_[0]))
    found = {}
    for lever in LEVERS[disease]:
        if lever.kind == "switch":
            gaps = {start: weight.get(f"categorical__{lever.feature}_{lever.switch_to}", 0.0)
                    - weight.get(f"categorical__{lever.feature}_{start}", 0.0) for start in lever.switch_from}
            found[lever.feature] = {"kind": "category gap (to minus from)", "values": {k: round(float(v), 4)
                                                                                       for k, v in gaps.items()}}
        elif f"numeric__{lever.feature}" in weight:
            found[lever.feature] = {"kind": "per standard deviation",
                                    "value": round(float(weight[f"numeric__{lever.feature}"]), 4)}
    return found


# =============================================================================
# DEMO EXAMPLES
# =============================================================================

def form_default(disease: str, feature: str):
    """What the screening form shows before anyone edits it (optional inputs start blank)."""
    if feature in OPTIONAL_INPUTS.get(disease, []):
        return None
    info = feature_summary(disease)[feature]
    if info["kind"] == "category":
        return info["default"]
    if info["binary"]:
        return info["mode"] or 0
    return int(round(info["median"])) if feature.lower() == "age" or info["integer"] else round(info["median"], 2)


def demo_examples(path: Path) -> list:
    """Each patient's latest saved screening per condition, from a COPY of the database opened read-only."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        people = {r["id"]: r["name"] for r in db.execute("SELECT id, name FROM users WHERE role='Patient'")}
        saved = db.execute("SELECT * FROM screenings ORDER BY created_at DESC, id DESC").fetchall()
    finally:
        db.close()
    latest = {}
    for r in saved:
        latest.setdefault((r["owner_id"], r["disease"]), r)
    examples = []
    for (owner, disease), r in sorted(latest.items()):
        if owner not in people:
            continue
        saved_values = json.loads(r["values_json"] or "{}")
        values, completed = {}, []
        for feature in features(disease):
            if feature in saved_values:
                values[feature] = saved_values[feature]
            else:
                values[feature] = form_default(disease, feature)
                if values[feature] is not None:
                    completed.append(feature)
        example = {"patient": people[owner], "disease": disease, "saved_at": r["created_at"],
                   "saved_model": r["model"], "saved_score": r["score"], "saved_threshold": r["threshold"],
                   "saved_values": saved_values, "completed_with_form_defaults": completed, "values": values}
        known = [m["model"] for m in meta(disease).get("validation_model_results", [])]
        result = score(disease, values, r["model"] if r["model"] in known else None)
        example["model"] = result["model"]
        example["score"] = result["score"]
        example["threshold"] = result["threshold"]
        example["flagged"] = result["flagged"]
        example["no_score"] = result["no_score"]
        example["missing"] = result["missing"]
        example["paths"] = {mode: suggest_path(disease, values, result["model"], owners=owners) for mode, owners in MODES.items()}
        examples.append(example)
    return examples


# =============================================================================
# REPORT
# =============================================================================

def pct(x) -> str:
    return "–" if x is None else f"{x:.1f}%"


def used(entry) -> str:
    """'n% (count)' of crossing paths that use a lever; '–' for a doctor lever in a habits-only search."""
    if entry is None:
        return "–"
    return "no paths" if entry["used_pct_of_paths"] is None else f"{entry['used_pct_of_paths']:.1f}% ({entry['used_in_paths']})"


def most_used(result: dict, mode: str) -> str:
    rows = [(e[mode]["used_pct_of_paths"], e["label"]) for e in result["levers"].values()
            if mode in e and e[mode]["used_pct_of_paths"]]
    return ", ".join(f"{label} ({share_:.0f}% of paths)" for share_, label in sorted(rows, key=lambda r: -r[0])[:3])


def iqr(q) -> str:
    return "–" if not q else f"{q['median']:g} (IQR {q['q1']:g}–{q['q3']:g})"


def count_pct(part: int, whole: int) -> str:
    return f"{part:,} of {whole:,}" + (f" ({100 * part / whole:.1f}%)" if whole else "")


def gate_words(result: dict) -> str:
    return ", ".join(f"{count} {GATE_WORDS[status]}" for status, count in sorted(result["gated"].items(),
                                                                                key=lambda kv: GATE_STATUSES.index(kv[0])))


def describe_plan(disease: str, plan: dict) -> str:
    if plan["status"] == "no_score":
        return (", ".join(pretty_label(f) for f in REQUIRED_FOR_SCORE.get(disease, []) if f in plan["missing"])
                + " left blank, so the Screenings page gives no estimate and there is no plan.")
    if plan["status"] in GATE_STATUSES:
        gate = plan["gates"][0]
        return f"No plan offered ({gate['rule']}). {gate['reason']}"
    if plan["status"] == "already_below":
        return (f"Not needed: the score ({plan['base_score'] * 100:.1f}%) is already below the "
                f"{plan['threshold'] * 100:g}% threshold minus the {plan['margin'] * 100:g}-point margin.")
    if plan["status"] == "no_levers":
        return "No allowed change applies to these values."
    if plan["status"] == "no_path":
        return "No allowed change lowers the model's score on its own."
    if plan["status"] == "negligible":
        return (f"Allowed changes lower the model's score by at most {plan['largest_drop'] * 100:.2f} points, below "
                f"the {MIN_DROP * 100:g}-point minimum, so no plan is offered.")
    changes = "; ".join(f"{c['label']}: {shown(disease, c['feature'], c['from'])} -> "
                        f"{shown(disease, c['feature'], c['to'])}{' (full limit)' if c['at_limit'] else ''}"
                        for c in plan["changes"])
    outcome = ("crosses below" if plan["crosses"] else "does not reach")
    return (f"{changes}. Model score {plan['base_score'] * 100:.1f}% -> {plan['new_score'] * 100:.1f}% "
            f"({outcome} the {plan['target'] * 100:g}% target"
            f"{'' if plan['crosses'] else '; best reduction found'}).")


def dataset_facts() -> dict:
    """Properties of the training data that shape the results, measured rather than assumed."""
    facts = {}
    diabetes = pd.read_csv(DATA_DIR / DISEASES["diabetes"]["dataset"])
    for feature in ("HbA1c_level", "blood_glucose_level"):
        rates = diabetes.groupby(feature)["diabetes"].mean()
        levels = sorted(rates.index)
        facts[feature] = {
            "distinct_values": len(levels),
            "all_with_diabetes_from": min((v for v in levels if all(rates[w] == 1 for w in levels if w >= v)),
                                          default=None),
            "none_with_diabetes_up_to": max((v for v in levels if all(rates[w] == 0 for w in levels if w <= v)),
                                            default=None)}
    facts["diabetes_rows_in_gate_range"] = round(float(in_value_range("diabetes", diabetes).mean()), 3)
    facts["diabetes_rows_under_18"] = round(float((diabetes["age"] < 18).mean()), 3)
    kidney = pd.read_csv(DATA_DIR / DISEASES["kidney"]["dataset"])
    target = DISEASES["kidney"]["target"]
    facts["kidney_lever_max_abs_correlation"] = round(float(max(
        abs(kidney[lever.feature].corr(kidney[target])) for lever in LEVERS["kidney"])), 3)
    facts["kidney_with_disease_share"] = round(float(kidney[target].mean()), 3)
    facts["kidney_rows_in_marker_range"] = round(float(in_value_range("kidney", kidney, ("kidney",)).mean()), 3)
    facts["kidney_rows_in_gate_range"] = round(float(in_value_range("kidney", kidney).mean()), 3)
    facts["kidney_highest_blood_pressure"] = f"{kidney['SystolicBP'].max():g}/{kidney['DiastolicBP'].max():g}"
    return facts


def in_value_range(disease: str, data: pd.DataFrame, kinds=None) -> pd.Series:
    """Rows where an entered value trips one of the condition's urgent value rules (age aside), optionally of *kinds*."""
    tripped = pd.Series(False, index=data.index)
    for rule in RULES[disease]:
        if rule.kind != ADULTS_ONLY and (kinds is None or rule.kind in kinds):
            tripped |= data[rule.feature].map(rule.trips).astype(bool)
    return tripped


def gate_summary() -> str:
    """The value gates in words, per condition, from the shared table in ahead/triage.py."""
    parts = []
    for disease in ORDER:
        rules = [rule.text() for rule in RULES[disease] if rule.kind != ADULTS_ONLY]
        if rules:
            listed = ", ".join(rules[:-1]) + ", or " + rules[-1] if len(rules) > 1 else rules[0]
            parts.append(f"on the {DISEASES[disease]['label'].lower()} screening: {listed}")
    return "; ".join(parts)


def direction_text(results: dict, coefficients: dict) -> list:
    """The plain-words reading of the direction tables, from measured figures."""
    lines = []
    heart = results["heart"]["direction"]
    sleep, alcohol = heart["SleepHours"], heart["AlcoholDrinkers"]
    linear = [f for f, d in heart.items() if d["reverses_along_steps"]]
    reversals = sum(heart[f]["reverses_along_steps"]["reverses"] for f in linear)
    checked = sum(heart[f]["reverses_along_steps"]["rows_with_several_steps"] for f in linear)
    if coefficients:
        sleep_weight = coefficients["SleepHours"]["value"]
        alcohol_gap = coefficients["AlcoholDrinkers"]["values"]["Yes"]
        lines.append(
            f"- **{DISEASES['heart']['label']}** ({best_model('heart')}): a logistic regression's score moves one way "
            f"only as an input rises, so this model is monotonic "
            f"by construction, and none of the {checked} rows with several BMI or sleep steps changes direction "
            f"({reversals} reversals). Its 'wrong way' figures come from the direction the model learnt, read from "
            f"its fitted coefficients. Sleep hours carry a coefficient of {sleep_weight:+.4f} per standard deviation, "
            f"so more sleep goes with a lower score: moving down toward 9 h raised the score in "
            f"{sleep['above_band']['full_change_raises']} of {sleep['above_band']['rows']} rows above 9 h, and moving "
            f"up toward 7 h raised it in {sleep['below_band']['full_change_raises']} of {sleep['below_band']['rows']} "
            f"rows below 7 h. Switching drinking from 'Yes' to 'No' adds {alcohol_gap:+.4f} to the log-odds, so "
            f"stopping alcohol raised this model's score in "
            f"{pct(alcohol['wrong_direction']['full_change_raises_pct'])} of the rows where it applies. These are "
            "associations in the survey data, not effects; in this model the 'down to 9 h' half of the sleep rule and "
            "'stop drinking' can never lower the score, so the engine never suggests them.")
    for disease in ("diabetes", "kidney"):
        found = [(results[disease]["levers"][f]["label"], d["reverses_along_steps"]["reverses_pct"],
                  d["reverses_along_steps"]["rows_with_several_steps"])
                 for f, d in results[disease]["direction"].items()
                 if d["reverses_along_steps"] and d["reverses_along_steps"]["rows_with_several_steps"]]
        if found:
            text = ", ".join(f"{label} {pct(p)} of {n}" for label, p, n in found)
            lines.append(
                f"- **{DISEASES[disease]['label']}** ({best_model(disease)}) is a tree ensemble trained without any "
                f"monotonic constraint, and it is not monotonic in practice: the score changes direction along the "
                f"25-100% steps in the same row for {text} rows with several steps. A small step can raise the score "
                "where a larger one lowers it, so a path can rest on a bump in the model rather than a consistent "
                "pattern.")
    return lines


def report(generated: str, checks: dict, results: dict, rows: list, demos: list, sample: int, facts: dict,
           coefficients: dict) -> str:
    hba1c, glucose = facts["HbA1c_level"], facts["blood_glucose_level"]
    lines = [
        "# Path to lower risk: offline evaluation (step 1 of 7)",
        "",
        f"Generated {generated} by `scripts/evaluate_risk_plans.py`. Engine: `ahead/risk_plan.py`. "
        "No user interface yet.",
        "",
        "## What was tested",
        "",
        "When AHEAD flags a screening, the new engine looks for the smallest realistic set of changes to inputs a "
        "person can change that would bring the model's score below the screening threshold, with a safety margin "
        "of 2 percentage points. It searches twice: once with the patient's own habits only (weight, smoking, "
        "physical activity, sleep, alcohol and, for kidney, diet), and once adding targets a doctor would set "
        "(HbA1c, blood glucose, blood pressure). Each numeric change is tried at 25%, 50%, 75% and 100% of the most "
        "it may change in one plan, each category change (such as current smoker to former smoker) as a single "
        "step, and plans have at most 3 changes. Age, sex, medical and family history, self-rated health and the "
        "kidney laboratory markers are never changed, and a value the person left blank is never filled in.",
        "",
        "Three safeguards decide whether a plan is offered at all:",
        "",
        "- **Gates.** No plan is offered to anyone under 18 (the rules are written for adults), or when an entered "
        f"value is already in a range that needs a doctor ({gate_summary()}). These results need a "
        "clinician, not a plan to lower a screening score. The gates are the urgent rules in `ahead/triage.py`, the "
        "same table that puts a Screenings result on the \"See a doctor soon\" track; the cut-offs are prototype "
        "values for a clinician to confirm (see *Gates* below).",
        "- **Every change helps on its own.** A change is only used if it lowers that person's score by itself; "
        "a change that only helps through an interaction with another change in the model is dropped. A crossing "
        "path also needs every one of its changes: dropping any one stops it crossing.",
        f"- **Minimum drop.** When no path crosses, the best reduction is offered only if it lowers the score by "
        f"at least {MIN_DROP * 100:g} percentage point; smaller drops are reported as *negligible* and no plan is "
        "offered.",
        "",
        f"**{CAVEAT}**",
        "",
        "## Is this the right test data?",
        "",
        "The test rows were rebuilt exactly as `train_model.py` builds them (20% held out, `random_state=42`, "
        "stratified by the outcome). To prove it is the same split, the recommended model's final test metrics were "
        "recomputed on these rows and compared with the figures recorded in `models/model_meta.json` when the "
        "models were trained.",
        "",
        "| Condition | Model | Test rows | Accuracy (recorded / recomputed) | Recall | Precision | ROC AUC | Match |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for disease in ORDER:
        c = checks[disease]
        rec, new = c["recorded"], c["recomputed"]
        lines.append(f"| {DISEASES[disease]['label']} | {c['model']} | {c['test_rows']:,} | "
                     f"{rec['accuracy']:.4f} / {new['accuracy']:.4f} | {rec['recall_disease']:.4f} / "
                     f"{new['recall_disease']:.4f} | {rec['precision_disease']:.4f} / {new['precision_disease']:.4f} | "
                     f"{rec['roc_auc']:.4f} / {new['roc_auc']:.4f} | "
                     f"{'yes (all 8 metrics identical)' if c['max_difference'] == 0 else ('yes' if c['matches'] else 'NO')} |")
    lines += [
        "",
        "Every recorded metric (accuracy, precision, recall, F1 for each class, macro F1, F2 and ROC AUC) was "
        "reproduced to 4 decimal places, so the evaluation below uses the same held-out test rows the models were "
        "scored on, rows the models never saw in training.",
        "",
        "## Headline results",
        "",
        f"Only test rows the recommended model flags are used (score at or above its threshold). Where a condition "
        f"had more than {sample:,} flagged rows, a fixed random sample of {sample:,} (seed 42) was used. *Has the "
        "condition* is the row's label in the dataset. *Eligible* rows are the ones no gate stops; path rates are "
        "shares of eligible rows.",
        "",
        "| Condition | Flagged test rows | Evaluated | Has the condition | No plan offered (gate) | Eligible "
        "| Path, habits only | Path, habits + doctor targets | Changes needed (habits + doctor, median) "
        "| Score drop on a path, points (habits + doctor) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for disease in ORDER:
        r = results[disease]
        p, b = r["modes"]["patient"], r["modes"]["both"]
        lines.append(
            f"| {DISEASES[disease]['label']} | {r['flagged_in_test']:,} of {r['test_rows']:,} | {r['sampled']:,} | "
            f"{count_pct(r['with_condition'], r['sampled'])} | {count_pct(r['gated_total'], r['sampled'])} | "
            f"{r['eligible']:,} | {pct(p['with_path_pct_of_eligible'])} ({p['with_path']}) | "
            f"{pct(b['with_path_pct_of_eligible'])} ({b['with_path']}) | {iqr(b['changes_needed'])} | "
            f"{iqr(b['score_drop_pp_with_path'])} |")
    lines += ["", "In plain words:", ""]
    for disease in ORDER:
        r = results[disease]
        p, b = r["modes"]["patient"], r["modes"]["both"]
        label = DISEASES[disease]["label"]
        gated = (f"{count_pct(r['gated_total'], r['sampled'])} get no plan because a gate trips "
                 f"({gate_words(r)}). " if r["gated_total"] else "No heart input trips a gate. ")
        split = "; ".join(
            f"{name.replace('_', ' ')}: habits only {count_pct(p['by_condition'][name]['with_path'], p['by_condition'][name]['eligible'])}, "
            f"habits + doctor {count_pct(b['by_condition'][name]['with_path'], b['by_condition'][name]['eligible'])}"
            for name in ("with_condition", "without_condition"))
        lines.append(
            f"- **{label}** ({r['sample_note']}): {count_pct(r['with_condition'], r['sampled'])} of the evaluated "
            f"flagged rows have the condition in the data. {gated}That leaves {r['eligible']:,} eligible rows "
            f"({r['eligible_with_condition']} with the condition). Among them, habit changes alone brought the "
            f"model's score below the target for {pct(p['with_path_pct_of_eligible'])} ({p['with_path']}), and "
            f"adding doctor targets for {pct(b['with_path_pct_of_eligible'])} ({b['with_path']}); over all evaluated "
            f"flagged rows that is {pct(p['with_path_pct_of_evaluated'])} and {pct(b['with_path_pct_of_evaluated'])}. "
            f"Paths by label among eligible rows: {split}. "
            + (f"With habits only, paths needed {iqr(p['changes_needed'])} changes and lowered the score by "
               f"{iqr(p['score_drop_pp_with_path'])} points. " if p["with_path"] else "")
            + (f"With doctor targets, paths needed {iqr(b['changes_needed'])} changes and lowered the score by "
               f"{iqr(b['score_drop_pp_with_path'])} points. " if b["with_path"] else "")
            + (f"Of the {b['not_crossing']} eligible rows where habits + doctor targets did not cross, "
               f"{b['best_reduction']} had a best reduction of at least {MIN_DROP * 100:g} point "
               f"({iqr(b['score_drop_pp_best_reduction'])} points) and {b['negligible']} only a negligible one; "
               f"counting every row without a plan as 0, the drop over all {b['not_crossing']} was "
               f"{iqr(b['score_drop_pp_not_crossing'])} points. " if b["not_crossing"] else "")
            + (f"Most used in paths with doctor targets: {most_used(r, 'both')}." if most_used(r, "both") else ""))
    lines += ["", "Runtime per eligible row, one search at a time (milliseconds; gated rows return at once):", "",
              "| Condition | Search | Rows | Median | 95th percentile | Slowest |", "|---|---|---|---|---|---|"]
    for disease in ORDER:
        for mode in MODES:
            t = results[disease]["modes"][mode]["runtime_ms"]
            if t:
                lines.append(f"| {DISEASES[disease]['label']} | {MODE_NAMES[mode]} | {t['rows']} | {t['median']:.0f} | "
                             f"{t['p95']:.0f} | {t['max']:.0f} |")
    lines += ["", "Detailed counts per outcome (rows):", "",
              "| Condition | Search | Crosses | Best reduction (1 point or more) | Negligible (under 1 point) "
              "| No change lowers the score on its own | No change applies | Gate: under 18 "
              "| Gate: diabetes diagnostic range | Gate: kidney markers | Gate: severe blood pressure |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for disease in ORDER:
        for mode in MODES:
            s = results[disease]["modes"][mode]["statuses"]
            lines.append(f"| {DISEASES[disease]['label']} | {MODE_NAMES[mode]} | {s.get('crosses', 0)} | "
                         f"{s.get('best_reduction', 0)} | {s.get('negligible', 0)} | {s.get('no_path', 0)} | "
                         f"{s.get('no_levers', 0)} | {s.get('not_for_children', 0)} | {s.get('diagnostic_range', 0)} | "
                         f"{s.get('kidney_markers_abnormal', 0)} | {s.get('severe_blood_pressure', 0)} |")
    lines += ["", "## Which changes the paths use", "",
              "*Applies* is the share of eligible rows where the change is allowed at all (for example, the BMI change "
              "applies only at a BMI of 25 or more). *Used* is the share of crossing paths that include it. *At full "
              "limit* is how often, when used, it had to go all the way to its maximum allowed change (category changes "
              "such as stopping smoking are always a full step).", ""]
    for disease in ORDER:
        lines += [f"**{DISEASES[disease]['label']}**", "",
                  "| Change | Who | Applies | Used (habits only) | Used (habits + doctor) | At full limit (habits + doctor) |",
                  "|---|---|---|---|---|---|"]
        for feature, entry in results[disease]["levers"].items():
            p, b = entry.get("patient"), entry["both"]
            lines.append(f"| {entry['label']} | {entry['owner']} | {pct(b['available_pct'])} | "
                         f"{used(p)} | {used(b)} | {pct(b['at_full_limit_pct_of_uses'])} |")
        lines.append("")
    lines += ["## Does a change move the score the wrong way?", "",
              "This describes the models, so it uses every evaluated flagged row, gated or not. For each change and "
              "each row where its rule applies, each allowed step was applied on its own. Two different things are "
              "measured:", "",
              "- *Wrong direction*: the smallest step, or the full allowed change, RAISES the score for that row. "
              "This can happen in any model, including a monotonic one, when the model learnt that the input goes "
              "the other way.",
              "- *Changes direction along the steps*: in the same row, the score goes up at one step and down at "
              "another between 0%, 25%, 50%, 75% and 100% of the allowed change. Only this shows the model is not "
              "monotonic for that input. A category change is a single step, so it shows '–'.", "",
              "The engine uses only changes that lower the score on their own, so neither kind of change can appear "
              "in a suggested path; the tables show how often that filter matters.", "",
              "| Condition | Change | Rows where it applies | Smallest step raises score | No effect | Lowers score "
              "| Full change raises score | Changes direction along the steps |",
              "|---|---|---|---|---|---|---|---|"]
    for disease in ORDER:
        for feature, d in results[disease]["direction"].items():
            w, rev = d["wrong_direction"], d["reverses_along_steps"]
            along = "–" if rev is None else f"{pct(rev['reverses_pct'])} of {rev['rows_with_several_steps']}"
            lines.append(f"| {DISEASES[disease]['label']} | {pretty_label(feature)} | {d['rows_where_lever_applies']} | "
                         f"{pct(w['smallest_step_raises_pct'])} | {pct(w['smallest_step_no_effect_pct'])} | "
                         f"{pct(w['smallest_step_lowers_pct'])} | {pct(w['full_change_raises_pct'])} | {along} |")
    lines += ["", *direction_text(results, coefficients), ""]
    lines += ["## Worked examples: demo patients", ""]
    if not demos:
        lines += ["No demo database was given (`--demo-db`), so there are no demo examples.", ""]
    else:
        lines += ["Each demo patient's latest saved screening was read from a copy of the demo database. The demo "
                  "seed saved only some inputs for some patients; inputs missing from a saved record were filled with "
                  "the values the screening form shows before anyone edits it (marked below), while optional lab and "
                  "blood-pressure values the form starts blank stay blank. The scores below are recomputed with the "
                  "current model, so they can differ from the hand-entered scores the demo seed stored.", ""]
        for e in demos:
            label = DISEASES[e["disease"]]["label"]
            lines.append(f"### {e['patient']}: {label}")
            lines.append("")
            entered = ", ".join(f"{pretty_label(k)} {shown(e['disease'], k, v)}" for k, v in e["saved_values"].items())
            lines.append(f"- Saved values: {entered}.")
            if e["completed_with_form_defaults"]:
                lines.append("- Filled with form defaults (not in the saved record): "
                             + ", ".join(f"{pretty_label(f)} {shown(e['disease'], f, e['values'][f])}"
                                         for f in e["completed_with_form_defaults"]) + ".")
            if e["missing"]:
                lines.append("- Left blank (never filled in by a plan): "
                             + ", ".join(pretty_label(f) for f in e["missing"]) + ".")
            if e["no_score"]:
                lines.append(f"- Result: no score. {describe_plan(e['disease'], e['paths']['patient'])}")
                if e["saved_score"] is not None:
                    lines.append(f"  (The demo seed stored a score of {e['saved_score'] * 100:.1f}% for this record; "
                                 "the current page would not produce one.)")
            else:
                saved = ("no saved score" if e["saved_score"] is None else
                         f"Saved score {e['saved_score'] * 100:.1f}% (threshold {e['saved_threshold'] * 100:g}%)")
                lines.append(f"- {saved}; "
                             f"recomputed with {e['model']}: {e['score'] * 100:.1f}% against a threshold of "
                             f"{e['threshold'] * 100:g}% ({'flagged' if e['flagged'] else 'not flagged'}).")
                lines.append(f"- Habits only: {describe_plan(e['disease'], e['paths']['patient'])}")
                lines.append(f"- Habits + doctor targets: {describe_plan(e['disease'], e['paths']['both'])}")
            lines.append("")
    lines += ["## Worked examples: held-out test rows", "",
              "For each condition, the first evaluated row (by row number in the dataset file, counting from 0) with "
              "a habits-only path, or if there is none, the first with a habits + doctor path.", ""]
    for disease in ORDER:
        mine = [r for r in rows if r["disease"] == disease]
        pick = next((r for r in mine if r["patient_crosses"]), None) or next((r for r in mine if r["both_crosses"]), None)
        if pick is None:
            lines += [f"- **{DISEASES[disease]['label']}**: no evaluated row had a crossing path.", ""]
            continue
        lines.append(f"- **{DISEASES[disease]['label']}**, dataset row {pick['dataset_row']} "
                     f"(score {pick['base_score'] * 100:.1f}%, threshold {pick['threshold'] * 100:g}%; "
                     f"{'has' if pick['has_condition'] else 'does not have'} the condition in the data):")
        for mode in MODES:
            path = pick[f"{mode}_path"] or "none"
            lines.append(f"  - {MODE_NAMES[mode]}: {path} -> score {pick['base_score'] * 100:.1f}% -> "
                         f"{pick[f'{mode}_new_score'] * 100:.1f}%"
                         f" ({'crosses' if pick[f'{mode}_crosses'] else pick[f'{mode}_status'].replace('_', ' ')}).")
        lines.append("")
    lines += ["## One plan across conditions", "",
              "A later step lets a patient and doctor agree one plan in shared terms and see its effect on every "
              "condition's score. `shared_plan()` maps each shared target onto each model's own input and coding, and "
              "each mapped change must pass that model's lever rule. A target past a limit is applied up to the "
              "strongest step the search itself would use (on the step grid and inside the training range) and marked "
              "as limited; when no such step exists, it is skipped with a reason. A condition whose values trip a gate "
              "gets no changes. It also reports each change's effect on its own, so a change that raises one model's "
              "score is visible.", "",
              "| Shared target | Diabetes model | Heart model | Kidney model |", "|---|---|---|---|"]
    for key, mapping in SHARED_TARGETS.items():
        lines.append(f"| {SHARED_WORDS[key]} | " + " | ".join(
            pretty_label(mapping[d]) if d in mapping else "not in this model" for d in ORDER) + " |")
    lines += ["", "Not shared on purpose: the diabetes data's blood glucose is read as a random (non-fasting) value and "
              "the kidney data's is fasting, so neither is mapped onto the other; sleep hours (heart) and the "
              "sleep-quality score (kidney) measure different things; blood pressure and diet quality exist only in "
              "the kidney model. \"Become active\" sets the heart answer to Yes, but the kidney data records activity "
              "as a 0-10 number, so there it is mapped (approximately) to the largest allowed increase. An alcohol "
              "target is a share of intake to cut; the heart data only records whether someone drinks, so only a full "
              "stop maps there, and in the heart model that change raises the score.", ""]
    lines += ["## Gates", "",
              "Prototype cut-offs for a clinician to confirm, not guideline citations. They are the urgent rules in "
              "`ahead/triage.py`; the Screenings track uses the same table, so an entered value that stops a plan here "
              "also puts the result on the \"See a doctor soon\" track. When any trips, no plan is offered; a blank value never trips a "
              "gate. The plan engine applies the age gate to every result, while the Screenings track counts age only "
              "for a flagged result. The heart inputs have none (every heart age category is 18 or older).", "",
              "| Condition | Input | No plan when | Status | Why |", "|---|---|---|---|---|"]
    for disease in ORDER:
        for row in gate_table(disease):
            lines.append(f"| {DISEASES[disease]['label']} | {row['input']} | {row['rule']} | `{row['status']}` | "
                         f"{row['reason']} |")
    lines += ["", "## Lever rules used", "",
              "These are prototype assumptions for a clinician to review, not guideline citations. Every category "
              "spelling, unit and range was checked against the training data. Every changed value also stays inside "
              "the range seen in the training data. One rule differs from the starting design: the diabetes data does "
              "not say its blood glucose was fasting, so it is read as random glucose and the target rule starts at "
              "140 mg/dL instead of the fasting 100 mg/dL cut-off.", ""]
    for disease in ORDER:
        lines += [f"**{DISEASES[disease]['label']}**", "", "| Input | Who | Shown as | Rule | Tracker goal | Note |",
                  "|---|---|---|---|---|---|"]
        for row in lever_table(disease):
            lines.append(f"| {row['input']} | {row['owner']} | {row['label']} | {row['rule']} | {row['goal'] or '–'} | "
                         f"{row['note'] or ''} |")
        lines.append("")
    kidney = results["kidney"]
    lines += [
        "## Limitations",
        "",
        "These figures describe how the trained models react to changed inputs, not what would happen to a person. "
        "The models learnt associations in historical or generated data, so a lower score after a change is not "
        "evidence that the change lowers anyone's risk, and the size of a score drop is not an effect size. The "
        "evaluation uses dataset rows rather than real patients, and only the rows the model flags. The lever rules "
        "and gate cut-offs are prototype assumptions written for a clinician to review; they are not personalised "
        "and do not consider time, cost, medication or how hard a change is for a given person. The search is coarse "
        "on purpose (four step sizes, at most three changes), so a finer or larger plan might exist.",
        "",
        f"The gates remove most flagged diabetes and kidney rows, so those results rest on few rows. In the diabetes "
        f"data, {facts['diabetes_rows_in_gate_range']:.0%} of all rows have HbA1c of 6.5% or more or glucose of 200 "
        f"mg/dL or more and {facts['diabetes_rows_under_18']:.0%} are under 18; the model flags mostly rows that "
        f"already meet a diagnostic cut-off. HbA1c and glucose there take only {hba1c['distinct_values']} and "
        f"{glucose['distinct_values']} distinct values, every record with HbA1c of "
        f"{hba1c['all_with_diabetes_from']:g}% or more or glucose of {glucose['all_with_diabetes_from']:g} mg/dL or "
        f"more has diabetes, and no record with HbA1c of {hba1c['none_with_diabetes_up_to']:g}% or less or glucose of "
        f"{glucose['none_with_diabetes_up_to']:g} mg/dL or less has it, so the model's score jumps at those gaps: a "
        "small lab target can move it a lot, which is a property of this dataset, not a clinical effect. In the "
        f"kidney data, {facts['kidney_rows_in_marker_range']:.0%} of all rows have eGFR below 60 or ACR of 30 mg/g or "
        f"more and {facts['kidney_rows_in_gate_range']:.0%} trip at least one value gate once HbA1c and fasting blood "
        "sugar are counted (no row reaches the blood pressure gate: the data stops at "
        f"{facts['kidney_highest_blood_pressure']} mmHg), so only {kidney['eligible']} of the {kidney['sampled']} "
        "evaluated flagged kidney test rows are eligible. Its inputs "
        "are spread evenly across their ranges and only weakly related to the diagnosis (no lever input's "
        f"correlation with it exceeds {facts['kidney_lever_max_abs_correlation']:.2f}), and "
        f"{facts['kidney_with_disease_share']:.0%} of its rows have the disease, so lifestyle changes barely move that "
        "model and kidney paths should not be shown to people without a warning. The heart model is a logistic "
        "regression over mostly yes/no answers, so a plan can only flip a few categories and lower BMI.",
        "",
        "In the test rows every input is filled in; the screening form lets people leave diabetes and kidney labs "
        "blank, which lowers the threshold and makes a path harder to reach, and that case is not measured here. "
        "Runtimes were measured one patient at a time on the development laptop, after loading the data and model "
        "once, and include rescoring the chosen plan through the screening page's own single-row path.",
        "",
        "## Method details",
        "",
        f"- Step sizes: {', '.join(f'{f:.0%}' for f in LEVEL_FRACTIONS)} of each numeric change's allowed range, "
        "rounded toward the current value (BMI, HbA1c, alcohol, activity, diet and sleep quality to 0.1; glucose, "
        "blood pressure, fasting blood sugar and sleep hours to whole numbers).",
        "- Target: threshold minus 2 points. The threshold is the one the Screenings page applies to the same values "
        "(including the lower threshold used when optional labs are blank).",
        "- Order of checks: no score (a required kidney test blank), then gates, then 'already below the target', "
        "then 'no change applies', then the search.",
        "- Each allowed step is scored on its own first; only steps that lower the score on their own are kept. "
        "If none is left, the outcome is 'no change lowers the score on its own'.",
        "- Choice: the fewest changes that cross; among those, the smallest total share of allowed change, then the "
        "lowest score. Without a crossing plan, the best reduction: among plans that lower the score by at least "
        f"{MIN_DROP * 100:g} point, the fewest changes among those within 0.1 point of the lowest score. When no plan "
        f"lowers it by {MIN_DROP * 100:g} point, the outcome is 'negligible' and no plan is offered.",
        "- The direction checks use every change whose rule applies, for patient and doctor alike, on every evaluated "
        "flagged row (gates aside).",
        "- Charts: not drawn, because matplotlib is not in `requirements.txt`.",
        "- Files: `summary.json` (all figures), `rows.csv` (one row per evaluated test row, with its gate if any).",
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=1000, help="most flagged test rows per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-db", type=Path, help="a COPY of an AHEAD database (opened read-only)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    checks, splits = {}, {}
    for disease in ORDER:
        X_test, y_test = test_split(disease)
        checks[disease] = check_split(disease, X_test, y_test)
        splits[disease] = (X_test, y_test)
        c = checks[disease]
        print(f"{disease:9} split check: {c['test_rows']:,} test rows, max metric difference {c['max_difference']:.6f} "
              f"-> {'OK' if c['matches'] else 'MISMATCH'}", flush=True)
    public_checks = {d: {k: v for k, v in c.items() if k not in ("proba", "predicted")} for d, c in checks.items()}
    if not all(c["matches"] for c in checks.values()):
        (args.out / "summary.json").write_text(json.dumps(jsonable(
            {"generated": generated, "error": "The recorded final test metrics could not be reproduced; nothing was "
             "evaluated.", "split_check": public_checks}), indent=2))
        (args.out / "report.md").write_text("# Path to lower risk: evaluation stopped\n\nThe held-out test split could "
                                            "not be reproduced (recorded and recomputed metrics differ), so nothing "
                                            "was evaluated. See summary.json.\n")
        print("Split check failed; nothing evaluated.", file=sys.stderr)
        sys.exit(1)

    results, rows = {}, []
    for disease in ORDER:
        X_test, y_test = splits[disease]
        c = checks[disease]
        if disease in ("diabetes", "kidney"):
            agree = int(((c["proba"] >= c["threshold"]).astype(int) == np.asarray(c["predicted"])).sum())
            c["flag_rule_agreement"] = f"{agree} of {len(X_test)} rows: score >= threshold equals pipeline.predict"
        start = time.perf_counter()
        n_flagged, mine, effects = evaluate_disease(disease, X_test, y_test, c["proba"], args.sample, args.seed)
        results[disease] = summarise(disease, len(X_test), n_flagged, mine, effects)
        rows += mine
        r = results[disease]
        print(f"{disease:9} evaluated {len(mine):,} of {n_flagged:,} flagged rows in {time.perf_counter() - start:.0f} s; "
              f"gated {r['gated_total']}, eligible {r['eligible']}; paths of eligible: patient "
              f"{r['modes']['patient']['with_path_pct_of_eligible']}%, both {r['modes']['both']['with_path_pct_of_eligible']}%",
              flush=True)

    demos = demo_examples(args.demo_db) if args.demo_db else []
    facts = dataset_facts()
    coefficients = linear_directions("heart")
    public_checks = {d: {k: v for k, v in c.items() if k not in ("proba", "predicted")} for d, c in checks.items()}
    summary = {
        "generated": generated, "caveat": CAVEAT,
        "settings": {"sample_per_condition": args.sample, "seed": args.seed, "margin": 0.02, "max_changes": 3,
                     "min_drop": MIN_DROP, "level_fractions": list(LEVEL_FRACTIONS),
                     "modes": {k: list(v) for k, v in MODES.items()}},
        "split_check": public_checks, "results": results, "dataset_facts": facts,
        "heart_coefficients": coefficients,
        "levers": {d: lever_table(d) for d in ORDER}, "gates": {d: gate_table(d) for d in ORDER if GATES[d]},
        "demo_examples": [{k: v for k, v in e.items() if k != "paths"} | {"paths": {
            m: {k: p[k] for k in ("status", "crosses", "gates", "base_score", "threshold", "target", "new_score",
                                  "largest_drop", "changes")}
            for m, p in e["paths"].items()}} for e in demos],
        "charts": "skipped: matplotlib is not in requirements.txt",
    }
    (args.out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_detail")} for r in rows])
    table.to_csv(args.out / "rows.csv", index=False)
    (args.out / "report.md").write_text(report(generated, public_checks, results, rows, demos, args.sample, facts,
                                               coefficients))
    print(f"Wrote {args.out / 'summary.json'}, {args.out / 'rows.csv'} and {args.out / 'report.md'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*serialized model.*")
    set_log_level("error")   # Streamlit's bare-mode cache warnings are noise in a script
    main()
