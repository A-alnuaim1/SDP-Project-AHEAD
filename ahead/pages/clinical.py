"""Doctor-only intake, validation, scoring, review queue (with each record's track, ahead/triage.py) and reports."""

import hashlib
import json
import math
from html import escape

import pandas as pd
import streamlit as st

from ahead.clinical_data import (dob_age, field_info, issue_text, model_frame, prepare_upload, shown_value,
                                  validate_row)
from ahead.visuals import summary_tile
from ahead.components import (current_user, empty_state, markdown_literal, model_label, model_selector, page_header, result_card,
                              section_heading, track_panel)
from ahead.config import AGE_FIELD, DISEASES, GATE_ONLY_FIELDS, OPTIONAL_INPUTS, form_fields, is_legacy, pretty_label
from ahead.reports import patient_report
from ahead.resources import feature_summary, features, model_test_metrics, predict_probability
from ahead.storage import insert_records, list_records, save_scoring_run, update_record
from ahead.screening import identify_risk_factors, individual_sensitivity, model_row
from ahead.platform_data import append_record_prediction, record_prediction_history, to_local
from ahead.triage import TRACKS, track_text, triage_saved

ID_COLUMNS = ["PatientID", "patient_id", "Patient ID", "ID", "id"]
MAX_ROWS = 5000
SOURCES = {"batch": "Cohort scoring", "corrected": "Corrected record"}
# A datetime column (not preformatted text) so a header click sorts chronologically.
LOCAL_TIME = {"Date": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm")}
TRACK_LABELS = [track_text(track, "doctor")["label"] for track in TRACKS]   # Urgent, At risk, Not flagged, No score
LEGACY_NOTE = ("This record was saved with an earlier version of this condition's form. It keeps its saved values, "
               "score and track; the current model never re-scores it. Upload it again in the current format to score "
               "it with the current model.")


def _legacy(record, disease):
    """A record saved with the condition's earlier form (config.LEGACY_FIELDS): shown and sorted, never re-scored."""
    return is_legacy(disease, json.loads(record["values_json"]))


def _when(stamp):
    moment = to_local(stamp)
    return f"{moment.day} {moment:%b %Y, %H:%M}"


def _issues(issues):
    return " · ".join(map(issue_text, issues))


def _status(record):
    """The status the doctor reads: a record scored below the threshold is 'Not flagged', not 'Ready' (unscored).
    Display only; the stored status stays 'Ready'."""
    return "Not flagged" if record["status"] == "Ready" and record["score"] is not None else record["status"]


def _track(record, disease):
    """The record's track (ahead/triage.py). The urgent rules need no score, so an unscored record is already Urgent
    or 'No score'. An incomplete record (never scored) is Urgent when an entered number trips a rule, otherwise None:
    a value past the model's input range (a blood pressure of 210/125, a glucose of 420) fails validation but is the
    most pressing kind, so it is never hidden. Blanks and non-numbers never trip a rule; _unchecked asks the doctor
    to confirm a value that failed validation."""
    if record["status"] == "Incomplete" or json.loads(record["issues_json"]):
        track = triage_saved(disease, record["values_json"])
        return track if track["urgent"] else None
    return triage_saved(disease, record["values_json"], record["score"], record["threshold"])


def _unchecked(track, issues):
    """The record panel's note when an urgent reason rests on a value that failed validation, so may be an entry
    error: 'Blood glucose (420 mg/dL) failed the input check above ...'. Empty otherwise."""
    failed = {issue.split(": ", 1)[0] for issue in issues}
    named = [f"{r['label']} ({r['value_text']})" for r in track["reasons"] if failed & set(r["features"])]
    if not named:
        return ""
    if len(named) == 1:
        return (f"{named[0]} failed the input check above and may be an entry error. Check it against the source record "
                "before acting on it.")
    return (f"{', '.join(named[:-1])} and {named[-1]} failed the input check above and may be entry errors. Check them "
            "against the source record before acting on them.")


def _editable(feature, value, info):
    """A stored value as the record editor shows it: words for 0/1 answers and categories, whole numbers without
    '.0'. Other numbers stay unrounded, so saving the record back never changes them."""
    if value is None:
        return ""
    whole = isinstance(value, (int, float)) and math.isfinite(value) and float(value).is_integer()
    return shown_value(feature, value, info) if whole or info["kind"] == "category" else str(value)


def _cohort_label(records):
    """'heart_a.csv · 4 Nov 2026, 14:05 · 6 rows'; cohorts saved before file names were kept read 'Upload · …'."""
    first, count = min(records, key=lambda r: r["created_at"]), len(records)
    return f"{first.get('source_name') or 'Upload'} · {_when(first['created_at'])} · {count} row{'s' if count != 1 else ''}"


def _upload():
    file = st.file_uploader("Upload patient records", type=["csv", "xlsx"],
                            help="Use fictional or de-identified records for this prototype.")
    if file is None:
        return None, None, None
    raw = file.getvalue()
    if len(raw) > 15 * 1024 * 1024:
        st.error("File exceeds the 15 MB upload limit.")
        return None, None, None
    try:
        # Text cells keep IDs such as 00042 intact (no lost zeros or '.0'); validation parses the numbers.
        frame = pd.read_csv(file, dtype=str) if file.name.lower().endswith(".csv") else pd.read_excel(file, dtype=str)
    except Exception:  # corrupt, encrypted or mislabelled files fail in many parser-specific ways
        st.error("Could not read this file. Check the CSV/XLSX format.")
        return None, None, None
    if frame.empty or len(frame) > MAX_ROWS:
        st.error(f"Upload between 1 and {MAX_ROWS:,} patient rows.")
        return None, None, None
    return frame, hashlib.sha256(raw).hexdigest()[:16], file.name[:120]


def example_row(disease, required, stats):
    """The example upload template's row: typical answers, and the optional labs and readings blank (a typical lab value
    is in the usual range, so it would set every row to Not flagged and the template would never show the model's
    flag). The heart history questions keep their 'No' default."""
    age_key, example = AGE_FIELD[disease], {"PatientID": "DEMO-001", "DOB": "1980-01-01"}
    for field in required:
        if field in OPTIONAL_INPUTS.get(disease, []) and field not in GATE_ONLY_FIELDS.get(disease, ()):
            example[field] = ""
        elif field != age_key:
            info = stats[field]
            example[field] = info["default"] if info["kind"] == "category" else (
                info["mode"] if info["binary"] else round(info["median"], 2))
    return example


def _intake(owner_id, disease, required, stats):
    section_heading("1 · Import and check records", "Age may be omitted at upload. Provide DOB to calculate it, "
                    "or enter age during individual review.")
    optional = [f for f in required if f in OPTIONAL_INPUTS.get(disease, [])]
    with st.expander("Upload columns"):
        st.code(", ".join(f for f in required if f not in optional), language=None)
        if optional:
            st.caption("Optional columns (a blank cell, or no column, means not provided): " + ", ".join(optional))
    example = example_row(disease, required, stats)
    st.download_button("Download example upload template", pd.DataFrame([example]).to_csv(index=False).encode(),
                       file_name=f"ahead_{disease}_template.csv", mime="text/csv")
    frame, fingerprint, name = _upload()
    if frame is None:
        return
    try:
        rows = prepare_upload(frame, disease, required, stats, ID_COLUMNS)
    except ValueError as error:
        st.error(str(error))
        return
    preview = pd.DataFrame([{
        "Row": i, "Patient": item["patient_id"], "Status": "Ready" if not item["issues"] else "Incomplete",
        "Issues": _issues(item["issues"]),
    } for i, item in enumerate(rows, 1)])
    st.dataframe(preview, width="stretch", hide_index=True, height=250)
    incomplete = sum(bool(r["issues"]) for r in rows)
    st.caption(f"{len(rows) - incomplete} ready · {incomplete} incomplete. No rows are scored during import.")
    if st.button("Save records to review queue", type="primary", key="clinical_import"):
        cohort = f"{disease}-{fingerprint}"
        if any(r["cohort"] == cohort for r in list_records(owner_id, disease)):
            st.info("This file is already in your review queue.")
        else:
            insert_records(owner_id, cohort, disease, rows, source_name=name)
            st.session_state[f"clinical_cohort_{disease}"] = cohort  # open the queue on the new upload
            st.toast(f"Records saved: {len(rows) - incomplete} ready, {incomplete} incomplete." if incomplete
                     else "Records saved. All rows are ready to score.")
            st.rerun()


def _edit_record(record, owner_id, disease, required, stats, pipeline, model_name, threshold):
    values, issues = json.loads(record["values_json"]), json.loads(record["issues_json"])
    st.markdown(f"**Record {record['source_row']} · {markdown_literal(record['patient_id'])}**")
    st.caption("Edit model inputs below. Enter DOB to calculate age automatically.")
    # 0/1 answers read Male/Female or No/Yes, as on the patient form; validate_row maps the words back to 0/1.
    shown = {f: _editable(f, values.get(f), stats[f]) for f in required}
    edited = pd.DataFrame([{"Field": pretty_label(f), "Input": f, "Value": shown[f], **({"Issue": next(
        (i.split(": ", 1)[1] for i in issues if i.startswith(f + ":")), "")} if issues else {})} for f in required])
    with st.form(f"edit_{record['id']}"):
        table = st.data_editor(edited, hide_index=True, disabled=["Field", "Input", "Issue"], width="stretch",
                               column_config={"Input": "Upload column"}, key=f"record_editor_{record['id']}")
        dob = st.text_input("Date of birth (YYYY-MM-DD, optional)", value=record["dob"] or "").strip()
        submitted = st.form_submit_button("Save and validate record")
    if submitted:
        incoming = dict(zip(table["Input"], table["Value"]))
        age_key = AGE_FIELD[disease]
        reference = to_local(record['created_at']).date()
        try:
            old = dob_age(record["dob"], disease, stats, reference)
        except (ValueError, TypeError):
            old = None
        # An entered DOB re-derives the age only while the untouched cell still holds the old DOB's age; an unreadable
        # new DOB blanks it too, so a typo never leaves a stale age behind. An uploaded or typed age is kept, so
        # validate_row flags a disagreement instead of silently replacing it.
        if old is not None and dob and values.get(age_key) == old and incoming.get(age_key) == shown[age_key]:
            incoming[age_key] = None
        clean, issues = validate_row(incoming, disease, required, stats, dob or None, reference)
        if not issues and record['score'] is not None:
            try:
                score = float(predict_probability(pipeline, model_frame(disease, [clean], features(disease))).iloc[0])
            except (ValueError, TypeError) as error:
                st.error(f'Could not update the prediction: {error}')
                return
            append_record_prediction(owner_id,record['id'],disease,clean,score,threshold,model_name,'corrected')
            update_record(owner_id, record['id'], values_json=json.dumps(clean), dob=dob or None,
                          issues_json='[]',status='Flagged' if score>=threshold else 'Ready',
                          score=score, model=model_name,threshold=threshold)
            _rerun_after_save()
        update_record(owner_id, record["id"], values_json=json.dumps(clean), dob=dob or None,
                      issues_json=json.dumps(issues), status="Incomplete" if issues else "Ready",
                      score=None, model=None, threshold=None)
        _rerun_after_save()


def _rerun_after_save():
    # The browser keeps the Record label it last saw until the server sets the value; with a stale label (old status)
    # its next click would match no option and the selection would fall back to Row 1.
    st.session_state.clinical_record_saved = True
    st.rerun()


def _summary(records, tracks):
    """The cohort at a glance, above the queue: intake on the first row, review progress on the second. The urgent
    count sits in the Records note, so the grid stays two even rows; it counts the rows the Track filter calls Urgent."""
    scored = [r for r in records if r["score"] is not None]
    urgent = sum(bool(track and track["urgent"]) for track in tracks.values())
    needs = sum(r["status"] == "Incomplete" for r in records)
    flagged, reviewed, excluded = (sum(r["status"] == s for r in records) for s in ("Flagged", "Reviewed", "Excluded"))
    pending = sum(r["status"] not in ("Reviewed", "Excluded") for r in scored)
    tiles = [("records", "Records", len(records), "Rows in this upload" + (f" · {urgent} urgent" if urgent else ""), "#0E9AA7",
              "uploads"),
             ("incomplete", "Incomplete", needs, "Missing or invalid inputs" if needs else "All inputs complete",
              "#C98F1E", "returning"),
             ("scored", "Scored", len(scored), "Model scores saved" if scored else "None scored yet", "#8E7CC3", "scored",
              len(records)),
             ("flagged", "Flagged", flagged, "At or above the threshold, not yet reviewed", "#C9706E", "flagged"),
             ("pending", "Pending review", pending, "Scored, awaiting a review decision", "#5B8FA3", "calendar"),
             ("reviewed", "Reviewed", reviewed, f"{excluded} excluded" if excluded else "Review decision saved"
              if reviewed else "No decisions saved yet", "#2E9477", "review")]
    for row in (tiles[:3], tiles[3:]):
        for column, (key, *tile) in zip(st.columns(3), row):
            with column:
                summary_tile(f"clinical_{key}", *tile)


def _queue(owner_id, disease, pipeline, model_name, threshold, required, stats):
    section_heading("2 · Review and score", "Score the complete records in one upload, then work through the queue.")
    all_records = list_records(owner_id, disease)
    if not all_records:
        empty_state("◉", "No records in the review queue yet",
                    "Upload a CSV or Excel file above and save it to start reviewing.")
        return
    groups = {}
    for r in all_records:
        groups.setdefault(r["cohort"], []).append(r)
    # Newest upload first. Labels can repeat across conditions, so the key (not the label) identifies the widget.
    cohorts = sorted(groups, key=lambda c: min(r["created_at"] for r in groups[c]), reverse=True)
    labels, seen = {}, {}
    for c in reversed(cohorts):  # a file corrected and uploaded again within the minute reads '… · upload 2'
        label = _cohort_label(groups[c])
        seen[label] = seen.get(label, 0) + 1
        labels[c] = label if seen[label] == 1 else f"{label} · upload {seen[label]}"
    cohort = st.selectbox("Uploaded cohort", cohorts, format_func=lambda c: labels.get(c, str(c)),
                          key=f"clinical_cohort_{disease}")
    records = sorted(groups[cohort], key=lambda r: r["source_row"])
    tracks = {r["id"]: _track(r, disease) for r in records}
    _summary(records, tracks)
    ready = [r for r in records if not json.loads(r["issues_json"]) and r["status"] not in ("Excluded", "Reviewed")
             and not _legacy(r, disease)]
    if st.button(f"Score {len(ready)} eligible record{'s' if len(ready) != 1 else ''} with {model_label(model_name)}",
                 disabled=not ready, type="primary"):
        with st.spinner(f"Checking and scoring {len(ready):,} records with {model_label(model_name)}…"):
            checked = [(r["id"], *validate_row(json.loads(r["values_json"]), disease, required, stats, r["dob"],
                                               to_local(r["created_at"]).date())) for r in ready]
            valid, scored, error = [(i, clean) for i, clean, issues in checked if not issues], [], None
            if valid:
                try:
                    scores = predict_probability(pipeline, model_frame(disease, [clean for _, clean in valid],
                                                                       features(disease)))
                    scored = [(i, clean, float(score)) for (i, clean), score in zip(valid, scores)]
                except (ValueError, TypeError) as failure:
                    error = failure
            # One transaction for the whole cohort; a connection per record took about a minute for 5,000 rows.
            save_scoring_run(owner_id, disease, model_name, threshold, scored, [row for row in checked if row[2]])
        if error:
            st.error(f"Scoring failed. Verify model inputs: {error}")
        else:
            if not valid:
                st.toast("No valid records remain after rechecking dates and inputs.")
            _rerun_after_save()
    rows = [{"Row": r["source_row"], "Patient": r["patient_id"], "Status": _status(r),
             "Track": track_text(tracks[r["id"]], "doctor")["label"] if tracks[r["id"]] else "",
             "Model score (%)": None if r["score"] is None else round(r["score"] * 100, 1),
             "Issues": _issues(json.loads(r["issues_json"])), "Model": model_label(r["model"] or "")}
            for r in records]
    table = pd.DataFrame(rows)
    # 'Ready' rows are complete but unscored; 'Not flagged' rows were scored below the threshold. An incomplete row has a
    # track only when an entered value is urgent (see _track); otherwise its Track is blank and it shows only under All.
    show_column, track_column = st.columns(2)
    with show_column:
        status_filter = st.selectbox("Show", ["All", "Incomplete", "Ready", "Flagged", "Not flagged", "Reviewed", "Excluded"])
    with track_column:
        track_filter = st.selectbox("Track", ["All", *TRACK_LABELS])
    shown = table if status_filter == "All" else table[table["Status"] == status_filter]
    shown = shown if track_filter == "All" else shown[shown["Track"] == track_filter]
    if shown.empty and track_filter != "All":
        st.caption(f"No records in this upload are on the {track_filter} track." if status_filter == "All"
                   else f"No records in this upload match Show: {status_filter} and Track: {track_filter}.")
    elif shown.empty:
        st.caption("No records scored below the threshold in this upload." if status_filter == "Not flagged"
                   else f"No {status_filter.lower()} records in this upload.")
    else:
        st.dataframe(shown, width="stretch", hide_index=True, height=320)
    reference = model_test_metrics(disease, model_name)
    st.info(f"{model_label(model_name)} test-set recall {reference.get('recall_disease', 0):.1%}, "
            f"precision {reference.get('precision_disease', 0):.1%} at its tuned "
            f"{reference.get('threshold', threshold) * 100:g}% threshold. These figures may differ on a hospital cohort.")
    safe_export = table.copy()
    for column in ("Patient", "Issues", "Model"):
        safe_export[column] = safe_export[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    st.download_button("Download review queue (CSV)", safe_export.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ahead_{disease}_review.csv", mime="text/csv")

    section_heading("3 · Individual review", "Correct a record's inputs, read its model output and save a review decision.")
    by_id, key = {r["id"]: r for r in records}, f"clinical_record_{disease}_{cohort}"
    # A stable key keeps the doctor on this record when its status (part of the label) changes; after a save the
    # value is set again so the browser receives the new label (see _rerun_after_save).
    if st.session_state.pop("clinical_record_saved", False) and st.session_state.get(key) in by_id:
        st.session_state[key] = st.session_state[key]
    selected_id = st.selectbox("Record", list(by_id), key=key,
                               format_func=lambda i: f"Row {by_id[i]['source_row']} · {by_id[i]['patient_id']} · {_status(by_id[i])}")
    record = by_id[selected_id]
    issues, legacy = json.loads(record["issues_json"]), _legacy(record, disease)
    if legacy:
        st.info(LEGACY_NOTE)
    elif issues:
        st.warning("Correct these inputs before scoring:\n" + "\n".join(f"- {issue_text(i)}" for i in issues))
    if not legacy and (record["status"] == "Incomplete" or st.toggle("Edit this record", key=f"edit_toggle_{selected_id}")):
        _edit_record(record, owner_id, disease, required, stats, pipeline, model_name, threshold)
    values, track = json.loads(record["values_json"]), tracks[selected_id]
    if record["score"] is not None:
        result_card(int(record["score"] >= record["threshold"]), record["score"],
                    DISEASES[disease]["long_label"], record["model"], record["threshold"], audience="doctor",
                    status=_status(record), disease=disease, track=track)
        if track:
            track_panel(track, "doctor")
        st.caption("Review the measurements and history independently before acting on this model output.")
        metrics = model_test_metrics(disease, record["model"])
        factors = identify_risk_factors(disease, values)
        # The last save of a reviewed or excluded record is the review decision, so its time is the review time.
        decided = record["status"] in ("Reviewed", "Excluded")
        st.download_button("Download patient PDF", patient_report(disease, record["patient_id"], values,
                           record["score"], record["threshold"], record["model"], metrics,
                           record["status"] if decided else "", record["note"],
                           metric_source="test",
                           audience="doctor", reviewer=current_user()["name"] if decided else "",
                           reviewed_at=record["updated_at"] if decided else None,
                           source=f"{record.get('source_name') or 'Uploaded file'}, row {record['source_row']}", factors=factors,
                           track=track),
                           file_name=f"ahead_record_{record['id']}.pdf", mime="application/pdf")
        if factors:
            # Not "values above typical ranges": the list also holds history answers such as a previous stroke.
            st.caption("Risk factors in this record")
            st.html("".join(f'<span class="factor-chip">{escape(f)}</span>' for f in factors))
        model_for_record = pipeline if record["model"] == model_name and not legacy else None
        if model_for_record is not None:
            try:
                # An entry that reads the same as the typical value once rounded (9.95 vs 9.9515) says nothing here.
                local = [(f, shown_value(f, v, stats[f]), shown_value(f, base, stats[f]), delta)
                         for f, v, base, delta in individual_sensitivity(model_for_record, model_row(disease, values),
                                                                         stats)]
                local = [row for row in local if row[1] != row[2]]
                if local:
                    st.caption("Illustrative model input sensitivity: one input replaced at a time by a typical training value; not a causal explanation.")
                    st.dataframe(pd.DataFrame([{"Input": pretty_label(f), "Entered": entered, "Typical value": typical,
                                                "Score change (points)": round(delta * 100, 1) + 0.0}  # no '-0.0'
                                               for f, entered, typical, delta in local]), hide_index=True, width="stretch")
                else:
                    st.info("The entered values match the typical training values, so there is no input comparison to show.")
            except (ValueError, TypeError):
                st.info("Individual model sensitivity is unavailable for this record.")
    elif track:   # the urgent rules work before scoring; an unscored record with no urgent value draws nothing
        track_panel(track, "doctor", _unchecked(track, issues))
    shown_fields = list(values) if legacy else required   # an earlier form's record shows its own inputs
    st.dataframe(pd.DataFrame({"Input": [pretty_label(f) for f in shown_fields],
                               "Value": [shown_value(f, values.get(f), field_info(stats, f)) for f in shown_fields]}),
                 hide_index=True, width="stretch")
    prediction_history=record_prediction_history(owner_id,selected_id)
    if prediction_history:
        with st.expander('Previous prediction versions'):
            st.dataframe(pd.DataFrame([{'Date':to_local(r['created_at']).replace(tzinfo=None),'Score (%)':round(r['score']*100,1),
                'Model':model_label(r['model']),'Reason':SOURCES.get(r['source'],r['source'].capitalize())} for r in prediction_history]),
                hide_index=True,column_config=LOCAL_TIME)
    with st.form(f"review_{selected_id}"):
        note = st.text_area("Doctor review note", value=record["note"], max_chars=2000)
        choices = list(dict.fromkeys([record["status"], "Reviewed", "Excluded"]))
        action = st.selectbox("Review decision", choices, format_func=lambda s: _status(record) if s == record["status"] else s)
        if st.form_submit_button("Save review"):
            if action == "Reviewed" and record["score"] is None:
                st.error("Correct and score this record before marking it reviewed.")
            else:
                update_record(owner_id, selected_id, note=note, status=action)
                _rerun_after_save()


def render_clinical_dashboard():
    user = current_user()
    if not user["is_doctor"] or not st.session_state.get("user", {}).get("id"):
        st.error("Doctor access required.")
        return
    page_header("AHEAD Clinical Decision Support", "Clinical Dashboard",
                "Import, validate, score and review de-identified patient records.")
    st.warning("Research prototype. Use fictional or de-identified records only; a qualified clinician makes decisions.")
    disease = st.selectbox("Disease screening", list(DISEASES), format_func=lambda d: DISEASES[d]["card_title"])
    pipeline, model_name, threshold = model_selector(disease, key=f"clinical_model_{disease}")
    if pipeline is None:
        st.error("No trained model is available.")
        return
    required, stats = form_fields(disease), feature_summary(disease)   # upload columns: the form's fields
    owner_id = st.session_state.user["id"]
    _intake(owner_id, disease, required, stats)
    _queue(owner_id, disease, pipeline, model_name, threshold, required, stats)
