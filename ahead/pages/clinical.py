"""Doctor-only intake, validation, scoring, review queue and reports."""

import hashlib
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from ahead.clinical_data import prepare_upload, validate_row
from ahead.visuals import metric
from ahead.components import current_user, model_selector, page_header, result_card
from ahead.config import DISEASES, pretty_label
from ahead.reports import patient_report
from ahead.resources import feature_summary, features, final_metrics, predict_probability, validation_results, best_model
from ahead.storage import insert_records, list_records, update_record
from ahead.screening import identify_risk_factors, individual_sensitivity
from ahead.platform_data import append_record_prediction, record_prediction_history

ID_COLUMNS = ["PatientID", "patient_id", "Patient ID", "ID", "id"]
MAX_ROWS = 5000


def _upload():
    file = st.file_uploader("Upload patient records", type=["csv", "xlsx"],
                            help="Use fictional or de-identified records for this prototype.")
    if file is None:
        return None, None
    raw = file.getvalue()
    if len(raw) > 15 * 1024 * 1024:
        st.error("File exceeds the 15 MB upload limit.")
        return None, None
    try:
        frame = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    except (ValueError, UnicodeError, pd.errors.ParserError, OSError):
        st.error("Could not read this file. Check the CSV/XLSX format.")
        return None, None
    if frame.empty or len(frame) > MAX_ROWS:
        st.error(f"Upload between 1 and {MAX_ROWS:,} patient rows.")
        return None, None
    return frame, hashlib.sha256(raw).hexdigest()[:16]


def _intake(owner_id, disease, required, stats):
    st.subheader("1 · Import and check records")
    st.caption("Age may be omitted at upload. Provide DOB to calculate it, or enter age during individual review.")
    with st.expander("Required model columns"):
        st.code(", ".join(required), language=None)
    age_key = "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    example = {"PatientID": "DEMO-001", "DOB": "1980-01-01"}
    for field in required:
        if field != age_key:
            info = stats[field]
            example[field] = info["default"] if info["kind"] == "category" else (
                info["mode"] if info["binary"] else round(info["median"], 2))
    st.download_button("Download example upload template", pd.DataFrame([example]).to_csv(index=False).encode(),
                       file_name=f"ahead_{disease}_template.csv", mime="text/csv")
    frame, fingerprint = _upload()
    if frame is None:
        return
    try:
        rows = prepare_upload(frame, disease, required, stats, ID_COLUMNS)
    except ValueError as error:
        st.error(str(error))
        return
    preview = pd.DataFrame([{
        "Row": i, "Patient": item["patient_id"], "Status": "Ready" if not item["issues"] else "Needs correction",
        "Issues": "; ".join(item["issues"]),
    } for i, item in enumerate(rows, 1)])
    st.dataframe(preview, width="stretch", hide_index=True, height=250)
    st.caption(f"{len(rows) - sum(bool(r['issues']) for r in rows)} ready · "
               f"{sum(bool(r['issues']) for r in rows)} needing correction. No rows are scored during import.")
    if st.button("Save records to review queue", type="primary", key="clinical_import"):
        cohort = f"{disease}-{fingerprint}"
        if any(r["cohort"] == cohort for r in list_records(owner_id, disease)):
            st.warning("This file is already in your review queue.")
        else:
            insert_records(owner_id, cohort, disease, rows)
            st.success("Records saved. Review incomplete rows before screening.")
            st.rerun()


def _edit_record(record, owner_id, disease, required, stats, pipeline, model_name, threshold):
    values = json.loads(record["values_json"])
    st.markdown(f"**Record {record['source_row']} · {record['patient_id']}**")
    st.caption("Edit model inputs below. Enter DOB to calculate age automatically.")
    edited = pd.DataFrame([{"Input": f, "Value": "" if values.get(f) is None else str(values[f])} for f in required])
    with st.form(f"edit_{record['id']}"):
        table = st.data_editor(edited, hide_index=True, disabled=["Input"], width="stretch",
                               key=f"record_editor_{record['id']}")
        dob = st.text_input("Date of birth (YYYY-MM-DD, optional)", value=record["dob"] or "")
        submitted = st.form_submit_button("Save and validate record")
    if submitted:
        incoming = dict(zip(table["Input"], table["Value"]))
        clean, issues = validate_row(incoming, disease, required, stats, dob or None,
                                     datetime.fromisoformat(record['created_at']).date())
        if not issues and record['score'] is not None:
            try:
                score = float(predict_probability(pipeline, pd.DataFrame([clean], columns=required)).iloc[0])
            except (ValueError, TypeError) as error:
                st.error(f'Could not update the prediction: {error}')
                return
            append_record_prediction(owner_id,record['id'],disease,clean,score,threshold,model_name,'corrected')
            update_record(owner_id, record['id'], values_json=json.dumps(clean), dob=dob or None,
                          issues_json='[]',status='Flagged' if score>=threshold else 'Ready',
                          score=score, model=model_name,threshold=threshold)
            st.rerun()
        update_record(owner_id, record["id"], values_json=json.dumps(clean), dob=dob or None,
                      issues_json=json.dumps(issues), status="Incomplete" if issues else "Ready",
                      score=None, model=None, threshold=None)
        st.rerun()


def _queue(owner_id, disease, pipeline, model_name, threshold, required, stats):
    st.subheader("2 · Review and score")
    all_records = list_records(owner_id, disease)
    if not all_records:
        st.info("Save an upload to create a review queue.")
        return
    cohorts = list(dict.fromkeys(r["cohort"] for r in all_records))
    cohort = st.selectbox("Uploaded cohort", cohorts)
    records = sorted((r for r in all_records if r["cohort"] == cohort), key=lambda r: r["source_row"])
    ready = [r for r in records if not json.loads(r["issues_json"]) and r["status"] not in ("Excluded", "Reviewed")]
    if st.button(f"Score {len(ready)} eligible records with {model_name}", disabled=not ready, type="primary"):
        refreshed = []
        for record in ready:
            clean, issues = validate_row(json.loads(record["values_json"]), disease, required, stats, record["dob"],
                                         datetime.fromisoformat(record['created_at']).date())
            if issues:
                update_record(owner_id, record["id"], values_json=json.dumps(clean),
                              issues_json=json.dumps(issues), status="Incomplete", score=None, model=None, threshold=None)
            else:
                update_record(owner_id, record["id"], values_json=json.dumps(clean))
                refreshed.append((record, clean))
        if not refreshed:
            st.warning("No valid records remain after rechecking dates and inputs.")
            st.rerun()
        frame = pd.DataFrame([values for _, values in refreshed], columns=required)
        try:
            scores = predict_probability(pipeline, frame)
        except (ValueError, TypeError) as error:
            st.error(f"Scoring failed. Verify model inputs: {error}")
        else:
            for (record, _), score in zip(refreshed, scores):
                append_record_prediction(owner_id,record['id'],disease,json.loads(record['values_json']),float(score),threshold,model_name)
                update_record(owner_id, record["id"], score=float(score), threshold=threshold, model=model_name,
                              status="Flagged" if score >= threshold else "Ready")
            st.rerun()
    rows = [{"Row": r["source_row"], "Patient": r["patient_id"], "Status": r["status"],
             "Model score (%)": None if r["score"] is None else round(r["score"] * 100, 1),
             "Issues": "; ".join(json.loads(r["issues_json"])), "Model": r["model"] or ""}
            for r in records]
    table = pd.DataFrame(rows)
    status_filter = st.selectbox("Show", ["All", "Incomplete", "Flagged", "Ready", "Reviewed", "Excluded"])
    shown = table if status_filter == "All" else table[table["Status"] == status_filter]
    st.dataframe(shown, width="stretch", hide_index=True, height=320)
    reference = next((m for m in validation_results(disease) if m["model"] == model_name), {})
    st.info(f"{model_name} validation-set recall {reference.get('recall_disease', 0):.1%}, "
            f"precision {reference.get('precision_disease', 0):.1%} at a 50% threshold. "
            "These figures may differ at a tuned threshold and on a hospital cohort.")
    a,b,c=st.columns(3)
    with a:metric('Records in review',len(records),'Current condition and cohort','#2563EB')
    with b:metric('Need input review',sum(bool(json.loads(r['issues_json'])) for r in records),'Missing or invalid entered data','#F59E0B')
    with c:metric('Scored records',sum(r['score'] is not None for r in records),'Available for clinical review','#8B5CF6')
    safe_export = table.copy()
    for column in ("Patient", "Issues", "Model"):
        safe_export[column] = safe_export[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    st.download_button("Download review queue (CSV)", safe_export.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ahead_{disease}_review.csv", mime="text/csv")

    st.subheader("3 · Individual review")
    selected_id = st.selectbox("Record", [r["id"] for r in records],
                               format_func=lambda key: next(f"Row {r['source_row']} · {r['patient_id']} · {r['status']}"
                                                     for r in records if r["id"] == key))
    record = next(r for r in records if r["id"] == selected_id)
    issues = json.loads(record["issues_json"])
    if issues:
        st.warning("Correct these inputs before screening: " + "; ".join(issues))
    if record["status"] == "Incomplete" or st.toggle("Edit this record", key=f"edit_toggle_{selected_id}"):
        _edit_record(record, owner_id, disease, required, stats, pipeline, model_name, threshold)
    values = json.loads(record["values_json"])
    if record["score"] is not None:
        result_card(int(record["score"] >= record["threshold"]), record["score"],
                    DISEASES[disease]["long_label"], record["model"], record["threshold"])
        st.caption("A model output, not a diagnosis. Review the measurements and history independently.")
        metrics = final_metrics(disease) if record["model"] == best_model(disease) else next(
            (m for m in validation_results(disease) if m["model"] == record["model"]), {})
        st.download_button("Download patient PDF", patient_report(disease, record["patient_id"], values,
                           record["score"], record["threshold"], record["model"], metrics,
                           record["status"], record["note"],
                           metric_source="test" if record["model"] == best_model(disease) else "validation"),
                           file_name=f"ahead_record_{record['id']}.pdf", mime="application/pdf")
        factors = identify_risk_factors(disease, values)
        if factors:
            st.write("Entered values to discuss with a clinician: " + ", ".join(factors))
        model_for_record = pipeline if record["model"] == model_name else None
        if model_for_record is not None:
            try:
                local = individual_sensitivity(model_for_record, values, stats)
                if local:
                    st.caption("Illustrative model input sensitivity: one input replaced at a time by a typical training value; not a causal explanation.")
                    st.dataframe(pd.DataFrame([{"Input": pretty_label(f), "Entered": str(v), "Comparison": str(base),
                                                "Score change (points)": round(delta * 100, 1)}
                                               for f, v, base, delta in local]), hide_index=True, width="stretch")
            except (ValueError, TypeError):
                st.info("Individual model sensitivity is unavailable for this record.")
    st.dataframe(pd.DataFrame({"Input": [pretty_label(f) for f in required],
                               "Value": [str(values.get(f) if values.get(f) is not None else "") for f in required]}),
                 hide_index=True, width="stretch")
    prediction_history=record_prediction_history(owner_id,selected_id)
    if prediction_history:
        with st.expander('Previous prediction versions'):
            st.dataframe(pd.DataFrame([{'Date':r['created_at'][:19],'Score (%)':round(r['score']*100,1),
                'Model':r['model'],'Reason':r['source']} for r in prediction_history]),hide_index=True)
    with st.form(f"review_{selected_id}"):
        note = st.text_area("Doctor review note", value=record["note"], max_chars=2000)
        choices = list(dict.fromkeys([record["status"], "Reviewed", "Excluded"]))
        action = st.selectbox("Review decision", choices)
        if st.form_submit_button("Save review"):
            if action == "Reviewed" and record["score"] is None:
                st.error("Correct and score this record before marking it reviewed.")
            else:
                update_record(owner_id, selected_id, note=note, status=action)
                st.rerun()


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
    required, stats = features(disease), feature_summary(disease)
    owner_id = st.session_state.user["id"]
    _intake(owner_id, disease, required, stats)
    _queue(owner_id, disease, pipeline, model_name, threshold, required, stats)
