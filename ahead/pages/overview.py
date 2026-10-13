"""
ahead/pages/overview.py
=======================
The signed-in home page: hero with quick-start, KPI cards, condition cards,
session risk distribution and reference-data visuals.
"""

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ahead.components import condition_cards, current_user, disclaimer, empty_state, go_to, kpi_card, open_assistant, section_heading
from ahead.config import DISEASES
from ahead.resources import best_model, final_metrics, n_samples, validation_results
from ahead.theme import show_chart, style_figure, tokens


def _risk_distribution_figure(history: list) -> go.Figure:
    t = tokens()
    frame = pd.DataFrame(history)
    bands = frame["prediction"].map({0: "Not flagged", 1: "Flagged"}).value_counts().reindex(
        ["Not flagged", "Flagged"]).fillna(0)
    elevated_pct = 100 * (frame["prediction"] == 1).mean()
    figure = go.Figure(
        go.Pie(
            labels=bands.index.tolist(),
            values=bands.values.tolist(),
            hole=0.62,
            marker=dict(colors=[t["success"], t["danger"]]),
            textinfo="none",
            sort=False,
        )
    )
    figure.add_annotation(
        text=f"<b>{elevated_pct:.0f}%</b><br><span style='font-size:11px'>Flagged</span>",
        showarrow=False, font=dict(size=22, color=t["title"]),
    )
    style_figure(figure, height=300, legend=True)
    figure.update_layout(title="Saved screening classifications")
    return figure


def _screenings_by_condition_figure(history: list) -> go.Figure:
    frame = pd.DataFrame(history)
    counts = (
        frame.groupby(["label", "prediction"]).size().reset_index(name="Count")
    )
    counts["Result"] = counts["prediction"].map({0: "Not flagged", 1: "Flagged"})
    t = tokens()
    figure = px.bar(
        counts, x="label", y="Count", color="Result", barmode="group", title="Screenings by condition",
        color_discrete_map={"Not flagged": t["success"], "Flagged": t["danger"]},
    )
    style_figure(figure, height=300, legend=True)
    figure.update_layout(xaxis_title="", yaxis_title="", yaxis=dict(dtick=1))
    return figure


def _model_quality_figure() -> go.Figure:
    rows = []
    for disease, info in DISEASES.items():
        m = final_metrics(disease)
        rows.append({"Condition": info["label"], "Metric": "Disease recall", "Value": 100 * m.get("recall_disease", 0)})
        rows.append({"Condition": info["label"], "Metric": "ROC-AUC", "Value": 100 * (m.get("roc_auc") or 0)})
    frame = pd.DataFrame(rows)
    t = tokens()
    figure = px.bar(
        frame, x="Condition", y="Value", color="Metric", barmode="group",
        title="Recommended model — test-set quality (%)",
        color_discrete_map={"Disease recall": t["accent"], "ROC-AUC": "#5B8FA3"},
    )
    style_figure(figure, height=300, legend=True)
    figure.update_layout(xaxis_title="", yaxis_title="", yaxis_range=[0, 100])
    return figure


def _reference_records_figure() -> go.Figure:
    frame = pd.DataFrame(
        {"Condition": [info["label"] for info in DISEASES.values()],
         "Records": [n_samples(d) for d in DISEASES]}
    )
    figure = px.bar(frame, x="Condition", y="Records", title="Reference records by condition", text_auto=".3s")
    figure.update_traces(marker_color=[info["colour"] for info in DISEASES.values()])
    style_figure(figure, height=300)
    figure.update_layout(xaxis_title="", yaxis_title="")
    return figure


def render_overview() -> None:
    user = current_user()
    history = st.session_state.get("screening_history", [])

    dashboard_copy = ("Review linked patient records, medication routines and assessment activity." if user["is_doctor"] else "Build healthier habits, keep track of your care, and make time for prevention.")
    # ------------------------------------------------------------------ hero
    with st.container(key="home_hero"):
        st.html(
            f"""
<div class="home-hero">
    <div>
        <div class="home-eyebrow">AHEAD Health Dashboard</div>
        <h1>Welcome back, {escape(user['first_name'])}</h1>
        <p>{escape(dashboard_copy)}</p>
    </div>
    <div class="hero-badge"><small>Our mission</small>Small screenings.<br>Big futures.</div>
</div>
"""
        )
        action_col, ask_col, _spacer = st.columns([0.26, 0.24, 0.5])
        with action_col:
            with st.popover("Start Screening", width="stretch", type="primary"):
                for disease, info in DISEASES.items():
                    if st.button(info["card_title"], key=f"hero_{disease}", width="stretch"):
                        go_to("screenings", disease)
        with ask_col:
            if st.button("Ask AHEAD Assistant", key="hero_ask", width="stretch"):
                open_assistant("overview")

    if not user["is_doctor"]:
        from ahead.pages.patient_dashboard import render_patient_dashboard
        render_patient_dashboard()
        return

    if st.button("Open Advanced Analytics & Patient Intelligence", type="primary"):
        go_to("intelligence")

    # ------------------------------------------------------------------ KPIs
    total_records = sum(n_samples(d) for d in DISEASES)
    families = max((len(validation_results(d)) for d in DISEASES), default=0)
    elevated = sum(1 for item in history if item["prediction"] == 1)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Saved screenings", str(len(history)),
                 "Available in your screening history" if history else "Run your first screening", "✚")
    with c2:
        kpi_card("Elevated results", str(elevated),
                 "Flagged for professional review" if elevated else "No elevated patterns yet", "!",
                 "danger" if elevated else "success")
    with c3:
        kpi_card("Reference records", f"{total_records:,}", "Across the three project datasets", "▤")
    with c4:
        kpi_card("Model families", str(families), "Benchmarked per condition", "⚙", "warning")

    # ------------------------------------------------------------------ conditions
    section_heading("Your screening options", "Three focused early-awareness tools using disease-specific machine-learning models.")
    condition_cards("overview")

    # ------------------------------------------------------------------ session visuals
    section_heading("Your screening history", "Saved results loaded for your account and screenings completed during this visit.")
    left, right = st.columns(2)
    if history:
        with left:
            show_chart(_risk_distribution_figure(history))
        with right:
            show_chart(_screenings_by_condition_figure(history))
        st.caption("Flagged means the model score reached that screening's threshold. These counts do not establish disease prevalence.")
        with st.expander("Recent screenings"):
            table = pd.DataFrame(
                [
                    {
                        "Time": item["time"].strftime("%H:%M"),
                        "Condition": item["label"],
                        "Model": item["model"],
                        "Score": f"{item['probability'] * 100:.1f}%",
                        "Result": "Elevated" if item["prediction"] else "Lower risk",
                    }
                    for item in reversed(history[-10:])
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
    else:
        with left:
            empty_state("◔", "No screenings yet",
                        "Complete a screening and your risk distribution will appear here.")
        with right:
            empty_state("▥", "Nothing to compare",
                        "Results by condition are charted once you have at least one screening.")

    # ------------------------------------------------------------------ project visuals
    section_heading("Project data at a glance", "Reference dataset sizes and how well each recommended model performs on unseen test data.")
    left, right = st.columns(2)
    with left:
        show_chart(_reference_records_figure())
    with right:
        show_chart(_model_quality_figure())
    st.caption(
        "Recall on the disease class and ROC-AUC are shown instead of plain accuracy because two of the "
        "datasets are heavily imbalanced, where accuracy alone is misleading. Recommended models: "
        + ", ".join(f"{info['label']} → {best_model(d)}" for d, info in DISEASES.items())
        + "."
    )

    disclaimer(
        "AHEAD provides educational screening estimates only. It does not provide medical diagnoses "
        "and does not replace professional medical evaluation.",
        label="Clinical notice",
    )
