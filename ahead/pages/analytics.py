"""
ahead/pages/analytics.py
========================
Dataset explorer and technical model performance per condition.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ahead.components import page_header, section_heading
from ahead.config import DISEASES, model_label, pretty_label
from ahead.resources import best_model, features, final_metrics, load_dataset, meta, validation_results
from ahead.theme import show_chart, style_figure, tokens
from ahead.visuals import summary_tile

METRIC_COLUMNS = {
    "model": "Algorithm",
    "accuracy": "Accuracy",
    "precision_disease": "Disease Precision",
    "recall_disease": "Disease Recall",
    "f1_disease": "Disease F1",
    "macro_f1": "Macro F1",
    "f2_disease": "Disease F2",
    "roc_auc": "ROC-AUC",
}


def _target_figure(dataset: pd.DataFrame, target: str) -> go.Figure:
    # Targets are stored as 0/1 or No/Yes; either way the positive class means the record has the condition.
    has = dataset[target].astype(str).isin(["1", "Yes"])
    counts = pd.DataFrame({"Outcome": ["Without", "With"], "Count": [int((~has).sum()), int(has.sum())]})
    figure = px.bar(counts, x="Outcome", y="Count", title="Records by outcome", text_auto=".3s")
    figure.update_traces(marker_color=[tokens()["accent"], "#C9706E"], hovertemplate="%{x} the condition: %{y:,}<extra></extra>")
    style_figure(figure, height=330)
    figure.update_layout(xaxis_title="", yaxis_title="")
    return figure


@st.cache_data(show_spinner=False)
def _feature_distribution(disease: str, feature: str) -> pd.DataFrame:
    """Binned counts for a feature — computed once, so the browser never receives raw rows."""
    series = load_dataset(disease)[feature].dropna()
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        counts, edges = np.histogram(series, bins=35)
        centres = (edges[:-1] + edges[1:]) / 2
        return pd.DataFrame({"Value": centres, "Count": counts, "width": np.diff(edges)})
    counts = series.astype(str).value_counts().head(20).reset_index()
    counts.columns = ["Value", "Count"]
    return counts


def _feature_figure(disease: str, feature: str) -> go.Figure:
    frame = _feature_distribution(disease, feature)
    title = pretty_label(feature)
    if "width" in frame.columns:
        figure = go.Figure(go.Bar(x=frame["Value"], y=frame["Count"], width=frame["width"], marker_color="#5B8FA3",
                                  hovertemplate="%{x:.3g}: %{y:,}<extra></extra>"))
        figure.update_layout(title=title, bargap=0.05)
    else:
        figure = px.bar(frame, x="Value", y="Count", title=title, text_auto=".3s")
        figure.update_traces(marker_color="#5B8FA3", hovertemplate="%{x}: %{y:,}<extra></extra>")
    style_figure(figure, height=330)
    figure.update_layout(xaxis_title="", yaxis_title="")
    return figure


def _threshold_figure(rows: list, chosen: float) -> go.Figure:
    frame = pd.DataFrame(rows)
    t = tokens()
    figure = go.Figure()
    for column, name, colour, dash in [("recall", "Recall", t["accent"], "solid"), ("precision", "Precision", "#8E7CC3", "solid"),
                                       ("f2", "F2 (recall counts double)", "#C98F1E", "dash")]:
        figure.add_trace(go.Scatter(x=frame["threshold"], y=frame[column], name=name, mode="lines",
                                    line=dict(color=colour, width=3, dash=dash, shape="spline"),
                                    hovertemplate=f"{name}: %{{y:.0%}}<extra></extra>"))
        # The chosen threshold's value on each line, so the trade-off reads straight off the chart.
        at = frame.iloc[(frame["threshold"] - chosen).abs().argmin()]
        figure.add_trace(go.Scatter(x=[at["threshold"]], y=[at[column]], mode="markers", showlegend=False, hoverinfo="skip",
                                    marker=dict(size=10, color=colour, line=dict(color=t["card"], width=2))))
    figure.add_vline(x=chosen, line=dict(color=t["muted"], width=1.5, dash="dot"))
    figure.add_annotation(x=chosen, y=1.08, yref="paper", text=f"Chosen threshold {chosen:.0%}", showarrow=False,
                          font=dict(size=12, color=t["title"]), bgcolor=t["card"], bordercolor=t["border"], borderpad=4)
    style_figure(figure, height=360, legend=True)
    figure.update_layout(title="Choosing the decision threshold", hovermode="x unified", margin=dict(t=80, r=36))
    figure.update_xaxes(title="Decision threshold", tickformat=".0%", range=[0, 1], hoverformat=".0%")
    figure.update_yaxes(title="", tickformat=".0%", range=[0, 1.05])
    return figure


def _model_comparison(disease: str) -> None:
    rows = validation_results(disease)
    if not rows:
        return
    section_heading(
        "Algorithm Comparison",
        "Validation-set performance of every model family at the default 0.5 cut-off (the tuned threshold "
        "above applies only to the recommended model's final test metrics).",
    )
    frame = pd.DataFrame(rows)
    columns = [c for c in METRIC_COLUMNS if c in frame.columns]
    frame = frame[columns].rename(columns=METRIC_COLUMNS)
    recommended = best_model(disease)
    frame.insert(0, "Recommended", frame["Algorithm"].map(lambda name: "★" if name == recommended else ""))
    numeric = [c for c in frame.columns if c not in {"Algorithm", "Recommended"}]
    frame[numeric] = frame[numeric].astype(float).round(4)
    st.dataframe(frame, width="stretch", hide_index=True)
    metric = meta(disease).get("selection_metric", "macro_f1")
    label = METRIC_COLUMNS.get(metric, metric)
    ranked = frame.sort_values(label, ascending=True)
    chart = px.bar(ranked, x=label, y="Algorithm", orientation="h", title=f"Selection metric — {label}")
    chart.update_traces(marker_color=[tokens()["accent"] if n == recommended else "#9DB6C6" for n in ranked["Algorithm"]])
    style_figure(chart, height=300)
    chart.update_layout(yaxis_title="")
    show_chart(chart)


def render_analytics() -> None:
    page_header(
        "AHEAD Analytics", "Data & Model Explorer",
        "Explore the datasets and review the technical performance of each recommended model.",
    )

    tabs = st.tabs([info["label"] for info in DISEASES.values()])
    for tab, (disease, info) in zip(tabs, DISEASES.items()):
        with tab:
            dataset = load_dataset(disease)
            target = info["target"]
            model_features = features(disease)
            details = meta(disease)

            positive = details.get("positive_cases", 0)
            share = 100 * positive / max(details.get("n_samples", 1), 1)
            tiles = [("records", "Records", f"{len(dataset):,}", "Rows in the reference dataset", "#0E9AA7", "uploads"),
                     ("positive", "Positive cases", f"{positive:,}", f"{share:.1f}% of records have the condition", "#C9706E", "flagged"),
                     ("features", "Model features", len(model_features), "Inputs the screening form asks for", "#8E7CC3", "review"),
                     # Spaced model names wrap between words instead of being cut off in a narrow card.
                     ("model", "Recommended model", model_label(best_model(disease)),
                      "Chosen on the validation results", "#2E9477", "scored")]
            for column, (key, label, value, note, colour, icon) in zip(st.columns(4), tiles):
                with column:
                    summary_tile(f"analytics_{disease}_{key}", label, value, note, colour, icon)

            section_heading("Explore the data", "How many records have the condition, and how each screening input is spread across the dataset.")
            # The picker gets its own row above the charts, so both charts start at the same height.
            selected_feature = st.columns(2)[0].selectbox(
                "Input to explore", model_features, format_func=pretty_label, key=f"explorer_{disease}",
            )
            left, right = st.columns(2)
            with left:
                show_chart(_target_figure(dataset, target))
            with right:
                show_chart(_feature_figure(disease, selected_feature))

            # ------------------------------------------------------------ final test metrics
            section_heading("Final Model Performance", "Measured once on the untouched test set with the recommended model.")
            metrics = final_metrics(disease)
            auc = metrics.get("roc_auc")
            scores = [("accuracy", "Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%", "Test records classified correctly", "#0E9AA7", "goals"),
                      ("recall", "Disease recall", f"{metrics.get('recall_disease', 0) * 100:.2f}%", "People with the condition the model flags", "#C9706E", "flagged"),
                      ("f1", "Disease F1", f"{metrics.get('f1_disease', 0):.4f}", "Balance of precision and recall", "#8E7CC3", "scored"),
                      ("macro_f1", "Macro-F1", f"{metrics.get('macro_f1', 0):.4f}", "F1 averaged over both outcomes", "#C98F1E", "review"),
                      ("auc", "ROC-AUC", f"{auc:.4f}" if auc is not None else "N/A", "How well scores rank cases first", "#2E9477", "activity")]
            for column, (key, label, value, note, colour, icon) in zip(st.columns(5), scores):
                with column:
                    summary_tile(f"analytics_{disease}_{key}", label, value, note, colour, icon)
            threshold = float(details.get("decision_threshold", 0.5))
            selection = METRIC_COLUMNS.get(details.get("selection_metric", ""), details.get("selection_metric", ""))
            st.caption(f"The recommended model flags a result when its score reaches {threshold:.0%}. "
                       f"It was picked from the model families by {selection} on the validation split.")

            if details.get("threshold_results"):
                show_chart(_threshold_figure(details["threshold_results"], threshold))
                st.caption(f"A higher threshold flags fewer people: precision rises and recall falls. {threshold:.0%} is where F2, "
                           "which counts recall twice as much as precision, peaks on the validation split.")

            _model_comparison(disease)

            with st.expander("View sample records"):
                preview = [c for c in model_features + [target] if c in dataset.columns]
                st.dataframe(dataset[preview].head(50), width="stretch")
