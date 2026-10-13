"""
ahead/resources.py
==================
Cached access to models, metadata, datasets, image assets and the Gemini
client.  Pages read their own stylesheet; everything else on disk or on the
network is reached through here.
"""

import base64
import json
import os
import pickle

import pandas as pd
import streamlit as st

from ahead.config import ASSETS_DIR, DATA_DIR, DISEASES, GEMINI_MODEL, ICONS_DIR, MODELS_DIR


# =============================================================================
# METADATA & DATASETS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with open(MODELS_DIR / "model_meta.json", "r") as file:
        return json.load(file)


@st.cache_resource(show_spinner="Loading reference data…")
def load_dataset(disease: str) -> pd.DataFrame:
    """
    The raw reference dataset.  Cached as a resource (shared object, no per-call
    copy — heart.csv alone is ~140 MB in memory), so callers must not mutate it.
    """
    return pd.read_csv(DATA_DIR / DISEASES[disease]["dataset"])


@st.cache_data(show_spinner=False)
def feature_summary(disease: str) -> dict:
    """
    Small per-feature statistics used to build the screening form, so the form
    never has to touch the full dataset on a rerun:
      categorical -> {"kind": "category", "options": [...], "default": mode}
      numeric     -> {"kind": "number", "min", "max", "median", "integer": bool,
                      "binary": bool, "mode": 0|1|None}
    """
    dataset = load_dataset(disease)
    summary = {}
    for feature in features(disease):
        series = dataset[feature].dropna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            values = series.astype(str)
            summary[feature] = {
                "kind": "category",
                "options": values.drop_duplicates().tolist(),
                "default": values.mode().iloc[0] if not values.empty else None,
            }
        else:
            unique = set(series.unique().tolist())
            summary[feature] = {
                "kind": "number",
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "integer": bool(pd.api.types.is_integer_dtype(series)) or bool(((series % 1) == 0).all()),
                "binary": unique.issubset({0, 1}),
                "mode": int(series.mode().iloc[0]) if unique.issubset({0, 1}) and not series.empty else None,
            }
    return summary


def meta(disease: str) -> dict:
    return load_metadata()[disease]


def features(disease: str) -> list:
    return meta(disease)["features"]


def best_model(disease: str) -> str:
    return meta(disease)["best_model"]


def available_models(disease: str) -> list:
    """Model names in benchmark order, recommended model first."""
    names = [row["model"] for row in meta(disease).get("validation_model_results", [])]
    recommended = best_model(disease)
    if recommended in names:
        names = [recommended] + [name for name in names if name != recommended]
    return names


def validation_results(disease: str) -> list:
    return meta(disease).get("validation_model_results", [])


def final_metrics(disease: str) -> dict:
    return meta(disease).get("final_test_metrics", {})


def decision_threshold(disease: str, model_name: str | None = None) -> float:
    """
    The tuned threshold only applies to the model it was tuned for.  Any other
    benchmark model falls back to the conventional 0.5 cut-off.
    """
    if model_name is not None and model_name != best_model(disease):
        return 0.5
    return float(meta(disease).get("decision_threshold", 0.5))


def n_samples(disease: str) -> int:
    return int(meta(disease).get("n_samples", 0))


def model_reason(disease: str) -> str:
    """Why the recommended model was chosen — derived from the metadata, not hard-coded."""
    info = meta(disease)
    name = info["best_model"]
    metric = info.get("selection_metric", "macro_f1")
    score = info.get("best_validation_score", 0)
    pretty = {"macro_f1": "macro-F1", "f2_disease": "F2 score"}.get(metric, metric)
    families = len(info.get("validation_model_results", [])) or "the"
    reason = (
        f"{name} is recommended because it achieved the strongest validation "
        f"{pretty} ({score:.3f}) among the {families} model families benchmarked on the "
        f"{DISEASES[disease]['label'].lower()} dataset."
    )
    if metric == "f2_disease":
        reason += (
            " The F2 score weights recall above precision, which reflects AHEAD's priority of not "
            f"missing at-risk individuals in {DISEASES[disease]['label'].lower()} screening."
        )
    if info.get("threshold_results"):
        reason += (
            f" Its probability threshold was tuned on the validation set to "
            f"{info.get('decision_threshold', 0.5):.2f}."
        )
    return reason


# =============================================================================
# MODELS
# =============================================================================

@st.cache_resource(show_spinner="Loading model…")
def load_pipeline(disease: str, model_name: str):
    """
    Load a benchmark pipeline saved by train_model.py.  The recommended model
    also exists as models/<disease>_model.pkl, which is used as a fallback if
    the all_models copy is missing.  Returns None if nothing is available.
    """
    safe_name = model_name.lower().replace(" ", "_")
    candidates = [MODELS_DIR / "all_models" / f"{disease}_{safe_name}.pkl"]
    if model_name == best_model(disease):
        candidates.append(MODELS_DIR / f"{disease}_model.pkl")
    for path in candidates:
        if path.exists():
            with open(path, "rb") as file:
                return pickle.load(file)
    return None


def predict_probability(pipeline, frame: pd.DataFrame) -> pd.Series:
    """Positive-class probability for every row of *frame*."""
    return pd.Series(pipeline.predict_proba(frame)[:, 1], index=frame.index)


# =============================================================================
# IMAGES
# =============================================================================

@st.cache_data(show_spinner=False)
def image_data_uri(relative_path: str) -> str:
    """Return a data: URI for a file under assets/ (svg or png)."""
    path = ASSETS_DIR / relative_path
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def icon_uri(name: str) -> str:
    return image_data_uri(str((ICONS_DIR / name).relative_to(ASSETS_DIR)))


def static_url(filename: str) -> str:
    """
    URL of a file in static/ served by Streamlit (server.enableStaticServing).
    Large images go through here instead of data URIs so they are fetched once
    and cached by the browser rather than re-sent inline on every rerun.
    """
    return f"app/static/{filename}"


# =============================================================================
# GEMINI
# =============================================================================

def gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def gemini_available() -> bool:
    return bool(gemini_api_key())


def gemini_client():
    """Bound each network attempt; do not silently retry while a user waits."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=gemini_api_key(), http_options=types.HttpOptions(
        timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))


def gemini_error_message(error: Exception) -> str:
    """Expose an actionable category, never raw responses, prompts or credentials."""
    import httpx
    code = str(getattr(error, 'code', ''))
    if isinstance(error, (TimeoutError, httpx.TimeoutException)) or code == '504':
        return 'Gemini timed out. Please try again later.'
    if code == '429':
        return 'Gemini quota or rate limit reached. Check the project usage and model limits in Google AI Studio.'
    if code in ('401', '403'):
        return 'Gemini access was denied. The app owner should check the server API key and project permissions.'
    if code == '404':
        return 'The configured Gemini model was not found or is unavailable to this project. The app owner needs to check the model setting.'
    if code == '400':
        return 'Gemini rejected the request. The app owner should check the API key, model and request configuration.'
    if code in ('500', '502', '503'):
        return 'Gemini is temporarily unavailable. Please try again later.'
    return 'Could not complete the Gemini request. Check the API project configuration and connection, then try again.'


def generate_text(prompt: str) -> str:
    """Single-turn generation. Raises if the client is not configured."""
    from google import genai

    client = gemini_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text or ""


def chat_reply(system_instruction: str, turns: list) -> str:
    """
    Multi-turn generation. *turns* is a list of {"role": "user"|"model", "text": ...};
    the system prompt travels as a system instruction so user text can never impersonate it.
    """
    from google import genai
    from google.genai import types

    client = gemini_client()
    contents = [types.Content(role=t["role"], parts=[types.Part.from_text(text=t["text"])]) for t in turns]
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text or ""
