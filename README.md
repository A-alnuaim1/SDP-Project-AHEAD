# ✚ AHEAD — Advanced Health Early Awareness and Disease Detection System

**Senior Design Project** · Python · Streamlit · scikit-learn · XGBoost · Plotly · Gemini

AHEAD is an educational early-awareness screening platform. A user (or a clinician) enters
lifestyle, medical-history and clinical values and receives a machine-learning screening
estimate for **diabetes**, **cardiovascular disease** or **chronic kidney disease** — together
with a gauge, next steps, the factors AHEAD noticed in the inputs and an optional AI-generated
explanation.

> AHEAD does not diagnose disease and must not be used to start, stop or change treatment.

---

## Table of contents

1. [Features](#features)
2. [Project structure](#project-structure)
3. [Setup](#setup)
4. [Running the app](#running-the-app)
5. [Demo accounts](#demo-accounts)
6. [AI features (Gemini)](#ai-features-gemini)
7. [Datasets](#datasets)
8. [Re-training the models](#re-training-the-models)
9. [How the machine learning works](#how-the-machine-learning-works)
10. [Deploying on Streamlit Cloud](#deploying-on-streamlit-cloud)
11. [Common errors](#common-errors)

---

## Features

| Area | What it does |
|---|---|
| **Sign-in** | Patient or Doctor/Admin demo accounts, public About / Contact / Help pages |
| **Overview** | Welcome hero with quick-start, KPI cards, session risk distribution, dataset and model-quality charts |
| **Screenings** | Guided three-section form per condition, six selectable model families (recommended first), gauge, next steps, risk-factor chips, urgent-care notice, AHEAD Insight (Gemini), model interpretation |
| **Data & Analytics** | Dataset explorer, final test metrics, threshold sweep (heart), algorithm comparison table |
| **Clinical Dashboard** *(doctors)* | Review CSV/XLSX rows for missing/invalid values, derive age from DOB, correct incomplete records, score a cohort, record clinician decisions, and download CSV/PDF summaries |
| **AI Assistant** | "Ask AHEAD" chat with suggested prompts; answers via Gemini (offline answers for the suggestions) |
| **Settings** | Persistent profile/password, Light/Dark appearance, English/partial Arabic patient labels, session history controls and account actions |

### Patient wellness and linked care (experimental)

The patient navigation also includes **Health & Lifestyle Tracker** (transparent five-part lifestyle score,
dated check-ins, trends, personal goals, manual 30×30 activity log), **Food Nutrition Analyzer**
(photo preview, optional real Gemini image estimate, mandatory manual confirmation, meal history),
**Medication Monitoring** (doctor-assigned plans and taken/skipped/unlogged dose tracking),
**Notifications** (in-app reminders when the website is open), and an optional browser-speech **Voice Assistant**.
Wellness is independent of the three disease models. Numeric wellness rules are educational heuristics, not
validated clinical scales. A food photo is never analyzed without a configured Gemini key; manual entry
remains available. The photo is sent to Google's model only when the user presses Analyze Food and is not
stored by AHEAD. Nutrition estimates must be checked and confirmed by the patient. Voice speech recognition
depends on browser support and is limited to short read-only answers; page buttons provide reliable navigation.

Patients explicitly add a doctor's account email under **Settings → Preferences → Care team** to grant
that doctor access to their account records; they can revoke access there. A separate unchecked permission
allows that doctor to send selected linked records to Gemini for an optional AI pre-consultation draft.
The draft must be verified against the source records. The doctor-only **Advanced
Analytics & Patient Intelligence** area reports only linked patients, their dated saved assessments,
medication adherence, and record-based summaries. Doctors create/edit/pause/resume/stop medication plans;
patients can only log their own scheduled doses. The original **Clinical Dashboard** handles imported
de-identified files separately. Imported rows are not automatically merged with registered patient accounts.
The default record-based summary and data-quality checklist do not use generative AI. Voice questions are
asked through the microphone in the **AI Assistant** chat bar: Gemini transcribes the recording (it is not
stored), page requests such as "open medication" and the patient's own status questions are answered inside
the app, and answers can be read aloud by the browser. The voice and photo flows have explicit fallback paths
and make no diagnostic or prescribing decisions.

Database migration is automatic and additive on app startup: the existing `screenings` table gains input
history columns and `ahead/platform_data.py` creates linked-care, wellness, food, medication, dose, notification,
activity, and prediction-version tables. Make a backup of `AHEAD_DB_PATH` before upgrading an existing
database. No manual migration command is required. `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) enables optional
food photo analysis and the existing AI assistant/insights. The database still defaults to local SQLite;
Streamlit Community Cloud does not provide durable application-file storage. Do not upload real patient data.
In-app reminders are calculated when pages are visited; there are no background jobs, push, SMS, or email.
Times in medication scheduling use `Asia/Dubai`; historical timestamps are stored in UTC.

---

## Project structure

```
SDP-Project-AHEAD/
├── app.py                  ← entry point: page config, routing, sidebar
├── ahead/
│   ├── config.py           ← disease registry, form layout, labels, navigation, demo accounts
│   ├── resources.py        ← cached models / metadata / datasets / images / Gemini client
│   ├── clinical_data.py    ← row-by-row upload validation and DOB calculation
│   ├── storage.py          ← local SQLite accounts and doctor review queue
│   ├── reports.py          ← downloadable individual screening PDFs
│   ├── i18n.py             ← patient-facing Arabic labels
│   ├── theme.py            ← colour tokens (light & dark), global CSS, Plotly styling
│   ├── components.py       ← shared UI pieces + session-state helpers
│   ├── screening.py        ← the guided screening flow (form → prediction → guidance)
│   └── pages/
│       ├── public.py       ← login, about, contact, help
│       ├── overview.py     ← dashboard
│       ├── screenings.py   ← condition picker + predictor
│       ├── analytics.py    ← data & model explorer
│       ├── clinical.py     ← doctor-only cohort screening
│       ├── assistant.py    ← Ask AHEAD chat
│       └── settings.py     ← settings tabs
├── train_model.py          ← benchmarks six model families per disease, saves models + metadata
├── login.css               ← styles for the public (pre-login) pages
├── assets/                 ← favicon and SVG icons (inlined as data URIs)
├── static/                 ← large background images, served by Streamlit at app/static/…
├── data/                   ← diabetes.csv · heart.csv · kidney_disease.csv
├── models/
│   ├── <disease>_model.pkl ← recommended pipeline per disease
│   ├── all_models/         ← every benchmarked pipeline (<disease>_<model>.pkl)
│   └── model_meta.json     ← features, metrics, thresholds, benchmark results
├── .streamlit/config.toml  ← Streamlit theme + server settings
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended.

Set `AHEAD_DB_PATH` to a protected durable volume to retain accounts and doctor review records across app restarts. By default the SQLite database is `ahead.sqlite3` in the project folder. Streamlit Cloud's local files may be lost on restart; use a persistent volume or a database service before relying on saved records. Keep the database private and use fictional or de-identified patient data for demonstrations.

---

## Running the app

```bash
streamlit run app.py
```

Open the printed URL (usually `http://localhost:8501`). Create a patient account on the sign-in page. To provision the first doctor account, set `AHEAD_DOCTOR_EMAIL` and a strong `AHEAD_DOCTOR_PASSWORD` (12+ characters) before the first run.

---

## Demo accounts

| Role | Email | Password |
|---|---|---|
| Patient | `user@ahead.demo` | `user123` |
| Doctor / Admin | `doctor@ahead.demo` | `doctor123` |

Demo accounts are disabled by default. Set `AHEAD_ENABLE_DEMO_ACCOUNTS=1` **only** for a classroom demo; never upload identifiable patient records when demo credentials are enabled. Accounts use salted password hashes and are stored in the local SQLite database. Doctor accounts see the Clinical Dashboard and records are scoped to their own account. This is a research prototype, not a production authentication or health-record system.

Clinical uploads require model feature columns except age. An optional `DOB`, `DateOfBirth`, `date_of_birth`, or `Date of Birth` column (YYYY-MM-DD) fills age or the heart model's age category. Missing and invalid values appear per row and are not scored until corrected. Correct a row and save it in the review queue, then score eligible records. Clinicians can mark a record reviewed or excluded and add a note. The file is limited to 5,000 rows and 15 MB. The PDF and CSV contain entered data; handle downloads appropriately.

Arabic currently covers navigation, key patient-screening controls, several input labels and the result summary. Long-form guidance, doctor tools and the generated PDF remain English; full translation and right-to-left layout are future work. No email, text or push notifications are sent. The map-search link does not show live appointment availability.

---

## AI features (Gemini)

AHEAD Insight (personalised result explanation) and the AI Assistant use Google Gemini
through the `google-genai` SDK. Provide an API key in **either** place:

```bash
export GEMINI_API_KEY="your-key"          # environment variable
```

or in `.streamlit/secrets.toml` (git-ignored):

```toml
GEMINI_API_KEY = "your-key"
```

Without a key the app still works: Insight shows a notice and the Assistant answers the
suggested questions from built-in text. The model name lives in `ahead/config.py`
(`GEMINI_MODEL`).

---

## Datasets

| Condition | File | Records | Target |
|---|---|---|---|
| Diabetes | `data/diabetes.csv` | 100,000 | `diabetes` |
| Cardiovascular | `data/heart.csv` | 246,022 | `HadHeartAttack` |
| Chronic kidney disease | `data/kidney_disease.csv` | 1,659 | `Diagnosis` |

Two of the datasets are strongly imbalanced (8.5 % positive for diabetes, 5.5 % for heart,
92 % positive for kidney), which is why the app reports disease recall, F1/F2 and ROC-AUC
alongside accuracy.

---

## Re-training the models

The trained pipelines are included. To rebuild everything:

```bash
python train_model.py
```

For each disease the script:

1. splits the data 64 % train / 16 % validation / 20 % test (stratified),
2. trains Random Forest, Logistic Regression, Gradient Boosting, SVM, Decision Tree and XGBoost
   inside a `Pipeline(preprocessor → classifier)`,
3. saves every pipeline to `models/all_models/` so the app can compare them,
4. picks the winner on the validation set (macro-F1 for diabetes/kidney, F2 for heart),
5. tunes the heart model's probability threshold on the validation set,
6. reports final metrics once on the untouched test set and writes `models/model_meta.json`.

---

## How the machine learning works

* **Preprocessing** — numeric columns are median-imputed and standardised; categorical columns
  are most-frequent-imputed and one-hot encoded (`ColumnTransformer`).
* **Prediction** — `predict_proba()` gives the probability of the disease class; the result is
  "elevated" when it reaches the model's decision threshold (0.5, or the tuned value for the
  recommended heart model).
* **Interpretation** — tree models expose `feature_importances_`, Logistic Regression uses
  absolute coefficient magnitude; RBF-SVM has no per-feature importance.
* **Rule-based factors** — independent of the model, the app highlights entered values that are
  commonly discussed with a clinician (e.g. HbA1c ≥ 5.7 %, BMI ≥ 30, eGFR < 60).

---

## Deploying on Streamlit Cloud

1. Push the repository to GitHub — `app.py`, the whole `ahead/` package, `assets/`, `static/`,
   `models/` and `data/` must all be committed (note: `data/heart.csv` is 82 MB, above GitHub's
   50 MB warning threshold; consider Git LFS).
2. On [share.streamlit.io](https://share.streamlit.io) create a new app pointing at `app.py`.
3. Add `GEMINI_API_KEY` under *Advanced settings → Secrets* if you want the AI features.

---

## Common errors

| Error | Fix |
|---|---|
| "No trained model is available" | Run `python train_model.py` to create `models/` |
| `ModuleNotFoundError` | Activate the virtual environment and `pip install -r requirements.txt` |
| A selected model says "has not been saved yet" | Run `python train_model.py` to generate `models/all_models/` |
| AI Assistant / Insight unavailable | Provide `GEMINI_API_KEY` (see above) |
| Background images missing | `server.enableStaticServing = true` must stay in `.streamlit/config.toml` and the PNGs in `static/` |
| Signed out after refreshing the page | Expected: the prototype keeps the session in memory only (no cookies) |
| Blank page after editing | Hard-refresh the browser (`Ctrl/Cmd + Shift + R`) |
