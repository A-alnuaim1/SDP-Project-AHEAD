# AHEAD visual redesign

Implemented on `codex/clinical-workflow-upgrade`. The application remains Streamlit; no model, authentication, database schema, storage service or API integration was replaced. There are no new dependencies, environment variables or migrations for this redesign.

## Design and pages

- Shared royal-blue, cyan, emerald, violet and orange palette; navy-gradient sidebar; grouped navigation; gradient page headers; rounded cards; consistent Plotly styling; stronger form focus and button feedback.
- Patient overview now reads the signed-in patient's stored assessments, lifestyle records, goals, challenge activity and medication schedule. It displays action cards, a wellness ring, category comparison, wellness/sleep trends, latest reported progress and recent assessments. Empty states invite a first check-in rather than generating sample records.
- Prediction results use an animated score ring and tabs for recommendations, key inputs, AHEAD Insight, model information and history. Disease color follows the original binary prediction/decision threshold (flagged red, not flagged green). No arbitrary intermediate medical category was introduced. A model score is not shown as clinical certainty or individual model accuracy. The existing illustrative sensitivity calculation has a horizontal chart and remains explicitly noncausal.
- Lifestyle tracker has category colors, score explanations, recommendation expanders, current/previous comparison, weight/activity/nutrition/sleep/hydration history, goal progress cards and a 30-day activity calendar. A weight goal displays the current weight, target and difference, not a misleading completion percentage. Existing scoring, goal and challenge logic remains intact.
- Food analyzer retains upload, image preview, API/fallback messaging, confirmation, editing and saving. Confirmed nutrient estimates receive colored cards and a macronutrient donut. Donut proportions are grams of protein/carbohydrates/fat, not percentages of calories.
- Medication pages have summary cards and a dose-status timeline. Notifications have unread/total cards and an unread filter. Existing assignment, dose logging, reminders and read actions remain intact.
- Doctor/Admin advanced intelligence uses scoped record metrics, disease and result distributions, dated assessment activity, model usage, per-model patient history and medication adherence visualizations. The clinical review page adds record/data-quality/scored-record metrics. Existing record search, filters and actions remain available.
- Public sign-in page has a new gradient presentation and feature introduction. Settings, assistant, voice, screening forms, existing analytics and public information pages receive shared styling or targeted presentation updates.

The existing application combines Doctor/Admin access. This update does not invent a separate unrestricted administrator role, appointment data, unsupported clinical categories, fake usage statistics or fabricated history.

## Interaction and accessibility

Hover elevation, pressed-button feedback, animated rings/progress bars, expandable score explanations, result tabs, chart tooltips, dashboard navigation actions and notification filtering are implemented. CSS provides visible keyboard focus, labeled status indicators, mobile wrapping, logical border directions and reduced-motion handling. Existing Arabic RTL behavior is retained and new core navigation/status labels are translated. Some new chart titles and descriptive text remain English; this is not a complete localization overhaul.

## Files

New reusable UI: `ahead/visuals.py`, `ahead/redesign.css`, `ahead/pages/patient_dashboard.py`.

Updated shared shell: `app.py`, `ahead/theme.py`, `ahead/i18n.py`, `login.css`.

Updated result view: `ahead/screening.py`.

Updated pages: `ahead/pages/overview.py`, `wellness.py`, `food.py`, `medications.py`, `notifications.py`, `doctor_intelligence.py`, `clinical.py`, `settings.py`, `public.py`, `assistant.py`, `voice.py` (all under `ahead/pages/`).

Verification and documentation: `scripts/check_redesign.py`, `docs/UI_REDESIGN.md`.

## Verification performed

- Existing pytest suite: **9 passed**. Covers existing storage, access/clinical validation and PDF behavior as defined by the repository tests.
- Streamlit AppTest: patient overview, wellness, food, medications, notifications, voice, settings, screenings, analytics and assistant; doctor overview, advanced intelligence and clinical review. Empty and populated states tested using an isolated temporary database.
- Actual patient and doctor sign-in form submissions passed.
- Lifestyle form submission persisted a new check-in.
- Patient navigation to the doctor intelligence page was redirected to the permitted overview; restricted navigation was absent.
- Arabic/dark-mode page render completed without exception. This is a render check, not a visual/accessibility audit.
- Original diabetes, heart and kidney pipelines all executed and their redesigned result pages rendered without exception.
- Python compilation and Git whitespace checks passed.
- Streamlit server started successfully in the test environment.

Browser launch was denied by the execution environment. An escalation request was rejected by the environment permission policy. Consequently, desktop/mobile screenshot review, keyboard/screen-reader evaluation, CSS contrast measurement and browser speech interaction are **not verified**. Gemini live API calls were not tested. Automated page checks do not constitute a complete end-to-end test of every unchanged form or a clinical validation of prediction models.

## Run locally

From the repository root, using the project's compatible Python version (tested here with Python 3.12):

```sh
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or macOS/Linux with `source .venv/bin/activate`, then:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Keep the existing database configuration and server-side API secrets. No secrets or database records are included in the redesign changes.

To repeat automated verification:

```sh
python -m pip install -r requirements-dev.txt pytest
python -m pytest -q
python scripts/check_redesign.py
```

The smoke script creates test users and sample records only inside a temporary database; it does not populate the production database.

## Review before presentation

1. Sign in as a patient; check Overview with both no history and several saved check-ins.
2. Run each existing disease assessment; inspect the score label, original threshold, result tabs and saved history.
3. Visit lifestyle history, goals and the challenge calendar; verify tooltips and values against stored records.
4. Review a confirmed meal and its gram-based donut; check medication and notification pages.
5. Sign in as Doctor/Admin; inspect linked-patient analytics, disease filters, patient history and clinical review.
6. Check English/Arabic, Light/Dark, keyboard navigation, reduced-motion preference and narrow/mobile widths in a real browser.

Deploy only from `codex/clinical-workflow-upgrade` after these changed files are committed and pushed there. A local code change or a Streamlit reboot alone does not publish files to GitHub. Do not merge into `main` for this update.

GitHub publication from this environment was rejected with `403: Resource not accessible by integration`. No remote branch was changed. The changed-files ZIP can be applied to a clean checkout of the requested branch by preserving its relative paths, reviewing the Git diff in GitHub Desktop, then committing and pushing that branch. If your teammate has edited these files since this handoff, use the included patch with `git apply --check` first and resolve differences rather than overwriting their work. The source baseline matched shared-branch commit `0479b9e98a1f00485703e1ef6d3a80f17667240b` when inspected.
