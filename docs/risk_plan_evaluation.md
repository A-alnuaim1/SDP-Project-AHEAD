> **Note (November 2026): this study ran on retired data.** Its diabetes and kidney figures come from the Kaggle
> `diabetes.csv` (origin and licence unknown) and the synthetic Kaggle `kidney_disease.csv`, which AHEAD no longer
> uses, and its cardiovascular figures from the earlier class-weighted heart model (see `docs/data_sources.md`).
> The engine and its script (`ahead/risk_plan.py`, `scripts/evaluate_risk_plans.py`) were removed, so the numbers
> below cannot be re-run. The conclusion still stands, and for reasons that do not depend on those files: a model
> score describes who in the data had an outcome, not what a change would do for one person, so a plan chosen to
> lower it is not evidence that it lowers anyone's risk. The current models make this plainer still. The diabetes
> and kidney models take no lab values, so a doctor's lab target cannot even be expressed as a model change; in the
> diabetes model, smoking barely moves the score; in the kidney model (gradient boosting), adding 5 to BMI lowers
> the score for 21% of test rows; and the heart model's label is a past heart attack. So AHEAD still sorts each
> result into a track and lets a doctor approve standard prevention advice instead of letting the model choose a plan.

> **Why AHEAD sorts results into tracks.** This study is why every screening result now goes on one track ("See a doctor soon", "At risk" or "Not flagged") instead of getting a plan the model chose to lower its score.
> It found that 98% of flagged diabetes and 95% of flagged kidney test rows already have the condition, most with labs already in diagnostic ranges, and that realistic changes rarely take a flagged score below the threshold.
> Its gates became the urgent rules in `ahead/triage.py`; the report below is unchanged output of `python scripts/evaluate_risk_plans.py --out <dir> --demo-db <copy of a demo database>`.

# Path to lower risk: offline evaluation (step 1 of 7)

Generated 2026-11-05 14:00 UTC by `scripts/evaluate_risk_plans.py`. Engine: `ahead/risk_plan.py`. No user interface yet.

## What was tested

When AHEAD flags a screening, the new engine looks for the smallest realistic set of changes to inputs a person can change that would bring the model's score below the screening threshold, with a safety margin of 2 percentage points. It searches twice: once with the patient's own habits only (weight, smoking, physical activity, sleep, alcohol and, for kidney, diet), and once adding targets a doctor would set (HbA1c, blood glucose, blood pressure). Each numeric change is tried at 25%, 50%, 75% and 100% of the most it may change in one plan, each category change (such as current smoker to former smoker) as a single step, and plans have at most 3 changes. Age, sex, medical and family history, self-rated health and the kidney laboratory markers are never changed, and a value the person left blank is never filled in.

Three safeguards decide whether a plan is offered at all:

- **Gates.** No plan is offered to anyone under 18 (the rules are written for adults), or when an entered value is already in a range that needs a doctor (on the diabetes screening: HbA1c 6.5% or more, or blood glucose 200 mg/dL or more; on the kidney health screening: eGFR below 60, urine ACR 30 mg/g or more, HbA1c 6.5% or more, fasting blood sugar 126 mg/dL or more, systolic blood pressure 180 mmHg or more, or diastolic blood pressure 120 mmHg or more). These results need a clinician, not a plan to lower a screening score. The gates are the urgent rules in `ahead/triage.py`, the same table that puts a Screenings result on the "See a doctor soon" track; the cut-offs are prototype values for a clinician to confirm (see *Gates* below).
- **Every change helps on its own.** A change is only used if it lowers that person's score by itself; a change that only helps through an interaction with another change in the model is dropped. A crossing path also needs every one of its changes: dropping any one stops it crossing.
- **Minimum drop.** When no path crosses, the best reduction is offered only if it lowers the score by at least 1 percentage point; smaller drops are reported as *negligible* and no plan is offered.

**These are the model's scores for changed values, not a guarantee or a causal effect. The model learnt patterns in past data; a lower score after a change does not show that making the change would lower this person's risk.**

## Is this the right test data?

The test rows were rebuilt exactly as `train_model.py` builds them (20% held out, `random_state=42`, stratified by the outcome). To prove it is the same split, the recommended model's final test metrics were recomputed on these rows and compared with the figures recorded in `models/model_meta.json` when the models were trained.

| Condition | Model | Test rows | Accuracy (recorded / recomputed) | Recall | Precision | ROC AUC | Match |
|---|---|---|---|---|---|---|---|
| Diabetes | GradientBoosting | 20,000 | 0.9723 / 0.9723 | 0.6900 / 0.6900 | 0.9775 / 0.9775 | 0.9794 / 0.9794 | yes (all 8 metrics identical) |
| Cardiovascular | LogisticRegression | 49,205 | 0.8117 / 0.8117 | 0.6368 / 0.6368 | 0.1711 / 0.1711 | 0.8279 / 0.8279 | yes (all 8 metrics identical) |
| Kidney Health | XGBoost | 332 | 0.9157 / 0.9157 | 0.9639 / 0.9639 | 0.9453 / 0.9453 | 0.8046 / 0.8046 | yes (all 8 metrics identical) |

Every recorded metric (accuracy, precision, recall, F1 for each class, macro F1, F2 and ROC AUC) was reproduced to 4 decimal places, so the evaluation below uses the same held-out test rows the models were scored on, rows the models never saw in training.

## Headline results

Only test rows the recommended model flags are used (score at or above its threshold). Where a condition had more than 1,000 flagged rows, a fixed random sample of 1,000 (seed 42) was used. *Has the condition* is the row's label in the dataset. *Eligible* rows are the ones no gate stops; path rates are shares of eligible rows.

| Condition | Flagged test rows | Evaluated | Has the condition | No plan offered (gate) | Eligible | Path, habits only | Path, habits + doctor targets | Changes needed (habits + doctor, median) | Score drop on a path, points (habits + doctor) |
|---|---|---|---|---|---|---|---|---|---|
| Diabetes | 1,200 of 20,000 | 1,000 | 980 of 1,000 (98.0%) | 974 of 1,000 (97.4%) | 26 | 46.2% (12) | 61.5% (16) | 1 (IQR 1–1.25) | 7.16 (IQR 4.4–10.97) |
| Cardiovascular | 10,002 of 49,205 | 1,000 | 175 of 1,000 (17.5%) | 0 of 1,000 (0.0%) | 1,000 | 4.2% (42) | 4.2% (42) | 1 (IQR 1–1) | 8.47 (IQR 8.28–8.58) |
| Kidney Health | 311 of 332 | 311 | 294 of 311 (94.5%) | 308 of 311 (99.0%) | 3 | 0.0% (0) | 0.0% (0) | – | – |

In plain words:

- **Diabetes** (random sample of 1,000 of 1,200 flagged test rows, seed 42): 980 of 1,000 (98.0%) of the evaluated flagged rows have the condition in the data. 974 of 1,000 (97.4%) get no plan because a gate trips (13 under 18, 961 HbA1c, glucose or fasting blood sugar in a diabetes diagnostic range). That leaves 26 eligible rows (13 with the condition). Among them, habit changes alone brought the model's score below the target for 46.2% (12), and adding doctor targets for 61.5% (16); over all evaluated flagged rows that is 1.2% and 1.6%. Paths by label among eligible rows: with condition: habits only 4 of 13 (30.8%), habits + doctor 7 of 13 (53.8%); without condition: habits only 8 of 13 (61.5%), habits + doctor 9 of 13 (69.2%). With habits only, paths needed 1 (IQR 1–1) changes and lowered the score by 6.88 (IQR 4.33–8.13) points. With doctor targets, paths needed 1 (IQR 1–1.25) changes and lowered the score by 7.16 (IQR 4.4–10.97) points. Of the 10 eligible rows where habits + doctor targets did not cross, 8 had a best reduction of at least 1 point (10.82 (IQR 9.38–14.09) points) and 0 only a negligible one; counting every row without a plan as 0, the drop over all 10 was 10.18 (IQR 6.21–13.29) points. Most used in paths with doctor targets: Body weight (BMI) (88% of paths), Blood glucose target (31% of paths), HbA1c target (12% of paths).
- **Cardiovascular** (random sample of 1,000 of 10,002 flagged test rows, seed 42): 175 of 1,000 (17.5%) of the evaluated flagged rows have the condition in the data. No heart input trips a gate. That leaves 1,000 eligible rows (175 with the condition). Among them, habit changes alone brought the model's score below the target for 4.2% (42), and adding doctor targets for 4.2% (42); over all evaluated flagged rows that is 4.2% and 4.2%. Paths by label among eligible rows: with condition: habits only 6 of 175 (3.4%), habits + doctor 6 of 175 (3.4%); without condition: habits only 36 of 825 (4.4%), habits + doctor 36 of 825 (4.4%). With habits only, paths needed 1 (IQR 1–1) changes and lowered the score by 8.47 (IQR 8.28–8.58) points. With doctor targets, paths needed 1 (IQR 1–1) changes and lowered the score by 8.47 (IQR 8.28–8.58) points. Of the 958 eligible rows where habits + doctor targets did not cross, 384 had a best reduction of at least 1 point (1.69 (IQR 1.26–5.02) points) and 484 only a negligible one; counting every row without a plan as 0, the drop over all 958 was 0 (IQR 0–1.45) points. Most used in paths with doctor targets: Stop smoking (100% of paths), Body weight (BMI) (7% of paths), Regular physical activity (7% of paths).
- **Kidney Health** (all 311 flagged test rows): 294 of 311 (94.5%) of the evaluated flagged rows have the condition in the data. 308 of 311 (99.0%) get no plan because a gate trips (26 HbA1c, glucose or fasting blood sugar in a diabetes diagnostic range, 282 eGFR or ACR in the chronic kidney disease range). That leaves 3 eligible rows (2 with the condition). Among them, habit changes alone brought the model's score below the target for 0.0% (0), and adding doctor targets for 0.0% (0); over all evaluated flagged rows that is 0.0% and 0.0%. Paths by label among eligible rows: with condition: habits only 0 of 2 (0.0%), habits + doctor 0 of 2 (0.0%); without condition: habits only 0 of 1 (0.0%), habits + doctor 0 of 1 (0.0%). Of the 3 eligible rows where habits + doctor targets did not cross, 2 had a best reduction of at least 1 point (4.21 (IQR 3.36–5.07) points) and 1 only a negligible one; counting every row without a plan as 0, the drop over all 3 was 2.5 (IQR 1.25–4.21) points. 

Runtime per eligible row, one search at a time (milliseconds; gated rows return at once):

| Condition | Search | Rows | Median | 95th percentile | Slowest |
|---|---|---|---|---|---|
| Diabetes | patient only | 26 | 45 | 63 | 64 |
| Diabetes | patient + doctor | 26 | 45 | 60 | 81 |
| Cardiovascular | patient only | 1000 | 46 | 67 | 148 |
| Cardiovascular | patient + doctor | 1000 | 46 | 67 | 158 |
| Kidney Health | patient only | 3 | 34 | 35 | 35 |
| Kidney Health | patient + doctor | 3 | 37 | 41 | 42 |

Detailed counts per outcome (rows):

| Condition | Search | Crosses | Best reduction (1 point or more) | Negligible (under 1 point) | No change lowers the score on its own | No change applies | Gate: under 18 | Gate: diabetes diagnostic range | Gate: kidney markers | Gate: severe blood pressure |
|---|---|---|---|---|---|---|---|---|---|---|
| Diabetes | patient only | 12 | 12 | 0 | 2 | 0 | 13 | 961 | 0 | 0 |
| Diabetes | patient + doctor | 16 | 8 | 0 | 2 | 0 | 13 | 961 | 0 | 0 |
| Cardiovascular | patient only | 42 | 384 | 484 | 47 | 43 | 0 | 0 | 0 | 0 |
| Cardiovascular | patient + doctor | 42 | 384 | 484 | 47 | 43 | 0 | 0 | 0 | 0 |
| Kidney Health | patient only | 0 | 2 | 1 | 0 | 0 | 0 | 26 | 282 | 0 |
| Kidney Health | patient + doctor | 0 | 2 | 1 | 0 | 0 | 0 | 26 | 282 | 0 |

## Which changes the paths use

*Applies* is the share of eligible rows where the change is allowed at all (for example, the BMI change applies only at a BMI of 25 or more). *Used* is the share of crossing paths that include it. *At full limit* is how often, when used, it had to go all the way to its maximum allowed change (category changes such as stopping smoking are always a full step).

**Diabetes**

| Change | Who | Applies | Used (habits only) | Used (habits + doctor) | At full limit (habits + doctor) |
|---|---|---|---|---|---|
| Body weight (BMI) | patient | 100.0% | 100.0% (12) | 87.5% (14) | 35.7% |
| Stop smoking | patient | 3.8% | 0.0% (0) | 0.0% (0) | – |
| HbA1c target | doctor | 76.9% | – | 12.5% (2) | 100.0% |
| Blood glucose target | doctor | 69.2% | – | 31.2% (5) | 0.0% |

**Cardiovascular**

| Change | Who | Applies | Used (habits only) | Used (habits + doctor) | At full limit (habits + doctor) |
|---|---|---|---|---|---|
| Body weight (BMI) | patient | 76.2% | 7.1% (3) | 7.1% (3) | 0.0% |
| Regular physical activity | patient | 38.9% | 7.1% (3) | 7.1% (3) | 100.0% |
| Stop smoking | patient | 17.6% | 100.0% (42) | 100.0% (42) | 100.0% |
| Sleep hours | patient | 44.1% | 4.8% (2) | 4.8% (2) | 100.0% |
| Stop drinking alcohol | patient | 34.4% | 0.0% (0) | 0.0% (0) | – |

**Kidney Health**

| Change | Who | Applies | Used (habits only) | Used (habits + doctor) | At full limit (habits + doctor) |
|---|---|---|---|---|---|
| Body weight (BMI) | patient | 33.3% | no paths | no paths | – |
| Stop smoking | patient | 0.0% | no paths | no paths | – |
| Alcohol intake | patient | 66.7% | no paths | no paths | – |
| Physical activity | patient | 100.0% | no paths | no paths | – |
| Diet quality | patient | 100.0% | no paths | no paths | – |
| Sleep quality | patient | 100.0% | no paths | no paths | – |
| Systolic blood pressure target | doctor | 33.3% | – | no paths | – |
| Diastolic blood pressure target | doctor | 66.7% | – | no paths | – |
| Fasting blood sugar target | doctor | 100.0% | – | no paths | – |
| HbA1c target | doctor | 66.7% | – | no paths | – |

## Does a change move the score the wrong way?

This describes the models, so it uses every evaluated flagged row, gated or not. For each change and each row where its rule applies, each allowed step was applied on its own. Two different things are measured:

- *Wrong direction*: the smallest step, or the full allowed change, RAISES the score for that row. This can happen in any model, including a monotonic one, when the model learnt that the input goes the other way.
- *Changes direction along the steps*: in the same row, the score goes up at one step and down at another between 0%, 25%, 50%, 75% and 100% of the allowed change. Only this shows the model is not monotonic for that input. A category change is a single step, so it shows '–'.

The engine uses only changes that lower the score on their own, so neither kind of change can appear in a suggested path; the tables show how often that filter matters.

| Condition | Change | Rows where it applies | Smallest step raises score | No effect | Lowers score | Full change raises score | Changes direction along the steps |
|---|---|---|---|---|---|---|---|
| Diabetes | Body Mass Index (BMI) | 904 | 13.7% | 60.1% | 26.2% | 58.0% | 43.3% of 903 |
| Diabetes | Smoking History | 107 | 0.0% | 66.4% | 33.6% | 0.0% | – |
| Diabetes | HbA1c Level (%) | 954 | 2.0% | 78.3% | 19.7% | 4.4% | 10.6% of 907 |
| Diabetes | Blood Glucose Level (mg/dL) | 845 | 0.1% | 74.9% | 25.0% | 7.9% | 13.3% of 845 |
| Cardiovascular | Body Mass Index (BMI) | 762 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% of 761 |
| Cardiovascular | Regular Physical Activity | 389 | 0.0% | 0.0% | 100.0% | 0.0% | – |
| Cardiovascular | Smoking Status | 176 | 0.0% | 0.0% | 100.0% | 0.0% | – |
| Cardiovascular | Average Sleep Hours | 441 | 13.8% | 0.0% | 86.2% | 13.8% | 0.0% of 170 |
| Cardiovascular | Alcohol Consumption Status | 344 | 100.0% | 0.0% | 0.0% | 100.0% | – |
| Kidney Health | Body Mass Index (BMI) | 182 | 18.7% | 66.5% | 14.8% | 36.8% | 15.9% of 182 |
| Kidney Health | Smoking | 79 | 2.5% | 0.0% | 97.5% | 2.5% | – |
| Kidney Health | Alcohol Consumption | 309 | 26.5% | 36.2% | 37.2% | 36.6% | 45.3% of 307 |
| Kidney Health | Physical Activity | 308 | 28.2% | 41.2% | 30.5% | 30.8% | 59.2% of 304 |
| Kidney Health | Diet Quality Score | 308 | 20.5% | 59.4% | 20.1% | 33.4% | 24.6% of 305 |
| Kidney Health | Sleep Quality Score | 305 | 34.4% | 38.4% | 27.2% | 45.2% | 52.8% of 301 |
| Kidney Health | Systolic Blood Pressure (mmHg) | 159 | 17.6% | 67.9% | 14.5% | 34.0% | 20.1% of 159 |
| Kidney Health | Diastolic Blood Pressure (mmHg) | 201 | 18.9% | 62.2% | 18.9% | 41.8% | 24.7% of 198 |
| Kidney Health | Fasting Blood Sugar (mg/dL) | 249 | 32.9% | 44.6% | 22.5% | 45.8% | 68.5% of 248 |
| Kidney Health | HbA1c (%) | 231 | 22.5% | 58.9% | 18.6% | 38.5% | 33.6% of 229 |

- **Cardiovascular** (LogisticRegression): a logistic regression's score moves one way only as an input rises, so this model is monotonic by construction, and none of the 931 rows with several BMI or sleep steps changes direction (0 reversals). Its 'wrong way' figures come from the direction the model learnt, read from its fitted coefficients. Sleep hours carry a coefficient of -0.0441 per standard deviation, so more sleep goes with a lower score: moving down toward 9 h raised the score in 61 of 61 rows above 9 h, and moving up toward 7 h raised it in 0 of 380 rows below 7 h. Switching drinking from 'Yes' to 'No' adds +0.2609 to the log-odds, so stopping alcohol raised this model's score in 100.0% of the rows where it applies. These are associations in the survey data, not effects; in this model the 'down to 9 h' half of the sleep rule and 'stop drinking' can never lower the score, so the engine never suggests them.
- **Diabetes** (GradientBoosting) is a tree ensemble trained without any monotonic constraint, and it is not monotonic in practice: the score changes direction along the 25-100% steps in the same row for Body weight (BMI) 43.3% of 903, HbA1c target 10.6% of 907, Blood glucose target 13.3% of 845 rows with several steps. A small step can raise the score where a larger one lowers it, so a path can rest on a bump in the model rather than a consistent pattern.
- **Kidney Health** (XGBoost) is a tree ensemble trained without any monotonic constraint, and it is not monotonic in practice: the score changes direction along the 25-100% steps in the same row for Body weight (BMI) 15.9% of 182, Alcohol intake 45.3% of 307, Physical activity 59.2% of 304, Diet quality 24.6% of 305, Sleep quality 52.8% of 301, Systolic blood pressure target 20.1% of 159, Diastolic blood pressure target 24.7% of 198, Fasting blood sugar target 68.5% of 248, HbA1c target 33.6% of 229 rows with several steps. A small step can raise the score where a larger one lowers it, so a path can rest on a bump in the model rather than a consistent pattern.

## Worked examples: demo patients

Each demo patient's latest saved screening was read from a copy of the demo database. The demo seed saved only some inputs for some patients; inputs missing from a saved record were filled with the values the screening form shows before anyone edits it (marked below), while optional lab and blood-pressure values the form starts blank stay blank. The scores below are recomputed with the current model, so they can differ from the hand-entered scores the demo seed stored.

### Alice Mansour: Diabetes

- Saved values: Gender Female, Age 58, History of Hypertension Yes, History of Heart Disease No, Smoking History Former, Body Mass Index (BMI) 31.2, HbA1c Level (%) Not provided, Blood Glucose Level (mg/dL) 182.
- Left blank (never filled in by a plan): HbA1c Level (%).
- Saved score 55.0% (threshold 41%); recomputed with GradientBoosting: 23.4% against a threshold of 10% (flagged).
- Habits only: Body weight (BMI): 31.2 -> 28.9. Model score 23.4% -> 15.2% (does not reach the 8% target; best reduction found).
- Habits + doctor targets: Body weight (BMI): 31.2 -> 28.9; Blood glucose target: 182 -> 152. Model score 23.4% -> 13.5% (does not reach the 8% target; best reduction found).

### Bob Haddad: Cardiovascular

- Saved values: Sex Male, Age Category Age 60 to 64, Body Mass Index (BMI) 27.5.
- Filled with form defaults (not in the saved record): General Health Very good, Poor Physical Health Days 0, Poor Mental Health Days 0, Regular Physical Activity Yes, Average Sleep Hours 7, History of Stroke No, History of Asthma No, History of Kidney Disease No, History of Arthritis No, History of Diabetes No, Smoking Status Never smoked, Alcohol Consumption Status Yes.
- Saved score 22.0% (threshold 50%); recomputed with LogisticRegression: 29.7% against a threshold of 60% (not flagged).
- Habits only: Not needed: the score (29.7%) is already below the 60% threshold minus the 2-point margin.
- Habits + doctor targets: Not needed: the score (29.7%) is already below the 60% threshold minus the 2-point margin.

### Sara Khalil: Kidney Health

- Saved values: Age 66, Estimated GFR 48, Albumin-to-Creatinine Ratio (ACR) 40, Systolic Blood Pressure (mmHg) 150, Diastolic Blood Pressure (mmHg) 92.
- Filled with form defaults (not in the saved record): Gender Female, Body Mass Index (BMI) 27.65, Smoking No, Alcohol Consumption 9.95, Physical Activity 5.07, Diet Quality Score 4.99, Sleep Quality Score 6.93, Family History of Kidney Disease No, Family History of Hypertension No, Family History of Diabetes No, Presence of Edema No.
- Left blank (never filled in by a plan): Serum Creatinine, Fasting Blood Sugar (mg/dL), HbA1c (%), Blood Urea Nitrogen (BUN), Protein in Urine, Hemoglobin Level.
- Result: no score. Serum Creatinine left blank, so the Screenings page gives no estimate and there is no plan.
  (The demo seed stored a score of 81.0% for this record; the current page would not produce one.)

### Omar Nasser: Diabetes

- Saved values: Gender Male, Age 41, History of Hypertension No, History of Heart Disease No, Smoking History Former, Body Mass Index (BMI) 24.1, HbA1c Level (%) 5.2, Blood Glucose Level (mg/dL) 95.
- Saved score 8.0% (threshold 50%); recomputed with GradientBoosting: 0.1% against a threshold of 50% (not flagged).
- Habits only: Not needed: the score (0.1%) is already below the 50% threshold minus the 2-point margin.
- Habits + doctor targets: Not needed: the score (0.1%) is already below the 50% threshold minus the 2-point margin.

## Worked examples: held-out test rows

For each condition, the first evaluated row (by row number in the dataset file, counting from 0) with a habits-only path, or if there is none, the first with a habits + doctor path.

- **Diabetes**, dataset row 9572 (score 50.6%, threshold 50%; does not have the condition in the data):
  - patient only: Body Mass Index (BMI): 31.36 -> 30.6 -> score 50.6% -> 46.7% (crosses).
  - patient + doctor: Body Mass Index (BMI): 31.36 -> 30.6 -> score 50.6% -> 46.7% (crosses).

- **Cardiovascular**, dataset row 14243 (score 65.8%, threshold 60%; does not have the condition in the data):
  - patient only: Smoking Status: Current smoker - now smokes every day -> Former smoker -> score 65.8% -> 57.5% (crosses).
  - patient + doctor: Smoking Status: Current smoker - now smokes every day -> Former smoker -> score 65.8% -> 57.5% (crosses).

- **Kidney Health**: no evaluated row had a crossing path.

## One plan across conditions

A later step lets a patient and doctor agree one plan in shared terms and see its effect on every condition's score. `shared_plan()` maps each shared target onto each model's own input and coding, and each mapped change must pass that model's lever rule. A target past a limit is applied up to the strongest step the search itself would use (on the step grid and inside the training range) and marked as limited; when no such step exists, it is skipped with a reason. A condition whose values trip a gate gets no changes. It also reports each change's effect on its own, so a change that raises one model's score is visible.

| Shared target | Diabetes model | Heart model | Kidney model |
|---|---|---|---|
| BMI target | Body Mass Index (BMI) | Body Mass Index (BMI) | Body Mass Index (BMI) |
| Stop smoking | Smoking History | Smoking Status | Smoking |
| Become physically active | not in this model | Regular Physical Activity | Physical Activity |
| HbA1c target (doctor) | HbA1c Level (%) | not in this model | HbA1c (%) |
| Cut or stop alcohol | not in this model | Alcohol Consumption Status | Alcohol Consumption |

Not shared on purpose: the diabetes data's blood glucose is read as a random (non-fasting) value and the kidney data's is fasting, so neither is mapped onto the other; sleep hours (heart) and the sleep-quality score (kidney) measure different things; blood pressure and diet quality exist only in the kidney model. "Become active" sets the heart answer to Yes, but the kidney data records activity as a 0-10 number, so there it is mapped (approximately) to the largest allowed increase. An alcohol target is a share of intake to cut; the heart data only records whether someone drinks, so only a full stop maps there, and in the heart model that change raises the score.

## Gates

Prototype cut-offs for a clinician to confirm, not guideline citations. They are the urgent rules in `ahead/triage.py`; the Screenings track uses the same table, so an entered value that stops a plan here also puts the result on the "See a doctor soon" track. When any trips, no plan is offered; a blank value never trips a gate. The plan engine applies the age gate to every result, while the Screenings track counts age only for a flagged result. The heart inputs have none (every heart age category is 18 or older).

| Condition | Input | No plan when | Status | Why |
|---|---|---|---|---|
| Diabetes | Age | Age below 18 years | `not_for_children` | The plan rules are written for adults: a child's BMI is read against age and sex percentiles, and a high HbA1c or glucose in a child needs a clinician, not a habit plan. |
| Diabetes | HbA1c Level (%) | HbA1c Level (%) 6.5 or more | `diagnostic_range` | An HbA1c of 6.5% or more is a common diabetes diagnostic cut-off, so this result needs a clinician to confirm it, not a plan to lower the screening score. |
| Diabetes | Blood Glucose Level (mg/dL) | Blood Glucose Level (mg/dL) 200 or more | `diagnostic_range` | A blood glucose of 200 mg/dL or more is a common diabetes diagnostic cut-off, so this result needs a clinician to confirm it, not a plan to lower the screening score. |
| Kidney Health | Age | Age below 18 years | `not_for_children` | The plan rules are written for adults: a child's BMI is read against age and sex percentiles, and a high HbA1c or glucose in a child needs a clinician, not a habit plan. |
| Kidney Health | Estimated GFR | Estimated GFR below 60 | `kidney_markers_abnormal` | An eGFR below 60 is in the chronic kidney disease range, so a clinician should review the kidney results; a habit plan would only move the screening score while the markers a plan never changes stay the same. |
| Kidney Health | Albumin-to-Creatinine Ratio (ACR) | Albumin-to-Creatinine Ratio (ACR) 30 mg/g or more | `kidney_markers_abnormal` | A urine ACR of 30 mg/g or more is in the chronic kidney disease range, so a clinician should review the kidney results; a habit plan would only move the screening score while the markers a plan never changes stay the same. |
| Kidney Health | HbA1c (%) | HbA1c (%) 6.5 or more | `diagnostic_range` | An HbA1c of 6.5% or more is a common diabetes diagnostic cut-off, so this result needs a clinician to confirm it, not a plan to lower the screening score. |
| Kidney Health | Fasting Blood Sugar (mg/dL) | Fasting Blood Sugar (mg/dL) 126 or more | `diagnostic_range` | A fasting blood sugar of 126 mg/dL or more is a common diabetes diagnostic cut-off, so this result needs a clinician to confirm it, not a plan to lower the screening score. |
| Kidney Health | Systolic Blood Pressure (mmHg) | Systolic Blood Pressure (mmHg) 180 or more | `severe_blood_pressure` | A systolic blood pressure of 180 mmHg or more is in the severely high range, so a clinician needs to see it soon, not a plan to lower the screening score. |
| Kidney Health | Diastolic Blood Pressure (mmHg) | Diastolic Blood Pressure (mmHg) 120 or more | `severe_blood_pressure` | A diastolic blood pressure of 120 mmHg or more is in the severely high range, so a clinician needs to see it soon, not a plan to lower the screening score. |

## Lever rules used

These are prototype assumptions for a clinician to review, not guideline citations. Every category spelling, unit and range was checked against the training data. Every changed value also stays inside the range seen in the training data. One rule differs from the starting design: the diabetes data does not say its blood glucose was fasting, so it is read as random glucose and the target rule starts at 140 mg/dL instead of the fasting 100 mg/dL cut-off.

**Diabetes**

| Input | Who | Shown as | Rule | Tracker goal | Note |
|---|---|---|---|---|---|
| Body Mass Index (BMI) | patient | Body weight (BMI) | down only; only if >= 25 kg/m²; at most 10% of the current value; never below 24.9 kg/m²; inside the training-data range | weight |  |
| Smoking History | patient | Stop smoking | 'current' -> 'former' only | – | 'ever', 'not current', 'never' and 'No Info' are left alone. |
| HbA1c Level (%) | doctor | HbA1c target | down only; only if >= 5.7 %; at most 1 % per plan; never below 5.7 %; inside the training-data range | – |  |
| Blood Glucose Level (mg/dL) | doctor | Blood glucose target | down only; only if >= 140 mg/dL; at most 40 mg/dL per plan; never below 140 mg/dL; inside the training-data range | – | The dataset does not say the sample was fasting, so it is read as random (non-fasting) glucose: the rule starts at 140 mg/dL, not the fasting 100 cut-off. A clinician confirms which it is. |

**Cardiovascular**

| Input | Who | Shown as | Rule | Tracker goal | Note |
|---|---|---|---|---|---|
| Body Mass Index (BMI) | patient | Body weight (BMI) | down only; only if >= 25 kg/m²; at most 10% of the current value; never below 24.9 kg/m²; inside the training-data range | weight |  |
| Regular Physical Activity | patient | Regular physical activity | 'No' -> 'Yes' only | exercise |  |
| Smoking Status | patient | Stop smoking | 'Current smoker - now smokes every day' or 'Current smoker - now smokes some days' -> 'Former smoker' only | – | 'Never smoked' and 'Former smoker' are left alone. |
| Average Sleep Hours | patient | Sleep hours | if below 7 h: up toward 7; if above 9 h: down toward 9; at most 2 h per plan; inside the training-data range | sleep | Whole hours, as the screening form and the dataset record them. |
| Alcohol Consumption Status | patient | Stop drinking alcohol | 'Yes' -> 'No' only | – |  |

**Kidney Health**

| Input | Who | Shown as | Rule | Tracker goal | Note |
|---|---|---|---|---|---|
| Body Mass Index (BMI) | patient | Body weight (BMI) | down only; only if >= 25 kg/m²; at most 10% of the current value; never below 24.9 kg/m²; inside the training-data range | weight |  |
| Smoking | patient | Stop smoking | 1 (yes) -> 0 (no) only | – |  |
| Alcohol Consumption | patient | Alcohol intake | down only; at most 50% of the current value; inside the training-data range | – | Dataset scale 0-20; its unit is not documented in AHEAD. |
| Physical Activity | patient | Physical activity | up only; at most 3 per plan; inside the training-data range | exercise | Dataset scale 0-10; its unit is not documented in AHEAD. |
| Diet Quality Score | patient | Diet quality | up only; at most 2 per plan; inside the training-data range | – | Score 0-10. |
| Sleep Quality Score | patient | Sleep quality | up only; at most 1.5 per plan; inside the training-data range | – | Score 4-10 in the data; a quality score, not hours, so no tracker goal. |
| Systolic Blood Pressure (mmHg) | doctor | Systolic blood pressure target | down only; only if >= 130 mmHg; at most 20 mmHg per plan; never below 120 mmHg; inside the training-data range | – |  |
| Diastolic Blood Pressure (mmHg) | doctor | Diastolic blood pressure target | down only; only if > 80 mmHg; at most 10 mmHg per plan; never below 80 mmHg; inside the training-data range | – |  |
| Fasting Blood Sugar (mg/dL) | doctor | Fasting blood sugar target | down only; only if >= 100 mg/dL; at most 30 mg/dL per plan; never below 100 mg/dL; inside the training-data range | – |  |
| HbA1c (%) | doctor | HbA1c target | down only; only if >= 5.7 %; at most 1 % per plan; never below 5.7 %; inside the training-data range | – |  |

## Limitations

These figures describe how the trained models react to changed inputs, not what would happen to a person. The models learnt associations in historical or generated data, so a lower score after a change is not evidence that the change lowers anyone's risk, and the size of a score drop is not an effect size. The evaluation uses dataset rows rather than real patients, and only the rows the model flags. The lever rules and gate cut-offs are prototype assumptions written for a clinician to review; they are not personalised and do not consider time, cost, medication or how hard a change is for a given person. The search is coarse on purpose (four step sizes, at most three changes), so a finer or larger plan might exist.

The gates remove most flagged diabetes and kidney rows, so those results rest on few rows. In the diabetes data, 28% of all rows have HbA1c of 6.5% or more or glucose of 200 mg/dL or more and 17% are under 18; the model flags mostly rows that already meet a diagnostic cut-off. HbA1c and glucose there take only 18 and 18 distinct values, every record with HbA1c of 6.8% or more or glucose of 220 mg/dL or more has diabetes, and no record with HbA1c of 5% or less or glucose of 100 mg/dL or less has it, so the model's score jumps at those gaps: a small lab target can move it a lot, which is a property of this dataset, not a clinical effect. In the kidney data, 93% of all rows have eGFR below 60 or ACR of 30 mg/g or more and 99% trip at least one value gate once HbA1c and fasting blood sugar are counted (no row reaches the blood pressure gate: the data stops at 179/119 mmHg), so only 3 of the 311 evaluated flagged kidney test rows are eligible. Its inputs are spread evenly across their ranges and only weakly related to the diagnosis (no lever input's correlation with it exceeds 0.10), and 92% of its rows have the disease, so lifestyle changes barely move that model and kidney paths should not be shown to people without a warning. The heart model is a logistic regression over mostly yes/no answers, so a plan can only flip a few categories and lower BMI.

In the test rows every input is filled in; the screening form lets people leave diabetes and kidney labs blank, which lowers the threshold and makes a path harder to reach, and that case is not measured here. Runtimes were measured one patient at a time on the development laptop, after loading the data and model once, and include rescoring the chosen plan through the screening page's own single-row path.

## Method details

- Step sizes: 25%, 50%, 75%, 100% of each numeric change's allowed range, rounded toward the current value (BMI, HbA1c, alcohol, activity, diet and sleep quality to 0.1; glucose, blood pressure, fasting blood sugar and sleep hours to whole numbers).
- Target: threshold minus 2 points. The threshold is the one the Screenings page applies to the same values (including the lower threshold used when optional labs are blank).
- Order of checks: no score (a required kidney test blank), then gates, then 'already below the target', then 'no change applies', then the search.
- Each allowed step is scored on its own first; only steps that lower the score on their own are kept. If none is left, the outcome is 'no change lowers the score on its own'.
- Choice: the fewest changes that cross; among those, the smallest total share of allowed change, then the lowest score. Without a crossing plan, the best reduction: among plans that lower the score by at least 1 point, the fewest changes among those within 0.1 point of the lowest score. When no plan lowers it by 1 point, the outcome is 'negligible' and no plan is offered.
- The direction checks use every change whose rule applies, for patient and doctor alike, on every evaluated flagged row (gates aside).
- Charts: not drawn, because matplotlib is not in `requirements.txt`.
- Files: `summary.json` (all figures), `rows.csv` (one row per evaluated test row, with its gate if any).
