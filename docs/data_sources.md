# Where AHEAD's data comes from

All three AHEAD models are trained on real public survey data collected by the US Centers for Disease
Control and Prevention (CDC). No training data is generated or synthetic. The two datasets AHEAD used
before (see [Retired datasets](#retired-datasets)) were removed because one was synthetic and the other's
origin could not be checked.

| Condition | Source | Licence | What the label means | Rows | Held-out test rows | Model | Test AUC |
|---|---|---|---|---|---|---|---|
| Diabetes | CDC NHANES 1999-2023 | Public domain (cite NHANES) | HbA1c 5.7% or higher at the exam, in adults never told they have diabetes | 53,903 | 4,966 (2021-2023 cycle) | Logistic regression | 0.735 |
| Kidney health | CDC NHANES 1999-2023 | Public domain (cite NHANES) | eGFR below 60 or urine ACR 30 mg/g or more at one visit | 60,396 | 5,551 (2021-2023 cycle) | Logistic regression | 0.776 |
| Cardiovascular | CDC BRFSS 2022 (Kaggle copy) | CC0 | Ever told they had a heart attack (a past event) | 246,022 | 49,205 (random 20%) | Logistic regression | 0.828 |

The whole chain runs from CDC's own files: `python scripts/fetch_nhanes.py` (download),
`python scripts/rebuild_nhanes.py` (merge and clean), `python scripts/build_nhanes_training.py` (training
files) and `python train_model.py` (models). Results are stored in `data/nhanes/build_summary.json` and
`models/model_meta.json`. The data checks, the earlier heart model's figures and the share of flagged heart
rows with a past event were measured separately, once, on the same files.

---

## The data

### NHANES (diabetes and kidney health)

The National Health and Nutrition Examination Survey (NHANES) is run by the CDC's National Center for
Health Statistics (NCHS). Each two-year cycle interviews a sample of the US population and examines them
in a mobile exam centre, where blood and urine are taken for lab tests. Public-use NHANES files are in the
public domain. Please cite them as:

> Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS).
> National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of Health and
> Human Services, Centers for Disease Control and Prevention, 1999-2023. https://www.cdc.gov/nchs/nhanes/

AHEAD's NHANES file, `data/nhanes/nhanes_1999_2023_research.csv.gz`, is built here from CDC's raw
public-use files (71,194 adults aged 18 or over, cycles 1999-2000 to 2021-2023; 2017-March 2020 replaces
2017-2018):

1. `scripts/fetch_nhanes.py` downloads 121 files (11 modules per cycle, about 283 MB) from
   wwwn.cdc.gov into `data/nhanes/raw/`, which git ignores, and logs every download.
2. `scripts/rebuild_nhanes.py` merges each cycle's modules on the participant number (SEQN) and keeps it,
   with exam status and the survey design columns, so every row can be traced back to CDC's files. It also:
   - corrects serum creatinine as CDC's laboratory notes advise: 1999-2000 (LAB18, "highly recommended":
     1.013 x creatinine + 0.147) and 2005-2006 (BIOPRO_D: -0.016 + 0.978 x creatinine). 2001-2002 stores
     creatinine as LBDSCR and needs no correction (L40_B);
   - computes eGFR with the race-free CKD-EPI 2021 equation;
   - sets the kidney marker only when both eGFR and urine ACR were measured at the visit;
   - leaves implausible values blank (for example BMI outside 10-90).

The column rules follow the NHANES extract and Colab notebooks a friend of the team prepared from the same
CDC files. We rebuilt that extract from the raw files without our corrections and compared it row by row
(`scripts/rebuild_nhanes.py --compare`): the number of people in every cycle is identical, and 71,172 of its
71,194 rows (99.97%) match on all 18 columns; the other 22 differ only in urine ACR. Two more checks against
CDC's own publications:

- "Doctor told you have diabetes" (DIQ010) "Yes" counts match the CDC codebooks: 1,073 adults against 1,081
  for all ages in 2021-2023, and 1,423 against 1,445 in 2017-March 2020 (the gap is children).
- The lab values behave like real lab values: eGFR falls as creatinine rises, HbA1c rises with blood glucose,
  age and BMI. The retired synthetic kidney file had no link between eGFR and creatinine (-0.003).

Compared with the friend's extract, our corrections change 152 kidney labels in 1999-2000 and 2005-2006
(the 1999-2000 share with eGFR below 60 goes from 3.5% to 7.4%), and recovering the 2001-2002 creatinine
adds 5,125 people to the kidney data. The diabetes data is identical.

`scripts/build_nhanes_training.py` turns the research file into two training files:

1. It drops the 5,192 people with no exam data at all (interviewed but not examined).
2. It recodes the answers to 0/1 (blank when there is no answer).
3. It keeps the people each model is about and writes:
   - `data/nhanes_diabetes.csv`: `cycle, gender, age, bmi, smoked_100, hypertension, heart_disease,
     hba1c_57_or_higher` (`heart_disease` is kept in the file but is not a diabetes model input)
   - `data/nhanes_kidney.csv`: `cycle, gender, age, bmi, smoked_100, physically_active, hypertension,
     diabetes, heart_disease, kidney_marker`

It also writes `data/nhanes/build_summary.json` with the row counts and the ADA comparison below.

What the columns mean:

| Column | Meaning |
|---|---|
| `smoked_100` | Has smoked at least 100 cigarettes in their life (1) or not (0) |
| `hypertension` | Has been told by a doctor they have high blood pressure |
| `heart_disease` | Has been told they have coronary heart disease (not asked of 18- and 19-year-olds, so blank for them) |
| `diabetes` | Has been told they have diabetes (borderline answers are blank) |
| `physically_active` | Reported moderate or vigorous leisure-time activity; blank for every 1999-2006 row (those cycles asked different questions) |
| `age` | Age in years; NHANES records older ages as 85 (1999-2006) or 80 (2007 on) |

### BRFSS 2022 (cardiovascular)

The Behavioral Risk Factor Surveillance System (BRFSS) is the CDC's yearly telephone survey of US adults.
AHEAD uses the cleaned 2022 version published on Kaggle as "Indicators of Heart Disease (2022 UPDATE)" by
Kamil Pytlak, released under CC0 (public domain dedication): `data/heart.csv`, 246,022 rows with every
answer present. Please credit both the CDC BRFSS 2022 and the Kaggle dataset.

---

## What each model learns

### Diabetes

- **Who is in it:** adults 18+ with exam data who have never been told they have diabetes (people who
  answered "borderline" or did not answer are left out) and had HbA1c measured. 53,903 people.
- **Label:** 1 if their HbA1c at the exam was 5.7% or higher (the prediabetes or diabetes range), else 0.
  25.8% of people have a 1. It is one lab value on one day, not a diagnosis.
- **Model inputs:** gender, age, BMI, ever smoked 100 cigarettes, told high blood pressure. Lab values are
  not model inputs: HbA1c defines the label, and the app checks entered HbA1c and glucose with fixed rules
  instead. "Told coronary heart disease" was dropped: among people never told they have diabetes it went
  with a slightly *lower* HbA1c (people with heart disease see doctors and get tested, so those with high
  sugar are usually already diagnosed), which the app must not show as protective. Dropping it changed the
  test AUC by 0.0001.

### Kidney health

- **Who is in it:** adults 18+ with both eGFR and urine albumin-to-creatinine ratio (ACR) measured. 60,396
  people.
- **Label:** 1 if eGFR (CKD-EPI 2021) was below 60, or urine ACR was 30 mg/g or more, at that single visit,
  else 0. 16.8% of people have a 1. This is a kidney marker on one visit, not confirmed chronic kidney
  disease, which needs the marker to last at least three months.
- **Model inputs:** gender, age, BMI, ever smoked 100 cigarettes, physically active, told high blood
  pressure, told diabetes, told coronary heart disease. Lab values (creatinine, eGFR, ACR) are not model
  inputs: they define the label, and the app checks entered values with fixed rules instead.

### Cardiovascular

- **Who is in it:** US adults who answered the BRFSS 2022 survey. 246,022 people.
- **Label:** 1 if they were ever told they had a heart attack, else 0 (5.5%). This is a **past** event. The
  model finds people whose answers look like the answers of people who have had a heart attack; it does
  not predict a future one. Among the test rows the model flags, 29% already report a heart attack, angina
  or a stroke, which is why the app also asks about a past heart attack and angina. A Yes to either, or to a
  past stroke, puts the result on "See a doctor soon" whatever the score, and no prevention plan is offered.
- **Model inputs:** unchanged (sex, general health, poor physical and mental health days, physical activity,
  sleep, stroke, asthma, kidney disease, arthritis, diabetes, smoking, age group, BMI, alcohol).

---

## How the models were trained and tested

- **Split (diabetes and kidney):** the most recent cycle, 2021-2023, is the test set. It is never used for
  fitting, choosing a model or choosing a threshold, so the test shows how the model does on people
  surveyed after all of its training data. The earlier cycles are split 80% training / 20% validation,
  stratified by the label (random_state=42). The cycle itself is never a model input.
- **Split (cardiovascular):** BRFSS 2022 is one survey year, so the split stays a stratified random
  64% training / 16% validation / 20% test (random_state=42).
- **Model families:** random forest, logistic regression, gradient boosting, support vector machine (trained
  on a 15,000-row sample), decision tree and XGBoost, each inside the same preprocessing pipeline.
- **No re-weighting:** the models are trained without `class_weight="balanced"`, so a score is close to the
  share of similar people in the data who had the outcome (see [Calibration](#calibration)).
- **Choosing a model:** logistic regression for every condition. Its score always moves the same way when
  an answer changes, and each answer's effect can be explained; the team's own study found the tree models
  reversing direction (for example a lower BMI raising the diabetes score). The other five families are
  trained as benchmarks. On validation, logistic regression was 0.003 below the best family for diabetes
  (0.764 against gradient boosting's 0.767), 0.001 for cardiovascular (0.834 against 0.835) and 0.019 for
  kidney (0.779 against 0.797).
- **Threshold:** for each model, the score cut-off that maximises Youden's J (sensitivity + specificity - 1)
  on the validation set, so a missed case and a false alarm count the same. An F2 rule (missed cases count
  more) flagged 75% of diabetes test people, which would make "At risk" meaningless.

---

## Results on the held-out test set

Sensitivity is the share of people with the label that the model flags; specificity is the share of people
without it that it does not flag; precision is the share of flagged people who have the label. Each model
uses its own validation threshold. The chosen model is in bold.

### Diabetes (2021-2023 cycle, 4,966 people, 28.2% with HbA1c 5.7% or higher)

| Model | Threshold | AUC | Sensitivity | Specificity | Precision | Share flagged |
|---|---|---|---|---|---|---|
| **Logistic regression** | **0.21** | **0.735** | **86.0%** | **48.3%** | **39.6%** | **61.4%** |
| Gradient boosting | 0.24 | 0.737 | 86.0% | 48.5% | 39.6% | 61.2% |
| Random forest | 0.23 | 0.732 | 85.7% | 46.8% | 38.8% | 62.4% |
| XGBoost | 0.19 | 0.699 | 85.9% | 44.2% | 37.7% | 64.3% |
| Support vector machine | 0.22 | 0.691 | 58.3% | 73.2% | 46.1% | 35.7% |
| Decision tree | 0.26 | 0.586 | 46.1% | 71.0% | 38.4% | 33.9% |

**Comparison with the ADA risk test.** The American Diabetes Association's risk test (Bang et al., *Annals
of Internal Medicine* 2009) adds points for age, sex, high blood pressure, physical inactivity and BMI, plus
family history and gestational diabetes. The extract has no family history or gestational diabetes, so we
scored the first five (a partial score; blank answers score 0). On the same 4,966 test people it reaches an
AUC of **0.722**, against **0.735** for AHEAD's model. AHEAD's model ranks people about as well as a
published points score, not better by much. That is expected: both use the same kind of everyday
information, and the honest message is that an HbA1c test answers the question far better than either.

### Kidney health (2021-2023 cycle, 5,551 people, 17.1% with a kidney marker)

| Model | Threshold | AUC | Sensitivity | Specificity | Precision | Share flagged |
|---|---|---|---|---|---|---|
| **Logistic regression** | **0.22** | **0.776** | **66.2%** | **77.5%** | **37.7%** | **30.0%** |
| Gradient boosting | 0.16 | 0.789 | 72.6% | 72.7% | 35.4% | 35.0% |
| Random forest | 0.17 | 0.788 | 71.9% | 73.3% | 35.7% | 34.4% |
| XGBoost | 0.16 | 0.743 | 64.8% | 72.0% | 32.2% | 34.3% |
| Support vector machine | 0.14 | 0.647 | 42.8% | 89.9% | 46.6% | 15.7% |
| Decision tree | 0.01 | 0.585 | 32.4% | 84.5% | 30.1% | 18.4% |

### Cardiovascular (random 20% of BRFSS 2022, 49,205 people, 5.5% ever had a heart attack)

| Model | Threshold | AUC | Sensitivity | Specificity | Precision | Share flagged |
|---|---|---|---|---|---|---|
| **Logistic regression** | **0.05** | **0.828** | **79.8%** | **71.0%** | **13.7%** | **31.8%** |
| Gradient boosting | 0.04 | 0.830 | 84.4% | 66.0% | 12.6% | 36.7% |
| Random forest | 0.06 | 0.818 | 74.7% | 74.1% | 14.3% | 28.5% |
| XGBoost | 0.03 | 0.803 | 81.6% | 64.2% | 11.6% | 38.3% |
| Support vector machine | 0.06 | 0.653 | 45.3% | 80.0% | 11.6% | 21.4% |
| Decision tree | 0.34 | 0.562 | 18.2% | 94.1% | 15.2% | 6.5% |

The support vector machine and the single decision tree are kept only for comparison. The random forest is
kept small (150 trees, depth 10) so its saved file stays under 10 MB.

### Calibration

A score should mean roughly what it says: among people who get about 20%, about 20 in 100 should have the
label. On the test set:

| Condition | Average score | Observed rate | Among flagged: average score | Among flagged: observed rate |
|---|---|---|---|---|
| Diabetes | 30.1% | 28.2% | 41.7% | 39.6% |
| Kidney health | 17.2% | 17.1% | 35.1% | 37.7% |
| Cardiovascular | 5.4% | 5.5% | 13.3% | 13.7% |

**Cardiovascular, before and after.** The earlier heart model was trained with `class_weight="balanced"`,
which pushes scores up to make the rare outcome as common as the other. People it flagged saw an average
score of 75.6% while 17.1% of them had had a heart attack; the average score over everyone was 35.0%
against 5.5% observed. Retrained without re-weighting, it ranks people the same (AUC 0.828 both
times) and the scores now match the observed rates (13.3% shown against 13.7% observed among flagged people
at the new threshold of 0.05). The new threshold rule catches more past heart attacks than before (79.8%
against 63.7%) and flags more people (31.8% against 20.3%).

---

## Limitations

- **Unweighted.** NHANES and BRFSS are designed to be analysed with survey weights, which make the
  sample represent the US population. AHEAD does not use them, so the rates on this page describe the
  people in the surveys, not the US population.
- **Kidney markers are from one visit.** The kidney label is a single eGFR and ACR result. Chronic kidney
  disease needs the result to last at least three months, so some people labelled 1 do not have CKD.
- **The heart label is a past heart attack.** The cardiovascular model describes who has had a heart attack,
  not who will have one. Its scores must not be read as a risk of a future heart attack.
- **Lab methods are not fully harmonised.** Serum creatinine is corrected as CDC advises, but HbA1c and
  urine albumin methods also changed over the years and are used as reported.
- **US data only.** Both surveys sample US residents. The models may do differently elsewhere.
- **Answers are self-reported.** Smoking, activity and "told by a doctor" histories are what people said.
  Someone with undiagnosed high blood pressure answers "no".
- **Survey questions and lab methods changed over 24 years.** For example, the share of people with HbA1c
  5.7% or higher is 15-18% in 1999-2006 and 27-33% from 2007 on. We did not adjust for this. The test on
  the newest cycle shows the result of these changes: diabetes AUC is 0.764 on validation (earlier cycles)
  and 0.735 on 2021-2023.
- **Activity is missing before 2007.** The kidney model's `physically_active` is blank for the 1999-2006
  cycles (26% of kidney rows); the pipeline fills those blanks with the training median, 0 (not active), so
  unknown activity in 1999-2006 is read as not active; a limitation.
- **Diabetes flags many people.** Prediabetes is common (28% of the 2021-2023 test people have HbA1c 5.7%
  or higher) and everyday answers separate it only moderately, so the diabetes model flags 61% of test
  people. A flag means "worth an HbA1c test", not "has the condition".
- **Logistic regression gives up a little accuracy for kidney.** Gradient boosting reached a test AUC of 0.789
  against 0.776, but its score sometimes moved the wrong way (a higher BMI giving a lower score for about a
  fifth of test people), which the app could not explain to a patient.

---

## Retired datasets

These files were removed from `data/` and are not used anywhere:

- **`kidney_disease.csv`** (1,659 rows). Its publisher describes it as synthetic. It also does not behave
  like real data: GFR and serum creatinine are unrelated (correlation -0.003, where real data shows a strong
  negative link), and 92% of rows have the disease.
- **`diabetes.csv`** (100,000 rows). Its source and licence are unknown (credited only to "Original
  Authors"), and its lab values look generated: HbA1c and blood glucose take only 18 distinct values each and
  are unrelated to each other within each outcome group.

---

## References

- CDC/NCHS. National Health and Nutrition Examination Survey Data, 1999-2023. https://www.cdc.gov/nchs/nhanes/
- CDC. Behavioral Risk Factor Surveillance System Survey Data, 2022. https://www.cdc.gov/brfss/
- Pytlak K. Indicators of Heart Disease (2022 UPDATE). Kaggle, CC0.
- Bang H, Edwards AM, Bomback AS, et al. Development and validation of a patient self-assessment score for
  diabetes risk. *Ann Intern Med.* 2009;151(11):775-783.
- Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C-based equations to estimate GFR
  without race. *N Engl J Med.* 2021;385:1737-1749.
