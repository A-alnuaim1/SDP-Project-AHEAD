"""Optional lab inputs: blank values must not crash factor rules, never reach the diabetes or kidney model (they are
not model inputs since the switch to NHANES), and the note above a result says so instead of claiming the model
filled them in. The old per-blank thresholds (scripts/missing_inputs.py profiles) were removed with them."""
import math
import unittest

from ahead.config import OPTIONAL_INPUTS
from ahead.resources import _missing_inputs, missing_inputs_profile
from ahead.screening import _model_frame, identify_risk_factors, missing_note


class OptionalLabTests(unittest.TestCase):
    def test_risk_factors_skip_blank_labs(self):
        diabetes = {'bmi': 31, 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never',
                    'HbA1c_level': None, 'blood_glucose_level': None}
        self.assertEqual(identify_risk_factors('diabetes', diabetes), ['Higher BMI'])
        kidney = {'age': 45, 'gender': 'Male', 'bmi': 24, 'smoking_history': 'never', 'physically_active': 1,
                  'hypertension': 0, 'diabetes': 0, 'heart_disease': 0, 'family_kidney': None,
                  'systolic_bp': None, 'diastolic_bp': 85, 'creatinine': None, 'egfr': None, 'uacr': 40}
        self.assertEqual(identify_risk_factors('kidney', kidney),
                         ['Higher entered blood pressure', 'Higher entered urine ACR'])
        # An older saved kidney screening's keys still give its factors.
        legacy = {'Smoking': 1, 'BMI': 31, 'FamilyHistoryKidneyDisease': 1, 'FamilyHistoryHypertension': 0,
                  'FamilyHistoryDiabetes': 0, 'Edema': 0, 'SystolicBP': None, 'DiastolicBP': 85, 'GFR': None, 'ACR': 40}
        self.assertEqual(identify_risk_factors('kidney', legacy),
                         ['Smoking history', 'Higher BMI', 'Family history of kidney disease',
                          'Higher entered blood pressure', 'Higher entered urine ACR'])

    def test_blank_values_become_nan_for_the_model(self):
        frame = _model_frame({'bmi': None, 'age': 52})
        self.assertTrue(math.isnan(frame.loc[0, 'bmi']))
        self.assertEqual(frame.loc[0, 'age'], 52)

    def test_no_condition_has_blank_input_profiles_any_more(self):
        # Optional values are labs, which no model reads, so there is nothing to measure with them blank.
        for disease in OPTIONAL_INPUTS:
            self.assertEqual(_missing_inputs()[disease]['models'], {}, disease)
            self.assertIsNone(missing_inputs_profile(disease, 'LogisticRegression', OPTIONAL_INPUTS[disease]))

    def test_note_says_labs_are_not_model_inputs(self):
        diabetes = {'values': {'age': 50, 'HbA1c_level': None, 'blood_glucose_level': None}}
        note = missing_note('diabetes', diabetes)
        self.assertTrue(note.startswith('You left HbA1c and blood glucose blank.'), note)
        self.assertIn("The model doesn't use test results", note)
        self.assertIn('An HbA1c test', note)
        self.assertNotIn('typical', note)
        doctor = missing_note('diabetes', diabetes, doctor=True)
        self.assertTrue(doctor.startswith('HbA1c and blood glucose were left blank.'), doctor)
        self.assertNotRegex(doctor, r'\b[Yy]ou(r)?\b')
        kidney = {'values': {'age': 50, 'gender': 'Female', 'creatinine': None, 'egfr': None, 'uacr': None,
                             'systolic_bp': None, 'diastolic_bp': None}}
        note = missing_note('kidney', kidney)
        self.assertTrue(note.startswith('You left creatinine and eGFR, and urine ACR blank.'), note)
        self.assertIn('You left blood pressure blank, so AHEAD could not check it.', note)
        # Labs entered: only the blood pressure line; everything entered, or a child: no note.
        entered = {**kidney['values'], 'egfr': 80, 'uacr': 10}
        self.assertEqual(missing_note('kidney', {'values': entered}),
                         'You left blood pressure blank, so AHEAD could not check it. A home or pharmacy reading is a '
                         'simple start.')
        self.assertEqual(missing_note('kidney', {'values': {**entered, 'systolic_bp': 120, 'diastolic_bp': 80}}), '')
        self.assertEqual(missing_note('diabetes', {'values': {'age': 12, 'HbA1c_level': None}}), '')


if __name__ == '__main__':
    unittest.main()
