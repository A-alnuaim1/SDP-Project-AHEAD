"""Optional lab inputs: blank values must not crash factor rules and must reach the model as missing."""
import math
import unittest

from ahead.config import OPTIONAL_INPUTS, REQUIRED_FOR_SCORE
from ahead.resources import _missing_inputs, best_model, decision_threshold, missing_inputs_profile
from ahead.screening import _model_frame, identify_risk_factors, missing_note


class OptionalLabTests(unittest.TestCase):
    def test_risk_factors_skip_blank_labs(self):
        diabetes = {'bmi': 31, 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never',
                    'HbA1c_level': None, 'blood_glucose_level': None}
        self.assertEqual(identify_risk_factors('diabetes', diabetes), ['Higher BMI'])
        kidney = {'Smoking': 0, 'BMI': 24, 'FamilyHistoryKidneyDisease': 0, 'FamilyHistoryHypertension': 0,
                  'FamilyHistoryDiabetes': 0, 'Edema': 0, 'SystolicBP': None, 'DiastolicBP': 85,
                  'GFR': None, 'ACR': 40}
        self.assertEqual(identify_risk_factors('kidney', kidney),
                         ['Higher entered blood pressure', 'Higher entered urine ACR'])

    def test_blank_values_become_nan_for_the_model(self):
        frame = _model_frame({'HbA1c_level': None, 'bmi': 27.5})
        self.assertTrue(math.isnan(frame.loc[0, 'HbA1c_level']))
        self.assertEqual(frame.loc[0, 'bmi'], 27.5)

    def test_profile_lookup_ignores_order(self):
        a = missing_inputs_profile('diabetes', 'GradientBoosting', ['HbA1c_level', 'blood_glucose_level'])
        b = missing_inputs_profile('diabetes', 'GradientBoosting', ['blood_glucose_level', 'HbA1c_level'])
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        self.assertLess(a['threshold'], 0.5)
        self.assertIsNotNone(missing_inputs_profile('kidney', 'XGBoost', ['DiastolicBP', 'SystolicBP']))
        self.assertIsNone(missing_inputs_profile('kidney', 'XGBoost', ['GFR']))

    def test_every_allowed_pattern_is_measured_and_never_raises_the_threshold(self):
        for disease, data in _missing_inputs().items():
            if disease == 'method':
                continue
            optional = [f for f in OPTIONAL_INPUTS[disease] if f not in REQUIRED_FOR_SCORE.get(disease, [])]
            for model, entries in data['models'].items():
                self.assertEqual(len(entries), 2 ** len(optional), (disease, model))
                usual = decision_threshold(disease, model)
                for key, entry in entries.items():
                    self.assertLessEqual(entry['threshold'], usual, (disease, model, key))

    def test_note_only_claims_compensation_when_the_threshold_drops(self):
        kidney_model = best_model('kidney')
        result = {'model': kidney_model, 'missing': ['SystolicBP', 'DiastolicBP'],
                  'threshold': decision_threshold('kidney', kidney_model),
                  'profile': missing_inputs_profile('kidney', kidney_model, ['SystolicBP', 'DiastolicBP'])}
        note = missing_note('kidney', result)
        self.assertNotIn('make up', note)
        self.assertIn('with every value entered', note)
        diabetes = {'model': 'GradientBoosting', 'missing': ['HbA1c_level'],
                    'profile': missing_inputs_profile('diabetes', 'GradientBoosting', ['HbA1c_level'])}
        diabetes['threshold'] = diabetes['profile']['threshold']
        self.assertIn('make up', missing_note('diabetes', diabetes))


if __name__ == '__main__':
    unittest.main()
