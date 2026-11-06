"""The test to ask for when a screening left a lab blank (ahead/triage.missing_tests): the helper's merging and wording,
and the three doctor surfaces that show it in the same words (the record view in Patient Records & History, the doctor
home's at-risk rows and the Prevention Plan strip). A screening under 18 asks for nothing, and a test a newer screening of
the same condition entered is no longer asked for. Also checks the Data Quality tab, whose one useful idea this was, is
gone. Every page check runs on a temporary database; users are inserted directly, since a password hash takes most of
a second."""
import html
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from streamlit.testing.v1 import AppTest

import ahead.pages.assistant as assistant
import ahead.pages.doctor_intelligence as page
from ahead.platform_data import grant_doctor, init_platform, local_today, save_plan
from ahead.storage import connection, init_db, save_screening
from ahead.triage import ADULTS_ONLY, LAB_TESTS, RULES, entered_later, missing_tests

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')
BLANKS = (None, '', '   ', float('nan'), 'nan')
HBA1C = {'features': ['HbA1c_level'], 'label': 'HbA1c', 'action': 'ask for an HbA1c test to confirm',
         'line': 'HbA1c not entered · ask for an HbA1c test to confirm',
         'doctor': 'HbA1c not entered. Ask for an HbA1c test to confirm.'}
KIDNEY = {'age': 50, 'gender': 'Female', 'egfr': 80, 'uacr': 10, 'creatinine': 1.0}   # the kidney tests, entered
EGFR_LINE = 'Creatinine and eGFR not entered · ask for a creatinine (eGFR) blood test'
ACR_LINE = 'Urine ACR not entered · ask for a urine ACR test'
BP_LINE = "Blood pressure not entered · check the patient's blood pressure"


def lines(disease, values, later=()):
    return [test['line'] for test in missing_tests(disease, values, later)]


class HelperTests(unittest.TestCase):
    def test_diabetes_with_both_labs_blank_asks_for_one_hba1c_test(self):
        for blank in BLANKS:
            self.assertEqual(missing_tests('diabetes', {'age': 50, 'HbA1c_level': blank, 'blood_glucose_level': blank}), [{
                'features': ['HbA1c_level', 'blood_glucose_level'], 'label': 'HbA1c and blood glucose',
                'action': 'ask for an HbA1c test to confirm',
                'line': 'HbA1c and blood glucose not entered · ask for an HbA1c test to confirm',
                'doctor': 'HbA1c and blood glucose not entered. Ask for an HbA1c test to confirm.'}])

    def test_diabetes_hba1c_blank_alone(self):
        for blank in BLANKS:
            self.assertEqual(missing_tests('diabetes', {'HbA1c_level': blank, 'blood_glucose_level': 120}), [HBA1C])

    def test_diabetes_with_hba1c_entered_asks_for_nothing_more(self):
        # The measured HbA1c sets the track, so a blank glucose is no longer a test to ask for (the separate
        # blood glucose ask was removed when the lab values stopped being model inputs).
        self.assertEqual(missing_tests('diabetes', {'HbA1c_level': 6.1, 'blood_glucose_level': None}), [])

    def test_entered_labs_ask_for_nothing(self):
        self.assertEqual(missing_tests('diabetes', {'HbA1c_level': 6.1, 'blood_glucose_level': 140}), [])
        self.assertEqual(missing_tests('diabetes', {'HbA1c_level': '6.1', 'blood_glucose_level': ' 140 '}), [])
        self.assertEqual(missing_tests('kidney', {**KIDNEY, 'systolic_bp': 130, 'diastolic_bp': 85}), [])

    def test_a_value_counts_as_blank_only_when_the_saved_inputs_include_it(self):
        # An older record saved without its inputs, or with only some of them, asks for nothing it did not ask about.
        for values in (None, {}, '', '{}', 'not json', '[1, 2]', {'age': 30}):
            self.assertEqual(missing_tests('diabetes', values), [])
            self.assertEqual(missing_tests('kidney', values), [])
        self.assertEqual(missing_tests('diabetes', {'age': 30, 'HbA1c_level': None}), [HBA1C])

    def test_saved_json_text_reads_like_the_dict(self):
        saved = json.dumps({'age': 50, 'HbA1c_level': float('nan'), 'blood_glucose_level': 120})
        self.assertEqual(missing_tests('diabetes', saved), [HBA1C])

    def test_kidney_blood_pressure_is_one_line(self):
        self.assertEqual(lines('kidney', {**KIDNEY, 'systolic_bp': None, 'diastolic_bp': ''}), [BP_LINE])
        self.assertEqual(missing_tests('kidney', {**KIDNEY, 'systolic_bp': None, 'diastolic_bp': None})[0]['doctor'],
                         "Blood pressure not entered. Check the patient's blood pressure.")
        # Half a reading names the number that is missing.
        self.assertEqual(lines('kidney', {**KIDNEY, 'systolic_bp': 135, 'diastolic_bp': None}),
                         ["Diastolic blood pressure not entered · check the patient's blood pressure"])
        self.assertEqual(lines('kidney', {**KIDNEY, 'systolic_bp': float('nan'), 'diastolic_bp': 85}),
                         ["Systolic blood pressure not entered · check the patient's blood pressure"])

    def test_kidney_asks_for_the_blood_test_and_the_urine_test(self):
        blank = {'age': 50, 'gender': 'Male', 'creatinine': None, 'egfr': None, 'uacr': None, 'systolic_bp': None,
                 'diastolic_bp': None}
        self.assertEqual(lines('kidney', blank), [EGFR_LINE, ACR_LINE, BP_LINE])
        self.assertEqual(missing_tests('kidney', blank)[0]['doctor'],
                         'Creatinine and eGFR not entered. Ask for a creatinine (eGFR) blood test.')
        self.assertEqual(missing_tests('kidney', blank)[1]['doctor'], 'Urine ACR not entered. Ask for a urine ACR test.')
        # Creatinine alone gives eGFR (CKD-EPI 2021), so only the urine test is left to ask for.
        self.assertEqual(lines('kidney', {**blank, 'creatinine': 1.1, 'systolic_bp': 120, 'diastolic_bp': 80}),
                         [ACR_LINE])
        # A lab eGFR with no creatinine is enough too; an ACR alone still asks for the blood test.
        self.assertEqual(lines('kidney', {**blank, 'egfr': 75, 'uacr': 12, 'systolic_bp': 120, 'diastolic_bp': 80}), [])
        self.assertEqual(lines('kidney', {**blank, 'uacr': 12, 'systolic_bp': 120, 'diastolic_bp': 80}), [EGFR_LINE])
        # Creatinine without an age or sex cannot give eGFR, so the blood test is still asked for, named by eGFR.
        self.assertEqual(lines('kidney', {**blank, 'age': None, 'creatinine': 1.1, 'uacr': 12, 'systolic_bp': 120,
                                          'diastolic_bp': 80}),
                         ['eGFR not entered · ask for a creatinine (eGFR) blood test'])

    def test_an_older_kidney_screening_asks_with_its_own_keys(self):
        # Saved before the switch to NHANES: GFR/ACR/SerumCreatinine were needed for a score, so they are entered; a
        # blank blood pressure still asks. Its blank HbA1c and fasting sugar no longer do (those rules were removed).
        legacy = {'Age': 50, 'GFR': 80, 'ACR': 10, 'SerumCreatinine': 1.0, 'HbA1c': None, 'FastingBloodSugar': None,
                  'SystolicBP': None, 'DiastolicBP': None}
        self.assertEqual(lines('kidney', legacy), [BP_LINE])
        self.assertEqual(lines('kidney', {**legacy, 'SystolicBP': 130, 'DiastolicBP': 80}), [])

    def test_heart_asks_for_nothing(self):
        self.assertEqual(missing_tests('heart', {'BMI': None, 'AgeCategory': '', 'PhysicalHealthDays': None}), [])
        with self.assertRaises(ValueError):
            missing_tests('liver', {})

    def test_the_table_covers_every_urgent_rule_value_a_screening_can_leave_blank(self):
        for disease, rules in RULES.items():
            values = {rule.feature for rule in rules if rule.kind != ADULTS_ONLY}
            tests = LAB_TESTS[disease]
            self.assertEqual({f for test in tests for f in test.asks}, values - {'blood_glucose_level'}, disease)
            self.assertLessEqual(values, {f for test in tests for f in test.covers}, disease)
            for test in tests:
                self.assertTrue(set(test.asks) <= set(test.covers), test)
            # Every asking value blank on its own gets exactly one test.
            for feature in {f for test in tests for f in test.asks}:
                self.assertEqual([t['features'] for t in missing_tests(disease, {feature: None})], [[feature]])

    def test_a_screening_under_18_asks_for_nothing(self):
        # The tests come from adult cut-offs; a flagged child is urgent, and its reason already asks for tests.
        for age in (6, 17.9, '12'):
            self.assertEqual(missing_tests('diabetes', {'age': age, 'HbA1c_level': None, 'blood_glucose_level': None}), [])
        self.assertEqual(missing_tests('kidney', {'age': 15, 'egfr': None, 'uacr': None, 'creatinine': None,
                                                  'systolic_bp': None, 'diastolic_bp': None}), [])
        self.assertEqual(missing_tests('kidney', {'Age': 15, 'GFR': None, 'SystolicBP': None}), [])   # older keys
        # 18, or an age left blank or not saved, still asks.
        for age in (18, 40, None, ''):
            self.assertEqual(missing_tests('diabetes', {'age': age, 'HbA1c_level': None, 'blood_glucose_level': 120}), [HBA1C])
        self.assertEqual(missing_tests('diabetes', {'HbA1c_level': None}), [HBA1C])
        self.assertEqual(lines('kidney', {**KIDNEY, 'age': 18, 'uacr': None}), [ACR_LINE])

    def test_a_value_a_newer_screening_entered_is_not_asked_for(self):
        both = {'age': 50, 'HbA1c_level': None, 'blood_glucose_level': None}
        self.assertEqual(missing_tests('diabetes', both, {'HbA1c_level', 'blood_glucose_level'}), [])
        # HbA1c done since: nothing left to ask for, since a blank glucose alone asks for nothing.
        self.assertEqual(lines('diabetes', both, {'HbA1c_level'}), [])
        self.assertEqual(lines('diabetes', both, ['blood_glucose_level']), [HBA1C['line']])
        self.assertEqual(lines('diabetes', both, ()), ['HbA1c and blood glucose not entered · ask for an HbA1c test to confirm'])
        # One blood pressure number taken since names the other one.
        self.assertEqual(lines('kidney', {**KIDNEY, 'systolic_bp': None, 'diastolic_bp': None}, {'systolic_bp'}),
                         ["Diastolic blood pressure not entered · check the patient's blood pressure"])

    def test_entered_later_collects_newer_values_per_patient_and_condition(self):
        rows = [  # newest first, as the pages read them
            {'id': 6, 'owner_id': 1, 'disease': 'kidney', 'values_json': {'age': 50, 'gender': 'Male', 'creatinine': 1.0}},
            {'id': 5, 'owner_id': 1, 'disease': 'diabetes', 'values_json': json.dumps({'HbA1c_level': 5.6, 'blood_glucose_level': None})},
            {'id': 4, 'owner_id': 2, 'disease': 'diabetes', 'values_json': {'blood_glucose_level': 110}},
            {'id': 3, 'owner_id': 1, 'disease': 'kidney', 'values_json': {'GFR': 55}},
            {'id': 2, 'owner_id': 1, 'disease': 'diabetes', 'values_json': {'blood_glucose_level': 140, 'age': float('nan')}},
            {'id': 1, 'owner_id': 1, 'disease': 'diabetes', 'values_json': 'not json'},
        ]
        later = entered_later(rows)
        self.assertEqual({k: v for k, v in later.items() if k != 3}, {
            6: set(), 5: set(), 4: set(), 2: {'HbA1c_level'}, 1: {'HbA1c_level', 'blood_glucose_level'}})
        # An eGFR worked out from a newer screening's creatinine counts as entered, under the current name.
        self.assertLessEqual({'egfr', 'creatinine', 'age'}, later[3])
        self.assertEqual(lines('kidney', {'GFR': None, 'ACR': 12}, later[3]), [])
        # Another patient's or another condition's screening clears nothing.
        self.assertEqual(missing_tests('diabetes', {'HbA1c_level': None, 'blood_glucose_level': None}, later[4]),
                         [{'features': ['HbA1c_level', 'blood_glucose_level'], 'label': 'HbA1c and blood glucose',
                           'action': 'ask for an HbA1c test to confirm',
                           'line': 'HbA1c and blood glucose not entered · ask for an HbA1c test to confirm',
                           'doctor': 'HbA1c and blood glucose not entered. Ask for an HbA1c test to confirm.'}])
        self.assertEqual(entered_later([]), {})

    def test_wording_is_plain_and_the_line_and_sentence_agree(self):
        cases = [('diabetes', {'HbA1c_level': None, 'blood_glucose_level': None}), ('diabetes', {'HbA1c_level': None}),
                 ('kidney', {'creatinine': None, 'egfr': None, 'uacr': None, 'systolic_bp': None, 'diastolic_bp': None}),
                 ('kidney', {'uacr': None, 'diastolic_bp': None})]
        for disease, values in cases:
            for test in missing_tests(disease, values):
                self.assertEqual(test['line'], f"{test['label']} not entered · {test['action']}")
                self.assertEqual(test['doctor'], f"{test['label']} not entered. {test['action'][:1].upper()}"
                                                 f"{test['action'][1:]}.")
                for text in test.values():
                    if isinstance(text, str):
                        # No longer says the model used a typical value: labs are not model inputs.
                        self.assertNotRegex(text, r'_|\byou\b|\byour\b|[Dd]iagnos|typical')


class RemovedTabTests(unittest.TestCase):
    def test_the_data_quality_tab_and_its_mentions_are_gone(self):
        self.assertEqual(page.TABS, ['Overview', 'Disease Analytics', 'Patient Records & History', 'Medication Monitoring',
                                     'Prevention Plan', 'Clinical Insights'])
        for name in ('_quality', '_upload_issue', 'UPLOAD_ISSUES', '_not_entered'):
            self.assertFalse(hasattr(page, name), name)
        # The doctor's canned replies too: the Patient Intelligence one listed the tab.
        for text in (assistant.DOCTOR_SYSTEM_PROMPT, *assistant.DOCTOR_LOCAL_ANSWERS.values(), *assistant.LOCAL_ANSWERS.values()):
            self.assertNotRegex(text, r'(?i)data.quality')
        answer = assistant._local_answer('show me patient intelligence', is_doctor=True)
        self.assertIsNotNone(answer)
        self.assertNotRegex(answer, r'(?i)data.quality')


class Fixture(unittest.TestCase):
    """A doctor and one linked patient on a fresh temporary database; each test saves its own screenings."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH'] = self.temp.name + '/app.db'
        os.environ.pop('AHEAD_ENABLE_DEMO_ACCOUNTS', None)
        init_db(); init_platform()
        self.d = self.user('d@test.com', 'Doctor', 'Dr. Sarah Ahmed')
        self.p = self.user('p@test.com', 'Patient', 'Alice Mansour', link=True)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop('AHEAD_DB_PATH', None)

    def user(self, email, role, name, link=False):
        with connection() as db:
            uid = db.execute('INSERT INTO users(email,password_hash,role,name) VALUES(?,?,?,?)', (email, '-', role, name)).lastrowid
        if link:grant_doctor(uid, 'd@test.com')
        return {'id': uid, 'email': email, 'role': role, 'name': name, 'institution': '', 'preferences': {}}

    def screening(self, patient, disease, score, values, days_ago=0):
        sid = save_screening(patient['id'], disease, 'GradientBoosting', score, int(score >= .5), .5, values)
        if days_ago:
            with connection() as db:
                db.execute('UPDATE screenings SET created_at=? WHERE id=?',
                           ((datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(), sid))
        return sid

    def page(self, nav, **state):
        at = AppTest.from_file(APP, default_timeout=120)
        for key, value in {'authenticated': True, 'user': self.d, 'nav': nav, **state}.items():at.session_state[key] = value
        return self.ok(at.run())

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def tab(self, at, label):
        return next(t for t in at.tabs if t.label == label)


class RecordViewTests(Fixture):
    def test_the_record_view_names_the_test_and_the_strip_says_the_same(self):
        at_risk = {'age': 50, 'bmi': 31.2, 'smoking_history': 'former', 'HbA1c_level': None, 'blood_glucose_level': 120}
        # Both labs blank, but the newer diabetes screenings below entered them since: nothing left to ask for.
        done = self.screening(self.p, 'diabetes', .6, {**at_risk, 'blood_glucose_level': None}, days_ago=5)
        older = self.screening(self.p, 'diabetes', .4, {}, days_ago=4)   # saved before inputs were kept
        entered = self.screening(self.p, 'diabetes', .6, {**at_risk, 'HbA1c_level': 6.1}, days_ago=3)
        kidney = self.screening(self.p, 'kidney', .6, {**KIDNEY, 'egfr': None, 'creatinine': None, 'uacr': None,
                                                      'systolic_bp': None, 'diastolic_bp': None}, days_ago=2)
        newest = self.screening(self.p, 'diabetes', .55, at_risk)
        at = self.page('intelligence', intelligence_focus={'tab': 'Patient Records & History', 'patient': self.p['id']})
        records = lambda:self.tab(at, 'Patient Records & History')
        picker = lambda:next(s for s in records().selectbox if s.label == 'Open historical assessment')
        notes = lambda:[i.value for i in records().info]
        # The newest assessment opens first: one next-step line, and the inputs table still marks the blank value.
        self.assertEqual(picker().value, newest)
        self.assertEqual(notes(), [HBA1C['doctor']])
        # No icon of its own: the theme draws every info box's icon, so a second one would sit beside it.
        self.assertEqual([i.proto.icon for i in records().info], [''])
        inputs = next(df.value for df in records().dataframe if 'Input' in df.value.columns)
        self.assertEqual(dict(zip(inputs['Input'], inputs['Value']))['HbA1c Level (%)'], page.BLANK)
        # The Prevention Plan strip for the same result: the facts line, then the same test in the home row's words.
        strip = html.unescape('\n'.join(h.proto.body for h in self.tab(at, 'Prevention Plan').get('html')))
        self.assertIn(f'<p class="track-panel-note">{HBA1C["line"]}</p>', strip)
        self.assertEqual(strip.count('class="track-panel-note"'), 2)
        # Kidney: the blood test, the urine test, then blood pressure, one paragraph each.
        picker().set_value(kidney); self.ok(at.run())
        self.assertEqual(notes(), ['Creatinine and eGFR not entered. Ask for a creatinine (eGFR) blood test.\n\n'
                                   'Urine ACR not entered. Ask for a urine ACR test.\n\n'
                                   "Blood pressure not entered. Check the patient's blood pressure."])
        # Every lab entered, or no saved inputs at all: no next-step line.
        picker().set_value(entered); self.ok(at.run())
        self.assertEqual(notes(), [])
        picker().set_value(older); self.ok(at.run())
        self.assertEqual(notes(), [])
        self.assertIn('This earlier assessment predates input-history storage; its values are unavailable.',
                      [c.value for c in records().caption])
        # Labs left blank here but entered by a newer screening of the same condition: no next-step line, though the
        # table still shows what this assessment saved.
        picker().set_value(done); self.ok(at.run())
        self.assertEqual(notes(), [])
        inputs = next(df.value for df in records().dataframe if 'Input' in df.value.columns)
        self.assertEqual(dict(zip(inputs['Input'], inputs['Value']))['HbA1c Level (%)'], page.BLANK)

    def test_a_child_gets_no_test_line_in_the_record_view_or_the_strip(self):
        # Not flagged, both labs blank, age 6: the strip says to agree any plan outside AHEAD and asks for no adult test.
        self.screening(self.p, 'diabetes', .2, {'age': 6, 'bmi': 16.0, 'smoking_history': 'never',
                                                'HbA1c_level': None, 'blood_glucose_level': None})
        at = self.page('intelligence', intelligence_focus={'tab': 'Patient Records & History', 'patient': self.p['id']})
        self.assertEqual([i.value for i in self.tab(at, 'Patient Records & History').info], [])
        strip = html.unescape('\n'.join(h.proto.body for h in self.tab(at, 'Prevention Plan').get('html')))
        self.assertIn('age 6 (under 18)', strip)
        self.assertNotIn('not entered', strip)

    def test_the_strip_drops_the_test_line_for_a_tracker_age_under_18(self):
        # A screening saved without an age: the strip reads the tracker log's age, so no adult test is asked for there.
        latest = {'disease': 'diabetes', 'score': .2, 'threshold': .5, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'values_json': json.dumps({'HbA1c_level': None, 'blood_glucose_level': 120})}
        values = json.loads(latest['values_json'])
        self.assertEqual(missing_tests('diabetes', values), [HBA1C])
        strip = html.unescape(page._strip_html('diabetes', latest, values, [{'values': {'age': 15}}]))
        self.assertIn('age 15 (under 18)', strip)
        self.assertNotIn('not entered', strip)
        self.assertIn(HBA1C['line'], html.unescape(page._strip_html('diabetes', latest, values, [{'values': {'age': 40}}])))


class DoctorHomeTests(Fixture):
    def rows(self, at):
        """{name: detail lines} of each attention row."""
        page_html = '\n'.join(h.proto.body for h in at.get('html'))
        found = re.findall(r'<strong class="attention-name">([^<]+)</strong>(.*?)</div>\s*<div class="attention-pills">',
                           page_html, re.S)
        return {html.unescape(name): [html.unescape(d) for d in re.findall(r'attention-detail">([^<]*)<', lines)]
                for name, lines in found}

    def test_an_at_risk_row_names_the_test_and_urgent_and_dose_rows_do_not(self):
        ben = self.user('b@test.com', 'Patient', 'Ben Cole', link=True)
        cara = self.user('c@test.com', 'Patient', 'Cara Diaz', link=True)
        dan = self.user('x@test.com', 'Patient', 'Dan Kidney', link=True)
        self.screening(self.p, 'diabetes', .55, {'age': 50, 'HbA1c_level': None, 'blood_glucose_level': 120})
        # Urgent by its glucose with HbA1c blank: the row gives the urgent reason and goes to a visit first.
        self.screening(ben, 'diabetes', .8, {'age': 50, 'HbA1c_level': None, 'blood_glucose_level': 250})
        self.screening(dan, 'kidney', .7, {**KIDNEY, 'uacr': None, 'systolic_bp': None, 'diastolic_bp': None})
        # Missed doses only: a 00:00 dose is already due, so unlogged it is missed today whatever the time of the run.
        save_plan(self.d['id'], cara['id'], {'name': 'Metformin', 'dosage': '500 mg', 'instructions': '', 'times': ['00:00'],
                  'days': list(range(7)), 'start_date': (local_today() - timedelta(days=1)).isoformat(),
                  'end_date': (local_today() + timedelta(days=2)).isoformat(), 'notes': ''})
        at = self.page('overview')
        rows = self.rows(at)
        self.assertEqual(rows['Alice Mansour'][1:], [HBA1C['line']])
        self.assertRegex(rows['Alice Mansour'][0], r'^Diabetes · 55% · today \d\d:\d\d$')
        self.assertEqual(rows['Dan Kidney'][1:], [ACR_LINE, BP_LINE])
        self.assertEqual(len(rows['Ben Cole']), 2)
        self.assertIn('blood glucose of 250 mg/dL', rows['Ben Cole'][1])
        self.assertNotIn('not entered', ' '.join(rows['Ben Cole']))
        self.assertEqual(rows['Cara Diaz'], ['Metformin · 1 dose missed today'])
        # Escaped in the page, as every row line is.
        self.assertIn('check the patient&#x27;s blood pressure', '\n'.join(h.proto.body for h in at.get('html')))

    def test_a_test_entered_by_a_newer_screening_is_no_longer_asked_for(self):
        ben = self.user('b@test.com', 'Patient', 'Ben Cole', link=True)
        cara = self.user('c@test.com', 'Patient', 'Cara Diaz', link=True)
        blank = {'age': 50, 'bmi': 31.2, 'HbA1c_level': None, 'blood_glucose_level': None}
        # Alice had both tests since; her newer result is not flagged, so the older at-risk result still leads the row.
        self.screening(self.p, 'diabetes', .55, blank, days_ago=2)
        self.screening(self.p, 'diabetes', .30, {**blank, 'HbA1c_level': 5.6, 'blood_glucose_level': 110})
        # Ben had the HbA1c test only: a blank glucose alone asks for nothing, so his row names no test.
        self.screening(ben, 'diabetes', .55, blank, days_ago=2)
        self.screening(ben, 'diabetes', .30, {**blank, 'HbA1c_level': 5.6})
        # Cara's newer screening is for another condition, so it clears nothing.
        self.screening(cara, 'diabetes', .55, {**blank, 'blood_glucose_level': 120}, days_ago=2)
        self.screening(cara, 'kidney', .30, {**KIDNEY, 'systolic_bp': 120, 'diastolic_bp': 80})
        at = self.page('overview')
        rows = self.rows(at)
        self.assertRegex(rows['Alice Mansour'][0], r'^Diabetes · 55% · ')
        self.assertEqual(rows['Alice Mansour'][1:], [])
        self.assertEqual(rows['Ben Cole'][1:], [])
        self.assertEqual(rows['Cara Diaz'][1:], [HBA1C['line']])
        # The record view the row opens and the Prevention Plan strip agree: nothing left to ask Alice for.
        at = self.page('intelligence', intelligence_focus={'tab': 'Patient Records & History', 'patient': self.p['id']})
        self.assertEqual([i.value for i in self.tab(at, 'Patient Records & History').info], [])
        self.assertNotIn('not entered', '\n'.join(h.proto.body for h in self.tab(at, 'Prevention Plan').get('html')))


if __name__ == '__main__':
    unittest.main()
