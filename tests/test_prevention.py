"""Prevention plans: the target catalog and its offer rules, baselines, progress and status rules, and the stored plan
lifecycle (drafts, approval, goal sync, notifications, edits, pause/stop, access and revocation)."""
import json
import os
import tempfile
import unittest
from datetime import date, timedelta

from ahead import prevention as pv
from ahead.platform_data import (
    active_plan, approve_prevention_plan, get_goals, grant_doctor, init_platform, local_today, notifications,
    notify_recheck_due, prevention_history, prevention_plan_status, prevention_plans, prevention_progress, revoke_doctor,
    save_lifestyle, save_prevention_draft, set_challenge_day, set_goals, update_prevention_plan,
)
from ahead.storage import connection, init_db, save_screening

LOG = {'weight_kg': 84, 'height_cm': 164, 'exercise_days': 2, 'exercise_minutes': 30, 'fruit_veg': 2,
       'sleep_hours': 6, 'steps': 5000, 'sleep_quality': 3, 'water_litres': 1.5, 'fast_food': 0, 'sugary_food': 1,
       'stress': 5, 'sedentary_hours': 7}   # BMI 31.2


def log(values, stamp):
    return {'values': values, 'created_at': stamp}


def keyed(rows):
    return {row['key']: row for row in rows}


class CatalogTests(unittest.TestCase):
    def test_weight_is_offered_only_at_a_bmi_of_25_or_more(self):
        # At 1.64 m: 66.9 kg is BMI 24.9 and 67.3 kg is 25.0 (the BMI is compared as shown, to 0.1).
        for weight, offered in ((66.9, False), (67.3, True), (84, True)):
            row = keyed(pv.template('diabetes', {}, [{**LOG, 'weight_kg': weight}]))['weight']
            self.assertEqual(row['offered'], offered, weight)
            self.assertEqual(row['enabled'], offered, weight)
        self.assertEqual(keyed(pv.template('diabetes', {}, [{**LOG, 'weight_kg': 66.9}]))['weight']['reason'],
                         'Not offered: BMI 24.9 is under 25')
        # The latest tracker log comes before the screening's BMI.
        self.assertFalse(keyed(pv.template('heart', {'BMI': 31}, [{**LOG, 'weight_kg': 60}]))['weight']['offered'])
        # No tracker log: the screening's BMI decides, but with no weight there is no default goal to tick.
        row = keyed(pv.template('kidney', {'BMI': 27.5}, []))['weight']
        self.assertEqual((row['offered'], row['enabled'], row['goal']), (True, False, None))
        self.assertEqual(row['reason'], 'No weight logged yet: enter a goal')
        self.assertEqual((row['bmi'], row['bmi_source']), (27.5, 'screening'))
        row = keyed(pv.template('diabetes', {'bmi': ''}, []))['weight']
        self.assertEqual((row['offered'], row['reason']), (False, 'Not offered: no BMI recorded yet'))

    def test_stop_smoking_is_offered_only_to_current_smokers(self):
        cases = [('diabetes', {'smoking_history': 'current'}, True), ('diabetes', {'smoking_history': 'former'}, False),
                 ('diabetes', {'smoking_history': 'not current'}, False), ('diabetes', {'smoking_history': 'ever'}, False),
                 ('diabetes', {'smoking_history': 'No Info'}, False), ('diabetes', {}, False),
                 ('heart', {'SmokerStatus': 'Current smoker - now smokes some days'}, True),
                 ('heart', {'SmokerStatus': 'Current smoker - now smokes every day'}, True),
                 ('heart', {'SmokerStatus': 'Former smoker'}, False), ('heart', {'SmokerStatus': 'Never smoked'}, False),
                 ('kidney', {'Smoking': 1}, True), ('kidney', {'Smoking': '1'}, True), ('kidney', {'Smoking': 0}, False),
                 ('kidney', {'Smoking': ''}, False)]
        for disease, values, offered in cases:
            row = keyed(pv.template(disease, values, [LOG]))['smoking']
            self.assertEqual((row['offered'], row['enabled']), (offered, offered), (disease, values))
        self.assertEqual(keyed(pv.template('diabetes', {'smoking_history': 'former'}, []))['smoking']['reason'],
                         'Not needed: former smoker')
        self.assertEqual(keyed(pv.template('diabetes', {'smoking_history': 'No Info'}, []))['smoking']['reason'],
                         'Not offered: smoking not recorded')
        self.assertEqual(pv.facts('diabetes', {'smoking_history': 'former', 'bmi': 31.24}, []), ['BMI 31.2', 'former smoker'])

    def test_no_target_is_offered_under_18(self):
        self.assertEqual(pv.ADULT_AGE, 18)   # triage's age rule
        child_log = {**LOG, 'weight_kg': 60, 'height_cm': 149}   # BMI 27.0
        for disease, values in (('diabetes', {'age': 12, 'bmi': 27, 'smoking_history': 'current'}),
                                ('kidney', {'Age': '17.5', 'BMI': 27, 'Smoking': 1}),
                                ('heart', {'BMI': 27, 'SmokerStatus': 'Current smoker - now smokes every day'}),
                                ('diabetes', {'bmi': 27})):
            # Heart has no numeric age, and the last diabetes screening left it blank: the tracker's age decides.
            rows = pv.template(disease, values, [{**child_log, 'age': 12}])
            self.assertEqual({(r['offered'], r['enabled'], r['reason']) for r in rows}, {(False, False, pv.CHILD_REASON)},
                             (disease, values))
            self.assertIsNone(keyed(rows)['weight']['goal'])   # no weight-loss number for a child
        self.assertEqual(pv.facts('diabetes', {'age': 12, 'bmi': 27}), ['age 12 (under 18)', 'BMI 27.0'])
        # 18 is an adult; heart's age categories all start at 18, whatever the tracker says; the screening's own
        # age comes before the tracker's.
        for disease, values, logs in (('diabetes', {'age': 18, 'bmi': 27}, [child_log]),
                                      ('heart', {'AgeCategory': 'Age 18 to 24', 'BMI': 27}, [{**child_log, 'age': 12}]),
                                      ('diabetes', {'age': 40, 'bmi': 27}, [{**child_log, 'age': 12}]),
                                      ('diabetes', {'bmi': 27}, [{**child_log, 'age': 30}])):
            self.assertTrue(keyed(pv.template(disease, values, logs))['activity']['enabled'], (disease, values))
            self.assertIsNone(pv.child_age(disease, values, logs))
        self.assertEqual(pv.child_age('diabetes', {'age': 40}, [{'age': 12}]), None)
        self.assertEqual(pv.child_age('kidney', {}, [{'age': 9}]), 9)

        # The doctor's strip: a child's line replaces the track's, and never claims a value is in a diagnostic range.
        from ahead.triage import triage
        flagged_child = triage('diabetes', {'age': 12, 'bmi': 27}, .9, .5)
        self.assertEqual((flagged_child['track'], flagged_child['child']), ('urgent', True))
        self.assertEqual(pv.fit_text(flagged_child), pv.FIT['urgent_child'])
        self.assertNotIn('diagnostic range', pv.fit_text(flagged_child))
        self.assertEqual(pv.fit_text(triage('diabetes', {'age': 12}, .1, .5)), pv.FIT['child'])
        self.assertEqual(pv.fit_text('at_risk', child=True), pv.FIT['child'])   # e.g. the tracker's age
        self.assertEqual(pv.fit_text(triage('diabetes', {'age': 40}, .9, .5)), pv.FIT['at_risk'])
        self.assertEqual(pv.fit_text('no_score'), pv.FIT['no_score'])

    def test_weight_goal_is_about_six_percent_lower_in_half_kilos(self):
        self.assertEqual([pv.default_weight_goal(w) for w in (84, 90, 100, 75, None)], [79.0, 84.5, 94.0, 70.5, None])
        for tenth in range(400, 2000):
            goal = pv.default_weight_goal(tenth / 10)
            self.assertEqual(goal * 2, int(goal * 2))
            self.assertLessEqual(abs(goal - tenth / 10 * 0.94), 0.25 + 1e-9)
        self.assertEqual(pv.why_text('diabetes', 'weight', 84, 79), 'About 6% lower (5–7% is the usual aim)')
        self.assertEqual(pv.why_text('diabetes', 'weight', 100, 95), 'About 5% lower (5–7% is the usual aim)')
        self.assertEqual(pv.why_text('diabetes', 'weight', 80, 80), 'Not below the latest weight (5–7% lower is the usual aim)')
        self.assertIn('potassium', pv.why_text('kidney', 'fruit_veg'))

    def test_baselines_come_from_the_latest_log_or_say_not_logged(self):
        older = {**LOG, 'weight_kg': 90, 'fruit_veg': 4}
        rows = keyed(pv.template('diabetes', {'smoking_history': 'current'},
                                 [log(LOG, '2026-11-05T08:00:00+00:00'), log(older, '2026-10-01T08:00:00+00:00')]))
        self.assertEqual([rows[k]['baseline'] for k in ('weight', 'activity', 'fruit_veg', 'sleep', 'smoking')],
                         [84, 60, 2, 6, 'current'])
        self.assertEqual([rows[k]['baseline_text'] for k in ('weight', 'activity', 'fruit_veg', 'sleep', 'smoking')],
                         ['84 kg', '60 min', '2 a day', '6 h', 'Current smoker'])
        self.assertEqual([rows[k]['goal'] for k in ('weight', 'activity', 'fruit_veg', 'sleep')], [79.0, 150, 5, [7.0, 9.0]])
        self.assertEqual([rows[k]['goal_text'] for k in ('weight', 'activity', 'fruit_veg', 'sleep', 'smoking')],
                         ['79 kg', '150 min a week', '5 a day', '7–9 h a night', 'Stop'])
        empty = keyed(pv.template('heart', {'BMI': 30}, []))
        self.assertEqual({k: r['baseline_text'] for k, r in empty.items() if k != 'smoking'},
                         dict.fromkeys(('weight', 'activity', 'fruit_veg', 'sleep'), 'Not logged yet'))
        self.assertTrue(all(empty[k]['enabled'] for k in ('activity', 'fruit_veg', 'sleep')))
        self.assertEqual(list(empty), list(pv.TARGETS))
        with self.assertRaises(ValueError):pv.template('lungs', {}, [])

    def test_checking_a_plan(self):
        today = date(2026, 11, 6)
        plan = {'disease': 'diabetes', 'screening_id': '4', 'note': '  Small steps.  ', 'recheck_date': '2027-01-29',
                'targets': [{'key': 'sleep', 'goal': (7, 9)}, {'key': 'weight', 'goal': 79.04},
                            {'key': 'activity', 'goal': 150.4}, {'key': 'smoking', 'goal': 'anything'}]}
        checked = pv.check_plan(plan, today)
        self.assertEqual(checked['targets'], [{'key': 'weight', 'goal': 79.0}, {'key': 'activity', 'goal': 150},
                                              {'key': 'sleep', 'goal': [7.0, 9.0]}, {'key': 'smoking', 'goal': None}])
        self.assertEqual((checked['note'], checked['screening_id'], checked['recheck_date']), ('Small steps.', 4, '2027-01-29'))
        for targets in ([], [{'key': 'steps', 'goal': 5}], [{'key': 'sleep', 'goal': [9, 7]}],
                        [{'key': 'sleep', 'goal': [5, 9]}], [{'key': 'weight', 'goal': 20}],
                        [{'key': 'activity', 'goal': 'lots'}], [{'key': 'fruit_veg', 'goal': 16}],
                        [{'key': 'fruit_veg', 'goal': 5}, {'key': 'fruit_veg', 'goal': 6}]):
            with self.assertRaises(ValueError, msg=targets):pv.check_plan({**plan, 'targets': targets}, today)
        for change in ({'recheck_date': '2026-11-06'}, {'recheck_date': 'soon'}, {'disease': 'flu'},
                       {'note': 'x' * 1001}):
            with self.assertRaises(ValueError, msg=change):pv.check_plan({**plan, **change}, today)
        self.assertEqual(pv.check_plan({**plan, 'recheck_date': date(2026, 11, 1)}, today, recheck_changed=False)['recheck_date'],
                         '2026-11-01')
        self.assertEqual(pv.tracker_goals(checked['targets']), {'weight': 79.0, 'exercise': 150, 'sleep': 8.0})
        self.assertEqual(pv.goals_from_plan(checked['targets'], {'weight': 79, 'exercise': 120, 'sleep': 8.0, 'steps': 1}),
                         {'weight', 'sleep'})
        self.assertEqual(pv.recheck_date(12, date(2026, 11, 6)), date(2027, 1, 29))
        with self.assertRaises(ValueError):pv.recheck_date(1)

    def test_display_text(self):
        self.assertEqual([pv.goal_text('activity', 150, full=False), pv.goal_text('sleep', [8, 8]), pv.goal_text('weight', None)],
                         ['150 min', '8 h a night', 'No goal set'])
        self.assertEqual([pv.value_text('weight', 82.45), pv.value_text('sleep', None), pv.value_text('smoking', 'never')],
                         ['82.45 kg', 'Not logged yet', 'Never smoked'])
        self.assertEqual([pv.summary_line(t) for t in ({'key': 'weight', 'goal': 79, 'baseline': 84},
                                                       {'key': 'weight', 'goal': 79, 'baseline': None},
                                                       {'key': 'fruit_veg', 'goal': 5}, {'key': 'smoking', 'goal': None})],
                         ['Body weight · 84 → 79 kg', 'Body weight · 79 kg', 'Fruit and vegetables · 5 a day',
                          'Stop smoking · checked at the re-check'])
        today = date(2026, 11, 6)
        self.assertEqual([pv.recheck_status(today + timedelta(days=d), today)['text'] for d in (84, 80, 14, 13, 1, 0, -3)],
                         ['Re-check in 12 weeks', 'Re-check in 11 weeks', 'Re-check in 2 weeks', 'Re-check in 13 days',
                          'Re-check in 1 day', 'Re-check due', 'Re-check due'])
        self.assertEqual(pv.recheck_status('2027-01-29', today)['date_text'], '29 Jan 2027')
        self.assertEqual(pv.history_text({'action': 'stopped', 'details': {'reason': 'replaced'}}),
                         'Stopped: replaced by a newer plan')
        self.assertEqual(pv.history_text({'action': 'active', 'details': {}}), 'Resumed')
        self.assertEqual(pv.fit_text('urgent'), pv.FIT['urgent'])
        # The plan is standard advice: no wording promises a lower model score or an outcome.
        words = ' '.join([*pv.FIT.values(), *(p.advice for p in pv.STARTING_POINTS.values()),
                          *(t.why for t in pv.TARGETS.values())])
        self.assertNotRegex(words, r"(?i)guarantee|lowers? (your |the )?(model's )?(score|risk)")


class ProgressTests(unittest.TestCase):
    TODAY = date(2026, 11, 20)
    PLAN = {'targets': [{'key': 'weight', 'goal': 79.0, 'baseline': 84.0}, {'key': 'activity', 'goal': 150, 'baseline': 60},
                        {'key': 'fruit_veg', 'goal': 5, 'baseline': 2}, {'key': 'sleep', 'goal': [7, 9], 'baseline': 6},
                        {'key': 'smoking', 'goal': None, 'baseline': 'current'}],
            'started_at': '2026-11-06T06:00:00+00:00', 'recheck_date': '2027-01-29'}

    def run_progress(self, logs, days=None, plan=None, today=None):
        return pv.progress(plan or self.PLAN, logs, days or {}, today or self.TODAY)

    def test_nothing_logged_since_the_start(self):
        before = log(LOG, '2026-11-05T08:00:00+00:00')   # the baseline log, before the plan started
        result = self.run_progress([before])
        items = keyed(result['targets'])
        self.assertEqual({k: i['status'] for k, i in items.items()},
                         {'weight': 'not_logged', 'activity': 'not_logged', 'fruit_veg': 'not_logged',
                          'sleep': 'not_logged', 'smoking': 'at_recheck'})
        self.assertEqual(items['weight']['note'], 'Started at 84 kg; log your weight to see progress')
        self.assertEqual((items['weight']['current_text'], items['weight']['share']), ('—', None))
        self.assertEqual((result['on_track'], result['tracked']), (0, 4))
        self.assertEqual(result['headline'], 'Log your first check-in to see your progress')
        self.assertEqual(result['message'], 'Log your body weight, activity, fruit and vegetables and sleep to see their progress.')

    def test_targets_on_track_and_behind(self):
        logs = [log({**LOG, 'weight_kg': 82.4, 'fruit_veg': 6, 'sleep_hours': 7.5}, '2026-11-20T05:00:00+00:00'),
                log({**LOG, 'weight_kg': 83, 'fruit_veg': 1, 'sleep_hours': 9}, '2026-11-20T03:00:00+00:00'),  # same day: older
                log({**LOG, 'fruit_veg': 4, 'sleep_hours': 6.5}, '2026-11-18T08:00:00+00:00'),
                log({**LOG, 'fruit_veg': 0, 'sleep_hours': 3}, '2026-11-13T08:00:00+00:00')]                   # 8 days ago
        days = {'2026-11-20': 40, '2026-11-17': 50, '2026-11-14': 70, '2026-11-13': 300}
        result = self.run_progress(logs, days)
        items = keyed(result['targets'])
        self.assertEqual((items['weight']['current'], items['weight']['status'], items['weight']['note']),
                         (82.4, 'on_track', 'Down 1.6 kg since 6 Nov'))
        self.assertEqual(items['weight']['share'], 0.32)
        self.assertEqual((items['activity']['current'], items['activity']['status'], items['activity']['note']),
                         (160, 'on_track', 'Last 7 days'))
        self.assertEqual((items['fruit_veg']['current'], items['fruit_veg']['status']), (5.0, 'on_track'))
        self.assertEqual((items['sleep']['current'], items['sleep']['status'], items['sleep']['note']),
                         (7.0, 'on_track', '7-night average'))
        self.assertEqual((items['activity']['current_text'], items['activity']['of_goal']), ('160', '/ 150 min'))
        self.assertEqual(items['sleep']['of_goal'], '/ 7–9 h')
        self.assertEqual(result['headline'], 'All four of your targets are on track')
        self.assertEqual(result['message'], 'Great work: keep it up until your re-check.')
        self.assertFalse(items['smoking']['tracked'])
        # Servings are logged in a tracker check-in (the food analyzer's meals hold none), so the card opens the tracker.
        self.assertEqual((items['fruit_veg']['page'], items['fruit_veg']['action']), ('wellness', 'Log servings'))
        self.assertEqual({t.page for t in pv.TARGETS.values() if t.page}, {'wellness'})

        # Behind: activity short of the goal, fruit below five, sleep outside the range.
        logs = [log({**LOG, 'weight_kg': 82.4, 'fruit_veg': 3, 'sleep_hours': 6.5}, '2026-11-19T08:00:00+00:00')]
        result = self.run_progress(logs, {'2026-11-19': 110})
        items = keyed(result['targets'])
        self.assertEqual([items[k]['status'] for k in ('weight', 'activity', 'fruit_veg', 'sleep')],
                         ['on_track', 'behind', 'behind', 'behind'])
        self.assertEqual(items['sleep']['note'], "From this week's check-in")
        self.assertEqual(result['headline'], 'One of your four targets is on track')
        self.assertEqual(result['message'],
                         'Keep going: give activity, fruit and vegetables and sleep a little more attention this week.')
        self.assertEqual(result['attention'], ['activity', 'fruit and vegetables', 'sleep'])
        self.assertEqual(result['recheck']['text'], 'Re-check in 10 weeks')
        late = keyed(self.run_progress([log({**LOG, 'sleep_hours': 10}, '2026-11-19T08:00:00+00:00')])['targets'])
        self.assertEqual((late['sleep']['status'], late['sleep']['share']), ('behind', 0.9))
        # One plural label still reads right.
        plan = {**self.PLAN, 'targets': [{'key': 'fruit_veg', 'goal': 5, 'baseline': 2}]}
        self.assertEqual(self.run_progress(logs, plan=plan)['message'],
                         'Keep going: give fruit and vegetables a little more attention this week.')

    def test_activity_falls_back_to_the_tracker_estimate(self):
        logs = [log({**LOG, 'exercise_days': 5, 'exercise_minutes': 40}, '2026-11-18T08:00:00+00:00')]
        item = keyed(self.run_progress(logs, {'2026-11-10': 500})['targets'])['activity']
        self.assertEqual((item['current'], item['status'], item['note']),
                         (200, 'on_track', "Weekly estimate from this week's check-in"))
        # A logged zero this week is a real entry, not a gap.
        item = keyed(self.run_progress(logs, {'2026-11-19': 0})['targets'])['activity']
        self.assertEqual((item['current'], item['status']), (0, 'behind'))

    def test_weight_pace_rule(self):
        plan = {**self.PLAN, 'targets': [{'key': 'weight', 'goal': 78.0, 'baseline': 84.0}]}   # 6 kg over 84 days
        check = lambda weight, day: keyed(self.run_progress(
            [log({**LOG, 'weight_kg': weight}, f'{day.isoformat()}T08:00:00+00:00')], plan=plan, today=day)['targets'])['weight']
        # Half the even pace after 42 of 84 days is 1.5 kg.
        self.assertEqual(check(82.5, date(2026, 12, 18))['status'], 'on_track')
        self.assertEqual(check(82.6, date(2026, 12, 18))['status'], 'behind')
        self.assertEqual(check(84.0, date(2026, 11, 7))['status'], 'on_track')   # steady on day one
        self.assertEqual(check(84.5, date(2026, 11, 7))['status'], 'behind')     # a gain is never on track
        self.assertEqual(check(84.5, date(2026, 11, 7))['note'], 'Up 0.5 kg since 6 Nov')
        self.assertEqual(check(77.9, date(2026, 11, 7))['status'], 'on_track')   # at the goal
        # No baseline: the first weight logged after the start is the starting weight.
        plan['targets'] = [{'key': 'weight', 'goal': 78.0, 'baseline': None}]
        logs = [log({**LOG, 'weight_kg': 82}, '2026-11-19T08:00:00+00:00'), log({**LOG, 'weight_kg': 83}, '2026-11-08T08:00:00+00:00')]
        item = keyed(self.run_progress(logs, plan=plan)['targets'])['weight']
        self.assertEqual((item['current'], item['note'], item['status']), (82, 'Down 1 kg since 8 Nov', 'on_track'))

    def test_a_weight_target_added_by_an_edit_counts_from_that_edit(self):
        # Started 6 Nov without weight; 90 kg logged 10 Nov; weight added 12 Nov from 90 to 84.5, re-check 4 Dec.
        plan = {**self.PLAN, 'recheck_date': '2026-12-04',
                'targets': [{'key': 'weight', 'goal': 84.5, 'baseline': 90, 'since': '2026-11-12T06:00:00+00:00'}]}
        before = log({**LOG, 'weight_kg': 90}, '2026-11-10T08:00:00+00:00')
        item = keyed(self.run_progress([before], plan=plan, today=date(2026, 11, 12))['targets'])['weight']
        self.assertEqual((item['status'], item['note']), ('not_logged', 'Started at 90 kg; log your weight to see progress'))
        # Pace and the note count from 12 Nov: 1 kg down in 8 of 22 days is ahead of half the even pace (1 kg).
        logs = [log({**LOG, 'weight_kg': 89}, '2026-11-20T08:00:00+00:00'), before]
        item = keyed(self.run_progress(logs, plan=plan)['targets'])['weight']
        self.assertEqual((item['status'], item['note']), ('on_track', 'Down 1 kg since 12 Nov'))
        # Without 'since' the same plan counts from 6 Nov and is behind (half the even pace by 20 Nov is 1.4 kg).
        plain = {**plan, 'targets': [{k: v for k, v in plan['targets'][0].items() if k != 'since'}]}
        self.assertEqual(keyed(self.run_progress(logs, plan=plain)['targets'])['weight']['status'], 'behind')

    def test_only_stop_smoking(self):
        plan = {**self.PLAN, 'targets': [{'key': 'smoking', 'goal': None, 'baseline': 'current'}]}
        result = self.run_progress([], plan=plan)
        self.assertEqual((result['tracked'], result['on_track'], result['message']), (0, 0, ''))
        self.assertEqual(result['headline'], 'Your doctor checks this plan at your re-check')
        self.assertEqual(result['targets'][0]['status_text'], 'Checked at re-check')


class StoredPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH'] = self.temp.name + '/app.db'
        os.environ.pop('AHEAD_ENABLE_DEMO_ACCOUNTS', None)
        init_db();init_platform();init_db();init_platform()
        # Inserted directly: sign-in is not under test here, and each password hash takes most of a second.
        with connection() as db:
            self.p, self.q, self.d, self.other = (
                db.execute('INSERT INTO users(email,password_hash,role,name) VALUES(?,?,?,?)', (email, '-', role, email)).lastrowid
                for email, role in [('p@test.com', 'Patient'), ('q@test.com', 'Patient'), ('d@test.com', 'Doctor'),
                                    ('other@test.com', 'Doctor')])
        grant_doctor(self.p, 'd@test.com')
        self.screening = save_screening(self.p, 'diabetes', 'Model', .55, 1, .5,
                                        {'smoking_history': 'current', 'bmi': 31.2, 'age': 50})
        save_lifestyle(self.p, LOG, {})
        self.recheck = (local_today() + timedelta(weeks=12)).isoformat()

    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH', None)

    def plan(self, **change):
        return {'disease': 'diabetes', 'screening_id': self.screening, 'note': 'Small steps count.',
                'recheck_date': self.recheck,
                'targets': [{'key': 'weight', 'goal': 79}, {'key': 'activity', 'goal': 150}, {'key': 'fruit_veg', 'goal': 5},
                            {'key': 'sleep', 'goal': [7, 9]}, {'key': 'smoking', 'goal': None}], **change}

    def plan_notes(self):
        return [n for n in notifications(self.p) if n['type'] == 'plan']

    def test_approval_activates_syncs_goals_and_notifies(self):
        set_goals(self.p, {'weight': 70, 'steps': 9000, 'water': 2.5, 'sleep': 6.5})
        plan_id = approve_prevention_plan(self.d, self.p, self.plan())
        current = active_plan(self.p)
        self.assertEqual((current['id'], current['status'], current['doctor_name']), (plan_id, 'active', 'd@test.com'))
        self.assertEqual(current['start_date'], local_today().isoformat())
        self.assertEqual(current['targets'][:2], [{'key': 'weight', 'goal': 79.0, 'baseline': 84},
                                                  {'key': 'activity', 'goal': 150, 'baseline': 60}])
        self.assertEqual(get_goals(self.p), {'weight': 79.0, 'steps': 9000, 'water': 2.5, 'sleep': 8.0, 'exercise': 150,
                                             'fruit_veg': 5})
        note, = self.plan_notes()
        self.assertEqual((note['title'], note['action']), ('New prevention plan', 'medications'))
        self.assertIn('diabetes prevention plan with 5 targets', note['description'])
        self.assertIn('Your tracker goals now match it.', note['description'])
        self.assertEqual(prevention_progress(self.p, current, self.d)['tracked'], 4)

        # A second approval from the same doctor stops the first, with history.
        second = approve_prevention_plan(self.d, self.p, self.plan(targets=[{'key': 'activity', 'goal': 180}]))
        self.assertEqual([p['id'] for p in prevention_plans(self.p)], [second])
        self.assertEqual(get_goals(self.p)['exercise'], 180)
        self.assertEqual(get_goals(self.p)['weight'], 79.0)   # a goal the new plan does not set stays
        first = next(p for p in prevention_plans(self.p, self.d) if p['id'] == plan_id)
        self.assertEqual(first['status'], 'stopped')
        self.assertEqual([h['text'] for h in first['history']], ['Approved and sent', 'Stopped: replaced by a newer plan'])
        self.assertIn('It replaces your earlier plan.', self.plan_notes()[0]['description'])

        # Another linked doctor's plan sits alongside; the newest active plan is the one shown.
        grant_doctor(self.p, 'other@test.com')
        third = approve_prevention_plan(self.other, self.p, self.plan(targets=[{'key': 'sleep', 'goal': [7, 8]}]))
        self.assertEqual([p['id'] for p in prevention_plans(self.p)], [third, second])
        self.assertEqual(active_plan(self.p)['id'], third)
        self.assertEqual([p['id'] for p in prevention_plans(self.p, self.d)], [second, plan_id])

    def test_drafts_are_invisible_to_patients(self):
        goals = {'weight': 70}
        set_goals(self.p, goals)
        draft = save_prevention_draft(self.d, self.p, self.plan())
        self.assertEqual(save_prevention_draft(self.d, self.p, self.plan(note='Edited.')), draft)   # one draft per doctor
        self.assertEqual((prevention_plans(self.p), active_plan(self.p), self.plan_notes()), ([], None, []))
        self.assertEqual(get_goals(self.p), goals)
        mine, = prevention_plans(self.p, self.d)
        self.assertEqual((mine['id'], mine['status'], mine['note'], mine['start_date']), (draft, 'draft', 'Edited.', None))
        with self.assertRaises(ValueError):prevention_plan_status(self.d, draft, 'paused')
        with self.assertRaises(ValueError):update_prevention_plan(self.d, draft, self.plan())
        self.assertEqual(approve_prevention_plan(self.d, self.p, self.plan(), draft), draft)
        self.assertEqual(active_plan(self.p)['id'], draft)
        self.assertEqual([h['action'] for h in prevention_history(self.d, draft)], ['drafted', 'draft_edited', 'approved'])
        with self.assertRaises(ValueError):approve_prevention_plan(self.d, self.p, self.plan(), draft)
        with self.assertRaises(ValueError):save_prevention_draft(self.d, self.p, self.plan(), draft)
        # A new draft alongside the active plan stays hidden; approving another plan discards it.
        hidden = save_prevention_draft(self.d, self.p, self.plan(note='Next time.'))
        self.assertEqual([p['id'] for p in prevention_plans(self.p)], [draft])
        approve_prevention_plan(self.d, self.p, self.plan())
        self.assertEqual([h['text'] for h in prevention_history(self.d, hidden)],
                         ['Draft saved', 'Draft discarded: another plan was approved'])
        # Stopping a draft discards it without telling the patient.
        dropped = save_prevention_draft(self.d, self.p, self.plan())
        before = len(self.plan_notes())
        self.assertTrue(prevention_plan_status(self.d, dropped, 'stopped'))
        self.assertEqual(len(self.plan_notes()), before)

    def test_only_a_linked_doctor_can_write_or_read(self):
        draft = save_prevention_draft(self.d, self.p, self.plan())
        live = approve_prevention_plan(self.d, self.p, self.plan())
        for call in (lambda: save_prevention_draft(self.other, self.p, self.plan()),
                     lambda: approve_prevention_plan(self.other, self.p, self.plan()),
                     lambda: approve_prevention_plan(self.d, self.q, self.plan()),
                     lambda: update_prevention_plan(self.other, live, self.plan()),
                     lambda: prevention_plan_status(self.other, live, 'paused'),
                     lambda: prevention_plans(self.p, self.other),
                     lambda: prevention_history(self.other, live),
                     lambda: prevention_progress(self.p, active_plan(self.p), self.other)):
            with self.assertRaises(PermissionError):call()
        # The doctor's own draft id cannot be used for another patient.
        grant_doctor(self.q, 'd@test.com')
        with self.assertRaises(PermissionError):approve_prevention_plan(self.d, self.q, self.plan(screening_id=None), draft)
        # A screening that is not this patient's, or not this condition, is refused.
        other_screening = save_screening(self.q, 'diabetes', 'Model', .6, 1, .5, {'bmi': 30})
        heart = save_screening(self.p, 'heart', 'Model', .6, 1, .5, {'BMI': 30})
        for screening in (other_screening, heart):
            with self.assertRaises(ValueError):approve_prevention_plan(self.d, self.p, self.plan(screening_id=screening))
        self.assertEqual(active_plan(self.p)['id'], live)

    def test_offer_rules_hold_when_saving(self):
        former = save_screening(self.p, 'diabetes', 'Model', .55, 1, .5, {'smoking_history': 'former', 'bmi': 31.2})
        with self.assertRaisesRegex(ValueError, r'Stop smoking is not offered for this patient \(former smoker\)'):
            approve_prevention_plan(self.d, self.p, self.plan(screening_id=former))
        live = approve_prevention_plan(self.d, self.p, self.plan())
        save_lifestyle(self.p, {**LOG, 'weight_kg': 60}, {})   # now BMI 22.3
        with self.assertRaisesRegex(ValueError, 'Body weight is not offered'):
            approve_prevention_plan(self.d, self.p, self.plan())
        # A target the plan already has stays editable after its rule stops applying, and keeps its baseline.
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(targets=[{'key': 'weight', 'goal': 59}])))
        self.assertEqual(active_plan(self.p)['targets'], [{'key': 'weight', 'goal': 59.0, 'baseline': 84}])

    def test_no_plan_for_a_child(self):
        grant_doctor(self.q, 'd@test.com')
        save_lifestyle(self.q, {**LOG, 'weight_kg': 60, 'height_cm': 149}, {})   # BMI 27.0
        child = save_screening(self.q, 'diabetes', 'Model', .2, 0, .5, {'age': 12, 'bmi': 27})
        targets = [{'key': 'activity', 'goal': 150}]
        for call in (approve_prevention_plan, save_prevention_draft):
            with self.assertRaisesRegex(ValueError, r'Physical activity is not offered for this patient \(under 18; '
                                                    r'these targets are written for adults\)'):
                call(self.d, self.q, self.plan(screening_id=child, targets=targets))
        # No screening for the condition: the tracker's age decides.
        save_lifestyle(self.q, {**LOG, 'age': 15}, {})
        with self.assertRaisesRegex(ValueError, 'under 18'):
            approve_prevention_plan(self.d, self.q, self.plan(disease='kidney', screening_id=None, targets=targets))
        self.assertEqual((prevention_plans(self.q, self.d), [n for n in notifications(self.q) if n['type'] == 'plan']), ([], []))

    def test_no_plan_after_a_past_heart_attack_angina_or_stroke(self):
        # The heart screening's urgent track for a past event: secondary prevention, agreed outside AHEAD.
        from ahead.triage import triage
        for answers in ({'HadHeartAttack': 'Yes'}, {'HadAngina': 'Yes'}, {'HadStroke': 'Yes'}):
            values = {'BMI': 31, 'SmokerStatus': 'Current smoker - now smokes every day', 'HadHeartAttack': 'No',
                      'HadAngina': 'No', 'HadStroke': 'No'} | answers
            rows = pv.template('heart', values, [LOG])
            self.assertEqual({(r['offered'], r['enabled'], r['reason']) for r in rows},
                             {(False, False, pv.HISTORY_REASON)}, answers)
            result = triage('heart', values, .02, .08)
            self.assertEqual((result['track'], pv.fit_text(result)), ('urgent', pv.FIT['urgent_history']))
        # Other conditions, and heart screenings without the answers (saved before the questions), are unchanged.
        self.assertTrue(keyed(pv.template('heart', {'BMI': 31, 'HadStroke': 'No'}, [LOG]))['activity']['offered'])
        self.assertTrue(keyed(pv.template('diabetes', {'bmi': 31, 'HadHeartAttack': 'Yes'}, [LOG]))['activity']['offered'])
        heart = save_screening(self.p, 'heart', 'Model', .2, 1, .08, {'BMI': 31, 'HadAngina': 'Yes'})
        with self.assertRaisesRegex(ValueError, r'Physical activity is not offered for this patient \(past heart '
                                                r'attack, angina or stroke reported\)'):
            approve_prevention_plan(self.d, self.p, self.plan(disease='heart', screening_id=heart,
                                                              targets=[{'key': 'activity', 'goal': 150}]))

    def test_edits_check_offers_against_the_newest_screening(self):
        # Approved from a screening with smoking not recorded; the patient then re-screens as a current smoker.
        first = save_screening(self.p, 'diabetes', 'Model', .55, 1, .5, {'smoking_history': 'No Info', 'bmi': 31.2})
        live = approve_prevention_plan(self.d, self.p, self.plan(screening_id=first, targets=[{'key': 'activity', 'goal': 150}]))
        save_screening(self.p, 'diabetes', 'Model', .6, 1, .5, {'smoking_history': 'current', 'bmi': 31.2})
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(
            targets=[{'key': 'activity', 'goal': 150}, {'key': 'smoking'}])))
        plan = active_plan(self.p)
        self.assertEqual([t['key'] for t in plan['targets']], ['activity', 'smoking'])
        self.assertEqual(plan['screening_id'], first)   # the source screening stays
        # And the reverse: once the newest screening says former smoker, stop smoking cannot be added again.
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(targets=[{'key': 'activity', 'goal': 150}])))
        save_screening(self.p, 'diabetes', 'Model', .6, 1, .5, {'smoking_history': 'former', 'bmi': 31.2})
        with self.assertRaisesRegex(ValueError, r'Stop smoking is not offered for this patient \(former smoker\)'):
            update_prevention_plan(self.d, live, self.plan(targets=[{'key': 'activity', 'goal': 150}, {'key': 'smoking'}]))
        # Another condition's newer screening does not count.
        save_screening(self.p, 'heart', 'Model', .6, 1, .5, {'SmokerStatus': 'Current smoker - now smokes every day'})
        with self.assertRaisesRegex(ValueError, 'former smoker'):
            update_prevention_plan(self.d, live, self.plan(targets=[{'key': 'smoking'}]))

    def test_a_target_added_by_an_edit_starts_at_the_edit(self):
        live = approve_prevention_plan(self.d, self.p, self.plan(targets=[{'key': 'activity', 'goal': 150}]))
        save_lifestyle(self.p, {**LOG, 'weight_kg': 90}, {})
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(
            targets=[{'key': 'weight', 'goal': 84.5}, {'key': 'activity', 'goal': 150}])))
        plan = active_plan(self.p)
        weight, activity = plan['targets']
        self.assertEqual((weight['baseline'], weight['since']), (90, plan['updated_at']))
        self.assertNotIn('since', activity)   # approved with the plan: counts from started_at
        item = keyed(prevention_progress(self.p, plan)['targets'])['weight']
        self.assertEqual((item['status'], item['note']), ('not_logged', 'Started at 90 kg; log your weight to see progress'))
        # A later edit keeps the target's start.
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(
            targets=[{'key': 'weight', 'goal': 85}, {'key': 'activity', 'goal': 150}])))
        self.assertEqual(active_plan(self.p)['targets'][0], {'key': 'weight', 'goal': 85.0, 'baseline': 90,
                                                             'since': weight['since']})

    def test_edits_pause_and_stop_keep_history(self):
        live = approve_prevention_plan(self.d, self.p, self.plan())
        self.assertFalse(update_prevention_plan(self.d, live, self.plan()))   # nothing changed
        save_lifestyle(self.p, {**LOG, 'weight_kg': 83, 'fruit_veg': 1}, {})
        later = (local_today() + timedelta(weeks=8)).isoformat()
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(
            note='Try a walk after dinner.', recheck_date=later,
            targets=[{'key': 'weight', 'goal': 78}, {'key': 'fruit_veg', 'goal': 4}, {'key': 'smoking'}])))
        plan = active_plan(self.p)
        self.assertEqual(plan['targets'], [{'key': 'weight', 'goal': 78.0, 'baseline': 84},
                                           {'key': 'fruit_veg', 'goal': 4, 'baseline': 2},
                                           {'key': 'smoking', 'goal': None, 'baseline': 'current'}])
        self.assertEqual((plan['note'], plan['recheck_date']), ('Try a walk after dinner.', later))
        self.assertEqual((get_goals(self.p)['weight'], get_goals(self.p)['fruit_veg']), (78.0, 4))
        edited = prevention_history(self.d, live)[-1]
        self.assertEqual(edited['action'], 'edited')
        self.assertEqual(edited['details']['before']['note'], 'Small steps count.')
        self.assertEqual(len(edited['details']['before']['targets']), 5)
        self.assertEqual(self.plan_notes()[0]['title'], 'Prevention plan updated')

        # Paused: still shown to the patient, edits do not touch the tracker goals until it resumes.
        self.assertTrue(prevention_plan_status(self.d, live, 'paused'))
        self.assertFalse(prevention_plan_status(self.d, live, 'paused'))
        self.assertEqual(active_plan(self.p)['status'], 'paused')
        set_goals(self.p, {'weight': 90, 'steps': 7000})
        update_prevention_plan(self.d, live, self.plan(recheck_date=later, targets=[{'key': 'weight', 'goal': 77}]))
        self.assertEqual(get_goals(self.p), {'weight': 90, 'steps': 7000})
        self.assertTrue(prevention_plan_status(self.d, live, 'active'))
        self.assertEqual(get_goals(self.p), {'weight': 77.0, 'steps': 7000})
        self.assertTrue(prevention_plan_status(self.d, live, 'stopped'))
        self.assertEqual((prevention_plans(self.p), active_plan(self.p)), ([], None))
        for call in (lambda: prevention_plan_status(self.d, live, 'active'),
                     lambda: update_prevention_plan(self.d, live, self.plan())):
            with self.assertRaises(ValueError):call()
        self.assertEqual([h['action'] for h in prevention_history(self.d, live)],
                         ['approved', 'edited', 'paused', 'edited', 'active', 'stopped'])
        self.assertEqual([n['title'] for n in self.plan_notes()][:4],
                         ['Prevention plan stopped', 'Prevention plan resumed', 'Prevention plan updated', 'Prevention plan paused'])
        # An unchanged re-check date in the past does not block an edit; a changed one must be after today.
        with connection() as db:
            db.execute("UPDATE prevention_plans SET status='active',recheck_date=? WHERE id=?",
                       ((local_today() - timedelta(days=1)).isoformat(), live))
        past = (local_today() - timedelta(days=1)).isoformat()
        self.assertTrue(update_prevention_plan(self.d, live, self.plan(recheck_date=past, note='Book a re-check.')))
        with self.assertRaises(ValueError):update_prevention_plan(self.d, live, self.plan(recheck_date=local_today().isoformat()))

    def test_revoking_access_stops_plans(self):
        live = approve_prevention_plan(self.d, self.p, self.plan())
        draft = save_prevention_draft(self.d, self.p, self.plan())
        revoke_doctor(self.p, self.d)
        self.assertEqual((prevention_plans(self.p), active_plan(self.p)), ([], None))
        with connection() as db:
            rows = {r['id']: r['status'] for r in db.execute('SELECT id,status FROM prevention_plans')}
            audit = {(r['plan_id'], r['action'], json.loads(r['details_json']).get('reason'))
                     for r in db.execute('SELECT * FROM prevention_audit')}
        self.assertEqual(rows, {live: 'stopped', draft: 'stopped'})
        self.assertLessEqual({(live, 'stopped', 'access removed'), (draft, 'discarded', 'access removed')}, audit)
        note = self.plan_notes()[0]
        self.assertEqual((note['title'], note['action']), ('Prevention plan stopped', 'medications'))
        self.assertEqual(sum(n['title'] == 'Prevention plan stopped' for n in self.plan_notes()), 1)   # not for the draft
        with self.assertRaises(PermissionError):approve_prevention_plan(self.d, self.p, self.plan())

        # Access removed outside revoke_doctor (an older build): start-up stops the plan too.
        grant_doctor(self.p, 'd@test.com')
        again = approve_prevention_plan(self.d, self.p, self.plan())
        with connection() as db:
            db.execute('DELETE FROM care_access WHERE patient_id=? AND doctor_id=?', (self.p, self.d))
        init_platform()
        self.assertEqual(prevention_plans(self.p), [])
        self.assertIn(f'prevention-status-{again}-stopped-unlinked', {n['dedupe_key'] for n in notifications(self.p)})

    def test_recheck_reminder_once_per_date(self):
        live = approve_prevention_plan(self.d, self.p, self.plan(recheck_date=(local_today() + timedelta(days=3)).isoformat()))
        notify_recheck_due(self.p)
        self.assertFalse(any(n['title'] == 'Time to screen again' for n in notifications(self.p)))
        due = local_today() + timedelta(days=3)
        notify_recheck_due(self.p, due);notify_recheck_due(self.p, due + timedelta(days=1))
        reminders = [n for n in notifications(self.p) if n['title'] == 'Time to screen again']
        self.assertEqual([(n['action'], n['dedupe_key']) for n in reminders],
                         [('screenings', f'prevention-recheck-{live}-{due.isoformat()}')])
        prevention_plan_status(self.d, live, 'paused')
        notify_recheck_due(self.p, due + timedelta(days=30))
        self.assertEqual(sum(n['title'] == 'Time to screen again' for n in notifications(self.p)), 1)

    def test_progress_reads_the_patients_logs(self):
        approve_prevention_plan(self.d, self.p, self.plan())
        set_challenge_day(self.p, local_today().isoformat(), 160)
        save_lifestyle(self.p, {**LOG, 'weight_kg': 83.5, 'fruit_veg': 5, 'sleep_hours': 7}, {})
        result = prevention_progress(self.p, active_plan(self.p))
        items = keyed(result['targets'])
        self.assertEqual({k: i['status'] for k, i in items.items()},
                         {'weight': 'on_track', 'activity': 'on_track', 'fruit_veg': 'on_track', 'sleep': 'on_track',
                          'smoking': 'at_recheck'})
        self.assertEqual(items['weight']['note'].split(' since ')[0], 'Down 0.5 kg')
        self.assertEqual(prevention_progress(self.q, {'targets': [], 'recheck_date': self.recheck})['tracked'], 0)

    def test_tables_migrate_idempotently(self):
        with connection() as db:
            tables = {r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {r['name'] for r in db.execute('PRAGMA table_info(prevention_plans)')}
        self.assertLessEqual({'prevention_plans', 'prevention_audit'}, tables)
        self.assertLessEqual({'patient_id', 'doctor_id', 'disease', 'screening_id', 'targets_json', 'note', 'recheck_date',
                              'status', 'started_at', 'created_at', 'updated_at'}, columns)
        approve_prevention_plan(self.d, self.p, self.plan())
        init_platform();init_db();init_platform()
        self.assertEqual(len(prevention_plans(self.p)), 1)
        # Deleting the patient's screening history is not blocked by the plan's source screening.
        from ahead.storage import delete_screenings
        delete_screenings(self.p)
        plan = active_plan(self.p)
        self.assertEqual(plan['screening_id'], self.screening)
        # The plan stays editable; only a new stop-smoking target would need the deleted screening.
        self.assertTrue(update_prevention_plan(self.d, plan['id'], self.plan(note='Still on.')))
        with self.assertRaisesRegex(ValueError, 'Choose one of'):approve_prevention_plan(self.d, self.p, self.plan())


class IconTests(unittest.TestCase):
    def test_new_dashboard_icons_load_as_white_stroke_svgs(self):
        import base64
        from ahead.resources import icon_uri
        for name in ('weight', 'food', 'sleep', 'smoking', 'plan'):
            uri = icon_uri(f'dashboard/{name}.svg')
            self.assertTrue(uri.startswith('data:image/svg+xml;base64,'), name)
            svg = base64.b64decode(uri.split(',', 1)[1]).decode()
            self.assertIn('viewBox="0 0 24 24"', svg)
            self.assertIn('stroke="white"', svg)
            self.assertIn('fill="none"', svg)
        self.assertEqual({t.icon for t in pv.TARGETS.values()}, {'weight', 'activity', 'food', 'sleep', 'smoking'})


if __name__ == '__main__':
    unittest.main()
