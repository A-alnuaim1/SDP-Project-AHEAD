"""The doctor's Prevention Plan tab in Patient Intelligence: the tab and its patient picker, the condition strip (a
child's included), the targets table filled from ahead/prevention.py's template (a ticked target that stops being
offered is cleared), the preview matching the patient's My plan page, approving (active plan, notification, tracker
goals), drafts the patient never sees, edits kept in the history, pause/resume/stop behind a confirmation, revoked
access and escaping.
Every check runs on a temporary database; users are inserted directly, since a password hash takes most of a second."""
import os
import re
import tempfile
import unittest
from datetime import timedelta
from html import unescape

from streamlit.testing.v1 import AppTest

import ahead.pages.doctor_intelligence as page
from ahead import prevention as pv
from ahead.components import markdown_literal
from ahead.platform_data import (active_plan, approve_prevention_plan, get_goals, grant_doctor, init_platform, local_today,
                                 notifications, prevention_history, prevention_plan_status, prevention_plans, revoke_doctor,
                                 save_lifestyle, save_prevention_draft, set_goals, update_prevention_plan)
from ahead.storage import connection, init_db, save_screening

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')
TAB = 'Prevention Plan'
LOG = {'weight_kg': 84, 'height_cm': 164, 'exercise_days': 2, 'exercise_minutes': 30, 'fruit_veg': 2, 'sleep_hours': 6}   # BMI 31.2
SCORES = {'overall': 62, 'categories': {}}
AT_RISK = {'age': 50, 'bmi': 31.2, 'smoking_history': 'former', 'HbA1c_level': None, 'blood_glucose_level': 120}
INCLUDE = {'weight': 'Include body weight', 'activity': 'Include physical activity', 'fruit_veg': 'Include fruit and vegetables',
           'sleep': 'Include sleep', 'smoking': 'Include stop smoking'}
GOAL = {'weight': 'Body weight goal (kg)', 'activity': 'Physical activity goal (minutes a week)',
        'fruit_veg': 'Fruit and vegetables goal (servings a day)', 'sleep': 'Sleep goal (hours a night)'}


def blocks(node):
    for child in getattr(node, 'children', {}).values():
        yield child
        yield from blocks(child)


class Fixture(unittest.TestCase):
    """A doctor and one linked patient with an at-risk diabetes screening and a tracker check-in."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH'] = self.temp.name + '/app.db'
        os.environ.pop('AHEAD_ENABLE_DEMO_ACCOUNTS', None)
        init_db(); init_platform()
        self.d = self.user('d@test.com', 'Doctor', 'Dr. Sarah Ahmed')
        self.p = self.user('p@test.com', 'Patient', 'Alice Mansour', link=True)
        self.screening = save_screening(self.p['id'], 'diabetes', 'GradientBoosting', .55, 1, .5, AT_RISK)
        save_lifestyle(self.p['id'], LOG, SCORES)

    def tearDown(self):
        self.temp.cleanup(); os.environ.pop('AHEAD_DB_PATH', None)

    def user(self, email, role, name, link=False):
        with connection() as db:
            uid = db.execute('INSERT INTO users(email,password_hash,role,name) VALUES(?,?,?,?)', (email, '-', role, name)).lastrowid
        if link:grant_doctor(uid, 'd@test.com')
        return {'id': uid, 'email': email, 'role': role, 'name': name, 'institution': '', 'preferences': {}}

    def plan(self, **change):
        return {'disease': 'diabetes', 'screening_id': self.screening, 'note': 'Small steps count.',
                'recheck_date': local_today() + timedelta(weeks=12),
                'targets': [{'key': 'weight', 'goal': 79}, {'key': 'activity', 'goal': 150}, {'key': 'fruit_veg', 'goal': 5},
                            {'key': 'sleep', 'goal': [7, 9]}], **change}

    def page(self, patient=None):
        at = AppTest.from_file(APP, default_timeout=120)
        state = {'authenticated': True, 'user': self.d, 'nav': 'intelligence',
                 'intelligence_focus': {'tab': TAB, 'patient': self.p['id'] if patient is None else patient}}
        for key, value in state.items():at.session_state[key] = value
        return self.ok(at.run())

    def ok(self, at):
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def tab(self, at):
        return next(t for t in at.tabs if t.label == TAB)

    def html(self, at):
        """The tab's HTML with entities decoded."""
        return unescape('\n'.join(h.proto.body for h in self.tab(at).get('html')))

    def widget(self, at, kind, label):
        return next(w for w in getattr(self.tab(at), kind) if w.label == label)

    def goal(self, at, key):
        return self.widget(at, 'slider' if key == 'sleep' else 'number_input', GOAL[key])

    def captions(self, at):
        return [c.value for c in self.tab(at).caption]

    def buttons(self, at):
        return [b.label for b in self.tab(at).button]

    def pills(self, at):
        return re.findall(r'<span class="visual-badge" style="--visual-color:(#\w+)">.*?</span> (\w+)</span>', self.html(at))

    def preview(self, at):
        """The patient preview's target lines, as text."""
        return [re.sub('<[^>]+>', '', line) for line in re.findall(r'<div class="plan-preview-item"[^>]*>(.*?)</div>', self.html(at))]

    def approve(self, at):
        return self.ok(at.button(key=f"prevention_approve_{self.p['id']}").click().run())

    def plan_notes(self):
        return [n for n in notifications(self.p['id']) if n['type'] == 'plan']


class TabTests(Fixture):
    def test_the_tab_follows_medication_monitoring_and_the_focus_selects_its_patient(self):
        self.assertEqual(page.TABS.index(TAB), page.TABS.index('Medication Monitoring') + 1)
        self.assertIn('plan_patient', page.PICKERS)
        ben = self.user('b@test.com', 'Patient', 'Ben Cole', link=True)
        at = self.page(ben['id'])
        box = next(b for b in blocks(at.main) if getattr(b, 'type', None) == 'tab_container')
        self.assertEqual(page.TABS[box.proto.tab_container.default_tab_index], TAB)
        self.assertEqual(at.selectbox(key='plan_patient').value, ben['id'])

    def test_the_picker_lists_only_linked_patients(self):
        self.user('c@test.com', 'Patient', 'Carl Diaz')
        ben = self.user('b@test.com', 'Patient', 'Ben Cole', link=True)
        self.assertEqual(self.page().selectbox(key='plan_patient').options,
                         [f"Alice Mansour · ID {self.p['id']} · p@test.com", f"Ben Cole · ID {ben['id']} · b@test.com"])

    def test_no_linked_patient_explains_how_to_link(self):
        revoke_doctor(self.p['id'], self.d['id'])
        self.assertIn(page.NO_PATIENTS, [i.value for i in self.tab(self.page()).info])


class StripTests(Fixture):
    def test_an_at_risk_result_says_a_plan_fits_with_the_facts_behind_the_targets(self):
        html = self.html(self.page())
        self.assertIn('<div class="track-panel at-risk">', html)
        self.assertIn(f'Diabetes · model score 55.0% · screened {pv.date_text(local_today())} '
                      '<span class="track-pill at-risk" title="At risk">At risk</span>', html)
        self.assertIn(pv.FIT['at_risk'], html)
        # The facts line, then the blank lab as its own line in the doctor home's words (triage.missing_tests).
        self.assertIn('<p class="track-panel-note">BMI 31.2 · former smoker</p>'
                      '<p class="track-panel-note">HbA1c not entered · ask for an HbA1c test</p>', html)

    def test_an_urgent_result_puts_a_visit_first_but_still_allows_a_plan(self):
        save_screening(self.p['id'], 'diabetes', 'GradientBoosting', .8, 1, .5, {**AT_RISK, 'HbA1c_level': 7.1})
        at = self.page()
        html = self.html(at)
        self.assertIn('<div class="track-panel urgent">', html)
        self.assertIn('<span class="track-pill urgent" title="Urgent">Urgent</span>', html)
        self.assertIn(pv.FIT['urgent'], html)
        self.assertIn("The patient's HbA1c of 7.1% is at or above 6.5%", html)
        self.assertNotIn('HbA1c not entered', html)
        approve = at.button(key=f"prevention_approve_{self.p['id']}")
        self.assertFalse(approve.disabled)
        self.approve(at)
        self.assertEqual(active_plan(self.p['id'])['status'], 'active')

    def test_a_child_gets_the_child_line_and_no_target(self):
        save_screening(self.p['id'], 'diabetes', 'GradientBoosting', .55, 1, .5, {**AT_RISK, 'age': 12})
        at = self.page()
        html = self.html(at)
        self.assertIn(pv.FIT['urgent_child'], html)
        self.assertNotIn(pv.FIT['at_risk'], html)
        # HbA1c was left blank, but the tests to ask for come from adult cut-offs, so a child gets none.
        self.assertNotIn('not entered', html)
        self.assertTrue(all(self.widget(at, 'checkbox', label).disabled for label in INCLUDE.values()))
        self.assertIn(pv.CHILD_REASON, self.captions(at))

    def test_the_starting_point_picks_the_condition_and_a_missing_screening_is_said_plainly(self):
        at = self.page()
        start = self.widget(at, 'selectbox', 'Starting point')
        self.assertEqual((start.value, start.options), ('diabetes', [pv.STARTING_POINTS[d].label for d in pv.STARTING_POINTS]))
        self.assertIn(pv.STARTING_POINTS['diabetes'].advice, self.captions(at))
        start.set_value('kidney').run()
        self.assertIn('No saved Kidney Health screening yet', self.html(at))
        self.assertIn('Start from standard advice for chronic kidney disease prevention', self.html(at))
        self.assertIn(pv.why_text('kidney', 'fruit_veg'), self.captions(at))
        # A newer at-risk result for another condition is where a new plan starts.
        save_screening(self.p['id'], 'heart', 'LogisticRegression', .1, 0, .5, {})
        save_screening(self.p['id'], 'kidney', 'RandomForest', .9, 1, .5, {'Age': 50, 'GFR': 90, 'ACR': 10, 'Smoking': 1})
        at = self.page()
        self.assertEqual(self.widget(at, 'selectbox', 'Starting point').value, 'kidney')
        self.assertEqual((self.widget(at, 'checkbox', INCLUDE['smoking']).value, self.widget(at, 'checkbox', INCLUDE['smoking']).disabled), (True, False))


class TemplateTests(Fixture):
    def test_the_targets_start_from_the_template(self):
        at = self.page()
        self.assertEqual({k: (self.widget(at, 'checkbox', label).value, self.widget(at, 'checkbox', label).disabled) for k, label in INCLUDE.items()},
                         {'weight': (True, False), 'activity': (True, False), 'fruit_veg': (True, False), 'sleep': (True, False),
                          'smoking': (False, True)})
        self.assertEqual([self.goal(at, k).value for k in GOAL], [79.0, 150, 5, (7.0, 9.0)])
        markdown = [m.value for m in self.tab(at).markdown]
        for now in ('84 kg', '60 min', '2 a day', '6 h', 'Former smoker'):self.assertIn(now, markdown)
        # The Now and Goal cells are keyed, so a narrow screen can label them once the header row is hidden.
        ids = [getattr(b.proto, 'id', '') for b in blocks(self.tab(at))]
        for cell in ('now', 'goal'):
            self.assertEqual([k for k in INCLUDE if any(i.endswith(f'-prevention_{cell}_{k}') for i in ids)], list(INCLUDE))
        for words in ('Not needed: former smoker', 'About 6% lower (5–7% is the usual aim)', 'Standard weekly guidance for adults'):
            self.assertIn(words, self.captions(at))
        recheck = pv.date_text(local_today() + timedelta(weeks=12))
        self.assertEqual(self.widget(at, 'number_input', 'Re-check (weeks)').value, 12)
        self.assertIn(f'Screen again on **{recheck}**', markdown)
        self.assertEqual(self.preview(at), ['Body weight · 84 → 79 kg', 'Activity · 150 min a week', 'Fruit and vegetables · 5 a day',
                                            'Sleep · 7–9 h a night'])
        html = self.html(at)
        self.assertIn('Your plan with Dr. Sarah Ahmed', html)
        self.assertIn(f'Agreed {pv.date_text(local_today())} · Re-check on {recheck}', html)
        self.assertIn('No earlier plans for this patient. Every change will be kept here.', self.captions(at))
        self.assertEqual(self.buttons(at), ['Approve and send to Alice', 'Save draft'])

    def test_the_why_and_the_preview_follow_the_doctor_edits(self):
        at = self.page()
        self.goal(at, 'weight').set_value(78.0).run()
        self.assertIn('About 7% lower (5–7% is the usual aim)', self.captions(at))
        self.widget(at, 'checkbox', INCLUDE['fruit_veg']).uncheck().run()
        self.assertTrue(self.goal(at, 'fruit_veg').disabled)
        self.widget(at, 'number_input', 'Re-check (weeks)').set_value(8).run()
        self.widget(at, 'text_area', 'Note for Alice').set_value('Walk after dinner.').run()
        self.assertEqual(self.preview(at), ['Body weight · 84 → 78 kg', 'Activity · 150 min a week', 'Sleep · 7–9 h a night'])
        html = self.html(at)
        self.assertIn(f'Re-check on {pv.date_text(local_today() + timedelta(weeks=8))}', html)
        self.assertIn('“Walk after dinner.” — Dr. Sarah Ahmed', html)
        self.approve(at)
        current = active_plan(self.p['id'])
        self.assertEqual([(t['key'], t['goal']) for t in current['targets']], [('weight', 78.0), ('activity', 150), ('sleep', [7.0, 9.0])])
        self.assertEqual((current['note'], current['recheck_date']), ('Walk after dinner.', (local_today() + timedelta(weeks=8)).isoformat()))

    def test_a_ticked_target_that_stops_being_offered_is_cleared_and_the_plan_still_approves(self):
        at = self.page()
        self.assertTrue(self.widget(at, 'checkbox', INCLUDE['weight']).value)
        # The patient logs a lower weight (BMI 24.5) while the doctor has the tab open; the doctor then edits the note.
        save_lifestyle(self.p['id'], {**LOG, 'weight_kg': 65.9}, SCORES)
        self.widget(at, 'text_area', 'Note for Alice').set_value('Keep going.').run()
        weight = self.widget(at, 'checkbox', INCLUDE['weight'])
        self.assertEqual((weight.value, weight.disabled), (False, True))
        self.assertIn('Not offered: BMI 24.5 is under 25', self.captions(at))
        self.assertEqual(self.preview(at), ['Activity · 150 min a week', 'Fruit and vegetables · 5 a day', 'Sleep · 7–9 h a night'])
        self.approve(at)
        self.assertEqual([e.value for e in self.tab(at).error], [])
        self.assertEqual([t['key'] for t in active_plan(self.p['id'])['targets']], ['activity', 'fruit_veg', 'sleep'])

    def test_a_weight_goal_is_needed_when_no_weight_is_logged(self):
        with connection() as db:db.execute('DELETE FROM lifestyle')
        at = self.page()
        weight = self.widget(at, 'checkbox', INCLUDE['weight'])
        self.assertEqual((weight.value, weight.disabled, self.goal(at, 'weight').value), (False, False, None))
        self.assertIn('No weight logged yet: enter a goal', self.captions(at))
        weight.check().run()
        self.approve(at)
        self.assertIn('Enter a body weight goal between 25 and 400 kg.', [e.value for e in self.tab(at).error])
        self.assertEqual(prevention_plans(self.p['id'], self.d['id']), [])
        self.goal(at, 'weight').set_value(80.0).run()
        self.approve(at)
        self.assertEqual(active_plan(self.p['id'])['targets'][0], {'key': 'weight', 'goal': 80.0, 'baseline': None})


class PreviewTests(Fixture):
    def test_the_preview_names_the_doctor_and_shows_the_note_as_my_plan_does(self):
        """'What <patient> will see' is what My plan prints, word for word: the doctor's name as stored (here without a
        title, where adding 'Dr.' on one side only would show) in the title and under the note, with its line breaks."""
        with connection() as db:db.execute('UPDATE users SET name=? WHERE id=?', ('Sarah Ahmed', self.d['id']))
        self.d['name'] = 'Sarah Ahmed'
        at = self.page()
        self.widget(at, 'text_area', 'Note for Alice').set_value('Walk after dinner.\nOne step at a time.').run()
        html = self.html(at)
        text = lambda markup:' '.join(re.sub('<[^>]+>', ' ', markup).split())
        title = text(re.search(r'<div class="plan-preview-title">(.*?)</div>', html).group(1))
        quote = text(re.search(r'<blockquote class="plan-preview-note">(.*?)</blockquote>', html, re.S).group(1))
        self.assertEqual((title, quote), ('Your plan with Sarah Ahmed', '“Walk after dinner. One step at a time.” — Sarah Ahmed'))
        self.assertIn('Walk after dinner.<br>One step at a time.', html)
        self.approve(at)
        mine = AppTest.from_file(APP, default_timeout=120)
        for key, value in {'authenticated': True, 'user': self.p, 'nav': 'medications'}.items():mine.session_state[key] = value
        patient = text(unescape('\n'.join(h.proto.body for h in self.ok(mine.run()).get('html'))))
        self.assertIn(title, patient)
        self.assertIn(quote, patient)


class ApproveTests(Fixture):
    def test_approving_activates_the_plan_notifies_and_writes_the_tracker_goals(self):
        set_goals(self.p['id'], {'steps': 9000, 'water': 2.5, 'sleep': 6.5})
        at = self.approve(self.page())
        plan, = prevention_plans(self.p['id'], self.d['id'])
        self.assertEqual((plan['status'], plan['disease'], plan['screening_id']), ('active', 'diabetes', self.screening))
        self.assertIn('Plan approved and sent to Alice. Their tracker goals now match it.', [s.value for s in self.tab(at).success])
        self.assertEqual([n['title'] for n in self.plan_notes()], ['New prevention plan'])
        self.assertEqual(get_goals(self.p['id']), {'steps': 9000, 'water': 2.5, 'sleep': 8.0, 'weight': 79.0, 'exercise': 150, 'fruit_veg': 5})
        # The plan stays open with its status, actions, progress and history.
        self.assertEqual(at.selectbox(key=f"prevention_plan_{self.p['id']}").value, plan['id'])
        self.assertEqual(self.pills(at), [('#2E9477', 'Active')])
        self.assertEqual(self.buttons(at), ['Pause', 'Stop this plan', 'Save changes and notify Alice'])
        self.assertTrue(self.widget(at, 'selectbox', 'Starting point').disabled)
        html = self.html(at)
        self.assertIn('<h2>Progress</h2>', html)
        tiles = re.findall(r'<span class="summary-label">([^<]+)</span><strong class="summary-value">([^<]+)</strong>\s*<p class="summary-note">([^<]*)</p>', html)
        self.assertEqual([t[:2] for t in tiles], [('Body weight', '— / 79 kg'), ('Activity', '60 / 150 min'), ('Fruit and vegetables', '2 / 5 a day'),
                                                  ('Sleep', '6 / 7–9 h')])
        self.assertEqual(tiles[0][2], 'Not logged yet since the target started')
        self.assertEqual(tiles[1][2], "Behind · Weekly estimate from this week's check-in")
        self.assertIn('<h2>What Alice sees</h2>', html)
        self.assertTrue(any(c.endswith(' · Approved and sent') for c in self.captions(at)))

    def test_a_new_plan_replaces_the_current_one_and_keeps_it_in_the_history(self):
        first = approve_prevention_plan(self.d['id'], self.p['id'], self.plan())
        at = self.page()
        box = at.selectbox(key=f"prevention_plan_{self.p['id']}")
        self.assertEqual(box.value, first)
        box.set_value(0).run()
        self.assertTrue(any('Approving replaces the current plan' in c for c in self.captions(at)))
        self.approve(at)
        current, old = prevention_plans(self.p['id'], self.d['id'])
        self.assertEqual((current['status'], old['id'], old['status']), ('active', first, 'stopped'))
        self.assertEqual(prevention_history(self.d['id'], first)[-1]['text'], 'Stopped: replaced by a newer plan')
        labels = at.selectbox(key=f"prevention_plan_{self.p['id']}").options
        self.assertTrue(labels[0].startswith('Diabetes · Active · since '))
        self.assertEqual(labels[1], 'New plan (replaces the current plan when approved)')
        self.assertTrue(labels[2].startswith('Diabetes · Stopped · '))
        self.assertIn('1 earlier plan, now stopped. Choose one under Prevention plan to read its history.', self.captions(at))


class DraftTests(Fixture):
    def test_a_draft_is_never_shown_to_the_patient_until_it_is_approved(self):
        at = self.page()
        self.goal(at, 'fruit_veg').set_value(6).run()
        self.ok(at.button(key=f"prevention_draft_{self.p['id']}").click().run())
        draft, = prevention_plans(self.p['id'], self.d['id'])
        self.assertEqual(draft['status'], 'draft')
        self.assertIn('Draft saved. Alice sees nothing until you approve it.', [s.value for s in self.tab(at).success])
        self.assertEqual((prevention_plans(self.p['id']), active_plan(self.p['id']), self.plan_notes(), get_goals(self.p['id'])), ([], None, [], {}))
        # The draft reopens with its goals and can be discarded behind a confirmation.
        at = self.page()
        self.assertEqual(self.pills(at), [('#8E7CC3', 'Draft')])
        self.assertEqual(self.goal(at, 'fruit_veg').value, 6)
        self.assertEqual(self.buttons(at), ['Discard this draft', 'Approve and send to Alice', 'Save draft'])
        self.approve(at)
        plan, = prevention_plans(self.p['id'], self.d['id'])
        self.assertEqual((plan['id'], plan['status'], plan['targets'][2]['goal']), (draft['id'], 'active', 6))
        self.assertEqual(len(self.plan_notes()), 1)

    def test_discarding_a_draft_returns_to_a_new_plan(self):
        draft = save_prevention_draft(self.d['id'], self.p['id'], self.plan())
        at = self.ok(self.page().button(key=f'prevention_discard_{draft}').click().run())
        self.assertIn('Draft discarded.', [s.value for s in self.tab(at).success])
        self.assertEqual(prevention_plans(self.p['id'], self.d['id'])[0]['status'], 'stopped')
        self.assertEqual((self.pills(at), self.buttons(at), self.plan_notes()), ([], ['Approve and send to Alice', 'Save draft'], []))


class EditTests(Fixture):
    def test_an_edit_keeps_the_earlier_version_in_the_history(self):
        pid = approve_prevention_plan(self.d['id'], self.p['id'], self.plan())
        at = self.page()
        save = lambda:self.ok(at.button(key=f'prevention_save_{pid}').click().run())
        save()
        self.assertIn('No changes to save.', [i.value for i in self.tab(at).info])
        self.goal(at, 'activity').set_value(180).run()
        self.widget(at, 'text_area', 'Note for Alice').set_value('Two short walks a day.').run()
        save()
        self.assertIn('Plan updated; Alice was notified.', [s.value for s in self.tab(at).success])
        history = prevention_history(self.d['id'], pid)
        self.assertEqual([h['action'] for h in history], ['approved', 'edited'])
        self.assertEqual(history[1]['details']['before']['targets'][1], {'key': 'activity', 'goal': 150, 'baseline': 60})
        self.assertEqual(get_goals(self.p['id'])['exercise'], 180)
        self.assertEqual([n['title'] for n in self.plan_notes()], ['Prevention plan updated', 'New prevention plan'])
        edited = next(c for c in self.captions(at) if ' · Edited' in c)
        self.assertTrue(edited.endswith(' · Edited: Activity 150 min → 180 min; note changed'), edited)
        self.assertEqual(self.preview(at)[1], 'Activity · 180 min a week')
        # A kept weight target keeps its starting value though the patient logged a lower weight since.
        save_lifestyle(self.p['id'], {**LOG, 'weight_kg': 60}, SCORES)
        at = self.page()
        weight = self.widget(at, 'checkbox', INCLUDE['weight'])
        self.assertEqual((weight.value, weight.disabled), (True, False))
        self.goal(at, 'weight').set_value(78.0).run()
        self.ok(at.button(key=f'prevention_save_{pid}').click().run())
        self.assertEqual(active_plan(self.p['id'])['targets'][0], {'key': 'weight', 'goal': 78.0, 'baseline': 84})


class StatusTests(Fixture):
    def test_pause_resume_and_stop_behind_a_confirmation(self):
        pid = approve_prevention_plan(self.d['id'], self.p['id'], self.plan())
        at = self.page()
        stop, = [p for p in self.tab(at).get('popover') if p.proto.popover.label == 'Stop']
        self.assertEqual(stop.proto.popover.icon, ':material/stop_circle:')
        self.assertEqual([b.label for b in stop.button], ['Stop this plan'])
        self.assertTrue(any('cannot be resumed' in m.value for m in stop.markdown))
        self.assertTrue(any(getattr(b.proto, 'id', '').endswith(f'-plan_stop_prevention_{pid}') for b in blocks(stop)))
        self.ok(at.button(key=f'prevention_pause_{pid}').click().run())
        self.assertEqual((active_plan(self.p['id'])['status'], self.pills(at)), ('paused', [('#B7821E', 'Paused')]))
        self.assertIn('Plan paused; Alice was notified. Their tracker goals stay as they are.', [s.value for s in self.tab(at).success])
        self.assertEqual(self.buttons(at), ['Resume', 'Stop this plan', 'Save changes and notify Alice'])
        set_goals(self.p['id'], {'exercise': 60})
        self.ok(at.button(key=f'prevention_resume_{pid}').click().run())
        self.assertEqual((active_plan(self.p['id'])['status'], get_goals(self.p['id'])['exercise']), ('active', 150))
        self.ok(at.button(key=f'prevention_stop_{pid}').click().run())
        self.assertIn('Plan stopped; Alice was notified.', [s.value for s in self.tab(at).success])
        self.assertEqual((prevention_plans(self.p['id']), prevention_plans(self.p['id'], self.d['id'])[0]['status']), ([], 'stopped'))
        self.assertEqual([n['title'] for n in self.plan_notes()],
                         ['Prevention plan stopped', 'Prevention plan resumed', 'Prevention plan paused', 'New prevention plan'])
        # A stopped plan stays selected and read-only.
        self.assertEqual((at.selectbox(key=f"prevention_plan_{self.p['id']}").value, self.pills(at)), (pid, [('#5B8FA3', 'Stopped')]))
        self.assertEqual(self.buttons(at), [])
        self.assertTrue(all(w.disabled for kind in ('checkbox', 'number_input', 'slider', 'text_area') for w in getattr(self.tab(at), kind)
                            if w.label != 'Search name or email'))
        self.assertIn('A stopped plan is read-only. Choose New plan to agree a new one; this one stays in the history.', self.captions(at))
        self.assertIn('<h2>What Alice saw</h2>', self.html(at))


class AccessTests(Fixture):
    def test_a_revoked_doctor_no_longer_sees_or_changes_the_plan(self):
        ben = self.user('b@test.com', 'Patient', 'Ben Cole', link=True)
        pid = approve_prevention_plan(self.d['id'], self.p['id'], self.plan())
        at = self.page()
        revoke_doctor(self.p['id'], self.d['id'])
        # A page opened before the revoke: the click finds Alice gone from the picker, so nothing is paused.
        self.ok(at.button(key=f'prevention_pause_{pid}').click().run())
        self.assertEqual(at.selectbox(key='plan_patient').options, [f"Ben Cole · ID {ben['id']} · b@test.com"])
        self.assertEqual(prevention_plans(self.p['id']), [])
        with connection() as db:
            actions = [r['action'] for r in db.execute('SELECT action FROM prevention_audit WHERE plan_id=? ORDER BY id', (pid,))]
        self.assertEqual(actions, ['approved', 'stopped'])
        self.assertNotIn(f'prevention_pause_{pid}', [b.key for b in at.button])
        for call in (lambda:update_prevention_plan(self.d['id'], pid, self.plan(note='Later')),
                     lambda:prevention_plan_status(self.d['id'], pid, 'active'),
                     lambda:prevention_plans(self.p['id'], self.d['id']),
                     lambda:approve_prevention_plan(self.d['id'], self.p['id'], self.plan())):
            with self.assertRaises(PermissionError):call()


class EscapingTests(Fixture):
    def test_names_and_notes_show_as_text(self):
        with connection() as db:db.execute("UPDATE users SET name=? WHERE id=?", ('<i>Eve</i>* Mansour', self.p['id']))
        self.d['name'] = 'Dr <b>Lee</b>'
        note = '<img src=x onerror=alert(1)> **bold** [link](javascript:alert(1))'
        at = self.page()
        self.widget(at, 'text_area', f"Note for {markdown_literal('<i>Eve</i>*')}").set_value(note).run()
        raw = '\n'.join(h.proto.body for h in self.tab(at).get('html'))
        self.assertNotIn('<i>Eve', raw)
        self.assertNotIn('<img src=x', raw)
        self.assertNotIn('<b>Lee', raw)
        self.assertIn('<h2>What &lt;i&gt;Eve&lt;/i&gt;* will see</h2>', raw)
        self.assertIn('“&lt;img src=x onerror=alert(1)&gt; **bold** [link](javascript:alert(1))” — Dr &lt;b&gt;Lee&lt;/b&gt;', raw)
        self.assertIn(f"Approve and send to {markdown_literal('<i>Eve</i>*')}", self.buttons(at))
        self.approve(at)
        self.assertIn(f"Plan approved and sent to {markdown_literal('<i>Eve</i>*')}. Their tracker goals now match it.",
                      [s.value for s in self.tab(at).success])
        self.assertEqual(active_plan(self.p['id'])['note'], note)


if __name__ == '__main__':unittest.main()
