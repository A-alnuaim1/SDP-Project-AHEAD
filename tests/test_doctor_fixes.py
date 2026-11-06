"""Doctor-side fixes: medication times, schedule history, pause timing, revoked doctors, the plan form and Patient Intelligence."""
import json
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date,datetime,timedelta,timezone
from unittest import mock
from zoneinfo import ZoneInfo

from streamlit.proto.Metric_pb2 import Metric
from streamlit.testing.v1 import AppTest

import ahead.platform_data as platform_data
from ahead.storage import init_db,create_user,authenticate,connection,insert_records,save_screening
from ahead.platform_data import (init_platform,grant_doctor,revoke_doctor,save_plan,plan_status,plans,
    scheduled_doses,mark_dose,notifications,local_today,activity_counts,event,open_doses)
from ahead.pages.doctor_intelligence import NO_PATIENTS
from ahead.pages.medications import refresh_medication_notifications

APP=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'app.py')


def readable(day):
    """A local date as the doctor pages show it, e.g. '4 Nov 2026'."""
    return f'{day.day} {day:%b %Y}'


@contextmanager
def clock(text):
    """Pretend the Dubai wall clock reads text ('YYYY-MM-DD HH:MM') for stamps and due checks."""
    moment=datetime.strptime(text,'%Y-%m-%d %H:%M').replace(tzinfo=ZoneInfo('Asia/Dubai'))
    with mock.patch.multiple(platform_data,now=lambda:moment.astimezone(timezone.utc).isoformat(),
                             local_now=lambda:moment,local_today=lambda:moment.date()):
        yield


class DoctorFixture(unittest.TestCase):
    """A patient linked to a doctor on a fresh temporary database."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH']=self.temp.name+'/app.db'
        init_db();init_platform()
        for email,role in [('p@test.com','Patient'),('d@test.com','Doctor')]:create_user(email,'safe-password-123',role,email)
        self.p=authenticate('p@test.com','safe-password-123','Patient')
        self.d=authenticate('d@test.com','safe-password-123','Doctor')
        grant_doctor(self.p['id'],'d@test.com')
    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH',None)
    def page(self,user,nav):
        at=AppTest.from_file(APP,default_timeout=120)
        for key,value in {'authenticated':True,'user':user,'nav':nav}.items():at.session_state[key]=value
        at.run();self.assertFalse(at.exception,[e.message for e in at.exception])
        return at


class DoctorFixTests(DoctorFixture):
    def plan(self,**changes):
        return {'name':'Metformin','dosage':'500 mg','instructions':'','times':['08:00'],'days':list(range(7)),
                'start_date':'2026-03-01','end_date':'2026-03-31','notes':'',**changes}
    def history(self,start,end,doctor=True):
        return [(x['date'],x['time'],x['status']) for x in scheduled_doses(self.p['id'],date.fromisoformat(start),date.fromisoformat(end),self.d['id'] if doctor else None)]

    def test_unpadded_times_are_stored_padded_and_loggable_on_their_day(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['20:30','8:00']))
        self.assertEqual(plans(self.p['id'])[0]['times'],['08:00','20:30'])
        with self.assertRaisesRegex(ValueError,'HH:MM'):save_plan(self.d['id'],self.p['id'],self.plan(times=['25:00']))
        with clock('2026-03-01 13:50'):
            mark_dose(self.p['id'],pid,'2026-03-01','8:00','taken')
            with self.assertRaisesRegex(ValueError,'not due yet'):mark_dose(self.p['id'],pid,'2026-03-01','20:30','taken')
            self.assertEqual(self.history('2026-03-01','2026-03-01'),[('2026-03-01','08:00','taken'),('2026-03-01','20:30','upcoming')])

    def test_legacy_unpadded_plan_is_read_and_logged_as_padded(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['08:00','20:30']))
        with connection() as db:
            db.execute("""UPDATE medication_plans SET times_json='["20:30", "8:00"]' WHERE id=?""",(pid,))
            db.execute("UPDATE medication_audit SET details_json=replace(details_json,'08:00','8:00') WHERE plan_id=?",(pid,))
        self.assertEqual(plans(self.p['id'])[0]['times'],['08:00','20:30'])
        with clock('2026-03-02 09:00'):
            mark_dose(self.p['id'],pid,'2026-03-02','8:00','taken')
            self.assertEqual(self.history('2026-03-02','2026-03-02'),[('2026-03-02','08:00','taken'),('2026-03-02','20:30','upcoming')])

    def test_editing_a_plan_keeps_past_doses_under_their_old_schedule(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan())
        for day in ('01','02','03'):
            with clock(f'2026-03-{day} 08:30'):mark_dose(self.p['id'],pid,f'2026-03-{day}','08:00','taken')
        # 4 March is a Wednesday; the new schedule is 09:00 on weekdays only.
        with clock('2026-03-04 10:00'):save_plan(self.d['id'],self.p['id'],self.plan(times=['09:00'],days=[0,1,2,3,4]),pid)
        with clock('2026-03-05 12:00'):
            self.assertEqual(self.history('2026-03-01','2026-03-09'),[
                ('2026-03-01','08:00','taken'),('2026-03-02','08:00','taken'),('2026-03-03','08:00','taken'),
                ('2026-03-04','08:00','missed'),('2026-03-05','09:00','missed'),('2026-03-06','09:00','upcoming'),('2026-03-09','09:00','upcoming')])
            save_plan(self.d['id'],self.p['id'],self.plan(times=['09:00'],start_date='2026-03-06'),pid)
            mark_dose(self.p['id'],pid,'2026-03-04','08:00','skipped')
            self.assertEqual(self.history('2026-03-01','2026-03-05')[-2:],[('2026-03-04','08:00','skipped'),('2026-03-05','09:00','missed')])

    def test_logged_dose_survives_a_same_day_correction_without_adding_missed_doses(self):
        with clock('2026-03-01 07:00'):
            pid=save_plan(self.d['id'],self.p['id'],self.plan(start_date='2026-02-26'))
            mark_dose(self.p['id'],pid,'2026-02-26','08:00','taken')
        with clock('2026-03-01 07:30'):save_plan(self.d['id'],self.p['id'],self.plan(start_date='2026-02-26',times=['09:00']),pid)
        with clock('2026-03-02 12:00'):
            self.assertEqual(self.history('2026-02-26','2026-03-02'),[('2026-02-26','08:00','taken'),('2026-03-01','09:00','missed'),('2026-03-02','09:00','missed')])

    def test_pause_keeps_doses_due_before_it_and_resume_restores_later_ones(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['00:00','20:00']))
        with clock('2026-03-02 13:50'):
            mark_dose(self.p['id'],pid,'2026-03-02','00:00','taken')
            plan_status(self.d['id'],pid,'paused')
            for doctor in (True,False):self.assertEqual(self.history('2026-03-02','2026-03-02',doctor),[('2026-03-02','00:00','taken')])
        with clock('2026-03-02 14:00'):
            mark_dose(self.p['id'],pid,'2026-03-01','20:00','skipped')
            with self.assertRaises(PermissionError):mark_dose(self.p['id'],pid,'2026-03-02','20:00','taken')
        with clock('2026-03-03 12:00'):
            plan_status(self.d['id'],pid,'active')
            self.assertEqual(self.history('2026-03-01','2026-03-03'),[('2026-03-01','00:00','missed'),('2026-03-01','20:00','skipped'),
                                                                      ('2026-03-02','00:00','taken'),('2026-03-03','20:00','upcoming')])

    def test_stop_between_midnight_and_4am_dubai_keeps_the_previous_day(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['00:00']))
        with clock('2026-03-02 01:00'):
            for day in ('2026-03-01','2026-03-02'):mark_dose(self.p['id'],pid,day,'00:00','taken')
        with clock('2026-03-03 02:30'):
            plan_status(self.d['id'],pid,'stopped')
            with connection() as db:self.assertTrue(db.execute("SELECT created_at FROM medication_audit WHERE plan_id=? ORDER BY id DESC",(pid,)).fetchone()['created_at'].startswith('2026-03-02T22:30'))
            self.assertEqual(self.history('2026-03-01','2026-03-04'),[('2026-03-01','00:00','taken'),('2026-03-02','00:00','taken'),('2026-03-03','00:00','missed')])

    def test_revoking_a_doctor_stops_their_plans(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['08:00','20:00']))
        with clock('2026-03-02 12:00'):
            revoke_doctor(self.p['id'],self.d['id'])
            self.assertEqual([x['status'] for x in plans(self.p['id'])],['stopped'])
            self.assertEqual(self.history('2026-03-02','2026-03-03',doctor=False),[('2026-03-02','08:00','missed')])
        with self.assertRaises(PermissionError):plan_status(self.d['id'],pid,'active')
        self.assertIn('Medication plan stopped',[n['title'] for n in notifications(self.p['id'])])
        with connection() as db:self.assertEqual(db.execute("SELECT action FROM medication_audit WHERE plan_id=? ORDER BY id DESC",(pid,)).fetchone()['action'],'stopped')

    def test_paused_and_stopped_plans_send_no_dose_reminders(self):
        today=local_today();span=dict(start_date=(today-timedelta(days=1)).isoformat(),end_date=(today+timedelta(days=5)).isoformat())
        a=save_plan(self.d['id'],self.p['id'],self.plan(name='Aspirin',times=['00:00','00:01'],**span))
        save_plan(self.d['id'],self.p['id'],self.plan(name='Warfarin',times=['00:00','00:01'],**span))
        plan_status(self.d['id'],a,'paused');revoke_doctor(self.p['id'],self.d['id'])
        refresh_medication_notifications(self.p['id'])
        self.assertFalse([n['description'] for n in notifications(self.p['id']) if n['type'] in ('medication_due','medication_missed')])

    def test_status_changes_only_notify_when_the_status_changes(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan())
        with clock('2026-03-01 07:05'):self.assertEqual([plan_status(self.d['id'],pid,s) for s in ('active','paused','paused','active','active','stopped','stopped')],[False,True,False,True,False,True,False])
        with connection() as db:self.assertEqual([r['action'] for r in db.execute('SELECT action FROM medication_audit WHERE plan_id=? ORDER BY id',(pid,))],['created','paused','active','stopped'])
        self.assertEqual(sum(n['title']=='Medication plan status' for n in notifications(self.p['id'])),3)
        for status in ('paused','active'):
            with self.assertRaisesRegex(ValueError,'stopped plan cannot be paused or resumed'):plan_status(self.d['id'],pid,status)

    def test_plans_orphaned_by_an_earlier_revoke_are_stopped_once(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan())
        with connection() as db:db.execute('DELETE FROM care_access')
        self.assertEqual(plans(self.p['id'])[0]['status'],'active')
        init_platform();init_platform()
        self.assertEqual(plans(self.p['id'])[0]['status'],'stopped')
        self.assertEqual(sum(n['title']=='Medication plan stopped' for n in notifications(self.p['id'])),1)
        with connection() as db:self.assertEqual(db.execute("SELECT COUNT(*) c FROM medication_audit WHERE plan_id=? AND action='stopped'",(pid,)).fetchone()['c'],1)

    def test_stopped_plan_cannot_be_edited_and_unchanged_save_sends_nothing(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan())
        before=len(notifications(self.p['id']))
        with clock('2026-03-01 07:05'):self.assertEqual(save_plan(self.d['id'],self.p['id'],self.plan(name=' Metformin ',times=['8:00']),pid),pid)
        self.assertEqual(len(notifications(self.p['id'])),before)
        with clock('2026-03-01 07:10'):plan_status(self.d['id'],pid,'stopped')
        before=len(notifications(self.p['id']))
        with clock('2026-03-01 07:15'),self.assertRaisesRegex(ValueError,'stopped plan cannot be edited'):
            save_plan(self.d['id'],self.p['id'],self.plan(dosage='1000 mg'),pid)
        self.assertEqual((plans(self.p['id'])[0]['dosage'],len(notifications(self.p['id']))),('500 mg',before))

    def test_doctor_form_confirms_save_keeps_the_plan_selected_and_avoids_duplicates(self):
        at=self.page(self.d,'intelligence')
        field=lambda label:next(t for t in at.text_input if t.label==label)
        plan_box=lambda:next(s for s in at.selectbox if s.label=='Plan')
        field('Medication name').set_value('Lisinopril');field('Dosage').set_value('10 mg')
        field('Exact times (HH:MM, comma separated)').set_value('8:00, 20:30')
        next(b for b in at.button if b.label=='Save medication plan').click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        saved=plans(self.p['id'])
        self.assertEqual([(x['name'],x['times']) for x in saved],[('Lisinopril',['08:00','20:30'])])
        self.assertIn('Medication plan saved and the patient notified.',[s.value for s in at.success])
        self.assertEqual((plan_box().value,field('Medication name').value),(saved[0]['id'],'Lisinopril'))
        next(b for b in at.button if b.label=='Save medication plan').click().run()
        self.assertEqual(len(plans(self.p['id'])),1)
        self.assertIn('No changes to save.',[i.value for i in at.info])
        self.assertEqual(sum('created a plan' in n['description'] for n in notifications(self.p['id'])),1)
        # Stop sits in a confirmation popover, so its button reads 'Stop this plan'.
        status_buttons=lambda:[b.label for b in at.button if b.label in ('Pause','Resume','Stop this plan')]
        self.assertEqual(status_buttons(),['Pause','Stop this plan'])
        next(b for b in at.button if b.label=='Pause').click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual((plans(self.p['id'])[0]['status'],plan_box().value),('paused',saved[0]['id']))
        self.assertIn('Lisinopril is now paused; the patient was notified.',[s.value for s in at.success])
        self.assertEqual(status_buttons(),['Resume','Stop this plan'])
        next(b for b in at.button if b.label=='Stop this plan').click().run()
        self.assertEqual((plans(self.p['id'])[0]['status'],plan_box().value,status_buttons()),('stopped',saved[0]['id'],[]))
        form=['Medication name','Dosage','Instructions','Exact times (HH:MM, comma separated)','Scheduled days','Start date','End date','Doctor notes']
        widgets=lambda:[w for w in [*at.text_input,*at.text_area,*at.multiselect,*at.date_input] if w.label in form]
        self.assertEqual([w.disabled for w in widgets()],[True]*8)
        self.assertTrue(next(b for b in at.button if b.label=='Save medication plan').disabled)
        self.assertTrue(any('stopped plan is read-only' in c.value for c in at.caption))
        plan_box().set_value(0).run()
        self.assertFalse(any(w.disabled for w in widgets()) or next(b for b in at.button if b.label=='Save medication plan').disabled)
        self.assertEqual(sum(n['title']=='Medication plan status' for n in notifications(self.p['id'])),2)

    def test_a_schedule_edit_starts_the_next_day_and_the_patient_is_told(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(times=['20:00']))
        with clock('2026-03-04 10:00'):save_plan(self.d['id'],self.p['id'],self.plan(times=['08:00']),pid)
        # Moving the dose earlier keeps today's 20:00 dose instead of leaving the day with none.
        with clock('2026-03-04 10:05'):self.assertEqual(self.history('2026-03-03','2026-03-05'),[
            ('2026-03-03','20:00','missed'),('2026-03-04','20:00','upcoming'),('2026-03-05','08:00','upcoming')])
        with clock('2026-03-04 10:10'):save_plan(self.d['id'],self.p['id'],self.plan(times=['08:00'],notes='With food'),pid)
        updates=[n['description'] for n in notifications(self.p['id']) if n['title']=='Medication plan updated']
        self.assertEqual(["today's doses stay as they were" in u for u in updates],[False,True,False])

    def test_moving_a_taken_dose_later_does_not_ask_for_a_second_dose_that_day(self):
        with clock('2026-03-01 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan())
        with clock('2026-03-04 08:30'):mark_dose(self.p['id'],pid,'2026-03-04','08:00','taken')
        with clock('2026-03-04 12:00'):
            save_plan(self.d['id'],self.p['id'],self.plan(times=['20:00']),pid)
            self.assertEqual(self.history('2026-03-04','2026-03-05'),[('2026-03-04','08:00','taken'),('2026-03-05','20:00','upcoming')])
            self.assertEqual(open_doses(self.p['id'],date(2026,3,4),date(2026,3,4)),[])
            with self.assertRaises(PermissionError):mark_dose(self.p['id'],pid,'2026-03-04','20:00','taken')

    def test_a_plan_created_today_can_be_corrected_for_today(self):
        with clock('2026-03-04 07:00'):pid=save_plan(self.d['id'],self.p['id'],self.plan(start_date='2026-03-04'))
        with clock('2026-03-04 07:30'):
            save_plan(self.d['id'],self.p['id'],self.plan(start_date='2026-03-04',times=['09:00']),pid)
            self.assertEqual(self.history('2026-03-04','2026-03-04'),[('2026-03-04','09:00','upcoming')])
        self.assertFalse(any("stay as they were" in n['description'] for n in notifications(self.p['id'])))

    def test_patient_page_explains_a_plan_whose_doctor_lost_access(self):
        today=local_today()
        save_plan(self.d['id'],self.p['id'],self.plan(times=['00:00'],start_date=(today-timedelta(days=1)).isoformat(),end_date=(today+timedelta(days=5)).isoformat()))
        revoke_doctor(self.p['id'],self.d['id'])
        # Today's 00:00 dose was due before the stop: it stays in the history but is no longer a to-do.
        self.assertEqual([(x['time'],x['status']) for x in scheduled_doses(self.p['id'],today,today)],[('00:00','missed')])
        self.assertEqual(open_doses(self.p['id'],today,today),[])
        at=self.page(self.p,'medications')
        self.assertIn('**Metformin · 500 mg** — stopped',[m.value for m in at.markdown])
        self.assertTrue(any('no longer has access' in c.value for c in at.caption))
        self.assertIn('This plan was stopped after this dose was due. Record it only if you already took it; do not take it now.',[w.value for w in at.warning])

    def test_rejected_new_plan_stays_with_its_patient(self):
        create_user('q@test.com','safe-password-123','Patient','Quinn');grant_doctor(authenticate('q@test.com','safe-password-123','Patient')['id'],'d@test.com')
        at=self.page(self.d,'intelligence')
        field=lambda label:next(t for t in at.text_input if t.label==label)
        picker=lambda:next(s for s in at.selectbox if s.key=='med_patient')
        field('Medication name').set_value('Lisinopril');field('Dosage').set_value('10 mg');field('Exact times (HH:MM, comma separated)').set_value('25:00')
        next(b for b in at.button if b.label=='Save medication plan').click().run()
        self.assertTrue(any('25:00' in e.value for e in at.error))
        picker().select_index(1-picker().index).run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual([field(label).value for label in ('Medication name','Dosage','Exact times (HH:MM, comma separated)')],['','','08:00'])


class PatientIntelligenceTests(DoctorFixture):
    def screening(self,disease,model,score,local=None,values=None):
        sid=save_screening(self.p['id'],disease,model,score,int(score>=.5),.5,{'age':30} if values is None else values)
        if local:
            stamp=datetime.strptime(local,'%Y-%m-%d %H:%M').replace(tzinfo=ZoneInfo('Asia/Dubai')).astimezone(timezone.utc).isoformat()
            with connection() as db:db.execute('UPDATE screenings SET created_at=? WHERE id=?',(stamp,sid))
        return sid
    def patient(self,email,name,link=True):
        create_user(email,'safe-password-123','Patient',name);user=authenticate(email,'safe-password-123','Patient')
        if link:grant_doctor(user['id'],'d@test.com')
        return user
    def score_metric(self,at):
        return next(((m.value,m.proto.delta,m.proto.color,m.proto.direction) for m in at.metric if m.label=='Latest vs previous model score'),None)
    def frame(self,at,column):
        return next(df.value for df in at.dataframe if column in df.value.columns)
    def xaxis(self,at,title):
        axis=next(json.loads(c.proto.spec)['layout']['xaxis'] for c in at.get('plotly_chart') if json.loads(c.proto.spec)['layout'].get('title',{}).get('text')==title)
        return axis.get('type'),axis.get('dtick')
    def tiles(self,at):
        html='\n'.join(h.proto.body for h in at.get('html'))
        return dict(re.findall(r'class="summary-label">([^<]+)</span><strong class="summary-value">(\d+)',html))
    def chart_titles(self,at):
        return [json.loads(c.proto.spec)['layout'].get('title',{}).get('text') for c in at.get('plotly_chart')]

    def test_score_change_compares_the_same_condition_and_model_with_rising_risk_in_red(self):
        first=self.screening('diabetes','GradientBoosting',.45);self.screening('kidney','RandomForest',.20);latest=self.screening('diabetes','GradientBoosting',.55)
        at=self.page(self.d,'intelligence')
        self.assertEqual(self.score_metric(at),('55.0%','+10.0 percentage points',Metric.RED,Metric.UP))
        self.assertTrue(any(f'#{latest} compared with #{first}' in c.value for c in at.caption))
        next(s for s in at.selectbox if s.label=='Filter disease').set_value('kidney').run()
        self.assertIsNone(self.score_metric(at))
        self.screening('diabetes','GradientBoosting',.40)
        at=self.page(self.d,'intelligence')
        self.assertEqual(self.score_metric(at),('40.0%','-15.0 percentage points',Metric.GREEN,Metric.DOWN))
        self.screening('heart','LogisticRegression',.30)
        at=self.page(self.d,'intelligence')
        self.assertIsNone(self.score_metric(at))
        self.assertTrue(any('no earlier result from the same condition and model' in c.value for c in at.caption))

    def test_unchanged_score_has_no_colour_or_arrow(self):
        self.screening('kidney','RandomForest',.20);self.screening('kidney','RandomForest',.1996)
        self.assertEqual(self.score_metric(self.page(self.d,'intelligence')),('20.0%','No change',Metric.GRAY,Metric.NONE))

    def test_day_counts_and_record_dates_use_dubai_time(self):
        today=local_today();week=today-timedelta(days=6)
        # Before 04:00 in Dubai the UTC date is still the previous day.
        ids=[self.screening('diabetes','GradientBoosting',.4,local) for local in (f'{week} 02:00',f'{today} 01:30',f'{today} 01:45')]
        at=self.page(self.d,'intelligence')
        tiles=self.tiles(at)
        self.assertEqual((tiles['Today'],tiles['Last 7 days']),('2','3'))
        self.assertEqual([d.date() for d in self.frame(at,'Condition')['Date']],[today,today,week])
        self.assertEqual([d.strftime('%Y-%m-%d %H:%M') for d in self.frame(at,'ID')['Date']],[f'{today} 01:45',f'{today} 01:30',f'{week} 02:00'])
        self.assertEqual(next(s for s in at.selectbox if s.label=='Open historical assessment').options[0],f'{readable(today)}, 01:45 · Diabetes · ID {ids[2]}')
        self.assertTrue(any(f'**Diabetes**: Gradient boosting score 40.0% (not flagged) on {readable(today)}, 01:45;' in m.value for m in at.markdown))

    def test_doctor_views_use_readable_names(self):
        self.screening('kidney','RandomForest',.2,f'{local_today()-timedelta(days=1)} 09:00');self.screening('diabetes','GradientBoosting',.4)
        event(self.p['id'],'food_saved')
        at=self.page(self.d,'intelligence')
        self.assertEqual(self.frame(at,'ID')['Condition'].tolist(),['Diabetes','Kidney Health'])
        history=next(json.loads(c.proto.spec) for c in at.get('plotly_chart') if 'Assessment history' in json.loads(c.proto.spec)['layout'].get('title',{}).get('text',''))
        self.assertEqual(sorted(t['name'] for t in history['data']),['Diabetes · Gradient boosting','Kidney Health · Random forest'])
        self.assertEqual(self.frame(at,'Threshold (%)')['Patient'].tolist(),[f"p@test.com (ID {self.p['id']})"])
        self.assertTrue(any(c.value.startswith(f'Recorded patient activity on {readable(local_today())}: Meal saved 1.') for c in at.caption))

    def test_records_chart_needs_two_distinct_minutes(self):
        self.screening('diabetes','GradientBoosting',.4,f'{local_today()} 10:15');self.screening('kidney','RandomForest',.2,f'{local_today()} 10:15')
        at=self.page(self.d,'intelligence')
        self.assertFalse([t for t in self.chart_titles(at) if t and 'Assessment history' in t])
        self.assertIn(f'All 2 assessments were saved at {readable(local_today())}, 10:15; the history chart appears once they span more than one minute.',[c.value for c in at.caption])

    def test_activity_counts_cover_every_event_in_the_dubai_30_day_window(self):
        q=self.patient('q@test.com','Q');r=self.patient('r@test.com','R',link=False);s=self.patient('s@test.com','S')
        with connection() as db:
            # 1 March 23:30 Dubai is outside the 30 days ending 31 March; 2 March 00:30 Dubai is inside.
            db.execute("INSERT INTO activity_events(patient_id,type,created_at) VALUES(?,'login','2026-03-01T19:30:00+00:00')",(q['id'],))
            db.execute("INSERT INTO activity_events(patient_id,type,created_at) VALUES(?,'food_saved','2026-03-01T20:30:00+00:00')",(s['id'],))
            # 30 March 23:59 Dubai stays on 30 March although it is counted by its UTC hour.
            db.execute("INSERT INTO activity_events(patient_id,type,created_at) VALUES(?,'food_saved','2026-03-30T19:59:00+00:00')",(q['id'],))
        with clock('2026-03-31 08:00'):
            for user in (self.p,r,self.p,r):event(user['id'],'login')
        with clock('2026-03-31 09:00'):
            for _ in range(520):event(q['id'],'food_saved')
        with clock('2026-03-31 12:00'):
            self.assertEqual(activity_counts(self.d['id'])[:2],(3,1))
            self.assertEqual(dict(activity_counts(self.d['id'])[2]),{(date(2026,3,2),'food_saved'):1,(date(2026,3,30),'food_saved'):1,(date(2026,3,31),'login'):2,(date(2026,3,31),'food_saved'):520})
        with clock('2026-04-01 12:00'):self.assertEqual(activity_counts(self.d['id'])[:2],(2,1))

    def test_overview_tiles_count_patients_beyond_the_newest_500_events(self):
        q=self.patient('q@test.com','Q')
        for _ in range(2):event(self.p['id'],'login')
        for _ in range(520):event(q['id'],'food_saved')
        at=self.page(self.d,'intelligence')
        tiles=self.tiles(at)
        self.assertEqual((tiles['Active (30 days)'],tiles['Returning']),('2','1'))
        self.assertTrue(any(c.value.startswith(f'Recorded patient activity on {readable(local_today())}: Meal saved 520, Sign-in 2.') for c in at.caption))

    def test_trends_need_two_days(self):
        self.screening('diabetes','GradientBoosting',.4);self.screening('diabetes','GradientBoosting',.6);event(self.p['id'],'login')
        at=self.page(self.d,'intelligence')
        self.assertFalse({'Saved assessment activity','Assessments over time','Recorded patient activity'}&set(self.chart_titles(at)))
        captions=[c.value for c in at.caption]
        for title in ('Saved assessment activity','Assessments over time'):self.assertIn(f'{title}: 2 assessments, all on {readable(local_today())}. The trend appears once assessments span two or more days.',captions)
        self.assertTrue(any(c.startswith(f'Recorded patient activity on {readable(local_today())}: Sign-in 1.') for c in captions))
        self.screening('diabetes','GradientBoosting',.5,f'{local_today()-timedelta(days=1)} 12:00')
        with connection() as db:db.execute("UPDATE activity_events SET created_at=?",((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),))
        event(self.p['id'],'login')
        at=self.page(self.d,'intelligence')
        # A few days get one tick per day rather than Plotly's hour ticks; longer spans keep automatic day-or-wider ticks.
        for title in ('Saved assessment activity','Assessments over time','Recorded patient activity'):self.assertEqual(self.xaxis(at,title),('date',86400000))
        self.screening('diabetes','GradientBoosting',.5,f'{local_today()-timedelta(days=20)} 12:00')
        at=self.page(self.d,'intelligence')
        for title in ('Saved assessment activity','Assessments over time'):self.assertEqual(self.xaxis(at,title),(None,None))

    def test_data_quality_reports_success_only_after_checking_data(self):
        create_user('e@test.com','safe-password-123','Doctor','Dr Empty')
        at=self.page(authenticate('e@test.com','safe-password-123','Doctor'),'intelligence')
        # One notice per tab (seven with Prevention Plan); the eighth info is the page's standing note about who appears here.
        self.assertEqual((len(at.success),[i.value for i in at.info].count(NO_PATIENTS),len(at.info)),(0,7,8))
        at=self.page(self.d,'intelligence')
        self.assertEqual(len(at.success),0)
        self.assertTrue(any('nothing to check' in i.value for i in at.info))
        self.screening('diabetes','GradientBoosting',.4)
        at=self.page(self.d,'intelligence')
        self.assertEqual([s.value for s in at.success],['No detected issues in linked patient assessments or uploaded records.'])

    def test_data_quality_patient_column_is_unambiguous_text(self):
        self.screening('diabetes','GradientBoosting',.4,values={'age':30,'HbA1c_level':None})
        a=self.patient('a@test.com','Sam Lee');b=self.patient('b@test.com','Sam Lee')
        # An uploaded PatientID equal to an account ID must not read as that account.
        insert_records(self.d['id'],'cohort','diabetes',[{'patient_id':pid,'values':{'age':40},'dob':None,'issues':['Missing value for bmi']} for pid in (str(self.p['id']),'P-1')])
        with self.assertNoLogs('streamlit.dataframe_util','INFO'):at=self.page(self.d,'intelligence')
        self.assertEqual(sorted(self.frame(at,'Severity')['Patient']),sorted([f"Sam Lee (ID {a['id']})",f"Sam Lee (ID {b['id']})",f"p@test.com (ID {self.p['id']})",f"Upload · {self.p['id']}",'Upload · P-1']))


if __name__=='__main__':unittest.main()
