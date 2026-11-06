"""Patient Intelligence restyle: the doctor-home focus contract, summary tiles, plan actions and readable records."""
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime,time,timedelta,timezone
from unittest import mock
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

import ahead.pages.doctor_intelligence as page
import ahead.platform_data as platform_data
from ahead.resources import icon_uri
from ahead.storage import init_db,create_user,authenticate,connection,save_screening
from ahead.platform_data import init_platform,grant_doctor,save_plan,plan_status,plans,mark_dose,local_today,event,ai_consent,save_lifestyle

APP=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'app.py')
FORM=['Medication name','Dosage','Instructions','Exact times (HH:MM, comma separated)','Scheduled days','Start date','End date','Doctor notes']


def blocks(node):
    for child in getattr(node,'children',{}).values():
        yield child
        yield from blocks(child)


def distance(a,b):
    """CIE76 colour difference between two hex colours."""
    def lab(colour):
        r,g,b=(c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in (int(colour[i:i+2],16)/255 for i in (1,3,5)))
        f=lambda t:t**(1/3) if t>.008856 else 7.787*t+16/116
        x,y,z=f((.4124*r+.3576*g+.1805*b)/.95047),f(.2126*r+.7152*g+.0722*b),f((.0193*r+.1192*g+.9505*b)/1.08883)
        return 116*y-16,500*(x-y),200*(y-z)
    return sum((p-q)**2 for p,q in zip(lab(a),lab(b)))**.5


@contextmanager
def clock(moment):
    """Pretend the Dubai wall clock reads moment for stamps, due checks, the page and this module."""
    today=lambda:moment.date()
    with mock.patch.multiple(platform_data,now=lambda:moment.astimezone(timezone.utc).isoformat(),local_now=lambda:moment,local_today=today),\
         mock.patch.object(page,'local_today',today),mock.patch.object(sys.modules[__name__],'local_today',today):yield


class IntelligenceFixture(unittest.TestCase):
    """A doctor with one linked patient on a fresh temporary database."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH']=self.temp.name+'/app.db'
        init_db();init_platform()
        create_user('d@test.com','safe-password-123','Doctor','Dr Dana');self.d=authenticate('d@test.com','safe-password-123','Doctor')
        self.p=self.patient('p@test.com','Alice Mansour')
    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH',None)
    def patient(self,email,name,link=True):
        create_user(email,'safe-password-123','Patient',name);user=authenticate(email,'safe-password-123','Patient')
        if link:grant_doctor(user['id'],'d@test.com')
        return user
    def screening(self,disease,model,score,values=None,days_ago=0):
        sid=save_screening(self.p['id'],disease,model,score,int(score>=.5),.5,{'age':30} if values is None else values)
        if days_ago:
            with connection() as db:db.execute('UPDATE screenings SET created_at=? WHERE id=?',((datetime.now(timezone.utc)-timedelta(days=days_ago)).isoformat(),sid))
        return sid
    def plan(self,name='Metformin',times=('00:00','00:01')):
        today=local_today()
        return save_plan(self.d['id'],self.p['id'],{'name':name,'dosage':'500 mg','instructions':'','times':list(times),'days':list(range(7)),
            'start_date':(today-timedelta(days=5)).isoformat(),'end_date':(today+timedelta(days=20)).isoformat(),'notes':''})
    def page(self,tab=None,patient=None):
        at=AppTest.from_file(APP,default_timeout=120)
        state={'authenticated':True,'user':self.d,'nav':'intelligence'}
        if tab:state['intelligence_focus']={'tab':tab,'patient':patient}
        for key,value in state.items():at.session_state[key]=value
        at.run();self.assertFalse(at.exception,[e.message for e in at.exception])
        return at
    def tab_box(self,at):
        return next(b for b in blocks(at.main) if getattr(b,'type',None)=='tab_container' and [t.label for t in b.children.values()]==page.TABS)
    def tabs_place(self,at):
        """The tabs' position in the page; the browser keys stateless tabs by it, so a move would remount them."""
        return next(i for i,b in at.main.children.items() if b is self.tab_box(at))
    def open_tab(self,at):
        return page.TABS[self.tab_box(at).proto.tab_container.default_tab_index]
    def pills(self,at):
        """(tone, label) of each plan status pill."""
        return re.findall(r'<span class="visual-badge" style="--visual-color:(#\w+)">.*?</span> (\w+)</span>',self.html(at))
    def html(self,at):
        """Rendered page HTML without the injected stylesheets."""
        return '\n'.join(h.proto.body for h in at.get('html') if not h.proto.body.lstrip().startswith('<style'))
    def tiles(self,at):
        """(label, tone, icon, value, note) for each summary tile, in page order."""
        return [(label,tone,icon,value,note) for tone,icon,label,value,note in re.findall(
            r'<div class="summary-inner" style="--tone:(#\w+)">\s*<div class="summary-icon"><img src="([^"]+)" alt=""></div>\s*'
            r'<span class="summary-label">([^<]+)</span><strong class="summary-value">(.*?)</strong>.*?<p class="summary-note">([^<]*)</p>',self.html(at),re.S)]
    def frame(self,at,column):
        return next(df.value for df in at.dataframe if column in df.value.columns)
    def chart(self,at,title):
        return next(json.loads(c.proto.spec) for c in at.get('plotly_chart') if json.loads(c.proto.spec)['layout'].get('title',{}).get('text')==title)
    def medication(self,pid=None):
        at=self.page('Medication Monitoring',self.p['id'])
        if pid:at.selectbox(key=f"med_plan_{self.p['id']}").set_value(pid).run()
        return at
    def actions(self,at):
        return [b.label for b in at.button if b.label in ('Pause','Resume','Stop','Stop this plan')]
    def stop_popover(self,at):
        return [b for b in at.get('popover') if b.proto.popover.label=='Stop']


class FocusTests(IntelligenceFixture):
    def test_focus_opens_the_tab_and_selects_the_patient_in_every_picker(self):
        ben=self.patient('b@test.com','Ben Cole')
        at=self.page('Medication Monitoring',ben['id'])
        self.assertEqual(self.open_tab(at),'Medication Monitoring')
        self.assertEqual({key:at.selectbox(key=key).value for key in page.PICKERS},dict.fromkeys(page.PICKERS,ben['id']))
        self.assertNotIn('intelligence_focus',at.session_state)
        # Stateless tabs (no widget id) switch in the browser without a rerun, and the browser keeps the open tab on
        # later runs while the tabs stay in place; the pickers keep the patient.
        place=self.tabs_place(at);at.run()
        self.assertEqual((self.tab_box(at).proto.tab_container.id,self.tabs_place(at),at.selectbox(key='records_patient').value),('',place,ben['id']))
        # Without a focus the page opens on Overview with the first patient, as before.
        at=self.page()
        self.assertEqual((self.open_tab(at),at.selectbox(key='med_patient').value),('Overview',self.p['id']))

    def test_a_patient_who_is_no_longer_linked_or_an_unknown_tab_does_not_fail(self):
        carl=self.patient('c@test.com','Carl Diaz',link=False)
        at=self.page('Clinical Insights',carl['id'])
        self.assertEqual((self.open_tab(at),at.selectbox(key='insight_patient').value),('Clinical Insights',self.p['id']))
        self.assertIn('That patient is no longer linked to you, so their records are not shown.',[i.value for i in at.info])
        # The notice goes on the next run without moving the tabs, so the browser does not reopen them on Overview.
        place=self.tabs_place(at);at.run()
        self.assertEqual(self.tabs_place(at),place)
        self.assertNotIn('That patient is no longer linked to you, so their records are not shown.',[i.value for i in at.info])
        self.assertEqual(self.open_tab(self.page('Somewhere else',None)),'Overview')


class DesignTests(IntelligenceFixture):
    def test_overview_tiles_headings_and_note_match_the_dashboard(self):
        self.screening('diabetes','GradientBoosting',.7);self.screening('kidney','RandomForest',.2);self.screening('diabetes','GradientBoosting',.3,days_ago=10)
        for _ in range(2):event(self.p['id'],'login')
        at=self.page()
        tiles=self.tiles(at)[:8]
        icons={'Linked patients':'patients','Active (30 days)':'activity','Returning':'returning','Saved assessments':'screenings',
               'Today':'calendar','Last 7 days':'scored','Last 30 days':'review','Flagged':'flagged'}
        self.assertEqual([t[0] for t in tiles],list(icons))
        self.assertEqual([t[2] for t in tiles],[icon_uri(f'dashboard/{i}.svg') for i in icons.values()])
        self.assertEqual(len(set(icons.values())),8)
        self.assertEqual([t[3] for t in tiles],['1','1','1','3','2','2','3','1'])
        self.assertEqual(len({t[4] for t in tiles}),8)
        self.assertIn(('Returning','Signed in on 2+ separate occasions'),[(t[0],t[4]) for t in tiles])
        self.assertTrue({t[1] for t in tiles}<={'#8E7CC3','#2E9477','#C98F1E','#C9706E','#0E9AA7','#14556B','#5B8FA3'})
        html=self.html(at)
        for old in ('visual-metric','✦','#2563EB','#8B5CF6','#06B6D4','#F97316'):self.assertNotIn(old,html)
        self.assertEqual(len(at.metric),0)
        self.assertEqual((len(at.subheader),len(at.warning),len(at.get('json'))),(0,0,0))
        self.assertTrue(at.info[0].value.startswith('Only patients who add your doctor email'))
        for heading in ('Saved assessments','Recent assessments','Medication plans','Pre-consultation record summary'):
            self.assertIn(f'<h2>{heading}</h2>',html)
        # The Data Quality tab was removed; its one useful idea (ask for a lab left blank) moved to the records view.
        self.assertNotIn('Data Quality',page.TABS);self.assertNotIn('Data-quality checklist',html)
        # Subtitles describe what is shown: all 3 assessments, and the assessment events stacked in the activity chart.
        self.assertIn('<h2>Recent assessments</h2><p>Every assessment your linked patients saved, newest first.</p>',html)
        self.assertIn('<p>Sign-ins, saved assessments, meals, lifestyle check-ins, challenge days and dose logs from linked patients.</p>',html)
        self.assertEqual(sorted(self.frame(at,'Model')['Model']),['Gradient boosting','Gradient boosting','Random forest'])
        # The page speaks to the doctor as 'you', never as 'this doctor'.
        captions=[c.value for c in at.caption]
        self.assertFalse([c for c in captions if 'this doctor' in c])
        self.assertIn('Counts include only patients who explicitly linked to you and screenings they chose to save. The classification is not a diagnosis.',captions)
        self.assertIn('Only records the patient allowed you to see are included.',captions)

    def test_overview_explains_when_no_assessment_is_saved_yet(self):
        at=self.page()
        overview=next(t for t in at.tabs if t.label=='Overview')
        self.assertIn('Your linked patients have not saved any disease assessments yet. The charts and the recent list appear after the first one.',
                      [i.value for i in overview.info])
        self.assertFalse(overview.get('plotly_chart'))
        self.screening('diabetes','GradientBoosting',.4)
        self.assertFalse([i for i in next(t for t in self.page().tabs if t.label=='Overview').info if 'not saved any' in i.value])

    def test_recent_assessments_say_when_only_the_newest_fifteen_are_listed(self):
        for _ in range(16):self.screening('diabetes','GradientBoosting',.4)
        at=self.page()
        self.assertIn('<h2>Recent assessments</h2><p>The 15 newest of the 16 assessments saved by your linked patients.</p>',self.html(at))
        self.assertEqual(len(self.frame(at,'Model')),15)

    def test_charts_use_explicit_ahead_colours_and_daily_axes(self):
        for days_ago in (0,1):
            self.screening('diabetes','GradientBoosting',.6,days_ago=days_ago);self.screening('kidney','RandomForest',.2,days_ago=days_ago)
        self.screening('diabetes','LogisticRegression',.4)
        pid=self.plan();mark_dose(self.p['id'],pid,(local_today()-timedelta(days=1)).isoformat(),'00:00','taken')
        for kind in ('login','lifestyle_completed','food_saved'):event(self.p['id'],kind)
        with connection() as db:db.execute('UPDATE activity_events SET created_at=? WHERE id=1',((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),))
        at=self.medication()
        allowed={*page.CHART,*page.CONDITION_COLORS.values()}
        for chart in at.get('plotly_chart'):
            spec=json.loads(chart.proto.spec)
            for trace in spec['data']:
                colours=[*trace.get('marker',{}).get('colors',[]),trace.get('marker',{}).get('color'),trace.get('line',{}).get('color')]
                self.assertTrue({c for c in colours if c}, spec['layout']['title']['text'])
                self.assertTrue({c.upper() for c in colours if c}<=allowed,(spec['layout']['title']['text'],colours))
        activity={t['name']:t['marker']['color'] for t in self.chart(at,'Recorded patient activity')['data']}
        self.assertEqual(activity['Lifestyle check-in'],'#8E7CC3')
        # Stacked activity segments stay apart: every pair at least 15 apart in CIELAB (Health assessment and Dose taken were 5.7).
        colours=list(page.ACTIVITY_COLORS.values())
        self.assertGreaterEqual(min(distance(a,b) for i,a in enumerate(colours) for b in colours[i+1:]),15)
        # A dose status has one colour on the page: the activity chart's dose events match the dose timeline.
        self.assertEqual((page.ACTIVITY_COLORS['Dose taken'],page.ACTIVITY_COLORS['Dose skipped']),(page.DOSE_COLORS['Taken'],page.DOSE_COLORS['Skipped']))
        doses=self.chart(at,'Dose log timeline')
        self.assertTrue({t['name'] for t in doses['data']}<={'Taken','Skipped','Missed (not logged)'})
        for trace in doses['data']:self.assertEqual(trace['marker']['color'],page.DOSE_COLORS[trace['name']])
        axis=doses['layout']['xaxis']
        self.assertEqual((axis.get('type'),axis.get('dtick'),axis.get('tickformat')),('date',86400000,'%-d %b'))


class MedicationTests(IntelligenceFixture):
    def test_status_pill_and_only_the_valid_actions_with_stop_behind_a_confirmation(self):
        pid=self.plan()
        at=self.medication(pid)
        # The design system's token-based pill (readable in dark mode), not st.badge.
        self.assertEqual((self.pills(at),[m.value for m in at.markdown if 'badge[' in m.value]),([('#2E9477','Active')],[]))
        self.assertEqual(self.actions(at),['Pause','Stop this plan'])
        icons=lambda:{b.label:b.proto.icon for b in at.button if b.label in ('Pause','Resume')}
        self.assertEqual(icons(),{'Pause':':material/pause_circle:'})
        stop,=self.stop_popover(at)
        self.assertEqual(stop.proto.popover.icon,':material/stop_circle:')
        self.assertEqual([b.label for b in stop.button],['Stop this plan'])
        self.assertTrue(any('cannot be resumed' in m.value for m in stop.markdown))
        # Keyed, so the stylesheet can style this confirmation apart from the Start Screening menu.
        self.assertTrue(any(b.proto.id.endswith(f'-plan_stop_{pid}') for b in blocks(stop)))
        at.button(key=f'Pause_{pid}').click().run()
        self.assertEqual((plans(self.p['id'])[0]['status'],self.actions(at)),('paused',['Resume','Stop this plan']))
        self.assertEqual((self.pills(at),icons()),([('#B7821E','Paused')],{'Resume':':material/play_circle:'}))
        at.button(key=f'Stop_{pid}').click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual(plans(self.p['id'])[0]['status'],'stopped')
        self.assertIn('Metformin is now stopped; the patient was notified.',[s.value for s in at.success])

    def test_the_newest_active_plan_opens_first(self):
        older=self.plan('Metformin');newer=self.plan('Lisinopril');stopped=self.plan('Atorvastatin');plan_status(self.d['id'],stopped,'stopped')
        at=self.medication()
        self.assertEqual((at.selectbox(key=f"med_plan_{self.p['id']}").value,self.pills(at)),(newer,[('#2E9477','Active')]))
        self.assertEqual(next(t for t in at.text_input if t.label=='Medication name').value,'Lisinopril')
        # New plan stays one choice away and is kept once chosen.
        at.selectbox(key=f"med_plan_{self.p['id']}").set_value(0).run().run()
        self.assertEqual((at.selectbox(key=f"med_plan_{self.p['id']}").value,self.pills(at)),(0,[]))
        for pid in (older,newer):plan_status(self.d['id'],pid,'stopped')
        self.assertEqual(self.medication().selectbox(key=f"med_plan_{self.p['id']}").value,0)

    def test_a_stopped_plan_is_read_only(self):
        pid=self.plan();plan_status(self.d['id'],pid,'stopped')
        at=self.medication(pid)
        self.assertEqual(self.pills(at),[('#5B8FA3','Stopped')])
        self.assertEqual((self.actions(at),self.stop_popover(at)),([],[]))
        widgets=[w for w in [*at.text_input,*at.text_area,*at.multiselect,*at.date_input] if w.label in FORM]
        self.assertEqual([w.disabled for w in widgets],[True]*8)
        self.assertTrue(next(b for b in at.button if b.label=='Save medication plan').disabled)
        self.assertTrue(any('stopped plan is read-only' in c.value for c in at.caption))

    def test_dose_history_is_readable_newest_first_with_counts_per_plan(self):
        # Noon, so today's 00:00 and 00:01 doses are both due whenever the suite runs.
        with clock(datetime.combine(local_today(),time(12),tzinfo=ZoneInfo('Asia/Dubai'))):self.dose_history()

    def dose_history(self):
        today=local_today();pid=self.plan()
        mark_dose(self.p['id'],pid,(today-timedelta(days=1)).isoformat(),'00:00','taken')
        mark_dose(self.p['id'],pid,(today-timedelta(days=2)).isoformat(),'00:01','skipped')
        at=self.medication()
        # Missed (not logged) keeps the doctor home's 'Missed doses today' look, since that card opens this tab.
        home=AppTest.from_file(APP,default_timeout=120)
        for key,value in {'authenticated':True,'user':self.d,'nav':'overview'}.items():home.session_state[key]=value
        home.run()
        looks={t[0]:t[1:3] for t in self.tiles(at)}
        self.assertEqual((looks['Skipped'],looks['Missed (not logged)']),(('#C9706E',icon_uri('dashboard/flagged.svg')),('#C98F1E',icon_uri('dashboard/doses.svg'))))
        self.assertEqual(looks['Missed (not logged)'],{t[0]:t[1:3] for t in self.tiles(home)}['Missed doses today'])
        doses=self.frame(at,'Time')
        self.assertEqual(list(doses.columns),['Medication','Dosage','Date','Time','Status'])
        stamps=[(str(d)[:10],t) for d,t in zip(doses['Date'],doses['Time'])]
        self.assertEqual(stamps,sorted(stamps,reverse=True))
        self.assertEqual(stamps[0],(today.isoformat(),'00:01'))
        self.assertEqual(set(doses['Status']),{'Taken','Skipped','Missed (not logged)'})
        counts=self.frame(at,'Due doses').iloc[0].to_dict()
        self.assertEqual({k:counts[k] for k in ('Medication','Plan','Due doses','Taken','Skipped','Missed (not logged)')},
                         {'Medication':'Metformin','Plan':'Active','Due doses':12,'Taken':1,'Skipped':1,'Missed (not logged)':10})
        self.assertIn(('Taken','1 <small>of 12</small>'),[(t[0],t[3]) for t in self.tiles(at)])

    def test_every_table_serialises_without_arrow_fallbacks(self):
        self.screening('diabetes','GradientBoosting',.4,{'age':30,'bmi':None,'HbA1c_level':None})
        pid=self.plan();mark_dose(self.p['id'],pid,(local_today()-timedelta(days=1)).isoformat(),'00:00','taken')
        save_lifestyle(self.p['id'],{'sleep_hours':7},{'overall':70,'categories':{}})
        with self.assertNoLogs('streamlit.dataframe_util','INFO'):self.medication(pid)


class RecordTests(IntelligenceFixture):
    def test_historical_inputs_are_a_readable_table(self):
        diabetes=self.screening('diabetes','GradientBoosting',.4,{'gender':'Female','age':52,'hypertension':1,'heart_disease':0,'smoking_history':'never',
                                                                     'bmi':30.456,'HbA1c_level':None,'blood_glucose_level':None})
        self.screening('kidney','RandomForest',.3,{'Age':52,'Gender':1,'Smoking':0,'BMI':27.0,'SystolicBP':None,'GFR':80.0,'SerumCreatinine':1.1,'ACR':20})
        at=self.page()
        self.assertFalse(at.get('json'))
        inputs=lambda:dict(zip(*self.frame(at,'Input')[['Input','Value']].T.values))
        self.assertEqual({k:inputs()[k] for k in ('Gender','Smoking','Body Mass Index (BMI)','Systolic Blood Pressure (mmHg)','Estimated GFR')},
                         {'Gender':'Female','Smoking':'No','Body Mass Index (BMI)':'27','Systolic Blood Pressure (mmHg)':page.BLANK,'Estimated GFR':'80'})
        next(s for s in at.selectbox if s.label=='Open historical assessment').set_value(diabetes).run()
        self.assertEqual({k:inputs()[k] for k in ('Gender','History of Hypertension','History of Heart Disease','Body Mass Index (BMI)','HbA1c Level (%)')},
                         {'Gender':'Female','History of Hypertension':'Yes','History of Heart Disease':'No','Body Mass Index (BMI)':'30.46','HbA1c Level (%)':page.BLANK})
        self.assertNotIn('HbA1c_level',' '.join(self.frame(at,'Input')['Input']))

    def test_score_history_uses_day_month_ticks_and_a_compact_change_card(self):
        self.screening('diabetes','GradientBoosting',.45,days_ago=2);self.screening('diabetes','GradientBoosting',.55)
        # AppTest keeps no element width, so the call itself is checked.
        with mock.patch.object(page.st,'metric',wraps=page.st.metric) as metric:at=self.page()
        history=self.chart(at,'Assessment history · separate condition / model series');axis=history['layout']['xaxis']
        self.assertEqual(([s['value'] for s in axis['tickformatstops']],axis['hoverformat']),(['%-d %b, %H:%M','%-d %b'],'%-d %b %Y, %H:%M'))
        self.assertEqual(([m.value for m in at.metric],metric.call_args.kwargs['width']),(['55.0%'],'content'))
        # Keyed, so the stylesheet can draw it like the summary tiles.
        self.assertTrue(any(getattr(b.proto,'id','').endswith('-score_change') and b.children for b in blocks(at.main)))
        # Models read as words, never as class names.
        self.assertEqual([t['name'] for t in history['data']],['Diabetes · Gradient boosting'])
        self.assertTrue(any(c.value.startswith('Diabetes · Gradient boosting: assessment #') for c in at.caption))
        self.assertEqual(list(self.frame(at,'ID')['Model']),['Gradient boosting']*2)


class InsightTests(IntelligenceFixture):
    def ask(self,at,question):
        next(t for t in at.text_input if t.label.startswith('Ask for latest')).set_value(question).run()
        return next(t for t in at.tabs if t.label=='Clinical Insights').info[-1].value

    def test_summary_and_answers_cover_every_condition_and_say_what_is_missing(self):
        self.screening('diabetes','GradientBoosting',.45,{'age':30,'HbA1c_level':None},days_ago=2)
        self.screening('diabetes','GradientBoosting',.55,{'age':30,'HbA1c_level':6.1},days_ago=1)
        self.screening('kidney','RandomForest',.7)
        at=self.page('Clinical Insights',self.p['id'])
        summary=next(m.value for m in at.markdown if '**Medication**' in m.value)
        self.assertIn('- **Kidney Health**: Random forest score 70.0% (flagged) on ',summary)
        self.assertIn('no earlier result from the same model to compare.',summary)
        self.assertRegex(summary,r'- \*\*Diabetes\*\*: Gradient boosting score 55\.0% \(flagged\) on .*; \+10\.0 percentage points since assessment #\d+')
        self.assertIn('- **Medication**: No medication plan from you for this patient.',summary)
        self.assertIn('- **Lifestyle**: no lifestyle check-in recorded.',summary)
        self.assertNotIn('0 taken',summary)
        self.assertEqual(self.ask(at,'medication doses'),'No medication plan from you for this patient.')
        change=self.ask(at,'what changed since the previous one?')
        self.assertIn('Diabetes · Gradient boosting: +10.0 percentage points',change)
        self.assertIn('HbA1c Level (%): not provided → 6.1',change)
        # A plan with nothing due yet is still not a count of zero doses taken.
        today=local_today()
        save_plan(self.d['id'],self.p['id'],{'name':'Later','dosage':'1 tab','instructions':'','times':['08:00'],'days':list(range(7)),
            'start_date':(today+timedelta(days=2)).isoformat(),'end_date':(today+timedelta(days=9)).isoformat(),'notes':''})
        at=self.page('Clinical Insights',self.p['id'])
        self.assertIn('- **Medication**: No doses from your plans fell due in the last 30 days.',next(m.value for m in at.markdown if '**Medication**' in m.value))

    def test_ai_notice_names_the_missing_requirement(self):
        with mock.patch.object(page,'gemini_available',return_value=True):
            notes=' '.join(i.value for i in self.page('Clinical Insights',self.p['id']).info)
            self.assertIn('This patient has not allowed AI summaries. They can turn it on in their Settings → Preferences → Care team access.',notes)
            self.assertNotIn('not configured',notes)
            ai_consent(self.p['id'],self.d['id'],True)
            at=self.page('Clinical Insights',self.p['id'])
            self.assertIn('Generate optional Gemini pre-consultation summary',[b.label for b in at.button])
        with mock.patch.object(page,'gemini_available',return_value=False):
            notes=' '.join(i.value for i in self.page('Clinical Insights',self.p['id']).info)
            self.assertIn('AI summaries are not configured on this server',notes)
            self.assertNotIn('has not allowed',notes)


if __name__=='__main__':unittest.main()
