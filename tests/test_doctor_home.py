"""Doctor home (Overview): linked-patient cards, the attention list and links into Patient Intelligence; the patient home is unchanged."""
import os
import re
import tempfile
import unittest
from datetime import date,datetime,timedelta,timezone
from unittest import mock

from streamlit.testing.v1 import AppTest

import ahead.pages.overview as overview
from ahead.storage import init_db,create_user,authenticate,save_screening,connection
from ahead.platform_data import init_platform,grant_doctor,save_plan,plan_status,local_today

APP=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'app.py')
REMOVED=('Saved screenings','Reference records','Model families','Your screening options','Your screening history','Project data at a glance')


class DoctorHomeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH']=self.temp.name+'/app.db'
        init_db();init_platform()
        create_user('d@test.com','safe-password-123','Doctor','Dr. Layla Hassan')
        self.d=authenticate('d@test.com','safe-password-123','Doctor')
    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH',None)

    def patient(self,email,name,link=True):
        create_user(email,'safe-password-123','Patient',name);p=authenticate(email,'safe-password-123','Patient')
        if link:grant_doctor(p['id'],'d@test.com')
        return p
    def screening(self,patient,score,days_ago=0):
        sid=save_screening(patient['id'],'diabetes','TestModel',score,int(score>=.5),.5,{'age':30})
        if days_ago:
            with connection() as db:db.execute('UPDATE screenings SET created_at=? WHERE id=?',((datetime.now(timezone.utc)-timedelta(days=days_ago)).isoformat(),sid))
    def plan(self,patient,name):
        # A 00:00 dose is already due, so unlogged it is missed today whatever the time of the run.
        return save_plan(self.d['id'],patient['id'],{'name':name,'dosage':'500 mg','instructions':'','times':['00:00'],'days':list(range(7)),
            'start_date':(local_today()-timedelta(days=1)).isoformat(),'end_date':(local_today()+timedelta(days=2)).isoformat(),'notes':''})
    def page(self,user,**state):
        at=AppTest.from_file(APP,default_timeout=120)
        for key,value in {'authenticated':True,'user':user,'nav':'overview',**state}.items():at.session_state[key]=value
        at.run();self.assertFalse(at.exception,[e.message for e in at.exception])
        return at
    def html(self,at):
        return '\n'.join(h.proto.body for h in at.get('html'))
    def cards(self,at):
        found=re.findall(r'summary-label">([^<]+)</span><strong class="summary-value">(\d+)',self.html(at))
        return {label:int(value) for label,value in found}
    def keys(self,at):
        return {b.key for b in at.button}

    def test_doctor_without_patients_sees_how_patients_link(self):
        at=self.page(self.d)
        html=self.html(at)
        self.assertIn('AHEAD clinician dashboard',html)
        self.assertRegex(html,r'Good (morning|afternoon|evening), Dr\. Hassan</h1>')
        self.assertIn('Patients appear here once they add your doctor email in their settings.',html)
        self.assertEqual(self.cards(at),{'Linked patients':0,'Flagged · 30 days':0,'Missed doses today':0,'Assessments · 7 days':0})
        self.assertIn('Settings → Preferences → Care team access',html)
        self.assertIn('<div class="attention-calm neutral">\n    <div class="summary-icon"><img',html);self.assertIn('No linked patients yet</strong>',html)
        self.assertNotIn('◔',html)
        # No one is linked, so no card may read as if patients were checked ("All caught up today", "No flagged results").
        self.assertEqual(html.count('summary-note">No patients linked yet<'),4);self.assertNotIn('Saved by your patients',html)
        self.assertNotIn('All caught up today',html);self.assertNotIn('No flagged results',html)
        self.assertNotIn('Nothing needs your attention',html)
        for text in REMOVED:self.assertNotIn(text,html)
        self.assertTrue({'hero_intelligence','hero_upload','doctor_patients','doctor_flagged','doctor_doses','doctor_activity',
                         'doctor_home_diabetes','doctor_home_heart','doctor_home_kidney'}<=self.keys(at))
        self.assertFalse([k for k in self.keys(at) if k and k.startswith('home_open_')])
        self.assertFalse([e for e in at.expander if e.label=='Screenings you ran'])
        self.assertIn('Clinical notice',html)

    def test_linked_patients_fill_the_cards_and_the_attention_list(self):
        amy=self.patient('a@test.com','Amy <Lee>');ben=self.patient('b@test.com','Ben Cole');dan=self.patient('c@test.com','Dan Calm')
        cara=self.patient('x@test.com','Cara Unlinked',link=False)
        self.screening(amy,.8);self.screening(amy,.2);self.screening(amy,.9,days_ago=40)
        self.screening(dan,.3,days_ago=3);self.screening(cara,.9)
        self.plan(ben,'Metformin')
        # A paused plan's earlier dose stays in the history but is not a to-do.
        plan_status(self.d['id'],self.plan(ben,'Paused pill'),'paused')
        history=[{'time':datetime.now(timezone.utc),'disease':'heart','label':'Cardiovascular','model':'M','probability':.7,'prediction':1,'threshold':.5},
                 {'time':datetime.now(timezone.utc)-timedelta(hours=1),'disease':'kidney','label':'Kidney Health','model':'M','probability':.1,'prediction':0,'threshold':.5}]
        at=self.page(self.d,screening_history=history)
        html=self.html(at)
        self.assertEqual(self.cards(at),{'Linked patients':3,'Flagged · 30 days':1,'Missed doses today':1,'Assessments · 7 days':3})
        self.assertEqual(html.count('Across 1 patient<'),2)
        self.assertIn('2 patients need a look today.',html)
        self.assertIn('Amy &lt;Lee&gt;',html);self.assertNotIn('Amy <Lee>',html)
        self.assertRegex(html,r'Diabetes · 80% · today \d\d:\d\d<')
        self.assertIn('Metformin · 1 dose missed today',html);self.assertNotIn('Paused pill',html)
        self.assertNotIn('Cara',html);self.assertNotIn('Dan Calm',html)
        # Newest first: Amy's flag is from just now, Ben's dose was due at 00:00.
        self.assertLess(html.index('Amy &lt;Lee&gt;'),html.index('Ben Cole'))
        self.assertEqual((html.count('track-pill at-risk" title="At risk">At risk'),html.count('attention-pill dose">Missed doses')),(1,1))
        opens={k for k in self.keys(at) if k and k.startswith('home_open_')}
        self.assertEqual(opens,{f"home_open_{amy['id']}",f"home_open_{ben['id']}"})
        # Each button names its patient, so a screen reader does not hear "Open, Open".
        self.assertEqual((at.button(key=f"home_open_{amy['id']}").label,at.button(key=f"home_open_{ben['id']}").label),('Open Amy','Open Ben'))
        self.assertFalse([e for e in at.expander if e.label.endswith(('more patient','more patients'))])
        ran=next(e for e in at.expander if e.label=='Screenings you ran')
        self.assertEqual(list(ran.dataframe[0].value['Result']),['Flagged','Not flagged'])
        self.assertTrue(ran.caption[0].value.startswith('Your most recent screenings, newest first (up to ten).'))
        for text in REMOVED:self.assertNotIn(text,html)

    def test_calm_card_when_nothing_needs_attention(self):
        self.screening(self.patient('a@test.com','Amy Lee'),.2)
        html=self.html(self.page(self.d))
        self.assertIn('Nothing needs your attention</strong>',html)
        self.assertIn('Nothing needs your attention right now.',html)
        self.assertIn('across 1 linked patient.',html)
        # The icon takes the card's colour from the CSS (--calm), so the tile and the tint never differ.
        self.assertRegex(html,r'<div class="attention-calm ">\s*<div class="summary-icon"><img')

    def test_cards_and_rows_open_patient_intelligence_on_the_matching_tab(self):
        amy=self.patient('a@test.com','Amy Lee');ben=self.patient('b@test.com','Ben Cole')
        self.screening(amy,.8);self.plan(ben,'Metformin')
        self.assertIn('Open overview',self.html(self.page(self.d)))
        cases={'doctor_patients':('Overview',None),'doctor_flagged':('Patient Records & History',amy['id']),
               'doctor_doses':('Medication Monitoring',ben['id']),'doctor_activity':('Overview',None),
               f"home_open_{amy['id']}":('Patient Records & History',amy['id']),f"home_open_{ben['id']}":('Medication Monitoring',ben['id']),
               f"home_plan_{amy['id']}":('Prevention Plan',amy['id'])}
        for key,(tab,patient) in cases.items():
            with self.subTest(key=key):
                self.assertEqual(self.click(key),{'tab':tab,'patient':patient})
        # More than five patients: five rows, the rest (here Amy and the doses-only Ben) in an expander on the same page,
        # since Patient Intelligence has no such list; cards for several patients name none.
        for i in range(5):self.screening(self.patient(f'p{i}@test.com',f'Pat {i}'),.7)
        at=self.page(self.d)
        more=next(e for e in at.expander if e.label=='2 more patients')
        self.assertEqual([b.key for b in more.button if b.key.startswith('home_open_')],[f"home_open_{amy['id']}",f"home_open_{ben['id']}"])
        # Only an at-risk row (no urgent value, not doses alone) offers the prevention-plan shortcut.
        self.assertEqual([b.key for b in more.button if b.key.startswith('home_plan_')],[f"home_plan_{amy['id']}"])
        self.assertEqual(len([k for k in self.keys(at) if k and k.startswith('home_open_')]),7)
        self.assertIn('7 patients need a look today.',self.html(at))
        self.assertNotIn('home_attention_all',self.keys(at))
        self.assertEqual(self.click(f"home_open_{ben['id']}"),{'tab':'Medication Monitoring','patient':ben['id']})
        self.assertEqual(self.click('doctor_flagged'),{'tab':'Patient Records & History','patient':None})

    def click(self,key):
        """The focus Overview hands to Patient Intelligence when *key* is clicked (read before that page runs)."""
        seen=[];real=overview.go_to
        def spy(nav,*args,**kwargs):
            seen.append((nav,dict(overview.st.session_state['intelligence_focus'])));return real(nav,*args,**kwargs)
        at=self.page(self.d)
        with mock.patch.object(overview,'go_to',spy):at.button(key=key).click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual((len(seen),seen[0][0],at.session_state.nav),(1,'intelligence','intelligence'))
        return seen[0][1]

    def test_patient_home_is_unchanged(self):
        create_user('p@test.com','safe-password-123','Patient','Sara Ali')
        at=self.page(authenticate('p@test.com','safe-password-123','Patient'))
        html=self.html(at)
        self.assertIn('AHEAD Health Dashboard',html);self.assertIn('Welcome back, Sara',html)
        self.assertNotIn('clinician dashboard',html)
        self.assertEqual(set(self.cards(at)),{'Screenings saved','Goals set','Challenge days','Doses today'})
        self.assertTrue({'hero_ask','dashboard_screenings','dashboard_goals','dashboard_challenge','dashboard_doses'}<=self.keys(at))
        self.assertFalse({'hero_intelligence','hero_upload','doctor_patients'}&self.keys(at))


class DoctorHomeHelperTests(unittest.TestCase):
    def test_greeting_uses_the_last_name_with_one_title(self):
        for name,expected in [('Dr. Layla Hassan','Dr. Hassan'),('dr layla hassan','Dr. hassan'),('Layla','Dr. Layla'),('Doctor Test','Dr. Test'),('','Doctor'),
                              ('Drew Smith','Dr. Smith'),('Prof. Dr. Layla Hassan','Dr. Hassan'),('Dr.Layla Hassan','Dr. Hassan')]:
            self.assertEqual(overview._greeting_name(name),expected)
    def test_greeting_skips_credentials_and_brackets(self):
        for name,expected in [('Layla Hassan, MD','Dr. Hassan'),('Layla Hassan MD','Dr. Hassan'),('Dr. Layla Hassan PhD','Dr. Hassan'),
                              ('Layla Hassan M.D., FRCP','Dr. Hassan'),('Ahmed Al Mansoori (GP)','Dr. Al Mansoori'),('John Smith Jr.','Dr. Smith'),
                              ('Anh Do','Dr. Do'),('Ben Cole DO','Dr. Cole'),('Dr.','Doctor'),('MD','Doctor')]:
            with self.subTest(name=name):self.assertEqual(overview._greeting_name(name),expected)
    def test_greeting_keeps_family_name_particles(self):
        for name,expected in [('Dr. Mohammed Abu Bakr','Dr. Abu Bakr'),('Dr. Sara El Amin','Dr. El Amin'),('Ahmed bin Rashid Al Maktoum','Dr. Al Maktoum'),
                              ('Mohammed bin Salman','Dr. bin Salman'),('Maria de la Cruz, MD','Dr. de la Cruz'),('Jan van der Berg','Dr. van der Berg'),
                              ('Ali Abdul Rahman','Dr. Abdul Rahman'),('Dr. Ahmed Al-Mansoori','Dr. Al-Mansoori'),
                              # The first word is always a given name ("Al" as in Al Smith), never a particle.
                              ('Al Smith','Dr. Smith'),('Dr. AHMED AL MANSOORI','Dr. AL MANSOORI')]:
            with self.subTest(name=name):self.assertEqual(overview._greeting_name(name),expected)
    def test_times_read_as_today_yesterday_or_a_date(self):
        today=date(2026,11,4)
        self.assertEqual(overview._when(datetime(2026,11,4,9,40),today),'today 09:40')
        self.assertEqual(overview._when(datetime(2026,11,3,18,5),today),'yesterday 18:05')
        self.assertEqual(overview._when(datetime(2026,11,2,8,0),today),'2 Nov')
    def test_initials(self):
        self.assertEqual([overview._initials(n) for n in ('Amy Lee','Dr. Ben Cole','dan','')],['AL','BC','D','?'])
        # Letters only: punctuation, bracket notes and credentials never reach the avatar.
        self.assertEqual([overview._initials(n) for n in ('(Amy) Lee','"Ben" Cole','Amy Lee (Mum)','<b>Amy</b> & "Co"','Layla Hassan MBBS','---')],
                         ['AL','BC','AL','BC','LH','?'])
    def test_label_text_escapes_markdown(self):
        self.assertEqual(overview.markdown_literal('amy_lee*@x.com'),r'amy\_lee\*\@x\.com')
        self.assertEqual(overview.markdown_literal('Layla'),'Layla')
        self.assertEqual(overview.markdown_literal('![t](https://x.example/p.png)'),r'\!\[t\]\(https\:\/\/x\.example\/p\.png\)')


if __name__=='__main__':
    unittest.main()
