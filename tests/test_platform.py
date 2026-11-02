"""Integrated tests of additive migration, consent, dose controls and wellness persistence."""
import os
import sqlite3
import tempfile
import unittest
from datetime import date,timedelta

from ahead.storage import init_db,create_user,authenticate,save_screening,load_screenings
from ahead.platform_data import (init_platform,grant_doctor,allowed,linked_patients,patient_screenings,
    save_lifestyle,lifestyle_history,set_goals,get_goals,set_challenge_day,challenge_days,
    save_food,food_history,food_on_day,to_local,save_plan,plans,scheduled_doses,mark_dose,notify,notifications,
    record_prediction_history,append_record_prediction)
from ahead.wellness import evaluate,challenge_summary
from ahead.clinical_data import validate_row


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH']=self.temp.name+'/app.db'
        os.environ.pop('AHEAD_ENABLE_DEMO_ACCOUNTS',None)
        init_db();init_platform();init_db();init_platform()
        for email,role in [('p@test.com','Patient'),('q@test.com','Patient'),('d@test.com','Doctor'),('other@test.com','Doctor')]:
            create_user(email,'safe-password-123',role,email)
        self.p=authenticate('p@test.com','safe-password-123','Patient')['id']
        self.q=authenticate('q@test.com','safe-password-123','Patient')['id']
        self.d=authenticate('d@test.com','safe-password-123','Doctor')['id']
        self.other=authenticate('other@test.com','safe-password-123','Doctor')['id']
    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH',None)

    def test_consent_and_history_migration(self):
        save_screening(self.p,'diabetes','Model',.61,1,.5,{'age':32})
        save_screening(self.p,'diabetes','Model',.49,0,.5,{'age':33})
        self.assertEqual(len(load_screenings(self.p)),2)
        self.assertEqual(load_screenings(self.p)[0]['values_json'],'{"age": 33}')
        with self.assertRaises(PermissionError):patient_screenings(self.d,self.p)
        self.assertTrue(grant_doctor(self.p,'d@test.com'))
        self.assertEqual(len(patient_screenings(self.d,self.p)),2)
        self.assertFalse(allowed(self.other,self.p))
        self.assertEqual(linked_patients(self.other),[])
        self.assertEqual(load_screenings(self.q),[])

    def test_wellness_and_history(self):
        v={'weight_kg':75,'height_cm':175,'exercise_days':3,'exercise_minutes':30,'steps':7000,
           'sleep_hours':7,'sleep_quality':3,'water_litres':2,'fruit_veg':3,'fast_food':0,
           'sugary_food':1,'stress':4,'sedentary_hours':6}
        outcome=evaluate(v)
        self.assertTrue(0<=outcome['overall']<=100)
        save_lifestyle(self.p,v,outcome)
        self.assertEqual(len(lifestyle_history(self.p)),1)
        self.assertEqual(lifestyle_history(self.q),[])
        set_goals(self.p,{'steps':10000});self.assertEqual(get_goals(self.p)['steps'],10000)
        set_challenge_day(self.p,date.today().isoformat(),32)
        self.assertEqual(challenge_summary(challenge_days(self.p))['completed'],1)
        save_food(self.p,['rice'],{'calories':300,'portion':'1 bowl'})
        self.assertEqual(food_history(self.p)[0]['foods'],['rice'])

    def test_meals_group_by_local_day(self):
        import json
        from ahead.storage import connection
        from ahead.visuals import amount
        day=date(2026,11,2)
        # 23:30 UTC on 1 November is 03:30 on 2 November in Dubai; 20:00 UTC on 2 November is the next local day.
        for stamp,food in [('2026-11-01T19:59:59.999999+00:00','before'),('2026-11-01T23:30:00+00:00','early'),
                           ('2026-11-02T19:59:59+00:00','late'),('2026-11-02T20:00:00+00:00','next')]:
            with connection() as db:
                db.execute("INSERT INTO food_entries(patient_id,foods_json,nutrients_json,source,created_at) VALUES(?,?,?,?,?)",
                           (self.p,json.dumps([food]),json.dumps({'calories':'x'}),'manual_estimate',stamp))
        self.assertEqual([m['foods'][0] for m in food_on_day(self.p,day)],['early','late'])
        self.assertEqual(food_on_day(self.q,day),[])
        self.assertEqual(to_local('2026-11-01T23:30:00+00:00').strftime('%Y-%m-%d %H:%M'),'2026-11-02 03:30')
        self.assertEqual([amount(v) for v in (None,'x',float('nan'),-5,'12.5',3)],[0.0,0.0,0.0,0.0,12.5,3.0])

    def test_daily_value_notes_match_the_rounded_amount(self):
        import re
        from ahead.visuals import _nutrient_row,percent,quantity
        row=lambda *a:' '.join(re.sub(r'<[^>]+>',' ',_nutrient_row(*a)).split())
        self.assertIn('Daily value reached',row('fiber','Fiber','g',28,'goal','#000',27.96,True))   # shown as 28 g
        self.assertIn('0.1 g to go',row('fiber','Fiber','g',28,'goal','#000',27.94,True))
        self.assertIn('99% of 50 g',row('protein','Protein','g',50,'reference','#000',49.8,True))   # never a premature 100%
        self.assertIn('Above the daily limit',row('sodium','Sodium','mg',2300,'limit','#000',2301,True))
        self.assertIn('Near the daily limit',row('sodium','Sodium','mg',2300,'limit','#000',2300,True))
        self.assertEqual((percent(1999,2000),quantity(1234567.4,'kcal'),quantity(12.25,'g')),(99,'1,234,567','12.2'))

    def test_medication_permissions_and_logs(self):
        data={'name':'Sample','dosage':'1 tablet','instructions':'As instructed','times':['00:00'],
              'days':list(range(7)),'start_date':(date.today()-timedelta(days=2)).isoformat(),
              'end_date':(date.today()+timedelta(days=2)).isoformat(),'notes':''}
        with self.assertRaises(PermissionError):save_plan(self.d,self.p,data)
        grant_doctor(self.p,'d@test.com')
        plan=save_plan(self.d,self.p,data)
        self.assertEqual(len(plans(self.p)),1)
        with self.assertRaises(PermissionError):plans(self.p,self.other)
        with self.assertRaises(PermissionError):mark_dose(self.q,plan,date.today().isoformat(),'00:00','taken')
        yesterday=(date.today()-timedelta(days=1)).isoformat()
        mark_dose(self.p,plan,yesterday,'00:00','taken')
        yesterday_doses=scheduled_doses(self.p,date.today()-timedelta(days=1),date.today()-timedelta(days=1))
        self.assertEqual(yesterday_doses[0]['status'],'taken')
        self.assertTrue(notifications(self.p))
        self.assertFalse(notifications(self.q))

    def test_dob_conflict_is_flagged(self):
        stats={'age':{'kind':'number','integer':True,'binary':False,'min':0,'max':120}}
        _,issues=validate_row({'age':22},'diabetes',['age'],stats,'1990-01-01',date(2026,9,25))
        self.assertTrue(any('conflicts with DOB' in issue for issue in issues))

    def test_existing_screening_schema_migrates_without_losing_rows(self):
        path=self.temp.name+'/legacy.db'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE screenings(id INTEGER PRIMARY KEY,owner_id INTEGER NOT NULL,disease TEXT NOT NULL,model TEXT NOT NULL,score REAL NOT NULL,prediction INTEGER NOT NULL,threshold REAL NOT NULL,created_at TEXT NOT NULL)')
            db.execute("INSERT INTO screenings(owner_id,disease,model,score,prediction,threshold,created_at) VALUES(1,'heart','Old',0.2,0,0.5,'2026-01-01')")
        os.environ['AHEAD_DB_PATH']=path
        init_db();init_platform()
        old=load_screenings(1)
        self.assertEqual(len(old),1)
        self.assertEqual(old[0]['source'],'legacy')
        self.assertEqual(old[0]['values_json'],'{}')

if __name__=='__main__':unittest.main()
