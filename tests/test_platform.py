"""Integrated tests of additive migration, consent, dose controls and wellness persistence."""
import os
import sqlite3
import tempfile
import unittest
from datetime import date,timedelta

from ahead.storage import init_db,create_user,authenticate,save_screening,load_screenings
from ahead.platform_data import (init_platform,grant_doctor,allowed,linked_patients,patient_screenings,
    save_lifestyle,lifestyle_history,set_goals,get_goals,set_challenge_day,challenge_days,
    save_food,food_history,save_plan,plans,scheduled_doses,mark_dose,notify,notifications,
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
