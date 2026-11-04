"""Doctor Overview 'Recent screenings': newest ten first, local (Dubai) date and time for saved and new results."""
import json
import os
import tempfile
import unittest
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

from ahead.storage import init_db,create_user,authenticate,save_screening,load_screenings,connection
from ahead.platform_data import init_platform

APP=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'app.py')
FORMAT='%d %b %Y, %H:%M'


class RecentScreeningsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        os.environ['AHEAD_DB_PATH']=self.temp.name+'/app.db'
        init_db();init_platform()
        create_user('doc@test.com','safe-password-123','Doctor','Doc Test')
        self.d=authenticate('doc@test.com','safe-password-123','Doctor')
        # Scores 1%..12%, one per day; the later ones fall on the next Dubai day.
        for i in range(1,13):
            sid=save_screening(self.d['id'],'diabetes','TestModel',i/100,0,.5,{'age':30})
            with connection() as db:db.execute('UPDATE screenings SET created_at=? WHERE id=?',(f'2026-10-{i:02d}T{i+10:02d}:30:00+00:00',sid))
    def tearDown(self):
        self.temp.cleanup();os.environ.pop('AHEAD_DB_PATH',None)

    def sign_in(self):
        at=AppTest.from_file(APP,default_timeout=120).run()
        at.segmented_control(key='login_account_type').set_value('Doctor / Admin')
        at.text_input(key='login_email').set_value('doc@test.com')
        at.text_input(key='login_password').set_value('safe-password-123')
        next(b for b in at.button if b.label=='Sign In').click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual(at.session_state.nav,'overview')
        return at
    def frame(self,at):
        return next(e for e in at.expander if e.label=='Recent screenings').dataframe[0]
    def table(self,at):
        table=self.frame(at).value.copy()
        table['Date']=table['Date'].dt.strftime(FORMAT)
        return table

    def test_saved_screenings_show_the_newest_ten_first_in_local_time(self):
        table=self.table(self.sign_in())
        self.assertEqual(list(table.columns[:1]),['Date'])
        self.assertEqual(list(table['Score']),[float(i) for i in range(12,2,-1)])
        self.assertEqual(table['Date'].iloc[0],'13 Oct 2026, 02:30')
        self.assertEqual(table['Date'].iloc[2],'11 Oct 2026, 00:30')
        self.assertEqual(table['Date'].iloc[3],'09 Oct 2026, 23:30')
        self.assertEqual(table['Date'].iloc[-1],'03 Oct 2026, 17:30')

    def test_date_column_sorts_chronologically_and_shows_day_month_year(self):
        with connection() as db:db.execute("UPDATE screenings SET created_at='2026-11-01T08:00:00+00:00' WHERE id=(SELECT MAX(id) FROM screenings)")
        frame=self.frame(self.sign_in())
        table=frame.value
        self.assertEqual(str(table['Date'].dt.tz),'Asia/Dubai')
        self.assertEqual(table['Date'].dt.strftime(FORMAT).iloc[0],'01 Nov 2026, 12:00')
        # As text, '01 Nov 2026' would sort before '03 Oct 2026'.
        self.assertEqual(list(table.sort_values('Date')['Score']),[float(i) for i in range(3,13)])
        self.assertNotEqual(list(table.sort_values('Date')['Score']),list(table.assign(Date=table['Date'].dt.strftime(FORMAT)).sort_values('Date')['Score']))
        self.assertEqual(json.loads(frame.proto.columns)['Date']['type_config'],{'type':'datetime','format':'DD MMM YYYY, HH:mm'})

    def test_score_column_sorts_numerically_and_shows_a_percent(self):
        frame=self.frame(self.sign_in())
        table=frame.value
        self.assertEqual(str(table['Score'].dtype),'float64')
        # As text, '10.0%' would sort before '3.0%'.
        self.assertEqual(list(table.sort_values('Score')['Score']),[float(i) for i in range(3,13)])
        self.assertEqual(json.loads(frame.proto.columns)['Score']['type_config'],{'type':'number','format':'%.1f%%'})

    def test_new_screening_goes_on_top_and_keeps_its_time_after_signing_in_again(self):
        at=self.sign_in()
        at.session_state['nav']='screenings';at.session_state['active_screening']='diabetes'
        at.run();next(b for b in at.button if 'Generate Screening Result' in (b.label or '')).click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        newest=at.session_state.screening_history[0]
        self.assertEqual(newest['time'].utcoffset(),timedelta(0))
        self.assertEqual(len(at.session_state.screening_history),13)
        saved=load_screenings(self.d['id'])[0]
        self.assertLess(abs(datetime.fromisoformat(saved['created_at'])-newest['time']),timedelta(seconds=5))
        at.session_state['nav']='overview';at.run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        table=self.table(at)
        self.assertEqual(len(table),10)
        self.assertEqual(table['Score'].iloc[0],round(saved['score']*100,1))
        self.assertEqual(table['Date'].iloc[0],newest['time'].astimezone(ZoneInfo('Asia/Dubai')).strftime(FORMAT))
        self.assertEqual(list(table['Score'].iloc[1:]),[float(i) for i in range(12,3,-1)])
        again=self.table(self.sign_in())
        self.assertEqual(again['Score'].iloc[0],table['Score'].iloc[0])
        shift=abs(datetime.strptime(again['Date'].iloc[0],FORMAT)-datetime.strptime(table['Date'].iloc[0],FORMAT))
        self.assertLessEqual(shift,timedelta(minutes=1))


if __name__=='__main__':
    unittest.main()
