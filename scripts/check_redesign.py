"""Run from the repository root: python scripts/check_redesign.py. Uses an isolated temporary DB."""
import os,tempfile,sys
sys.path.insert(0,os.getcwd())
from streamlit.testing.v1 import AppTest
from ahead.storage import init_db,create_user,authenticate,save_screening
from ahead.platform_data import init_platform,save_lifestyle,set_goals,set_challenge_day,save_food,grant_doctor,save_plan,local_today
from ahead.wellness import evaluate
from datetime import timedelta
test_directory=tempfile.TemporaryDirectory();root=test_directory.name;os.environ['AHEAD_DB_PATH']=root+'/test.db'
init_db();init_platform()
for email,role in [('patient@test.local','Patient'),('doctor@test.local','Doctor')]:create_user(email,'test-password-123',role,role+' Test')
p=authenticate('patient@test.local','test-password-123','Patient');d=authenticate('doctor@test.local','test-password-123','Doctor')
v={'age':30,'height_cm':175,'weight_kg':75,'steps':6500,'activity_level':'Some movement','exercise_days':3,'exercise_minutes':30,'sleep_hours':7,'sleep_quality':3,'water_litres':1.5,'fruit_veg':3,'sugary_food':2,'fast_food':2,'balanced_meals':True,'smoking':False,'stress':3,'sedentary_hours':6}

def pages(user,names):
    for page in names:
        at=AppTest.from_file(os.path.abspath('app.py'),default_timeout=90)
        
        for key,value in {'authenticated':True,'user':user,'nav':page}.items():at.session_state[key]=value
        at.run()
        if at.exception:
            print('FAIL',page,[x.message for x in at.exception]);raise RuntimeError(page)
        print('PASS',user['role'],page,flush=True)
    return at
pages(p,['overview','wellness','food','medications','notifications','voice','settings','screenings','analytics','assistant'])
pages(d,['overview','intelligence','clinical'])
for i in range(2):save_lifestyle(p['id'],v,evaluate(v))
set_goals(p['id'],{'steps':8000,'weight':72,'exercise':150,'water':2,'sleep':8,'fruit_veg':5})
set_challenge_day(p['id'],local_today().isoformat(),35)
save_food(p['id'],['Test meal'],{'portion':'test','calories':500,'protein':25,'carbs':50,'fat':20,'sugar':5,'fiber':3,'sodium':400})
grant_doctor(p['id'],'doctor@test.local')
save_plan(d['id'],p['id'],{'name':'Demo only','dosage':'Test dose','instructions':'Test only','times':['00:00'],'days':list(range(7)),'start_date':(local_today()-timedelta(days=1)).isoformat(),'end_date':(local_today()+timedelta(days=2)).isoformat(),'notes':''})
for val in [.2,.6]:save_screening(p['id'],'diabetes','TestModel',val,int(val>=.5),.5,{'age':30})
pages(p,['overview','wellness','food','medications','notifications'])
pages(d,['intelligence'])
# Forms keep their original callbacks and persistence.
at=pages(p,['wellness']);before=len(__import__('ahead.platform_data',fromlist=['lifestyle_history']).lifestyle_history(p['id']))
next(b for b in at.button if b.label=='Save lifestyle check-in').click().run()
assert not at.exception
assert len(__import__('ahead.platform_data',fromlist=['lifestyle_history']).lifestyle_history(p['id']))==before+1
print('PASS lifestyle form save',flush=True)
# Each model runs using its own feature defaults; render the complete saved result.
from ahead.resources import features,feature_summary,load_pipeline,best_model,decision_threshold,predict_probability
for disease in ['diabetes','heart','kidney']:
    stats=feature_summary(disease)
    values={f:(stats[f]['default'] if stats[f]['kind']=='category' else stats[f]['mode'] if stats[f]['binary'] else stats[f]['median']) for f in features(disease)}
    pipeline=load_pipeline(disease,best_model(disease));prob=float(predict_probability(pipeline,__import__('pandas').DataFrame([values],columns=features(disease))).iloc[0]);threshold=decision_threshold(disease,best_model(disease))
    at=AppTest.from_file(os.path.abspath('app.py'),default_timeout=120)
    
    for key,value in {'authenticated':True,'user':p,'nav':'screenings','active_screening':disease,'last_results':{disease:{'values':values,'probability':prob,'prediction':int(prob>=threshold),'model':best_model(disease),'threshold':threshold}}}.items():at.session_state[key]=value
    at.run();assert not at.exception,[e.message for e in at.exception]
    print('PASS original prediction and result',disease,prob,flush=True)
# Exercise the actual sign-in form, not just injected authenticated sessions.
for account, role in [('patient', 'Patient'), ('doctor', 'Doctor / Admin')]:
    at = AppTest.from_file(os.path.abspath('app.py'), default_timeout=90).run()
    assert not at.exception
    at.segmented_control(key='login_account_type').set_value(role)
    at.text_input(key='login_email').set_value(f'{account}@test.local')
    at.text_input(key='login_password').set_value('test-password-123')
    next(b for b in at.button if b.label == 'Sign In').click().run()
    assert not at.exception
    assert at.session_state.authenticated
    assert at.session_state.user['id'] == (p if account == 'patient' else d)['id']
    print('PASS sign-in form', account, flush=True)

at = AppTest.from_file(os.path.abspath('app.py'), default_timeout=90)
for key, value in {'authenticated': True, 'user': p, 'nav': 'intelligence',
                   'appearance_mode': 'Dark', 'pref_language': 'العربية'}.items():
    at.session_state[key] = value
at.run()
assert not at.exception
assert at.session_state.nav == 'overview'
assert not any(b.key == 'nav_intelligence' for b in at.button)
print('PASS patient restricted navigation and Arabic/dark render', flush=True)
print('SMOKE COMPLETE')
