"""Patient-owned wellness and care records; every doctor read checks an explicit grant."""
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ahead.storage import connection


def now():
    return datetime.now(timezone.utc).isoformat()


def local_today():
    return datetime.now(ZoneInfo('Asia/Dubai')).date()


def local_now():
    return datetime.now(ZoneInfo('Asia/Dubai'))


def to_local(stamp):
    """A stored UTC timestamp as local (Dubai) time."""
    moment=datetime.fromisoformat(stamp)
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(ZoneInfo('Asia/Dubai'))


def init_platform():
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS care_access (
          patient_id INTEGER NOT NULL REFERENCES users(id), doctor_id INTEGER NOT NULL REFERENCES users(id),
          created_at TEXT NOT NULL, PRIMARY KEY(patient_id,doctor_id));
        CREATE TABLE IF NOT EXISTS lifestyle (
          id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES users(id),
          values_json TEXT NOT NULL, scores_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS lifestyle_patient ON lifestyle(patient_id,created_at);
        CREATE TABLE IF NOT EXISTS lifestyle_goals (
          patient_id INTEGER PRIMARY KEY REFERENCES users(id), goals_json TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS challenge_days (
          patient_id INTEGER NOT NULL REFERENCES users(id), day TEXT NOT NULL,minutes INTEGER NOT NULL,
          source TEXT NOT NULL DEFAULT 'manual', updated_at TEXT NOT NULL,PRIMARY KEY(patient_id,day));
        CREATE TABLE IF NOT EXISTS food_entries (
          id INTEGER PRIMARY KEY,patient_id INTEGER NOT NULL REFERENCES users(id),
          foods_json TEXT NOT NULL,nutrients_json TEXT NOT NULL,source TEXT NOT NULL,
          created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS medication_plans (
          id INTEGER PRIMARY KEY,patient_id INTEGER NOT NULL REFERENCES users(id),
          doctor_id INTEGER NOT NULL REFERENCES users(id), name TEXT NOT NULL,dosage TEXT NOT NULL,
          instructions TEXT NOT NULL,times_json TEXT NOT NULL,days_json TEXT NOT NULL,
          start_date TEXT NOT NULL,end_date TEXT NOT NULL,notes TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'active',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS medication_patient ON medication_plans(patient_id,status);
        CREATE TABLE IF NOT EXISTS medication_audit (
          id INTEGER PRIMARY KEY,plan_id INTEGER NOT NULL REFERENCES medication_plans(id),
          doctor_id INTEGER NOT NULL REFERENCES users(id),action TEXT NOT NULL,details_json TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dose_logs (
          id INTEGER PRIMARY KEY,plan_id INTEGER NOT NULL REFERENCES medication_plans(id),
          patient_id INTEGER NOT NULL REFERENCES users(id),scheduled_date TEXT NOT NULL,scheduled_time TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('taken','skipped')),
          logged_at TEXT NOT NULL, UNIQUE(plan_id,scheduled_date,scheduled_time));
        CREATE TABLE IF NOT EXISTS notifications (
          id INTEGER PRIMARY KEY,patient_id INTEGER NOT NULL REFERENCES users(id),type TEXT NOT NULL,
          title TEXT NOT NULL,description TEXT NOT NULL,action TEXT NOT NULL DEFAULT '',
          dedupe_key TEXT,created_at TEXT NOT NULL,read_at TEXT,
          UNIQUE(patient_id,dedupe_key));
        CREATE TABLE IF NOT EXISTS activity_events (
          id INTEGER PRIMARY KEY,patient_id INTEGER NOT NULL REFERENCES users(id),type TEXT NOT NULL,
          created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS activity_patient ON activity_events(patient_id,created_at);
        CREATE TABLE IF NOT EXISTS record_predictions (
          id INTEGER PRIMARY KEY, record_id INTEGER NOT NULL REFERENCES records(id),
          owner_id INTEGER NOT NULL REFERENCES users(id),disease TEXT NOT NULL,
          values_json TEXT NOT NULL,score REAL NOT NULL,threshold REAL NOT NULL,
          model TEXT NOT NULL,source TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS record_predictions_owner ON record_predictions(owner_id,record_id,created_at);
        """)
        if 'ai_allowed' not in {r['name'] for r in db.execute('PRAGMA table_info(care_access)')}:
            db.execute('ALTER TABLE care_access ADD COLUMN ai_allowed INTEGER NOT NULL DEFAULT 0')


def event(patient_id, kind):
    with connection() as db:
        db.execute("INSERT INTO activity_events(patient_id,type,created_at) VALUES(?,?,?)", (patient_id,kind,now()))


def grant_doctor(patient_id, email):
    with connection() as db:
        doctor = db.execute("SELECT id FROM users WHERE email=? AND role='Doctor'",(email.strip().lower(),)).fetchone()
        if not doctor:
            return False
        db.execute("INSERT OR IGNORE INTO care_access(patient_id,doctor_id,created_at) VALUES(?,?,?)",(patient_id,doctor['id'],now()))
        return True


def revoke_doctor(patient_id, doctor_id):
    with connection() as db:
        db.execute("DELETE FROM care_access WHERE patient_id=? AND doctor_id=?",(patient_id,doctor_id))


def doctors_for(patient_id):
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT u.id,u.name,u.email,a.ai_allowed FROM care_access a JOIN users u ON a.doctor_id=u.id WHERE a.patient_id=?",(patient_id,))]


def ai_consent(patient_id, doctor_id, enabled=None):
    with connection() as db:
        if enabled is not None:
            db.execute('UPDATE care_access SET ai_allowed=? WHERE patient_id=? AND doctor_id=?',(int(bool(enabled)),patient_id,doctor_id))
        row=db.execute('SELECT ai_allowed FROM care_access WHERE patient_id=? AND doctor_id=?',(patient_id,doctor_id)).fetchone()
        return bool(row and row['ai_allowed'])


def allowed(doctor_id, patient_id):
    with connection() as db:
        return bool(db.execute("SELECT 1 FROM care_access WHERE doctor_id=? AND patient_id=?",(doctor_id,patient_id)).fetchone())


def linked_patients(doctor_id):
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT u.id,u.name,u.email,a.created_at FROM care_access a JOIN users u ON u.id=a.patient_id WHERE a.doctor_id=? ORDER BY u.name",(doctor_id,))]


def save_lifestyle(patient_id, values, scores):
    with connection() as db:
        db.execute("INSERT INTO lifestyle(patient_id,values_json,scores_json,created_at) VALUES(?,?,?,?)",(patient_id,json.dumps(values),json.dumps(scores),now()))
    event(patient_id,'lifestyle_completed')


def lifestyle_history(patient_id):
    with connection() as db:
        return [{**dict(r),'values':json.loads(r['values_json']),'scores':json.loads(r['scores_json'])} for r in db.execute("SELECT * FROM lifestyle WHERE patient_id=? ORDER BY created_at DESC,id DESC LIMIT 365",(patient_id,))]


def set_goals(patient_id, goals):
    with connection() as db:
        db.execute("INSERT INTO lifestyle_goals VALUES(?,?,?) ON CONFLICT(patient_id) DO UPDATE SET goals_json=excluded.goals_json,updated_at=excluded.updated_at",(patient_id,json.dumps(goals),now()))


def get_goals(patient_id):
    with connection() as db:
        row=db.execute("SELECT goals_json FROM lifestyle_goals WHERE patient_id=?",(patient_id,)).fetchone()
        return json.loads(row['goals_json']) if row else {}


def set_challenge_day(patient_id, day, minutes):
    if date.fromisoformat(day)>local_today() or not 0<=int(minutes)<=1440:
        raise ValueError('Activity date or minutes are invalid.')
    with connection() as db:
        db.execute("INSERT INTO challenge_days(patient_id,day,minutes,updated_at) VALUES(?,?,?,?) ON CONFLICT(patient_id,day) DO UPDATE SET minutes=excluded.minutes,updated_at=excluded.updated_at",(patient_id,day,int(minutes),now()))
    event(patient_id,'challenge_activity')


def challenge_days(patient_id):
    with connection() as db:
        return {r['day']:r['minutes'] for r in db.execute("SELECT day,minutes FROM challenge_days WHERE patient_id=?",(patient_id,))}


def save_food(patient_id, foods, nutrients, source='manual'):
    with connection() as db:
        db.execute("INSERT INTO food_entries(patient_id,foods_json,nutrients_json,source,created_at) VALUES(?,?,?,?,?)",(patient_id,json.dumps(foods),json.dumps(nutrients),source,now()))
    event(patient_id,'food_saved')


def food_on_day(patient_id, day):
    """Meals saved on one local (Dubai) day, earliest first, with no history limit."""
    start,end=(datetime.combine(d,datetime.min.time(),tzinfo=ZoneInfo('Asia/Dubai')).astimezone(timezone.utc).isoformat()
               for d in (day,day+timedelta(days=1)))
    with connection() as db:
        return [{**dict(r),'foods':json.loads(r['foods_json']),'nutrients':json.loads(r['nutrients_json'])} for r in db.execute(
            "SELECT * FROM food_entries WHERE patient_id=? AND created_at>=? AND created_at<? ORDER BY created_at,id",(patient_id,start,end))]


def food_history(patient_id):
    with connection() as db:
        return [{**dict(r),'foods':json.loads(r['foods_json']),'nutrients':json.loads(r['nutrients_json'])} for r in db.execute("SELECT * FROM food_entries WHERE patient_id=? ORDER BY created_at DESC,id DESC LIMIT 180",(patient_id,))]


def notify(patient_id, kind, title, description, action='', dedupe_key=None):
    with connection() as db:
        db.execute("INSERT OR IGNORE INTO notifications(patient_id,type,title,description,action,dedupe_key,created_at) VALUES(?,?,?,?,?,?,?)",(patient_id,kind,title,description,action,dedupe_key,now()))


def notifications(patient_id):
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM notifications WHERE patient_id=? ORDER BY created_at DESC,id DESC LIMIT 100",(patient_id,))]


def mark_notification(patient_id, notification_id):
    with connection() as db:
        db.execute("UPDATE notifications SET read_at=? WHERE id=? AND patient_id=?",(now(),notification_id,patient_id))


def _authorized_patient(db, doctor_id, patient_id):
    return bool(db.execute("SELECT 1 FROM care_access WHERE doctor_id=? AND patient_id=?",(doctor_id,patient_id)).fetchone())


def save_plan(doctor_id, patient_id, plan, plan_id=None):
    if not plan['name'].strip() or not plan['dosage'].strip() or not plan['times'] or plan['end_date'] < plan['start_date']:
        raise ValueError('Enter a medication, dosage, valid dates and at least one scheduled time.')
    if not plan['days']:
        raise ValueError('Choose at least one day.')
    with connection() as db:
        if not _authorized_patient(db,doctor_id,patient_id):
            raise PermissionError('Patient access has not been granted.')
        stamp=now()
        fields=(plan['name'].strip(),plan['dosage'].strip(),plan['instructions'].strip(),json.dumps(plan['times']),json.dumps(plan['days']),plan['start_date'],plan['end_date'],plan['notes'].strip(),stamp)
        if plan_id is None:
            cursor=db.execute("""INSERT INTO medication_plans(patient_id,doctor_id,name,dosage,instructions,times_json,days_json,start_date,end_date,notes,created_at,updated_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(patient_id,doctor_id,*fields[:-1],stamp,stamp))
            plan_id=cursor.lastrowid
            action='created'
        else:
            cursor=db.execute("""UPDATE medication_plans SET name=?,dosage=?,instructions=?,times_json=?,days_json=?,start_date=?,end_date=?,notes=?,updated_at=?
                      WHERE id=? AND patient_id=? AND doctor_id=?""",(*fields,plan_id,patient_id,doctor_id))
            if not cursor.rowcount:
                raise PermissionError('Only the prescribing doctor can edit this plan.')
            action='edited'
        db.execute("INSERT INTO medication_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",(plan_id,doctor_id,action,json.dumps(plan),stamp))
    notify(patient_id,'medication','Medication plan updated',f"Your doctor {action} a plan for {plan['name']}.",'medications',f'plan-{plan_id}-{stamp}')
    return plan_id


def plan_status(doctor_id, plan_id, status):
    if status not in ('active','paused','stopped'):
        raise ValueError('Invalid plan status.')
    with connection() as db:
        row=db.execute("SELECT * FROM medication_plans WHERE id=? AND doctor_id=?",(plan_id,doctor_id)).fetchone()
        if not row or not _authorized_patient(db,doctor_id,row['patient_id']):
            raise PermissionError('Plan access denied.')
        if row['status']=='stopped' and status!='stopped':
            raise ValueError('A stopped plan cannot be resumed; create a new plan instead.')
        db.execute("UPDATE medication_plans SET status=?,updated_at=? WHERE id=?",(status,now(),plan_id))
        db.execute("INSERT INTO medication_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",(plan_id,doctor_id,status,'{}',now()))
    notify(row['patient_id'],'medication','Medication plan status',f"{row['name']} is now {status}.",'medications',f'plan-status-{plan_id}-{status}-{now()}')


def plans(patient_id, doctor_id=None):
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        query='SELECT * FROM medication_plans WHERE patient_id=?'
        args=[patient_id]
        if doctor_id is not None:
            query+=' AND doctor_id=?';args.append(doctor_id)
        return [{**dict(r),'times':json.loads(r['times_json']),'days':json.loads(r['days_json'])} for r in db.execute(query+' ORDER BY created_at DESC',args)]


def dose_entries(patient_id, doctor_id=None):
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT l.*,p.name FROM dose_logs l JOIN medication_plans p ON p.id=l.plan_id WHERE l.patient_id=? ORDER BY scheduled_date DESC,scheduled_time DESC",(patient_id,))]


def mark_dose(patient_id, plan_id, day, time, status):
    if status not in ('taken','skipped') or day > local_today().isoformat():
        raise ValueError('Only due or past doses can be marked taken or skipped.')
    with connection() as db:
        p=db.execute("SELECT * FROM medication_plans WHERE id=? AND patient_id=?",(plan_id,patient_id)).fetchone()
        if not p or p['status']!='active' or not p['start_date']<=day<=p['end_date'] or time not in json.loads(p['times_json']) or date.fromisoformat(day).weekday() not in json.loads(p['days_json']):
            raise PermissionError('This is not a scheduled dose for your account.')
        if day==local_today().isoformat() and time>local_now().strftime('%H:%M'):
            raise ValueError('This dose is not due yet.')
        db.execute("INSERT INTO dose_logs(plan_id,patient_id,scheduled_date,scheduled_time,status,logged_at) VALUES(?,?,?,?,?,?) ON CONFLICT(plan_id,scheduled_date,scheduled_time) DO UPDATE SET status=excluded.status,logged_at=excluded.logged_at",(plan_id,patient_id,day,time,status,now()))
    event(patient_id,'medication_'+status)


def scheduled_doses(patient_id, start, end, doctor_id=None):
    ps=plans(patient_id,doctor_id)
    logs={(x['plan_id'],x['scheduled_date'],x['scheduled_time']):x['status'] for x in dose_entries(patient_id,doctor_id)}
    with connection() as db:
        transitions={p['id']:[(r['created_at'],r['action']) for r in db.execute(
            "SELECT created_at,action FROM medication_audit WHERE plan_id=? ORDER BY id",(p['id'],))]
            for p in ps}
    current=start
    result=[]
    while current<=end:
        ds=current.isoformat()
        for p in ps:
            status='active'
            for stamp,action in transitions[p['id']]:
                if stamp[:10]<=ds and action in ('active','paused','stopped'):
                    status=action
            if status=='active' and p['start_date']<=ds<=p['end_date'] and current.weekday() in p['days']:
                for time in p['times']:
                    status=logs.get((p['id'],ds,time))
                    if not status:
                        status='missed' if datetime.combine(current,datetime.strptime(time,'%H:%M').time(),tzinfo=ZoneInfo('Asia/Dubai'))<local_now() else 'upcoming'
                    result.append({'plan_id':p['id'],'name':p['name'],'dosage':p['dosage'],'instructions':p['instructions'],'date':ds,'time':time,'status':status})
        current+=timedelta(days=1)
    return result


def patient_screenings(doctor_id, patient_id):
    if not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM screenings WHERE owner_id=? ORDER BY created_at DESC,id DESC",(patient_id,))]


def analytics(doctor_id):
    people=linked_patients(doctor_id)
    ids=[p['id'] for p in people]
    if not ids:
        return people,[],[]
    with connection() as db:
        marks=','.join('?' for _ in ids)
        assessments=[dict(r) for r in db.execute(f'SELECT * FROM screenings WHERE owner_id IN ({marks}) ORDER BY created_at DESC',ids)]
        events=[dict(r) for r in db.execute(f'SELECT * FROM activity_events WHERE patient_id IN ({marks}) ORDER BY created_at DESC LIMIT 500',ids)]
    return people,assessments,events


def append_record_prediction(owner_id, record_id, disease, values, score, threshold, model, source='batch'):
    with connection() as db:
        if not db.execute('SELECT 1 FROM records WHERE id=? AND owner_id=? AND disease=?',(record_id,owner_id,disease)).fetchone():
            raise PermissionError('Record access denied.')
        db.execute('INSERT INTO record_predictions(record_id,owner_id,disease,values_json,score,threshold,model,source,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                   (record_id,owner_id,disease,json.dumps(values),score,threshold,model,source,now()))


def record_prediction_history(owner_id, record_id):
    with connection() as db:
        return [dict(r) for r in db.execute('SELECT * FROM record_predictions WHERE owner_id=? AND record_id=? ORDER BY id DESC',(owner_id,record_id))]
