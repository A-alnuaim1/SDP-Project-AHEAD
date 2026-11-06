"""Patient-owned wellness and care records; every doctor read checks an explicit grant."""
import json
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ahead import prevention
from ahead.config import DISEASES
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


def _since(days=30):
    """UTC stamp of the local midnight that starts the last `days` local days, today included."""
    return datetime.combine(local_today()-timedelta(days=days-1),datetime.min.time(),tzinfo=ZoneInfo('Asia/Dubai')).astimezone(timezone.utc).isoformat()


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
        DROP INDEX IF EXISTS activity_patient; -- superseded by activity_patient_type, which also covers type
        CREATE INDEX IF NOT EXISTS activity_patient_type ON activity_events(patient_id,created_at,type);
        CREATE TABLE IF NOT EXISTS record_predictions (
          id INTEGER PRIMARY KEY, record_id INTEGER NOT NULL REFERENCES records(id),
          owner_id INTEGER NOT NULL REFERENCES users(id),disease TEXT NOT NULL,
          values_json TEXT NOT NULL,score REAL NOT NULL,threshold REAL NOT NULL,
          model TEXT NOT NULL,source TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS record_predictions_owner ON record_predictions(owner_id,record_id,created_at);
        -- screening_id has no foreign key: deleting a patient's screening history must not be blocked by a plan.
        CREATE TABLE IF NOT EXISTS prevention_plans (
          id INTEGER PRIMARY KEY,patient_id INTEGER NOT NULL REFERENCES users(id),
          doctor_id INTEGER NOT NULL REFERENCES users(id),disease TEXT NOT NULL,screening_id INTEGER,
          targets_json TEXT NOT NULL,note TEXT NOT NULL DEFAULT '',recheck_date TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','active','paused','stopped')),
          started_at TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS prevention_patient ON prevention_plans(patient_id,doctor_id,status);
        CREATE TABLE IF NOT EXISTS prevention_audit (
          id INTEGER PRIMARY KEY,plan_id INTEGER NOT NULL REFERENCES prevention_plans(id),
          doctor_id INTEGER NOT NULL REFERENCES users(id),action TEXT NOT NULL,details_json TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS prevention_audit_plan ON prevention_audit(plan_id,id);
        """)
        if 'ai_allowed' not in {r['name'] for r in db.execute('PRAGMA table_info(care_access)')}:
            db.execute('ALTER TABLE care_access ADD COLUMN ai_allowed INTEGER NOT NULL DEFAULT 0')
        stopped=_stop_unlinked_plans(db);stopped_prevention=_stop_unlinked_prevention(db)
    _notify_unlinked(stopped);_notify_unlinked_prevention(stopped_prevention)


def _stop_unlinked_plans(db, patient_id=None):
    """Stop active/paused plans whose prescriber no longer has access; nobody could change them otherwise."""
    rows=db.execute("""SELECT * FROM medication_plans p WHERE status IN ('active','paused') AND (? IS NULL OR patient_id=?)
        AND NOT EXISTS(SELECT 1 FROM care_access a WHERE a.patient_id=p.patient_id AND a.doctor_id=p.doctor_id)""",(patient_id,patient_id)).fetchall()
    stopped=[]
    for r in rows:
        stamp=now()
        if db.execute("UPDATE medication_plans SET status='stopped',updated_at=? WHERE id=? AND status IN ('active','paused')",(stamp,r['id'])).rowcount:
            db.execute("INSERT INTO medication_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",(r['id'],r['doctor_id'],'stopped',json.dumps({'reason':'prescriber access removed'}),stamp))
            stopped.append(r)
    return stopped


def _notify_unlinked(rows):
    for r in rows:
        notify(r['patient_id'],'medication','Medication plan stopped',f"{r['name']} was stopped because the prescribing doctor no longer has access. Talk to a clinician before changing how you take it.",'medications',f"plan-status-{r['id']}-stopped-unlinked")


def _stop_unlinked_prevention(db, patient_id=None):
    """Stop prevention plans (drafts included) whose doctor no longer has access; returns the ones the patient saw."""
    rows=db.execute("""SELECT * FROM prevention_plans p WHERE status IN ('draft','active','paused') AND (? IS NULL OR patient_id=?)
        AND NOT EXISTS(SELECT 1 FROM care_access a WHERE a.patient_id=p.patient_id AND a.doctor_id=p.doctor_id)""",(patient_id,patient_id)).fetchall()
    stopped=[]
    for r in rows:
        stamp=now()
        if db.execute("UPDATE prevention_plans SET status='stopped',updated_at=? WHERE id=? AND status=?",(stamp,r['id'],r['status'])).rowcount:
            _prevention_audit(db,r['id'],r['doctor_id'],'discarded' if r['status']=='draft' else 'stopped',{'reason':'access removed'},stamp)
            if r['status']!='draft':stopped.append(r)
    return stopped


def _notify_unlinked_prevention(rows):
    for r in rows:
        notify(r['patient_id'],'plan','Prevention plan stopped','Your prevention plan was stopped because the doctor who agreed it no longer has access. '
               'Your tracker goals stay as they are.','medications',f"prevention-status-{r['id']}-stopped-unlinked")


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
        stopped=_stop_unlinked_plans(db,patient_id);stopped_prevention=_stop_unlinked_prevention(db,patient_id)
    _notify_unlinked(stopped);_notify_unlinked_prevention(stopped_prevention)


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


def clock_time(value):
    """A dose time as zero-padded HH:MM; an unpadded '8:00' would sort and compare after '13:50'."""
    try:return datetime.strptime(str(value).strip(),'%H:%M').strftime('%H:%M')
    except ValueError:raise ValueError(f"'{str(value).strip()}' is not a valid time; use HH:MM, for example 08:00.") from None


def _clock(value):
    try:return clock_time(value)
    except ValueError:return str(value)


def _plan_row(r):
    return {**dict(r),'times':sorted({_clock(t) for t in json.loads(r['times_json'])}),'days':sorted(json.loads(r['days_json']))}


def save_plan(doctor_id, patient_id, plan, plan_id=None):
    times=sorted({clock_time(t) for t in plan['times']})
    if not plan['name'].strip() or not plan['dosage'].strip() or not times or plan['end_date'] < plan['start_date']:
        raise ValueError('Enter a medication, dosage, valid dates and at least one scheduled time.')
    days=sorted({int(d) for d in plan['days']})
    if not days or not set(days)<=set(range(7)):
        raise ValueError('Choose at least one day.')
    values={'name':plan['name'].strip(),'dosage':plan['dosage'].strip(),'instructions':plan['instructions'].strip(),'times':times,'days':days,
            'start_date':plan['start_date'],'end_date':plan['end_date'],'notes':plan['notes'].strip()}
    with connection() as db:
        if not _authorized_patient(db,doctor_id,patient_id):
            raise PermissionError('Patient access has not been granted.')
        stamp=now()
        fields=(*[json.dumps(v) if k in ('times','days') else v for k,v in values.items()],stamp)
        later=False
        if plan_id is None:
            cursor=db.execute("""INSERT INTO medication_plans(patient_id,doctor_id,name,dosage,instructions,times_json,days_json,start_date,end_date,notes,created_at,updated_at)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(patient_id,doctor_id,*fields[:-1],stamp,stamp))
            plan_id=cursor.lastrowid
            action='created'
        else:
            row=db.execute("SELECT * FROM medication_plans WHERE id=? AND patient_id=? AND doctor_id=?",(plan_id,patient_id,doctor_id)).fetchone()
            if not row:
                raise PermissionError('Only the prescribing doctor can edit this plan.')
            if row['status']=='stopped':
                raise ValueError('A stopped plan cannot be edited; create a new plan instead.')
            if all(_plan_row(row)[k]==v for k,v in values.items()):
                return plan_id
            later=(to_local(row['created_at']).date()<local_today()
                   and any(_plan_row(row)[k]!=values[k] for k in ('times','days','start_date','end_date')))
            db.execute("""UPDATE medication_plans SET name=?,dosage=?,instructions=?,times_json=?,days_json=?,start_date=?,end_date=?,notes=?,updated_at=?
                      WHERE id=?""",(*fields,plan_id))
            action='edited'
        db.execute("INSERT INTO medication_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",(plan_id,doctor_id,action,json.dumps(values),stamp))
    notify(patient_id,'medication','Medication plan updated',f"Your doctor {action} a plan for {values['name']}."+(
        " The new schedule starts tomorrow; today's doses stay as they were." if later else ''),'medications',f'plan-{plan_id}-{stamp}')
    return plan_id


def plan_status(doctor_id, plan_id, status):
    """Change a plan's status; False, with no audit row or notification, when it already has that status."""
    if status not in ('active','paused','stopped'):
        raise ValueError('Invalid plan status.')
    with connection() as db:
        row=db.execute("SELECT * FROM medication_plans WHERE id=? AND doctor_id=?",(plan_id,doctor_id)).fetchone()
        if not row or not _authorized_patient(db,doctor_id,row['patient_id']):
            raise PermissionError('Plan access denied.')
        if row['status']==status:
            return False
        if row['status']=='stopped':
            raise ValueError('A stopped plan cannot be paused or resumed; create a new plan instead.')
        db.execute("UPDATE medication_plans SET status=?,updated_at=? WHERE id=?",(status,now(),plan_id))
        db.execute("INSERT INTO medication_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",(plan_id,doctor_id,status,'{}',now()))
    notify(row['patient_id'],'medication','Medication plan status',f"{row['name']} is now {status}.",'medications',f'plan-status-{plan_id}-{status}-{now()}')
    return True


def plans(patient_id, doctor_id=None):
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        query='SELECT * FROM medication_plans WHERE patient_id=?'
        args=[patient_id]
        if doctor_id is not None:
            query+=' AND doctor_id=?';args.append(doctor_id)
        return [_plan_row(r) for r in db.execute(query+' ORDER BY created_at DESC',args)]


def dose_entries(patient_id, doctor_id=None):
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT l.*,p.name FROM dose_logs l JOIN medication_plans p ON p.id=l.plan_id WHERE l.patient_id=? ORDER BY scheduled_date DESC,scheduled_time DESC",(patient_id,))]


def _history(db, p):
    """Plan p's schedule versions and status changes, each with the local moment it took effect, oldest first."""
    keys=('times','days','start_date','end_date')
    rows=[(to_local(r['created_at']),r['action'],json.loads(r['details_json'] or '{}')) for r in db.execute(
        "SELECT created_at,action,details_json FROM medication_audit WHERE plan_id=? ORDER BY id",(p['id'],))]
    saved=[(at,{k:details.get(k,p[k]) for k in keys}) for at,action,details in rows if action in ('created','edited')]
    # The plan row is the latest schedule; plans saved before the audit trail existed have only that.
    versions=[*((at,{**v,'times':sorted({_clock(t) for t in v['times']}),'days':[int(d) for d in v['days']]}) for at,v in saved[:-1]),
              (saved[-1][0] if saved else None,{k:p[k] for k in keys})]
    changes=[(at,action) for at,action,_ in rows if action in ('active','paused','stopped')]
    if (changes[-1][1] if changes else 'active')!=p['status']:changes.append((to_local(p['updated_at']),p['status']))
    return versions,changes


def _doses_on(versions, changes, day):
    """(time, local due moment) of the doses due on day. A day keeps the schedule in force when it began, so a schedule
    edit starts the next day and can't add or drop a dose mid-day; the day a plan is created follows its last version
    saved that day. Pauses and stops take effect at the moment they are made."""
    first=versions[0][0];ds=day.isoformat();due=[];zone=ZoneInfo('Asia/Dubai')
    in_force=lambda when:next(v for at,v in reversed(versions) if at is None or at<=when)
    if first and day<first.date():
        # Before the plan existed a dose needs both its first version and the creation day's last one,
        # so a same-day correction can drop such a dose but never add one.
        schedules=[versions[0][1],in_force(datetime.combine(first.date(),datetime.max.time(),tzinfo=zone))]
    else:
        schedules=[in_force(datetime.combine(day,datetime.max.time() if first and day==first.date() else datetime.min.time(),tzinfo=zone))]
    for time in sorted({t for v in schedules for t in v['times']}):
        moment=datetime.combine(day,datetime.strptime(time,'%H:%M').time(),tzinfo=zone)
        if (all(time in v['times'] and day.weekday() in v['days'] and v['start_date']<=ds<=v['end_date'] for v in schedules)
                and next((s for at,s in reversed(changes) if at<=moment),'active')=='active'):
            due.append((time,moment))
    return due


def mark_dose(patient_id, plan_id, day, time, status):
    time=_clock(time)
    if status not in ('taken','skipped') or day > local_today().isoformat():
        raise ValueError('Only due or past doses can be marked taken or skipped.')
    with connection() as db:
        row=db.execute("SELECT * FROM medication_plans WHERE id=? AND patient_id=?",(plan_id,patient_id)).fetchone()
        due=dict(_doses_on(*_history(db,_plan_row(row)),date.fromisoformat(day))) if row else {}
        logged=row and db.execute("SELECT 1 FROM dose_logs WHERE plan_id=? AND scheduled_date=? AND scheduled_time=?",(plan_id,day,time)).fetchone()
        if time not in due and not logged:
            raise PermissionError('This is not a scheduled dose for your account.')
        if time in due and due[time]>local_now():
            raise ValueError('This dose is not due yet.')
        db.execute("INSERT INTO dose_logs(plan_id,patient_id,scheduled_date,scheduled_time,status,logged_at) VALUES(?,?,?,?,?,?) ON CONFLICT(plan_id,scheduled_date,scheduled_time) DO UPDATE SET status=excluded.status,logged_at=excluded.logged_at",(plan_id,patient_id,day,time,status,now()))
    event(patient_id,'medication_'+status)


def scheduled_doses(patient_id, start, end, doctor_id=None):
    """Each day's doses under the schedule in force when they were due; a logged dose always stays in the history."""
    ps=plans(patient_id,doctor_id)
    logs={}
    for x in sorted(dose_entries(patient_id,doctor_id),key=lambda x:x['logged_at']):
        logs.setdefault((x['plan_id'],x['scheduled_date']),{})[_clock(x['scheduled_time'])]=x['status']
    with connection() as db:
        history={p['id']:_history(db,p) for p in ps}
    current=start
    result=[]
    moment_now=local_now()
    while current<=end:
        ds=current.isoformat()
        for p in ps:
            due=dict(_doses_on(*history[p['id']],current));logged=logs.get((p['id'],ds),{})
            for time in sorted({*due,*logged}):
                status=logged.get(time) or ('missed' if due[time]<moment_now else 'upcoming')
                result.append({'plan_id':p['id'],'name':p['name'],'dosage':p['dosage'],'instructions':p['instructions'],'date':ds,'time':time,'status':status})
        current+=timedelta(days=1)
    return result


def open_doses(patient_id, start, end):
    """Unlogged doses still to act on; a paused or stopped plan's earlier doses stay in the history, not as a to-do."""
    active={p['id'] for p in plans(patient_id) if p['status']=='active'}
    return [d for d in scheduled_doses(patient_id,start,end) if d['plan_id'] in active and d['status'] in ('upcoming','missed')]


# Prevention plans hold the doctor-approved targets of ahead/prevention.py. Per patient a doctor has at most one draft
# (never shown to the patient) and one active-or-paused plan; every change is kept in prevention_audit.

_PREVENTION_SELECT="SELECT p.*,u.name doctor_name FROM prevention_plans p JOIN users u ON u.id=p.doctor_id"


def _prevention_audit(db, plan_id, doctor_id, action, details, stamp=None):
    db.execute("INSERT INTO prevention_audit(plan_id,doctor_id,action,details_json,created_at) VALUES(?,?,?,?,?)",
               (plan_id,doctor_id,action,json.dumps(details),stamp or now()))


def _prevention_row(r):
    """A stored plan with its targets decoded and start_date, the local day it was approved (None for a draft)."""
    return {**dict(r),'targets':json.loads(r['targets_json']),
            'start_date':to_local(r['started_at']).date().isoformat() if r['started_at'] else None}


def _prevention_history(db, plan_id):
    rows=[{'action':r['action'],'details':json.loads(r['details_json'] or '{}'),'created_at':r['created_at']}
          for r in db.execute("SELECT action,details_json,created_at FROM prevention_audit WHERE plan_id=? ORDER BY id",(plan_id,))]
    return [{**r,'text':prevention.history_text(r)} for r in rows]


def _lifestyle_logs(db, patient_id, limit=365):
    return [{'values':json.loads(r['values_json']),'created_at':r['created_at']} for r in db.execute(
        "SELECT values_json,created_at FROM lifestyle WHERE patient_id=? ORDER BY created_at DESC,id DESC LIMIT ?",(patient_id,limit))]


def _plan_details(values):
    return {k:values[k] for k in ('disease','screening_id','targets','note','recheck_date')}


def _offer_values(db, patient_id, disease, screening_id, edit):
    """The screening inputs the offer rules read. A new plan or a draft: its source screening, which must be this
    patient's own for this condition. An edit: the patient's newest screening for the condition (none: no values),
    the result the doctor's tab builds its table from, so a re-screen (a patient who now smokes, a new BMI) can add a
    target to an approved plan."""
    if edit:
        row=db.execute("SELECT values_json FROM screenings WHERE owner_id=? AND disease=? ORDER BY created_at DESC,id DESC LIMIT 1",
                       (patient_id,disease)).fetchone()
    elif screening_id is None:
        return {}
    else:
        row=db.execute("SELECT disease,values_json FROM screenings WHERE id=? AND owner_id=?",(screening_id,patient_id)).fetchone()
        if not row or row['disease']!=disease:
            raise ValueError("Choose one of this patient's own screenings for this condition.")
    return json.loads(row['values_json'] or '{}') if row else {}


def _with_baselines(db, patient_id, values, kept=None, since=None):
    """values with each target's baseline, refusing a target the catalog does not offer this patient (weight under
    BMI 25, stop smoking for anyone but a current smoker, anything under 18). On an edit (kept: {key: stored target}),
    a target the plan already had keeps its recorded baseline (and 'since') and stays allowed, so an edit never drops
    it because the patient lost weight; a target the edit adds gets since (the edit's UTC stamp), the time its
    progress counts from."""
    screening=_offer_values(db,patient_id,values['disease'],values['screening_id'],kept is not None)
    suggested={t['key']:t for t in prevention.template(values['disease'],screening,_lifestyle_logs(db,patient_id,1))}
    targets=[]
    for t in values['targets']:
        if t['key'] in (kept or {}):
            old=kept[t['key']]
            targets.append({**t,'baseline':old.get('baseline'),**({'since':old['since']} if old.get('since') else {})});continue
        s=suggested.get(t['key'])
        if not s or not s['offered']:
            detail=s['reason'].split(': ',1)[-1] if s and s['reason'] else 'not in this starting point'
            raise ValueError(f"{prevention.TARGETS[t['key']].label} is not offered for this patient ({detail}).")
        targets.append({**t,'baseline':s['baseline'],**({'since':since} if since else {})})
    return {**values,'targets':targets}


def _sync_goals(db, patient_id, targets):
    """Write the plan's tracker goals over the patient's saved goals, leaving every other goal key as it was."""
    goals=prevention.tracker_goals(targets)
    if goals:
        row=db.execute("SELECT goals_json FROM lifestyle_goals WHERE patient_id=?",(patient_id,)).fetchone()
        db.execute("INSERT INTO lifestyle_goals VALUES(?,?,?) ON CONFLICT(patient_id) DO UPDATE SET goals_json=excluded.goals_json,updated_at=excluded.updated_at",
                   (patient_id,json.dumps({**(json.loads(row['goals_json']) if row else {}),**goals}),now()))
    return goals


def _plan_fields(values):
    return (values['disease'],values['screening_id'],json.dumps(values['targets']),values['note'],values['recheck_date'])


def _own_draft(db, doctor_id, patient_id, plan_id):
    row=db.execute("SELECT status FROM prevention_plans WHERE id=? AND patient_id=? AND doctor_id=?",(plan_id,patient_id,doctor_id)).fetchone()
    if not row:
        raise PermissionError('Only your own draft for this patient can be changed or approved.')
    if row['status']!='draft':
        raise ValueError('This plan is no longer a draft; edit the plan itself instead.')


def save_prevention_draft(doctor_id, patient_id, plan, plan_id=None):
    """Save the doctor's draft and return its id; the patient never sees a draft. Without plan_id the doctor's
    existing draft for this patient is updated when there is one, so each doctor keeps one draft per patient."""
    values=prevention.check_plan(plan,local_today())
    with connection() as db:
        if not _authorized_patient(db,doctor_id,patient_id):
            raise PermissionError('Patient access has not been granted.')
        if plan_id is None:
            row=db.execute("SELECT id FROM prevention_plans WHERE patient_id=? AND doctor_id=? AND status='draft' ORDER BY id DESC",(patient_id,doctor_id)).fetchone()
            plan_id=row['id'] if row else None
        else:
            _own_draft(db,doctor_id,patient_id,plan_id)
        values=_with_baselines(db,patient_id,values)
        stamp=now()
        if plan_id is None:
            plan_id=db.execute("""INSERT INTO prevention_plans(patient_id,doctor_id,disease,screening_id,targets_json,note,recheck_date,status,created_at,updated_at)
                                  VALUES(?,?,?,?,?,?,?,'draft',?,?)""",(patient_id,doctor_id,*_plan_fields(values),stamp,stamp)).lastrowid
            action='drafted'
        else:
            db.execute("UPDATE prevention_plans SET disease=?,screening_id=?,targets_json=?,note=?,recheck_date=?,updated_at=? WHERE id=?",
                       (*_plan_fields(values),stamp,plan_id))
            action='draft_edited'
        _prevention_audit(db,plan_id,doctor_id,action,_plan_details(values),stamp)
    return plan_id


def approve_prevention_plan(doctor_id, patient_id, plan, plan_id=None):
    """
    Approve a plan (a new one, or the doctor's draft plan_id) and return its id. It becomes this doctor's one active
    plan for the patient: their earlier active or paused plan stops and their other drafts are discarded, both kept in
    the history. Baselines are the patient's latest tracker values now; the plan's goals are written to the tracker
    goals (other goal keys untouched) and the patient is notified.
    """
    values=prevention.check_plan(plan,local_today())
    with connection() as db:
        if not _authorized_patient(db,doctor_id,patient_id):
            raise PermissionError('Patient access has not been granted.')
        if plan_id is not None:
            _own_draft(db,doctor_id,patient_id,plan_id)
        values=_with_baselines(db,patient_id,values)
        stamp=now()
        if plan_id is None:
            plan_id=db.execute("""INSERT INTO prevention_plans(patient_id,doctor_id,disease,screening_id,targets_json,note,recheck_date,status,started_at,created_at,updated_at)
                                  VALUES(?,?,?,?,?,?,?,'active',?,?,?)""",(patient_id,doctor_id,*_plan_fields(values),stamp,stamp,stamp)).lastrowid
        else:
            db.execute("UPDATE prevention_plans SET disease=?,screening_id=?,targets_json=?,note=?,recheck_date=?,status='active',started_at=?,updated_at=? WHERE id=?",
                       (*_plan_fields(values),stamp,stamp,plan_id))
        replaced=False
        for r in db.execute("SELECT id,status FROM prevention_plans WHERE patient_id=? AND doctor_id=? AND id<>? AND status IN ('draft','active','paused')",
                            (patient_id,doctor_id,plan_id)).fetchall():
            db.execute("UPDATE prevention_plans SET status='stopped',updated_at=? WHERE id=?",(stamp,r['id']))
            draft=r['status']=='draft';replaced|=not draft
            _prevention_audit(db,r['id'],doctor_id,'discarded' if draft else 'stopped',
                              {'reason':'another plan approved' if draft else 'replaced','plan_id':plan_id},stamp)
        _prevention_audit(db,plan_id,doctor_id,'approved',_plan_details(values),stamp)
        goals=_sync_goals(db,patient_id,values['targets'])
    count=len(values['targets'])
    notify(patient_id,'plan','New prevention plan',
           f"Your doctor sent you a {DISEASES[values['disease']]['long_label']} prevention plan with {count} target{'s'*(count!=1)}."
           +(' It replaces your earlier plan.' if replaced else '')+(' Your tracker goals now match it.' if goals else '')
           +f" Re-check on {prevention.date_text(values['recheck_date'])}.",'medications',f'prevention-{plan_id}-approved')
    return plan_id


def update_prevention_plan(doctor_id, plan_id, plan):
    """
    Edit an active or paused plan's targets, note or re-check date (its condition and source screening stay); False,
    with no history row or notification, when nothing changed. Kept targets keep their baselines; an added target is
    checked against the patient's newest screening for the condition and counts its progress from this edit. The
    earlier version stays in the history. An active plan's goals are written to the tracker again, and the patient is
    notified.
    """
    when=plan.get('recheck_date')
    with connection() as db:
        row=db.execute("SELECT * FROM prevention_plans WHERE id=? AND doctor_id=?",(plan_id,doctor_id)).fetchone()
        if not row or not _authorized_patient(db,doctor_id,row['patient_id']):
            raise PermissionError('Plan access denied.')
        if row['status']=='draft':
            raise ValueError('This plan is still a draft; save the draft or approve it instead.')
        if row['status']=='stopped':
            raise ValueError('A stopped plan cannot be edited; approve a new plan instead.')
        changed=(when.isoformat() if isinstance(when,date) else str(when))!=row['recheck_date']
        values=prevention.check_plan({**plan,'disease':row['disease'],'screening_id':row['screening_id']},local_today(),changed)
        before={'targets':json.loads(row['targets_json']),'note':row['note'],'recheck_date':row['recheck_date']}
        stamp=now()
        values=_with_baselines(db,row['patient_id'],values,{t['key']:t for t in before['targets']},stamp)
        if all(values[k]==v for k,v in before.items()):
            return False
        db.execute("UPDATE prevention_plans SET targets_json=?,note=?,recheck_date=?,updated_at=? WHERE id=?",
                   (json.dumps(values['targets']),values['note'],values['recheck_date'],stamp,plan_id))
        _prevention_audit(db,plan_id,doctor_id,'edited',{**_plan_details(values),'before':before},stamp)
        if row['status']=='active':
            _sync_goals(db,row['patient_id'],values['targets'])
    notify(row['patient_id'],'plan','Prevention plan updated','Your doctor updated your prevention plan. Open My plan to see the targets.',
           'medications',f'prevention-{plan_id}-edited-{stamp}')
    return True


_PREVENTION_NOTES={
    'paused':('Prevention plan paused','Your doctor paused your prevention plan. Your tracker goals stay as they are.'),
    'active':('Prevention plan resumed','Your doctor resumed your prevention plan, and your tracker goals match it again.'),
    'stopped':('Prevention plan stopped','Your doctor stopped your prevention plan. Your tracker goals stay as they are.'),
}


def prevention_plan_status(doctor_id, plan_id, status):
    """Pause, resume ('active') or stop a plan; False, with no history row or notification, when it already has that
    status. Stopping a draft discards it without telling the patient. Resuming writes the goals to the tracker again."""
    if status not in ('active','paused','stopped'):
        raise ValueError('Invalid plan status.')
    with connection() as db:
        row=db.execute("SELECT * FROM prevention_plans WHERE id=? AND doctor_id=?",(plan_id,doctor_id)).fetchone()
        if not row or not _authorized_patient(db,doctor_id,row['patient_id']):
            raise PermissionError('Plan access denied.')
        if row['status']==status:
            return False
        if row['status']=='stopped':
            raise ValueError('A stopped plan cannot be paused or resumed; approve a new plan instead.')
        if row['status']=='draft' and status!='stopped':
            raise ValueError('A draft cannot be paused or resumed; approve it to start it.')
        stamp=now()
        db.execute("UPDATE prevention_plans SET status=?,updated_at=? WHERE id=?",(status,stamp,plan_id))
        _prevention_audit(db,plan_id,doctor_id,'discarded' if row['status']=='draft' else status,{},stamp)
        if row['status']=='draft':
            return True
        if status=='active':
            _sync_goals(db,row['patient_id'],json.loads(row['targets_json']))
    notify(row['patient_id'],'plan',*_PREVENTION_NOTES[status],'medications',f'prevention-status-{plan_id}-{status}-{stamp}')
    return True


def prevention_plans(patient_id, doctor_id=None):
    """
    The patient's view (doctor_id None): active and paused plans from every doctor, newest approved first; never a
    draft or a stopped plan. The doctor's view: only their own plans for this linked patient, drafts and stopped ones
    included, current ones first, each with its 'history'. Rows add targets, start_date and doctor_name.
    """
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        if doctor_id is None:
            return [_prevention_row(r) for r in db.execute(
                _PREVENTION_SELECT+" WHERE p.patient_id=? AND p.status IN ('active','paused') ORDER BY p.started_at DESC,p.id DESC",(patient_id,))]
        rows=db.execute(_PREVENTION_SELECT+""" WHERE p.patient_id=? AND p.doctor_id=?
            ORDER BY CASE p.status WHEN 'active' THEN 0 WHEN 'paused' THEN 0 WHEN 'draft' THEN 1 ELSE 2 END,p.created_at DESC,p.id DESC""",
            (patient_id,doctor_id)).fetchall()
        return [{**_prevention_row(r),'history':_prevention_history(db,r['id'])} for r in rows]


def prevention_history(doctor_id, plan_id):
    """One of the doctor's plans' history, oldest first: [{'action', 'details', 'created_at', 'text'}]."""
    with connection() as db:
        row=db.execute("SELECT patient_id FROM prevention_plans WHERE id=? AND doctor_id=?",(plan_id,doctor_id)).fetchone()
        if not row or not _authorized_patient(db,doctor_id,row['patient_id']):
            raise PermissionError('Plan access denied.')
        return _prevention_history(db,plan_id)


def active_plan(patient_id):
    """The prevention plan the patient's My plan page shows: the newest active one, else the newest paused one."""
    current=prevention_plans(patient_id)
    return next((p for p in current if p['status']=='active'),next(iter(current),None))


def prevention_progress(patient_id, plan, doctor_id=None, today=None):
    """prevention.progress() for a stored plan with the patient's tracker logs and 30x30 activity days; a doctor
    (doctor_id) must still be linked."""
    if doctor_id is not None and not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        logs=_lifestyle_logs(db,patient_id)
    return prevention.progress(plan,logs,challenge_days(patient_id),today or local_today())


def notify_recheck_due(patient_id, today=None):
    """Remind the patient, once per re-check date, to screen again when an active plan's re-check date has come."""
    today=today or local_today()
    for p in prevention_plans(patient_id):
        if p['status']=='active' and p['recheck_date']<=today.isoformat():
            notify(patient_id,'plan','Time to screen again',f"Your prevention plan's re-check date ({prevention.date_text(p['recheck_date'])}) has come. "
                   'Screen again so you and your doctor can see how things have changed.','screenings',f"prevention-recheck-{p['id']}-{p['recheck_date']}")


def patient_screenings(doctor_id, patient_id):
    if not allowed(doctor_id,patient_id):
        raise PermissionError('Patient access denied.')
    with connection() as db:
        return [dict(r) for r in db.execute("SELECT * FROM screenings WHERE owner_id=? ORDER BY created_at DESC,id DESC",(patient_id,))]


def analytics(doctor_id):
    """Linked patients and their saved assessments."""
    people=linked_patients(doctor_id)
    ids=[p['id'] for p in people]
    if not ids:
        return people,[]
    with connection() as db:
        marks=','.join('?' for _ in ids)
        assessments=[dict(r) for r in db.execute(f'SELECT * FROM screenings WHERE owner_id IN ({marks}) ORDER BY created_at DESC',ids)]
    return people,assessments


def activity_counts(doctor_id):
    """(linked patients active in the last 30 local days, linked patients with 2+ logins, a Counter of those 30 days' events by (local date, type)) over every stored event."""
    since=_since()
    linked='patient_id IN (SELECT patient_id FROM care_access WHERE doctor_id=?)'
    with connection() as db:
        active=db.execute(f'SELECT COUNT(DISTINCT patient_id) FROM activity_events WHERE {linked} AND created_at>=?',(doctor_id,since)).fetchone()[0]
        returning=db.execute(f"SELECT COUNT(*) FROM (SELECT patient_id FROM activity_events WHERE {linked} AND type='login' GROUP BY patient_id HAVING COUNT(*)>=2)",(doctor_id,)).fetchone()[0]
        # Counted per UTC hour in SQL so only the buckets are converted; every UTC hour lies within one Dubai day.
        hours=db.execute(f'SELECT substr(created_at,1,13) hour,type,COUNT(*) n FROM activity_events WHERE {linked} AND created_at>=? GROUP BY hour,type',(doctor_id,since)).fetchall()
    daily=Counter()
    for r in hours:daily[(to_local(r['hour']).date(),r['type'])]+=r['n']
    return active,returning,daily


def append_record_prediction(owner_id, record_id, disease, values, score, threshold, model, source='batch'):
    with connection() as db:
        if not db.execute('SELECT 1 FROM records WHERE id=? AND owner_id=? AND disease=?',(record_id,owner_id,disease)).fetchone():
            raise PermissionError('Record access denied.')
        db.execute('INSERT INTO record_predictions(record_id,owner_id,disease,values_json,score,threshold,model,source,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                   (record_id,owner_id,disease,json.dumps(values),score,threshold,model,source,now()))


def record_prediction_history(owner_id, record_id):
    with connection() as db:
        return [dict(r) for r in db.execute('SELECT * FROM record_predictions WHERE owner_id=? AND record_id=? ORDER BY id DESC',(owner_id,record_id))]
