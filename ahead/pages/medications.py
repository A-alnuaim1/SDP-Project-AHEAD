"""
The patient's My plan page. Its nav key stays 'medications', so every notification's Open still lands here.

It shows the prevention plan a linked doctor approved (ahead/prevention.py) with its progress from the tracker, then the
medicines a doctor prescribed: today's doses to record, the plans and their 30-day history. Plans and prescriptions are
doctor controlled, a draft never shows here, and doctors never get this page (app.py keeps it patient-only).
"""
from datetime import timedelta
from html import escape

import pandas as pd
import streamlit as st

from ahead.components import current_user,go_to,markdown_literal,page_header,section_heading,track_panel
from ahead.config import DISEASES
from ahead.pages.settings import open_care_team
from ahead.platform_data import (doctors_for,local_today,mark_dose,notify,notify_recheck_due,open_doses,plans,
                                 prevention_plans,prevention_progress,scheduled_doses,to_local)
from ahead.prevention import date_text
from ahead.resources import icon_uri
from ahead.storage import load_screenings
from ahead.triage import open_urgent
from ahead.visuals import adherence_chart

WEEKDAYS=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# Today's doses as pills: (text, palette tone). An unlogged dose past its time reads 'Due' while its plan is active,
# and 'Missed' once a later dose of the same plan has come due (taking both together would double up); one that fell
# due before its plan was paused or stopped reads 'Not logged', with the warning under it.
DOSE_PILLS={'taken':('Taken','#2E9477'),'skipped':('Skipped','#C9706E'),'missed':('Due','#C98F1E'),
            'late':('Missed','#C98F1E'),'closed':('Not logged','#C98F1E'),'upcoming':('Due at {time}','#5B8FA3')}


def refresh_medication_notifications(patient_id):
    """The My plan reminders: due and missed doses of active medication plans, and the re-check of an active prevention
    plan once its date has come. Each is sent once (dedupe keys)."""
    today=local_today()
    for dose in open_doses(patient_id,today-timedelta(days=1),today):
        kind='medication_due' if dose['date']==today.isoformat() else 'medication_missed'
        notify(patient_id,kind,'Medication '+('due' if kind=='medication_due' else 'missed'),
               f"{dose['name']} · {dose['dosage']} · {dose['date']} {dose['time']}",'medications',
               f"{kind}-{dose['plan_id']}-{dose['date']}-{dose['time']}")
    notify_recheck_due(patient_id,today)


def _listed(names):
    return names[0] if len(names)==1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _plan_label(plan):
    """A plan in the chooser shown when plans from more than one doctor are current."""
    return f"{plan['doctor_name']} · {DISEASES[plan['disease']]['long_label']}"+(' · paused' if plan['status']=='paused' else '')


def render_medications():
    if current_user()['is_doctor']:
        st.error('Doctor access required.');return
    owner=st.session_state.user['id']
    refresh_medication_notifications(owner)
    today=local_today()
    # Active plans first (newest approved first), then paused ones: the first is the one platform_data.active_plan picks.
    current=sorted(prevention_plans(owner),key=lambda p:p['status']!='active')
    plan=next((p for p in current if p['id']==st.session_state.get('plan_choice')),current[0] if current else None)
    medicines=plans(owner)
    if saved:=st.session_state.pop('dose_saved',None):st.toast(saved,icon=':material/check_circle:')
    if not plan:
        page_header('My plan','My plan','The plan your doctor agrees with you and the medicines they prescribe, in one place.')
        _no_plan(owner)
        if medicines:_medicines(owner,medicines,today)
    else:
        condition=DISEASES[plan['disease']]['long_label']
        agreed=f"Agreed on {date_text(plan['start_date'])}" if plan['start_date'] else 'Agreed'
        page_header('My plan',f"Your plan with {plan['doctor_name']}",
                    (f"{agreed} after your {condition} screening." if plan['screening_id'] is not None else
                     f"{agreed} as your {condition} prevention plan.")
                    +' Your doctor chose these targets from standard health advice.')
        if len(current)>1:
            st.segmented_control('Plan',[p['id'] for p in current],default=plan['id'],required=True,key='plan_choice',
                                 format_func=lambda pid:markdown_literal(_plan_label(next(p for p in current if p['id']==pid))),
                                 label_visibility='collapsed')
        if plan['status']=='paused':
            _notice('Paused by your doctor',f"{plan['doctor_name']} paused this plan. Your targets and progress stay here "
                    'and your tracker goals stay as they are, until your doctor resumes it.')
        urgent=_urgent(owner,plan)
        progress=prevention_progress(owner,plan,today=today)
        if urgent:   # the plan goes alongside the visit, so the hero never points only at the re-check
            progress={**progress,'message':progress['message'].replace('until your re-check','alongside your doctor visit')}
        _hero(owner,plan,progress,medicines,today)
        _targets(progress)
        if medicines:
            left,right=st.columns([1.2,1],gap='large')
            with left:_medicines(owner,medicines,today)
            with right:
                section_heading('From your doctor','Their note and your next re-check.')
                _note(plan);_recheck(plan,progress['recheck'],urgent)
        else:
            section_heading('From your doctor','Their note and your next re-check.')
            left,right=st.columns(2)
            with left:_note(plan)
            with right:_recheck(plan,progress['recheck'],urgent)
    if medicines:_history(owner,medicines,today)


def _urgent(owner,plan):
    """
    A result for the plan's condition still on the urgent track (triage.open_urgent: only a later screening that
    re-entered the value clears it), as its track panel above the hero, so the plan never reads as the next step
    instead of the doctor's visit that result needs. Returns True when shown.
    """
    row,result=next(((row,result) for row,result in open_urgent(load_screenings(owner))
                     if row['disease']==plan['disease']),(None,None))
    if not row:return False
    moment=to_local(row['created_at'])
    track_panel(dict(result,reasons=result['reasons'][:1]),'patient',
                note=f"From your {DISEASES[row['disease']]['card_title'].lower()} on {moment.day} {moment:%b %Y}. "
                     'This plan goes alongside that visit, not instead of it.')
    return True


def _notice(title,text,calm=False):
    st.html(f'''<div class="plan-notice{' calm' if calm else ''}"><div class="summary-icon" aria-hidden="true">
<img src="{icon_uri('dashboard/plan.svg')}" alt=""></div><div><strong>{escape(title)}</strong><p>{escape(text)}</p></div></div>''')


def _no_plan(owner):
    """The calm empty state: where a plan comes from, and the share step when no doctor is linked."""
    _notice('No prevention plan yet','Plans come from a doctor you have linked. When a screening comes back at risk, '
            'they can agree a few targets with you from standard health advice, such as being active, eating fruit and '
            'vegetables and sleeping well. Your plan shows here once they approve it.',calm=True)
    names=sorted((d['name'] for d in doctors_for(owner)),key=str.lower)
    if names:
        st.caption(f"Your linked doctor{'s' if len(names)>1 else ''}, {markdown_literal(_listed(names))}, can see your "
                   'saved screenings and can set up a plan with you.')
    else:
        with st.container(key='plan_share_row',horizontal=True,vertical_alignment='center'):
            if st.button('Share with your doctor',key='plan_share',type='primary'):open_care_team()
            st.caption('No doctor can see your results yet. Add your doctor under Care team access in Settings.')


def _hero(owner,plan,progress,medicines,today):
    """'n of N on track' ring, the encouraging line naming what needs attention, and the doses and re-check figures."""
    tracked,on=progress['tracked'],progress['on_track']
    if tracked:
        ring=(f'<div class="plan-ring" style="--share:{on/tracked*100:.0f}%" role="img" aria-label="{on} of {tracked} '
              f'targets on track"><div class="plan-ring-center"><strong>{on}/{tracked}</strong><small>on track</small></div></div>')
    else:   # only stop smoking, which the doctor checks at the re-check
        ring=(f'<div class="plan-ring" aria-hidden="true"><div class="plan-ring-center">'
              f'<img src="{icon_uri("dashboard/plan.svg")}" alt=""></div></div>')
    stats=[]
    if medicines:
        week=[d for d in scheduled_doses(owner,today-timedelta(days=6),today) if d['status']!='upcoming']
        stats.append(('Doses this week',f"{sum(d['status']=='taken' for d in week)} of {len(week)}" if week else 'None due',
                      'Doses taken of those due in the last 7 days'))
    if recheck:=progress['recheck']:
        stats.append(('Re-check','Due',f"Re-check date: {recheck['date_text']}") if recheck['due'] else
                     ('Re-check in',recheck['short'],f"Re-check on {recheck['date_text']}"))
    badge='<span class="plan-hero-badge">Paused</span>' if plan['status']=='paused' else ''
    figures=''.join(f'<div class="plan-stat" title="{escape(tip)}">{escape(label)}<strong>{escape(value)}</strong></div>'
                    for label,value,tip in stats)
    st.html(f'''<div class="plan-hero"><div class="plan-hero-main">{ring}<div class="plan-hero-text">{badge}
<div class="plan-hero-title">{escape(progress['headline'])}</div><p>{escape(progress['message'])}</p></div></div>
<div class="plan-hero-stats">{figures}</div></div>''')


def _targets(progress):
    section_heading('Your targets','Progress comes from your tracker check-ins and activity log.')
    items=progress['targets']
    per_row=max(1,len(items)) if len(items)<=4 else 3   # five cards (a current smoker) as rows of three and two, never five narrow ones
    for start in range(0,len(items),per_row):
        for column,item in zip(st.columns(per_row),items[start:start+per_row]):
            with column:_target_card(item,progress['recheck'])


def _target_card(item,recheck):
    """One target in the summary-card style: value / goal, a progress bar, its status and note, and the button to
    where it is logged. Stop smoking has no daily figure, so it is a tile with the re-check date."""
    share=item['share']
    if item['tracked']:
        value=f"{escape(item['current_text'])} <small>{escape(item['of_goal'])}</small>"
        width=0 if share is None else max(3,share*100)
        bar=(f'<div class="summary-bar" role="progressbar" aria-label="{escape(item["label"])}" aria-valuemin="0" '
             f'aria-valuemax="100" aria-valuenow="{0 if share is None else round(share*100)}"><span style="width:{width:.0f}%"></span></div>')
    else:
        value=f"{recheck['date'].day} {recheck['date']:%b} <small>re-check</small>" if recheck else escape(item['goal_text'])
        bar=''
    go=(f'<div class="summary-go" aria-hidden="true"><span>{escape(item["action"])}</span><span>→</span></div>'
        if item['page'] else '')
    body=f'''<div class="summary-inner plan-card" style="--tone:{item['color']}">
<div class="plan-card-head"><div class="summary-icon"><img src="{icon_uri(f"dashboard/{item['icon']}.svg")}" alt=""></div>
<span class="plan-status" style="--pill:{item['tone']}">{escape(item['status_text'])}</span></div>
<span class="summary-label">{escape(item['label'])}</span><strong class="summary-value">{value}</strong>{bar}
<p class="summary-note">{escape(item['note'])}</p>{go}</div>'''
    if not item['page']:
        with st.container(key=f"plan_{item['key']}_summary_tile"):st.html(body)
        return
    with st.container(key=f"plan_{item['key']}_summary_card"):
        st.html(body)
        if st.button(f"{item['action']} →",key=f"plan_{item['key']}",width='stretch'):
            if item['page']=='wellness':st.session_state.tracker_hint=item['key']   # where to log it (wellness.TRACKER_HINTS)
            go_to(item['page'])


def _note(plan):
    if plan['note']:
        text=escape(plan['note']).replace('\n','<br>')
        st.html(f'<div class="plan-note"><p>“{text}”</p><span>— {escape(plan["doctor_name"])}</span></div>')
    else:
        st.html(f'<div class="plan-note"><p>{escape(plan["doctor_name"])} did not add a note to this plan.</p></div>')


def _recheck(plan,recheck,urgent=False):
    """The re-check date, or 'Re-check due' once it has come, with the button that opens this condition's screening.
    With an urgent result shown above (urgent), a re-check still to come says it is not the next step."""
    if not recheck:return
    if recheck['due']:
        title,text=('Re-check due',f"Your re-check date ({recheck['date_text']}) has come. Screen again so you and your "
                    'doctor can see how things have changed.')
    else:
        title=f"Re-check on {recheck['date_text']}"
        text=("This is your plan's re-check, not the next step: see a doctor about the result above first." if urgent else
              "We'll remind you to screen again." if plan['status']=='active' else
              'Screen again on or after this date.')
    with st.container(key='plan_recheck_due' if recheck['due'] else 'plan_recheck',horizontal=True,vertical_alignment='center'):
        st.html(f'<div class="plan-recheck"><strong>{escape(title)}</strong><span>{escape(text)}</span></div>')
        if st.button('Screen again',key='plan_screen_again',type='primary' if recheck['due'] else 'secondary'):
            go_to('screenings',plan['disease'])


def _medicines(owner,medicines,today):
    """Today's doses from every medication plan, to record as taken or skipped (shown only once a doctor prescribed)."""
    section_heading('Medicines today','Prescribed by your doctor. Record each dose when you take it.')
    st.info('Only a doctor you have linked can create or change a plan, and removing a doctor’s access stops their plans. If a dose is missed, do not double up without advice from your clinician.')
    status={p['id']:p['status'] for p in medicines}
    doses=scheduled_doses(owner,today,today)
    latest={}   # each plan's latest dose time that has come due today ('HH:MM' strings sort as times)
    for dose in doses:
        if dose['status']!='upcoming':latest[dose['plan_id']]=max(latest.get(dose['plan_id'],''),dose['time'])
    with st.container(key='plan_medicines'):
        if not doses:st.caption('No doses are scheduled today.')
        for dose in doses:_dose(owner,dose,status[dose['plan_id']],latest.get(dose['plan_id'],''))
    st.caption('A due dose appears in the in-app notification center. External push, email and SMS reminders are not configured.')


def _dose(owner,dose,plan_status,latest=''):
    """
    One dose: Take/Skip once it is due (mark_dose refuses one that is not), and a logged dose can be changed. An
    unlogged dose whose plan has a later dose due (latest, 'HH:MM') counts as missed, and one that fell due before its
    plan was paused or stopped keeps the warning not to take it now: both get record wording (Taken/Skipped).
    """
    pid,time,state=dose['plan_id'],dose['time'],dose['status']
    closed=plan_status!='active' and state=='missed'
    late=state=='missed' and not closed and latest>time
    text,tone=DOSE_PILLS['closed' if closed else 'late' if late else state]
    if state in ('taken','skipped'):
        choices=[('skipped','Mark skipped')] if state=='taken' else [('taken','Mark taken')]
    elif state=='missed':
        record=closed or late
        choices=[('taken','Taken' if record else 'Take'),('skipped','Skipped' if record else 'Skip')]
    else:
        choices=[]   # not due yet
    slot=f"{pid}_{time.replace(':','')}"
    problem=None
    with st.container(key=f'dose_block_{slot}'):
        with st.container(key=f'dose_row_{slot}',horizontal=True,vertical_alignment='center'):
            how=f"<span>{escape(dose['instructions'])}</span>" if dose['instructions'] else ''
            st.html(f'<div class="dose-main"><strong>{escape(dose["name"])} · {escape(dose["dosage"])}</strong>'
                    f'<span>{escape(time)}</span>{how}</div>')
            st.html(f'<span class="plan-status" style="--pill:{tone}">{escape(text.format(time=time))}</span>')
            for value,label in choices:
                primary=value=='taken' and state=='missed' and not (closed or late)
                if st.button(label,key=f'{value}_{pid}_{time}',type='primary' if primary else 'secondary'):
                    try:
                        mark_dose(owner,pid,dose['date'],time,value)
                        st.session_state.dose_saved=f"{markdown_literal(dose['name'])} at {time} recorded as {value}."
                        st.rerun()
                    except (ValueError,PermissionError) as error:problem=str(error)
        if closed:st.warning(f"This plan was {plan_status} after this dose was due. Record it only if you already took it; do not take it now.")
        elif late:st.caption(f'Record it only if you already took it. If you missed it, do not take it together with your '
                             f'{latest} dose; ask your clinician if you are unsure.')
        if problem:st.error(problem)


def _history(owner,medicines,today):
    """Every medication plan and the last 30 days' dose log, kept from the Medication Monitoring page."""
    linked={d['id'] for d in doctors_for(owner)}
    with st.expander('Medication plans and 30-day history'):
        for p in medicines:
            with st.container(border=True):
                st.markdown(f"**{markdown_literal(p['name'])} · {markdown_literal(p['dosage'])}** — {p['status']}")
                st.markdown(f"Times: {', '.join(p['times'])} · Days: {', '.join(WEEKDAYS[d] for d in p['days'])}")
                st.caption(' · '.join([f"{p['start_date']} to {p['end_date']}",
                                       *(markdown_literal(p[k]) for k in ('instructions','notes') if p[k])]))
                if p['doctor_id'] not in linked:st.caption('The prescribing doctor no longer has access to this plan. Talk to a clinician before changing how you take this medication.')
        start=today-timedelta(days=29)
        schedule=scheduled_doses(owner,start,today)
        past=[r for r in schedule if r['status']!='upcoming']
        taken=sum(x['status']=='taken' for x in past)
        st.metric('Last 30 days adherence',f'{taken/len(past):.0%}' if past else 'No due doses')
        adherence_chart(past)
        if not schedule:st.info('No doses in the last 30 days.')
        else:
            st.dataframe(pd.DataFrame(past).rename(columns={'date':'Date','time':'Time','status':'Status','name':'Medication'}),hide_index=True,width='stretch')
            st.caption('Missed is inferred from the schedule when no dose was logged; it is not proof of whether medication was taken.')
