"""Patient medication schedules and explicit dose logs. Prescriptions are doctor controlled."""
from datetime import date,timedelta

import pandas as pd
import streamlit as st

from ahead.components import page_header
from ahead.i18n import tr
from ahead.visuals import metric,adherence_chart
from ahead.platform_data import dose_entries,mark_dose,notify,plans,scheduled_doses,local_today


def refresh_medication_notifications(patient_id):
    today=local_today()
    for dose in scheduled_doses(patient_id,today-timedelta(days=1),today):
        if dose['status'] in ('upcoming','missed'):
            kind='medication_due' if dose['date']==today else 'medication_missed'
            notify(patient_id,kind,'Medication '+('due' if kind=='medication_due' else 'missed'),
                   f"{dose['name']} · {dose['dosage']} · {dose['date']} {dose['time']}",'medications',
                   f"{kind}-{dose['plan_id']}-{dose['date']}-{dose['time']}")


def render_medications():
    owner=st.session_state.user['id']
    page_header('Patient care','Medication Monitoring','Follow plans assigned by your doctor and record doses.')
    st.info('Only your linked doctor can create or change a plan. If a dose is missed, do not double up without advice from your clinician.')
    refresh_medication_notifications(owner)
    active=plans(owner)
    today=local_today()
    today_doses=scheduled_doses(owner,today,today)
    cols=st.columns(3)
    for col,args in zip(cols,[('Active plans',sum(p['status']=='active' for p in active),'Assigned by your doctor','#2563EB'),('Scheduled today',len(today_doses),'Your daily care routine','#8B5CF6'),('Logged taken',sum(d['status']=='taken' for d in today_doses),'Today’s recorded doses','#10B981')]):
        with col:metric(*args)
    tab1,tab2,tab3=st.tabs([tr('Today'),tr('Plans'),tr('Adherence history')])
    with tab1:
        doses=scheduled_doses(owner,today,today)
        if not doses:st.info('No active medication doses scheduled today.')
        for d in doses:
            with st.container(border=True):
                st.write(f"**{d['name']} · {d['dosage']}**  — {d['time']}  · {d['status'].title()}")
                st.caption(d['instructions'])
                c1,c2=st.columns(2)
                if c1.button(tr('Taken'),key=f"taken_{d['plan_id']}_{d['time']}"):
                    try:mark_dose(owner,d['plan_id'],d['date'],d['time'],'taken');st.rerun()
                    except (ValueError,PermissionError) as error:st.error(str(error))
                if c2.button(tr('Skipped'),key=f"skipped_{d['plan_id']}_{d['time']}"):
                    try:mark_dose(owner,d['plan_id'],d['date'],d['time'],'skipped');st.rerun()
                    except (ValueError,PermissionError) as error:st.error(str(error))
        st.caption('A due dose appears in the in-app notification center. External push, email and SMS reminders are not configured.')
    with tab2:
        if not active:st.info('No doctor-assigned medication plans yet. Give your doctor access under Settings → Preferences.')
        for p in active:
            with st.container(border=True):
                st.write(f"**{p['name']} · {p['dosage']}** — {p['status']}")
                st.write(f"Times: {', '.join(p['times'])} · Days: {', '.join(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d] for d in p['days'])}")
                st.caption(f"{p['start_date']} to {p['end_date']} · {p['instructions']} · {p['notes']}")
    with tab3:
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
