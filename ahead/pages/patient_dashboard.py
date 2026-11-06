"""Personal dashboard drawn exclusively from the signed-in patient's saved records."""
import json
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from ahead.components import condition_cards, empty_state, go_to, section_heading, disclaimer, track_panel
from ahead.config import DISEASES
from ahead.platform_data import lifestyle_history,get_goals,challenge_days,open_doses,local_today,to_local
from ahead.screening import share_with_doctor
from ahead.storage import load_screenings
from ahead.triage import open_urgent, track_text
from ahead.wellness import challenge_summary
from ahead.visuals import summary_card,week_strip,score_ring,progress_card,category_chart,trend,COLORS

RECENT_DAYS=30   # an urgent result older than this no longer gets the home banner


def _screening_name(disease):
    return DISEASES[disease]['card_title'].lower()


def _urgent_banner(assessments,today):
    """
    One calm banner when a screening saved in the last RECENT_DAYS local days is still on the urgent track (see
    open_urgent): the newest such result with its first open reason, and who can see it. A result for another
    condition never hides it.
    """
    since=today-timedelta(days=RECENT_DAYS-1)
    urgent=open_urgent([row for row in assessments if to_local(row['created_at']).date()>=since])
    if not urgent:return
    (row,result),others=urgent[0],urgent[1:]
    moment=to_local(row['created_at'])
    note=f"From your {_screening_name(row['disease'])} on {moment.day} {moment:%b %Y}."
    later=sum(r['disease']==row['disease'] for r in assessments[:assessments.index(row)])   # newest first
    if later:   # say why an older result still shows
        note+=f" Your later {_screening_name(row['disease'])}{'s'*(later>1)} did not re-check this value."
    if others:
        dated=[f"{_screening_name(r['disease'])} on {to_local(r['created_at']).day} {to_local(r['created_at']):%b %Y}"
               for r,_ in others]
        note+=(f" Your {' and '.join(dated)} {'is' if len(others)==1 else 'are'} also marked "
               f"\"{track_text('urgent')['label']}\".")
    with st.container(key='home_track_banner'):
        track_panel(dict(result,reasons=result['reasons'][:1]),'patient',note=note)
        share_with_doctor('home_share')


def render_patient_dashboard():
    owner=st.session_state.user['id']
    records=lifestyle_history(owner); goals=get_goals(owner); days=challenge_days(owner)
    assessments=load_screenings(owner); today=local_today(); challenge=challenge_summary(days)
    _urgent_banner(assessments,today)
    due=open_doses(owner,today,today)
    goal_count=min(len(goals),6)
    cards=[('screenings','Screenings saved',len(assessments),None,'#8E7CC3','screenings',
            'Your saved screening results' if assessments else 'Take your first screening to start your history',
            'View screenings' if assessments else 'Start a screening'),
           ('goals','Goals set',goal_count,6,'#2E9477','wellness','Targets for steps, sleep, water and more',
            'Review goals' if goal_count else 'Set goals'),
           ('challenge','Challenge days',challenge['completed'],30,'#C98F1E','wellness','Days with 30+ active minutes','Log activity'),
           ('doses','Doses today',len(due),None,'#C9706E','medications',
            'Scheduled today, not yet logged' if due else 'All caught up for today','My plan')]
    for col,(icon,label,value,total,color,page,note,action) in zip(st.columns(4),cards):
        with col:
            if summary_card(f'dashboard_{icon}',label,value,note,action,color,icon,total):go_to(page)
    section_heading('Your wellness, at a glance','Small check-ins help you see how your habits change.')
    if not records:
        empty_state('◉','Your story starts with one check-in','Record your lifestyle to unlock your score, trends and personal progress.')
        if st.button('Complete your first check-in',type='primary'):go_to('wellness')
    else:
        latest=records[0];result=latest['scores'];v=latest['values']
        left,right=st.columns([1,2])
        with left:score_ring(result['overall'],'Lifestyle / Wellness Score',note=f"Latest check-in: {latest['created_at'][:10]} · Educational wellness score")
        with right:
            category_chart(result['categories'])
            if len(records)>1:st.caption(f"Change since previous check-in: {result['overall']-records[1]['scores']['overall']:+d} points")
        df=pd.DataFrame([{'Date':r['created_at'],'Wellness score':r['scores']['overall'],'Sleep (hours)':r['values']['sleep_hours'],'Sleep quality (1–5)':r['values']['sleep_quality']} for r in reversed(records)])
        if len(records)>1:
            a,b=st.columns(2)
            with a:trend(df,'Date','Wellness score','Your wellness journey',area=True)
            with b:trend(df,'Date','Sleep (hours)','Sleep duration','#14556B')
        else:st.info('One check-in saved. Save another to reveal your personal trends.')
        for col,args in zip(st.columns(3),[
            ('Daily steps',v['steps'],goals.get('steps',8000),'steps','#0E9AA7'),
            ('Water',v['water_litres'],goals.get('water',2),'L','#5B8FA3'),
            ('Fruit / vegetables',v['fruit_veg'],goals.get('fruit_veg',5),'servings','#2E9477')]):
            with col:progress_card(*args)
        st.caption('Progress reflects the latest self-reported check-in. Default targets are shown until you save your own goals.')
    with st.container(key='home_week'):
        week_strip(days,today)
        with st.container(horizontal=True,vertical_alignment='center',horizontal_alignment='distribute'):
            st.caption(f"{challenge['completed']} of 30 days in your 30×30 challenge" if days
                       else 'Start the 30×30 challenge: 30 active minutes a day for 30 days.')
            if st.button('Log activity →',type='primary',key='home_log_activity'):go_to('wellness')
    section_heading('Make time for prevention','Choose a condition to start your guided screening.')
    condition_cards('patient_dashboard')
    if assessments:
        section_heading('Recent saved assessments','Model scores are educational outputs, not diagnoses.')
        st.dataframe(pd.DataFrame([{'Date':r['created_at'][:16],'Condition':r['disease'].title(),'Model':r['model'],'Score (%)':round(r['score']*100,1),'Threshold (%)':round(r['threshold']*100,1),'Result':'Flagged' if r['prediction'] else 'Not flagged'} for r in assessments[:8]]),hide_index=True,width='stretch')
    disclaimer()
