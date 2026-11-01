"""Personal dashboard drawn exclusively from the signed-in patient's saved records."""
from datetime import date
import pandas as pd
import streamlit as st
from ahead.components import condition_cards, empty_state, go_to, section_heading, disclaimer
from ahead.platform_data import lifestyle_history,get_goals,challenge_days,scheduled_doses,local_today
from ahead.storage import load_screenings
from ahead.wellness import challenge_summary
from ahead.visuals import summary_card,score_ring,progress_card,category_chart,trend,calendar,COLORS


def render_patient_dashboard():
    owner=st.session_state.user['id']
    records=lifestyle_history(owner); goals=get_goals(owner); days=challenge_days(owner)
    assessments=load_screenings(owner); today=local_today(); challenge=challenge_summary(days)
    due=[d for d in scheduled_doses(owner,today,today) if d['status'] in ('upcoming','missed')]
    goal_count=min(len(goals),6)
    cards=[('screenings','Screenings saved',len(assessments),None,'#8E7CC3','screenings',
            'Your saved screening results' if assessments else 'Take your first screening to start your history',
            'View screenings' if assessments else 'Start a screening'),
           ('goals','Goals set',goal_count,6,'#2E9477','wellness','Targets for steps, sleep, water and more',
            'Review goals' if goal_count else 'Set goals'),
           ('challenge','Challenge days',challenge['completed'],30,'#C98F1E','wellness','Days with 30+ active minutes','Log activity'),
           ('doses','Doses today',len(due),None,'#C9706E','medications',
            'Scheduled today, not yet logged' if due else 'All caught up for today','Medications')]
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
    with st.expander('Your 30×30 activity calendar',expanded=bool(days)):
        if days:calendar(days,challenge['start'],today)
        else:
            st.write('Record your first activity day in the tracker to begin your 30-day view.')
        if st.button('Open activity challenge'):go_to('wellness')
    section_heading('Make time for prevention','Choose a condition to start your guided screening.')
    condition_cards('patient_dashboard')
    if assessments:
        section_heading('Recent saved assessments','Model scores are educational outputs, not diagnoses.')
        st.dataframe(pd.DataFrame([{'Date':r['created_at'][:16],'Condition':r['disease'].title(),'Model':r['model'],'Score (%)':round(r['score']*100,1),'Result':'Flagged' if r['prediction'] else 'Not flagged'} for r in assessments[:8]]),hide_index=True,width='stretch')
    disclaimer()
