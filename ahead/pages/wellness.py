"""Patient-only lifestyle assessment, longitudinal goals and 30×30 activity."""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from ahead.components import page_header
from ahead.platform_data import (challenge_days,food_history,to_local,get_goals,lifestyle_history,
                                 save_lifestyle,set_challenge_day,set_goals)
from ahead.wellness import challenge_summary,evaluate
from ahead.platform_data import local_today
from ahead.visuals import score_ring, category_cards, category_chart, progress_card, calendar, trend, metric, COLORS
from ahead.theme import style_figure


def render_wellness():
    owner=st.session_state.user['id']
    page_header('Patient wellness','Health & Lifestyle Tracker','Check in on your lifestyle, follow your progress and build healthier routines.')
    st.info('Lifestyle scores are educational summaries of your entries, not diagnoses or clinical risk scores.')
    goals=get_goals(owner)
    tracker,history,goal_tab,challenge=st.tabs(['Check-in','Progress','Goals','30×30 Challenge'])
    with tracker:
        with st.form('wellness_form'):
            a,b,c=st.columns(3)
            with a:
                age=st.number_input('Age',1,120,30)
                height=st.number_input('Height (cm)',90,250,170)
                weight=st.number_input('Weight (kg)',25.0,400.0,75.0,step=0.5)
                steps=st.number_input('Daily steps',0,100000,5000,step=500)
                active=st.selectbox('Daily activity level',['Mostly sitting','Some movement','Active','Very active'])
            with b:
                days=st.number_input('Exercise days per week',0,7,2)
                minutes=st.number_input('Minutes per exercise day',0,360,30)
                sleep=st.number_input('Hours of sleep',0.0,24.0,7.0,step=0.5)
                quality=st.slider('Sleep quality (1–5)',1,5,3)
                water=st.number_input('Water (litres/day)',0.0,12.0,1.5,step=0.25)
            with c:
                fruit=st.number_input('Fruit/vegetable servings per day',0,20,2)
                sugar=st.number_input('Sugary foods/drinks per day',0,20,1)
                fast=st.number_input('Fast food meals per day',0,10,0)
                stress=st.slider('Stress (0–10)',0,10,5)
                sedentary=st.number_input('Hours sitting per day',0.0,24.0,7.0,step=0.5)
            submitted=st.form_submit_button('Save lifestyle check-in',type='primary')
        if submitted:
            values={'age':age,'height_cm':height,'weight_kg':weight,'steps':steps,'activity_level':active,
                    'exercise_days':days,'exercise_minutes':minutes,'sleep_hours':sleep,'sleep_quality':quality,
                    'water_litres':water,'fruit_veg':fruit,'sugary_food':sugar,'fast_food':fast,
                    'stress':stress,'sedentary_hours':sedentary}
            result=evaluate(values,goals)
            save_lifestyle(owner,values,result)
            st.success('Check-in saved. Your lifestyle summary is below.')
        records=lifestyle_history(owner)
        if records:
            latest=records[0]; result=latest['scores']
            left,right=st.columns([1,2])
            with left:
                score_ring(result['overall'],'Lifestyle / Wellness Score',note='Educational lifestyle assessment · not a medical diagnosis')
            with right:
                category_chart(result['categories'])
                st.caption(f"BMI: {result['bmi']} · Calculated from your entered height and weight; not part of the wellness score.")
                if len(records)>1: st.caption(f"Since previous check-in: {result['overall']-records[1]['scores']['overall']:+d} points")
            category_cards(result)
            st.subheader('Practical suggestions')
            for index,advice in enumerate(result['recommendations'],1):
                with st.expander(f'{index:02d} · Practical next step',expanded=True):st.write(advice)
    with history:
        records=lifestyle_history(owner)
        if not records:st.info('Save your first check-in to see progress.')
        else:
            df=pd.DataFrame([{'Date':r['created_at'][:10],'Score':r['scores']['overall'],
                              'Weight (kg)':r['values']['weight_kg'],'Activity minutes':r['scores']['exercise_weekly_minutes'],
                              'Sleep (h)':r['values']['sleep_hours'],'Sleep quality (1–5)':r['values']['sleep_quality'],'Hydration (L)':r['values']['water_litres'],
                              'Nutrition':r['scores']['categories']['Nutrition']} for r in reversed(records)])
            interval=st.segmented_control('View',['Weekly','Monthly'],default='Weekly')
            df['Period']=pd.to_datetime(df['Date']).dt.to_period('W' if interval=='Weekly' else 'M').astype(str)
            st.dataframe(df.iloc[::-1],hide_index=True,width='stretch')
            chart_columns=st.columns(2)
            for i,field in enumerate(['Score','Weight (kg)','Activity minutes','Sleep (h)','Nutrition','Hydration (L)','Sleep quality (1–5)']):
                chart=df.groupby('Period',as_index=False)[field].mean()
                with chart_columns[i%2]:
                    trend(chart,'Period',field,field,list(COLORS.values())[i%5])
        meals=food_history(owner)
        if meals:
            st.subheader('Confirmed food entries')
            st.caption('Nutrition figures below come from your confirmed estimates, not from the wellness scoring formula.')
            food_frame=pd.DataFrame([{'Date':to_local(r['created_at']).date().isoformat(), 'Calories (estimated)':r['nutrients'].get('calories',0),
                 'Protein (g, estimated)':r['nutrients'].get('protein',0),'Food':', '.join(r['foods'])} for r in meals])
            st.dataframe(food_frame,hide_index=True,width='stretch')
    with goal_tab:
        with st.form('wellness_goals'):
            c1,c2,c3=st.columns(3)
            weight_goal=c1.number_input('Target weight (kg)',25.0,400.0,float(goals.get('weight',75.0)))
            step_goal=c2.number_input('Daily step goal',1000,100000,int(goals.get('steps',8000)),step=500)
            exercise_goal=c3.number_input('Weekly exercise minutes',30,2000,int(goals.get('exercise',150)))
            water_goal=c1.number_input('Daily water target (L)',0.5,10.0,float(goals.get('water',2.0)),step=0.25)
            sleep_goal=c2.number_input('Sleep target (h)',6.0,12.0,float(goals.get('sleep',8.0)),step=0.5)
            fruit_goal=c3.number_input('Fruit/vegetable servings target',1,15,int(goals.get('fruit_veg',5)))
            if st.form_submit_button('Save goals'):
                goals={'weight':weight_goal,'steps':step_goal,'exercise':exercise_goal,'water':water_goal,'sleep':sleep_goal,'fruit_veg':fruit_goal}
                set_goals(owner,goals);st.success('Goals saved.')
        records=lifestyle_history(owner)
        if records:
            v=records[0]['values']
            goal_columns=st.columns(3)
            for i,(name,amount,target,unit) in enumerate([('Daily steps',v['steps'],step_goal,'steps'),('Weekly activity',v['exercise_days']*v['exercise_minutes'],exercise_goal,'min'),('Water',v['water_litres'],water_goal,'L'),('Sleep',v['sleep_hours'],sleep_goal,'h'),('Fruit/vegetables',v['fruit_veg'],fruit_goal,'servings')]):
                with goal_columns[i%3]:progress_card(name,amount,target,unit,['#0E9AA7','#C98F1E','#5B8FA3','#14556B','#2E9477'][i])
            with goal_columns[2]:metric('Weight target',f"{weight_goal:g} kg",f"Latest: {v['weight_kg']:g} kg · difference: {v['weight_kg']-weight_goal:+g} kg",'#14556B','◎')
            st.caption(f"Current weight: {v['weight_kg']:g} kg · target: {weight_goal:g} kg. Weight is a personal goal, not part of the score.")
        meals=food_history(owner)
        if meals:st.caption(f'{len(meals)} confirmed meal entries are saved in your nutrition history. Food photos do not automatically change the lifestyle score; use them to inform your next check-in.')
    with challenge:
        st.markdown("**Move for at least 30 minutes a day, 30 days in a row.** Walking, cycling, sports or anything else "
                    "that gets you moving counts. Log your minutes each day: every day you reach 30 minutes is marked "
                    "complete on the calendar and adds to your streak.")
        days=challenge_days(owner)
        start=min((date.fromisoformat(day) for day in days),default=local_today())
        status=challenge_summary(days,start)
        cols=st.columns(4)
        for col,label,value in zip(cols,['Day','Completed','Current streak','Longest streak'],[f"{status['day']}/30",f"{status['completed']}/30",status['streak'],status['longest']]):col.metric(label,value)
        st.progress(status['percent']/100,text=f"{status['percent']}% of 30 days")
        if saved:=st.session_state.pop('activity_saved',None):st.toast(saved,icon=':material/check_circle:')
        with st.container(key='activity_log'):
            st.markdown("**:material/directions_run: Log activity**")
            today=local_today()
            when=st.pills('When',['Today','Yesterday','Other date'],default='Today',required=True,key='activity_when')
            if when=='Other date':
                chosen=st.date_input('Pick a day',value=today-timedelta(days=2),min_value=today-timedelta(days=29),max_value=today,format='DD/MM/YYYY',width=260)
            else:
                chosen=today-timedelta(days=1 if when=='Yesterday' else 0)
            logged=int(days.get(chosen.isoformat(),0))
            length=st.pills('How long',['15 min','30 min','45 min','60 min','Custom'],default='30 min',required=True,key='activity_length')
            activity=st.number_input('Minutes',0,1440,logged or 30,width=260) if length=='Custom' else int(length.split()[0])
            if logged:st.caption(f"Already logged for this day: {logged} min. Saving replaces it.")
            with st.container(horizontal=True,vertical_alignment='center'):
                clicked=st.button('Save activity',type='primary')
                st.caption(':material/check_circle: 30 minutes or more counts as a complete day.')
            if clicked:
                if (chosen-start).days>=30:st.error('This challenge began more than 30 days before the selected date. Choose a day within your 30-day window.')
                else:
                    set_challenge_day(owner,chosen.isoformat(),activity)
                    st.session_state.activity_saved=f"Saved {activity} min for {chosen:%A}, {chosen.day} {chosen:%B}."
                    st.rerun()
        calendar(days,start,local_today())
        if status['completed']>=7:st.success('Achievement: 7 active days!')
        if status['completed']==30:st.balloons()
