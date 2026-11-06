"""Patient-only lifestyle assessment, longitudinal goals and 30×30 activity."""
import math
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from ahead import prevention as pv
from ahead.components import markdown_literal, page_header
from ahead.platform_data import (challenge_days,food_history,to_local,get_goals,lifestyle_history,prevention_plans,
                                 save_lifestyle,set_challenge_day,set_goals)
from ahead.wellness import challenge_summary,evaluate
from ahead.platform_data import local_today
from ahead.visuals import score_ring, category_cards, category_chart, progress_card, calendar, trend, metric, COLORS
from ahead.theme import style_figure


# The Goals form's fields a prevention plan can set (prevention.tracker_goals), in the form's order.
PLAN_GOALS={'weight':'target weight','exercise':'weekly exercise minutes','sleep':'sleep target','fruit_veg':'fruit/vegetable servings'}
# The Check-in form starts from the latest check-in, so saving after changing one field (say sleep, from a My plan card)
# never logs a made-up weight or activity; these values are used only before the first check-in, or for a missing field.
CHECK_IN_DEFAULTS={'age':30,'height_cm':170,'weight_kg':75.0,'steps':5000,'exercise_days':2,'exercise_minutes':30,
                   'sleep_hours':7.0,'sleep_quality':3,'water_litres':1.5,'fruit_veg':2,'sugary_food':1,'fast_food':0,
                   'stress':5,'sedentary_hours':7.0}
ACTIVITY_LEVELS=['Mostly sitting','Some movement','Active','Very active']
# Where a My plan card's target is logged, shown once when the patient arrives from that card (medications.py).
TRACKER_HINTS={'weight':'update Weight (kg) in the Check-in below',
               'sleep':'update Hours of sleep in the Check-in below',
               'fruit_veg':'update Fruit/vegetable servings per day in the Check-in below',
               'activity':"log each day's minutes in the 30×30 Challenge tab (My plan counts the last 7 days), or update "
                          'exercise days and minutes in the Check-in below'}


def check_in_start(values, key, low, high):
    """A Check-in field's starting value: the latest check-in's (values, {} before the first), inside the field's
    bounds and of its default's type (whole numbers stay whole), else CHECK_IN_DEFAULTS[key]."""
    default=CHECK_IN_DEFAULTS[key]
    try:number=float(values[key])
    except (KeyError,TypeError,ValueError):return default
    if not math.isfinite(number):return default
    number=min(max(number,low),high)
    return round(number) if isinstance(default,int) else float(number)


def plan_goals(owner, goals):
    """{tracker goal key: the current prevention plan that set it}, for goals still at that plan's value (the patient
    may have changed them since). Plans come newest approved first, so a value two plans share names the newer one."""
    found={}
    for plan in prevention_plans(owner):
        for key in pv.goals_from_plan(plan['targets'],goals):found.setdefault(key,plan)
    return found


def _plan_note(found):
    """Which goals came from which plan and doctor, for the top of the Goals tab (Markdown, names escaped)."""
    lines=[]
    for plan in {id(p):p for p in found.values()}.values():
        names=[label for key,label in PLAN_GOALS.items() if found.get(key) is plan]
        listed=names[0] if len(names)==1 else f"{', '.join(names[:-1])} and {names[-1]}"
        line=(f"Set by your prevention plan with {markdown_literal(plan['doctor_name'])}"
              f"{' (paused)' if plan['status']=='paused' else ''}: {listed}.")
        sleep=next((t['goal'] for t in plan['targets'] if t['key']=='sleep'),None)
        if 'sleep target' in names and sleep and sleep[0]!=sleep[1]:
            line+=f" The sleep target is the middle of the plan's {pv.goal_text('sleep',sleep,full=False)} range."
        lines.append(line)
    return ('\n\n'.join(lines)+'\n\nYou can change these goals here. My plan still shows your progress against the '
            'goals your doctor set.') if lines else ''


def render_wellness():
    owner=st.session_state.user['id']
    page_header('Patient wellness','Health & Lifestyle Tracker','Check in on your lifestyle, follow your progress and build healthier routines.')
    st.info('Lifestyle scores are educational summaries of your entries, not diagnoses or clinical risk scores.')
    goals=get_goals(owner)
    last=next(iter(lifestyle_history(owner)),{'values':{}})['values']
    if hint:=TRACKER_HINTS.get(st.session_state.pop('tracker_hint',None)):
        # Only say the form starts from the last check-in when there is one; a first check-in starts from defaults.
        st.info(f'From My plan: {hint}.'+(' The Check-in starts from your last one, so change only what is new.' if last
                                         else ' Fill in the whole Check-in once; later ones start from it.'))
    tracker,history,goal_tab,challenge=st.tabs(['Check-in','Progress','Goals','30×30 Challenge'])
    with tracker:
        start=lambda key,low,high:check_in_start(last,key,low,high)
        with st.form('wellness_form'):
            if last:st.caption('Filled in from your last check-in. Change what is new, then save.')
            a,b,c=st.columns(3)
            with a:
                age=st.number_input('Age',1,120,start('age',1,120))
                height=st.number_input('Height (cm)',90,250,start('height_cm',90,250))
                weight=st.number_input('Weight (kg)',25.0,400.0,start('weight_kg',25,400),step=0.5)
                steps=st.number_input('Daily steps',0,100000,start('steps',0,100000),step=500)
                level=last.get('activity_level')
                active=st.selectbox('Daily activity level',ACTIVITY_LEVELS,
                                    index=ACTIVITY_LEVELS.index(level) if level in ACTIVITY_LEVELS else 0)
            with b:
                days=st.number_input('Exercise days per week',0,7,start('exercise_days',0,7))
                minutes=st.number_input('Minutes per exercise day',0,360,start('exercise_minutes',0,360))
                sleep=st.number_input('Hours of sleep',0.0,24.0,start('sleep_hours',0,24),step=0.5)
                quality=st.slider('Sleep quality (1–5)',1,5,start('sleep_quality',1,5))
                water=st.number_input('Water (litres/day)',0.0,12.0,start('water_litres',0,12),step=0.25)
            with c:
                fruit=st.number_input('Fruit/vegetable servings per day',0,20,start('fruit_veg',0,20))
                sugar=st.number_input('Sugary foods/drinks per day',0,20,start('sugary_food',0,20))
                fast=st.number_input('Fast food meals per day',0,10,start('fast_food',0,10))
                stress=st.slider('Stress (0–10)',0,10,start('stress',0,10))
                sedentary=st.number_input('Hours sitting per day',0.0,24.0,start('sedentary_hours',0,24),step=0.5)
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
        if st.session_state.pop('goals_saved',False):st.toast('Goals saved.',icon=':material/check_circle:')
        from_plan=plan_goals(owner,goals)
        if note:=_plan_note(from_plan):st.info(note)
        hint=lambda key:(f"Set by your prevention plan with {markdown_literal(from_plan[key]['doctor_name'])}."
                         if key in from_plan else None)
        with st.form('wellness_goals'):
            c1,c2,c3=st.columns(3)
            weight_goal=c1.number_input('Target weight (kg)',25.0,400.0,float(goals.get('weight',75.0)),help=hint('weight'))
            step_goal=c2.number_input('Daily step goal',1000,100000,int(goals.get('steps',8000)),step=500)
            exercise_goal=c3.number_input('Weekly exercise minutes',30,2000,int(goals.get('exercise',150)),help=hint('exercise'))
            water_goal=c1.number_input('Daily water target (L)',0.5,10.0,float(goals.get('water',2.0)),step=0.25)
            sleep_goal=c2.number_input('Sleep target (h)',6.0,12.0,float(goals.get('sleep',8.0)),step=0.5,help=hint('sleep'))
            fruit_goal=c3.number_input('Fruit/vegetable servings target',1,15,int(goals.get('fruit_veg',5)),help=hint('fruit_veg'))
            if st.form_submit_button('Save goals'):
                goals={'weight':weight_goal,'steps':step_goal,'exercise':exercise_goal,'water':water_goal,'sleep':sleep_goal,'fruit_veg':fruit_goal}
                # Rerun so the plan note above reads the saved goals (a toast survives the rerun; st.success would not).
                set_goals(owner,goals);st.session_state.goals_saved=True;st.rerun()
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
