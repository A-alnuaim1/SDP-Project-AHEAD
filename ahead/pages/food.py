"""Food photo preview and optional real Gemini image analysis with manual confirmation."""
import base64
import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from ahead.components import page_header
from ahead.platform_data import food_history,food_on_day,local_today,save_food,to_local
from ahead.resources import gemini_api_key,gemini_interaction,gemini_error_message
from ahead.visuals import amount,day_totals

FIELDS={'calories':'kcal','protein':'g','carbs':'g','fat':'g','sugar':'g','fiber':'g','sodium':'mg'}
DAY_KEY='food_day_picker'
EARLIEST=365   # days back the day picker reaches


def analyze_photo(data,mime):
    key=gemini_api_key()
    if not key:raise RuntimeError('Image analysis is not configured. Enter the food and nutrition values manually.')
    prompt=('Identify visible foods and approximate portions in this photo. Return ONLY JSON with keys '
            '"foods" (array of visible food names), "portion" (short estimate), "calories", "protein", '
            '"carbs", "fat", "sugar", "fiber", "sodium" (numbers, use null if unknown), '
            '"summary" (one short factual sentence), "uncertainty" (what cannot be determined). '
            'Nutrition is an uncertain estimate; do not invent exact ingredients or assume hidden oils/sauces. '
            'If this is not food, return an empty foods array and null nutrients.')
    raw=gemini_interaction([
        {"type":"text","text":prompt},
        {"type":"image","data":base64.b64encode(data).decode("ascii"),"mime_type":mime},
    ], response_format={"type":"text","mime_type":"application/json"})
    parsed=json.loads(raw)
    if not isinstance(parsed,dict) or not isinstance(parsed.get('foods'),list):raise ValueError('No usable foods detected. Enter them manually.')
    return parsed


def _shift_day(step):
    """Previous/next-day buttons: move the day picker one day, within its range."""
    today=local_today()
    day=st.session_state.get(DAY_KEY,today)+timedelta(days=step)
    st.session_state[DAY_KEY]=min(today,max(today-timedelta(days=EARLIEST),day))


def _day_title(day,today,meals):
    name={today:'Today',today-timedelta(days=1):'Yesterday'}.get(day)
    count=f"{len(meals)} meal{'s' if len(meals)!=1 else ''}" if meals else 'no meals'
    when=f'{day.day} {day:%B %Y}'
    return (f'<div class="food-day-title"><strong>{name or f"{day:%A}"}</strong>'
            f'<span>{f"{day:%A} " if name else ""}{when} · {count}</span></div>')


def render_daily_totals(owner):
    """Daily totals card: pick a day, see its meals added up against general daily values."""
    today=local_today()
    earliest=today-timedelta(days=EARLIEST)
    # Keep the picker inside its range, e.g. when the page stays open past midnight; the widget
    # raises if its stored value falls outside min/max, and it has not been created yet in this run.
    st.session_state[DAY_KEY]=min(today,max(earliest,st.session_state.get(DAY_KEY,today)))
    day=st.session_state[DAY_KEY]
    meals=food_on_day(owner,day)
    st.subheader('Daily totals')
    with st.container(key='food_day'):
        with st.container(horizontal=True,vertical_alignment='center',horizontal_alignment='distribute'):
            with st.container(horizontal=True,vertical_alignment='center',gap='small',key='food_day_nav'):
                st.button('Previous day',key='food_prev_day',on_click=_shift_day,args=(-1,),disabled=day<=earliest)
                st.html(_day_title(day,today,meals),width='content')
                st.button('Next day',key='food_next_day',on_click=_shift_day,args=(1,),disabled=day>=today)
            with st.popover('Pick a day',icon=':material/calendar_month:'):
                st.date_input('Day',key=DAY_KEY,min_value=earliest,max_value=today,format='DD/MM/YYYY',
                              label_visibility='collapsed')
        if not meals:
            st.html('<div class="food-empty">'+('No meals logged today yet. Meals you confirm above add up here.'
                    if day==today else 'No meals were logged on this day.')+'</div>')
        else:
            totals={field:sum(amount(m['nutrients'].get(field)) for m in meals) for field in FIELDS}
            day_totals(totals,[{'time':to_local(m['created_at']).strftime('%H:%M'),'foods':', '.join(map(str,m['foods'])),
                                'portion':str(m['nutrients'].get('portion','')),'calories':amount(m['nutrients'].get('calories')),
                                'photo':m['source']=='gemini_estimate_confirmed'} for m in meals],day==today)
    if meals:
        st.caption('Daily values are general references for adults on a 2,000-calorie diet (US FDA). Your needs depend on '
                   'age, body size, activity and health; a doctor or dietitian can suggest personal targets. '
                   'Totals add up your confirmed estimates, so they are approximate.')


def render_food():
    owner=st.session_state.user['id']
    page_header('Patient nutrition','Food Nutrition Analyzer',"Understand what's on your plate: log meals, review their nutrients and track how you eat over time.")
    st.info('All nutrients are rough estimates; photos cannot reveal exact ingredients or portion weights. Images are not stored in AHEAD.')
    uploaded=st.file_uploader('Upload or take a food photo',type=['jpg','jpeg','png','webp'],key='meal_photo')
    if uploaded:
        if uploaded.size>8*1024*1024:st.error('Photo exceeds 8 MB.');return
        st.image(uploaded,width=400)
        if not gemini_api_key():st.warning('Image analysis is unavailable on this deployment. Use manual entry below; no AI estimate will be shown.')
        elif st.button('Analyze Food',type='primary'):
            with st.spinner('Analyzing the photo with Gemini…'):
                try:
                    result=analyze_photo(uploaded.getvalue(),uploaded.type)
                    st.session_state.meal_analysis=result
                    st.session_state.meal_photo_signature=(uploaded.name,uploaded.size)
                except Exception as exc:
                    st.error(gemini_error_message(exc))
    result=st.session_state.get('meal_analysis',{})
    if uploaded and st.session_state.get('meal_photo_signature')!=(uploaded.name,uploaded.size):result={}
    if result:
        st.caption('Gemini image estimate · please correct every value before saving.')
        if result.get('summary'):st.info(str(result['summary']))
        if result.get('uncertainty'):st.warning(str(result['uncertainty']))
    with st.form('confirm_meal'):
        foods=st.text_input('Foods (comma separated)',value=', '.join(map(str,result.get('foods',[]))))
        portion=st.text_input('Portion description',value=str(result.get('portion') or ''))
        estimates={}
        cols=st.columns(4)
        for i,(field,unit) in enumerate(FIELDS.items()):
            value=result.get(field)
            try:value=max(0.0,float(value))
            except (ValueError,TypeError):value=0.0
            estimates[field]=cols[i%4].number_input(field.title()+f' ({unit})',min_value=0.0,max_value=100000.0,value=value)
        submitted=st.form_submit_button('Confirm and save meal')
    if submitted:
        selected=[x.strip() for x in foods.split(',') if x.strip()]
        if not selected or not portion.strip():st.error('Enter foods and a portion before saving.')
        else:
            source='gemini_estimate_confirmed' if result else 'manual_estimate'
            save_food(owner,selected,{**estimates,'portion':portion.strip()},source)
            st.session_state.pop('meal_analysis',None)
            st.session_state[DAY_KEY]=local_today()   # show the day the meal was added to
            st.success('Confirmed meal saved. The numbers remain approximate.')
            labels=[]
            if estimates['protein']>=20:labels.append('Contains at least 20 g estimated protein')
            if estimates['fiber']>=5:labels.append('Contains at least 5 g estimated fiber')
            if estimates['sugar']>=20:labels.append('Contains at least 20 g estimated sugar')
            if estimates['sodium']>=600:labels.append('Contains at least 600 mg estimated sodium')
            if labels:st.info('Per estimated portion: '+', '.join(labels)+'. Check packaging or a food database for reliable numbers.')
    render_daily_totals(owner)
    records=food_history(owner)
    st.subheader('Meal history')
    if not records:st.info('No confirmed meals yet.')
    else:
        st.dataframe(pd.DataFrame([{'Date':to_local(r['created_at']).strftime('%Y-%m-%d %H:%M'),'Foods':', '.join(r['foods']),
            'Portion':r['nutrients']['portion'],**{k:r['nutrients'].get(k) for k in FIELDS},'Source':r['source']} for r in records]),hide_index=True,width='stretch')
        st.caption('High/low nutrition labels require reliable ingredient and portion data; please review estimates yourself.')
