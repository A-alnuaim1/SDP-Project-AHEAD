"""Food photo preview and optional real Gemini image analysis with manual confirmation."""
import json

import pandas as pd
import streamlit as st

from ahead.components import page_header
from ahead.i18n import tr
from ahead.platform_data import food_history,save_food
from ahead.resources import gemini_api_key,gemini_client,gemini_error_message
from ahead.visuals import metric,macros

FIELDS={'calories':'kcal','protein':'g','carbs':'g','fat':'g','sugar':'g','fiber':'g','sodium':'mg'}


def analyze_photo(data,mime):
    from google import genai
    from google.genai import types
    key=gemini_api_key()
    if not key:raise RuntimeError('Image analysis is not configured. Enter the food and nutrition values manually.')
    client=gemini_client()
    prompt=('Identify visible foods and approximate portions in this photo. Return ONLY JSON with keys '
            '"foods" (array of visible food names), "portion" (short estimate), "calories", "protein", '
            '"carbs", "fat", "sugar", "fiber", "sodium" (numbers, use null if unknown), '
            '"summary" (one short factual sentence), "uncertainty" (what cannot be determined). '
            'Nutrition is an uncertain estimate; do not invent exact ingredients or assume hidden oils/sauces. '
            'If this is not food, return an empty foods array and null nutrients.')
    response=client.models.generate_content(model='gemini-2.5-flash',contents=[prompt,types.Part.from_bytes(data=data,mime_type=mime)],
       config=types.GenerateContentConfig(response_mime_type='application/json'))
    raw=response.text or ''
    parsed=json.loads(raw)
    if not isinstance(parsed,dict) or not isinstance(parsed.get('foods'),list):raise ValueError('No usable foods detected. Enter them manually.')
    return parsed


def render_food():
    owner=st.session_state.user['id']
    page_header('Patient nutrition','Food Nutrition Analyzer','Review and confirm meal entries for your food history.')
    st.info('Photo analysis requires a configured Gemini vision API. All nutrients are rough estimates; photos cannot reveal exact ingredients or portion weights. Images are not stored in AHEAD.')
    uploaded=st.file_uploader(tr('Upload or take a food photo'),type=['jpg','jpeg','png','webp'],key='meal_photo')
    if uploaded:
        if uploaded.size>8*1024*1024:st.error('Photo exceeds 8 MB.');return
        st.image(uploaded,width=400)
        if not gemini_api_key():st.warning('Image analysis is unavailable on this deployment. Use manual entry below; no AI estimate will be shown.')
        elif st.button(tr('Analyze Food'),type='primary'):
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
        foods=st.text_input(tr('Foods (comma separated)'),value=', '.join(map(str,result.get('foods',[]))))
        portion=st.text_input(tr('Portion description'),value=str(result.get('portion') or ''))
        estimates={}
        cols=st.columns(4)
        for i,(field,unit) in enumerate(FIELDS.items()):
            value=result.get(field)
            try:value=max(0.0,float(value))
            except (ValueError,TypeError):value=0.0
            estimates[field]=cols[i%4].number_input(tr(field.title())+f' ({unit})',min_value=0.0,max_value=100000.0,value=value)
        submitted=st.form_submit_button(tr('Confirm and save meal'))
    if submitted:
        selected=[x.strip() for x in foods.split(',') if x.strip()]
        if not selected or not portion.strip():st.error('Enter foods and a portion before saving.')
        else:
            source='gemini_estimate_confirmed' if result else 'manual_estimate'
            save_food(owner,selected,{**estimates,'portion':portion.strip()},source)
            st.session_state.pop('meal_analysis',None)
            st.success('Confirmed meal saved. The numbers remain approximate.')
            labels=[]
            if estimates['protein']>=20:labels.append('Contains at least 20 g estimated protein')
            if estimates['fiber']>=5:labels.append('Contains at least 5 g estimated fiber')
            if estimates['sugar']>=20:labels.append('Contains at least 20 g estimated sugar')
            if estimates['sodium']>=600:labels.append('Contains at least 600 mg estimated sodium')
            if labels:st.info('Per estimated portion: '+', '.join(labels)+'. Check packaging or a food database for reliable numbers.')
    records=food_history(owner)
    if records:
        latest=records[0]['nutrients']
        st.subheader('Latest confirmed meal')
        st.caption(', '.join(records[0]['foods']) + ' · ' + latest.get('portion',''))
        cols=st.columns(4)
        colors=['#F97316','#2563EB','#F59E0B','#8B5CF6','#EC4899','#10B981','#EF4444']
        for i,(field,unit) in enumerate(FIELDS.items()):
            with cols[i%4]:metric(field.title(),f"{latest.get(field,0):g} {unit}",'Confirmed estimate',colors[i])
        macros(latest)
    st.subheader(tr('Meal history'))
    if not records:st.info('No confirmed meals yet.')
    else:
        st.dataframe(pd.DataFrame([{'Date':r['created_at'][:16],'Foods':', '.join(r['foods']),
            'Portion':r['nutrients']['portion'],**{k:r['nutrients'].get(k) for k in FIELDS},'Source':r['source']} for r in records]),hide_index=True,width='stretch')
        st.caption('High/low nutrition labels require reliable ingredient and portion data; please review estimates yourself.')
