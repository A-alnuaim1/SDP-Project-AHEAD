"""Progressive-enhancement browser speech interface; no medical decision making."""
from html import escape
from datetime import date

import streamlit as st

from ahead.components import go_to,page_header
from ahead.platform_data import lifestyle_history,scheduled_doses,local_today


def render_voice():
    owner=st.session_state.user['id']
    page_header('Patient access','Voice Assistant','Speak simple navigation and account-status questions.')
    st.info('Browser speech recognition may be unavailable. Your browser may process speech through its own service. Typed navigation below always works. No diagnosis or prescription advice is provided.')
    latest=lifestyle_history(owner)
    score=f"Your latest lifestyle score is {latest[0]['scores']['overall']} out of 100." if latest else 'There is no saved lifestyle score yet.'
    doses=scheduled_doses(owner,local_today(),local_today())
    due=[d for d in doses if d['status'] in ('upcoming','missed')]
    meds=f"You have {len(due)} scheduled doses to check today in Medication Monitoring." if due else 'No active medication doses are scheduled today.'
    phrases={'medication':meds,'score':score,'lifestyle':score,'assessment':'Open Disease Screenings using the button below.',
             'food':'Open Food Nutrition Analyzer using the button below.'}
    script='''<!doctype html><html><body style="font:16px sans-serif;color:#153b52">
<button id="talk" style="background:linear-gradient(120deg,#2563EB,#8B5CF6);color:white;border:0;border-radius:14px;padding:14px 24px;cursor:pointer;font-size:16px">🎙 Speak</button>
<p id="out" aria-live="polite">Press Speak and say: open medication, latest lifestyle score, or open assessments.</p>
<script>
const Speech=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!Speech){document.getElementById('talk').disabled=true;document.getElementById('out').textContent='Speech recognition unavailable in this browser. Use the page buttons.';}
else document.getElementById('talk').onclick=()=>{let r=new Speech();r.lang='en-US';r.onresult=e=>{
let q=e.results[0][0].transcript.toLowerCase();let response='I can help with your medication schedule, lifestyle score, food page, or assessments.';
const phrases=PHRASES;
for(const [word,answer] of Object.entries(phrases)){if(q.includes(word)){response=answer;break;}}
document.getElementById('out').textContent='Heard: '+q+' — '+response;
if(window.speechSynthesis){let u=new SpeechSynthesisUtterance(response);window.speechSynthesis.speak(u);}
};r.onerror=()=>document.getElementById('out').textContent='Could not hear that. Try again or use the buttons below.';r.start();};
</script></body></html>'''.replace('PHRASES',__import__('json').dumps(phrases).replace('</','<\\/'))
    st.iframe(script,height=140)
    st.caption('Voice recognition is optional. Use these buttons to navigate reliably:')
    cols=st.columns(4)
    for col,label,page in zip(cols,['Disease Assessments','Health Tracker','Medication','Food Analyzer'],['screenings','wellness','medications','food']):
        if col.button(label,width='stretch'):go_to(page)
    st.caption('Field dictation and Arabic speech recognition are not yet supported; those controls remain editable with the keyboard.')
