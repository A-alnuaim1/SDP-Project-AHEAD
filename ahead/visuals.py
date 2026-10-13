"""Shared, data-driven presentation components. No inference or storage mutations."""
from datetime import timedelta
from html import escape
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ahead.i18n import tr
from ahead.theme import style_figure, show_chart

COLORS = {'Nutrition':'#10B981','Activity':'#2563EB','Sleep':'#8B5CF6','Hydration':'#06B6D4','Habits':'#F97316'}
PALETTE = list(COLORS.values())

def wellness_status(score):
    if score < 50: return '#EF4444', 'Needs improvement'
    if score < 80: return '#F59E0B', 'Moderate'
    return '#10B981', 'Good'

def score_ring(value, label, status=None, color=None, note=''):
    value = max(0., min(100., float(value)))
    default_color, default_status = wellness_status(value)
    color, status = color or default_color, status or default_status
    st.html(f'''<div class="score-panel" style="--visual-color:{color}">
    <div class="score-ring" role="img" aria-label="{escape(label)}: {value:.1f} / 100, {escape(status)}">
    <svg viewBox="0 0 160 160" aria-hidden="true"><circle class="ring-track" cx="80" cy="80" r="68"/>
    <circle class="ring-value" cx="80" cy="80" r="68" pathLength="100" style="--ring-offset:{100-value};stroke-dashoffset:{100-value}"/></svg>
    <div class="ring-center"><strong>{value:.0f}<small>/100</small></strong></div></div>
    <h3>{escape(tr(label))}</h3><span class="visual-badge">{escape(tr(status))}</span><p>{escape(note)}</p></div>''')

def metric(label, value, note='', color='#2563EB', icon='✦'):
    st.html(f'''<div class="visual-metric" style="--visual-color:{color}"><span class="visual-icon" aria-hidden="true">{escape(icon)}</span>
    <span class="visual-label">{escape(tr(label))}</span><strong>{escape(str(value))}</strong><small>{escape(note)}</small></div>''')

def progress_card(label, amount, target, unit='', color='#2563EB'):
    percentage = min(100, max(0, amount / target * 100)) if target else 0
    status = 'Target reached' if percentage >= 100 else 'In progress'
    st.html(f'''<div class="goal-card" style="--visual-color:{color}"><span class="visual-label">{escape(tr(label))}</span>
    <strong>{amount:g} <small>/ {target:g} {escape(unit)}</small></strong>
    <div class="visual-track" role="progressbar" aria-valuenow="{percentage:.0f}" aria-valuemin="0" aria-valuemax="100" aria-label="{escape(label)}"><span style="width:{percentage}%"></span></div>
    <small>{percentage:.0f}% · {escape(tr(status))}</small></div>''')

def category_cards(result):
    for col, (name, value) in zip(st.columns(5), result['categories'].items()):
        with col:
            metric(name, f'{value}/100', wellness_status(value)[1], COLORS[name], {'Nutrition':'◈','Activity':'↗','Sleep':'☾','Hydration':'◉','Habits':'✦'}[name])
            with st.expander(tr('Why this score?')): st.write(result['explanations'][name])

def category_chart(categories):
    df=pd.DataFrame({'Category':list(categories),'Score':list(categories.values())})
    fig=px.bar(df,x='Score',y='Category',orientation='h',color='Category',color_discrete_map=COLORS,text='Score',range_x=[0,100],title='Your lifestyle balance')
    show_chart(style_figure(fig,300))

def trend(frame, x, y, title, color='#2563EB', area=False):
    fig=(px.area if area else px.line)(frame,x=x,y=y,markers=True,title=title)
    fig.update_traces(line_color=color)
    if area: fig.update_traces(fillcolor='rgba(37,99,235,0.10)')
    fig.update_layout(hovermode='x unified')
    show_chart(style_figure(fig,280))

def calendar(days,start,today):
    cells=[]
    for i in range(30):
        day=start+timedelta(days=i); minutes=days.get(day.isoformat(),0)
        state='future' if day>today else ('complete' if minutes>=30 else ('today' if day==today else 'incomplete'))
        label='Upcoming' if day>today else ('Complete' if minutes>=30 else 'Below 30 min')
        cells.append(f'<div class="calendar-day {state}" title="{day.isoformat()}: {minutes} minutes · {label}"><small>{day:%d %b}</small><strong>{"✓" if state=="complete" else i+1}</strong><span>{minutes} min</span><small>{label}</small></div>')
    st.html('<div class="challenge-calendar" aria-label="30 day activity calendar">'+''.join(cells)+'</div>')
    st.caption('✓ Complete: at least 30 minutes · Outlined: today · Pale: upcoming · Amber: past day below target')

def macros(nutrients):
    values=[float(nutrients.get(key) or 0) for key in ('protein','carbs','fat')]
    if any(not math.isfinite(v) or v<0 for v in values) or not any(values):
        st.info('Add confirmed protein, carbohydrate and fat estimates to see a macronutrient chart.')
        return
    fig=go.Figure(go.Pie(labels=['Protein (g)','Carbohydrates (g)','Fat (g)'],values=values,hole=.68,marker_colors=['#2563EB','#F59E0B','#8B5CF6'],textinfo='percent',hovertemplate='%{label}: %{value:g}<br>%{percent}<extra></extra>'))
    fig.update_layout(title='Macronutrient proportions by mass')
    show_chart(style_figure(fig,300,legend=True))
    st.caption('Shares use grams of these three macronutrients, not calorie percentages. All values are estimates.')

def adherence_chart(doses):
    if not doses: return
    df=pd.DataFrame(doses)
    counts=df.groupby(['date','status']).size().reset_index(name='Doses')
    fig=px.bar(counts,x='date',y='Doses',color='status',title='Dose log timeline',color_discrete_map={'taken':'#10B981','skipped':'#F59E0B','missed':'#EF4444','upcoming':'#2563EB'})
    show_chart(style_figure(fig,280,legend=True))
