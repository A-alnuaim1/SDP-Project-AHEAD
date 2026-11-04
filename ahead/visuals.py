"""Shared, data-driven presentation components. No inference or storage mutations."""
from datetime import date, timedelta
from html import escape
import math
import pandas as pd
import plotly.express as px
import streamlit as st
from ahead.resources import icon_uri
from ahead.theme import style_figure, show_chart

COLORS = {'Nutrition':'#2E9477','Activity':'#0E9AA7','Sleep':'#14556B','Hydration':'#5B8FA3','Habits':'#C98F1E'}
PALETTE = list(COLORS.values())

def wellness_status(score):
    if score < 50: return '#C75D5D', 'Needs improvement'
    if score < 80: return '#C98F1E', 'Moderate'
    return '#2E9477', 'Good'

def score_ring(value, label, status=None, color=None, note=''):
    value = max(0., min(100., float(value)))
    default_color, default_status = wellness_status(value)
    color, status = color or default_color, status or default_status
    st.html(f'''<div class="score-panel" style="--visual-color:{color}">
    <div class="score-ring" role="img" aria-label="{escape(label)}: {value:.1f} / 100, {escape(status)}">
    <svg viewBox="0 0 160 160" aria-hidden="true"><circle class="ring-track" cx="80" cy="80" r="68"/>
    <circle class="ring-value" cx="80" cy="80" r="68" pathLength="100" style="--ring-offset:{100-value};stroke-dashoffset:{100-value}"/></svg>
    <div class="ring-center"><strong>{value:.0f}<small>/100</small></strong></div></div>
    <h3>{escape(label)}</h3><span class="visual-badge">{escape(status)}</span><p>{escape(note)}</p></div>''')

def metric(label, value, note='', color='#0E9AA7', icon='✦'):
    st.html(f'''<div class="visual-metric" style="--visual-color:{color}"><span class="visual-icon" aria-hidden="true">{escape(icon)}</span>
    <span class="visual-label">{escape(label)}</span><strong>{escape(str(value))}</strong><small>{escape(note)}</small></div>''')

def _summary_html(label, value, note, color, icon, total=None, action=''):
    out_of = f' <small>of {total}</small>' if total else ''
    bar = (f'<div class="summary-bar"><span style="width:{max(3, min(100, value / total * 100)):.0f}%"></span></div>'
           if total else '')
    go = f'<div class="summary-go" aria-hidden="true"><span>{escape(action)}</span><span>→</span></div>' if action else ''
    return f'''<div class="summary-inner" style="--tone:{color}">
<div class="summary-icon"><img src="{icon_uri(f'dashboard/{icon}.svg')}" alt=""></div>
<span class="summary-label">{escape(label)}</span><strong class="summary-value">{escape(str(value))}{out_of}</strong>{bar}
<p class="summary-note">{escape(note)}</p>{go}
</div>'''

def summary_card(key, label, value, note, action, color, icon, total=None):
    """A tinted dashboard card; the whole card is a button. Returns True when it is clicked."""
    with st.container(key=f'{key}_summary_card'):
        st.html(_summary_html(label, value, note, color, icon, total, action))
        return st.button(f'{action} →', key=key, width='stretch')

def summary_tile(key, label, value, note, color, icon, total=None):
    """The same tinted card without an action, for figures that don't lead anywhere."""
    with st.container(key=f'{key}_summary_tile'):
        st.html(_summary_html(label, value, note, color, icon, total))

def progress_card(label, amount, target, unit='', color='#0E9AA7'):
    percentage = min(100, max(0, amount / target * 100)) if target else 0
    status = 'Target reached' if percentage >= 100 else 'In progress'
    st.html(f'''<div class="goal-card" style="--visual-color:{color}"><span class="visual-label">{escape(label)}</span>
    <strong>{amount:g} <small>/ {target:g} {escape(unit)}</small></strong>
    <div class="visual-track" role="progressbar" aria-valuenow="{percentage:.0f}" aria-valuemin="0" aria-valuemax="100" aria-label="{escape(label)}"><span style="width:{percentage}%"></span></div>
    <small>{percentage:.0f}% · {escape(status)}</small></div>''')

def category_cards(result):
    for col, (name, value) in zip(st.columns(5), result['categories'].items()):
        with col:
            metric(name, f'{value}/100', wellness_status(value)[1], COLORS[name], {'Nutrition':'◈','Activity':'↗','Sleep':'☾','Hydration':'◉','Habits':'✦'}[name])
            with st.expander('Why this score?'): st.write(result['explanations'][name])

def category_chart(categories):
    df=pd.DataFrame({'Category':list(categories),'Score':list(categories.values())})
    fig=px.bar(df,x='Score',y='Category',orientation='h',color='Category',color_discrete_map=COLORS,text='Score',range_x=[0,100],title='Your lifestyle balance')
    show_chart(style_figure(fig,300))

def trend(frame, x, y, title, color='#0E9AA7', area=False):
    fig=(px.area if area else px.line)(frame,x=x,y=y,markers=True,title=title)
    fig.update_traces(line_color=color)
    if area: fig.update_traces(fillcolor='rgba(14,154,167,0.10)')
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

def week_strip(days, today):
    """Monday-to-Sunday view of the current week: a tick for 30+ active minutes, plus the current streak."""
    start = today - timedelta(days=today.weekday())
    first = min((date.fromisoformat(d) for d in days), default=today)   # days before the first log aren't "missed"
    day = today if days.get(today.isoformat(), 0) >= 30 else today - timedelta(days=1)
    streak = 0
    while days.get(day.isoformat(), 0) >= 30:
        streak += 1
        day -= timedelta(days=1)
    cells = []
    for i in range(7):
        day = start + timedelta(days=i)
        minutes = days.get(day.isoformat(), 0)
        state = ('done' if minutes >= 30 else 'future' if day > today else 'open' if day == today
                 else 'missed' if day >= first else 'before')
        mark = '✓' if state == 'done' else str(minutes) if minutes else '·' if state != 'missed' else '–'
        today_class = ' is-today' if day == today else ''
        cells.append(f'<div class="week-day {state}{today_class}" title="{day:%A} {day.day} {day:%B}: {minutes} active minutes">'
                     f'<span>{"Today" if day == today else f"{day:%a}"}</span><b>{mark}</b></div>')
    streak_text = f'{streak}-day streak' if streak else 'No streak yet'
    st.html(f'''<div class="week-head"><div><div class="week-title">This week</div>
<div class="week-sub">Days with 30+ active minutes</div></div>
<span class="week-streak" style="--flame:url('{icon_uri('dashboard/challenge.svg')}')">{streak_text}</span></div>
<div class="week-days" aria-label="Active days this week">{''.join(cells)}</div>''')

# General daily values for adults on a 2,000-calorie diet (US FDA Nutrition Facts label). A 'goal' is
# worth reaching, a 'limit' worth staying under; total sugar has no daily value (only added sugar does).
CALORIE_REFERENCE = 2000
NUTRIENT_GROUPS = [
    ('Macronutrients', [('protein', 'Protein', 'g', 50, 'reference', '#2E9477'),
                        ('carbs', 'Carbs', 'g', 275, 'reference', '#C98F1E'),
                        ('fat', 'Fat', 'g', 78, 'reference', '#5B8FA3')]),
    ('Fiber, sugar and salt', [('fiber', 'Fiber', 'g', 28, 'goal', '#2E9477'),
                               ('sodium', 'Sodium', 'mg', 2300, 'limit', '#C9706E'),
                               ('sugar', 'Sugar', 'g', None, None, '#8E7CC3')]),
]


def amount(value):
    """A stored nutrient as a number; missing or invalid entries count as 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def shown(value, unit):
    """The amount as displayed: whole kcal, mg and values from 100 up, otherwise one decimal."""
    return round(value) if unit in ('kcal', 'mg') or value >= 100 else round(value, 1)


def quantity(value, unit):
    amount = shown(value, unit)
    return f'{amount:,}' if isinstance(amount, int) else f'{amount:g}'


def percent(part, whole):
    """Whole percent that never rounds up to 100% before the amount gets there."""
    share = part / whole * 100
    return math.floor(share) if share < 100 else round(share)


def _pill(text, tone):
    return f'<span class="food-pill" style="--tone:{tone}">{escape(text)}</span>'


def _nutrient_row(key, label, unit, reference, kind, tone, total, is_today):
    icon = f'<i class="food-icon" style="--icon:url({icon_uri(f"food/{key}.svg")})" aria-hidden="true"></i>'
    head = f'{icon}<span>{escape(label)}</span><b>{quantity(total, unit)} {unit}</b>'
    if reference is None:
        return f'<div class="food-row" style="--tone:{tone}">{head}<span class="food-note food-wide">No daily value for total {label.lower()}</span></div>'
    # Judge the amount the row shows, so "28 g" never sits next to "0 g to go".
    share = percent(shown(total, unit), reference)
    limit = f'{reference:,} {unit}'
    if kind == 'goal' and share >= 100:
        note = _pill('Daily value reached', tone)
    elif kind == 'goal' and is_today:
        note = _pill(f'{quantity(reference - shown(total, unit), unit)} {unit} to go', tone)
    elif kind == 'limit' and shown(total, unit) > reference:
        note = _pill('Above the daily limit', '#C9706E')
    elif kind == 'limit' and share >= 80:
        note = _pill('Near the daily limit', '#C98F1E')
    elif kind == 'reference' and share >= 100:
        note = _pill(f'{share:.0f}% of {limit}', tone)
    else:
        note = f'<span class="food-note">{share:.0f}% of {limit}{" limit" if kind == "limit" else ""}</span>'
    bar = (f'<div class="visual-track food-track" role="progressbar" aria-valuenow="{min(100, share):.0f}" aria-valuemin="0" '
           f'aria-valuemax="100" aria-label="{escape(label)}"><span style="width:{min(100, share):.0f}%"></span></div>')
    return f'<div class="food-row" style="--tone:{tone}">{head}{bar}{note}</div>'


def day_totals(totals, meals, is_today):
    """
    One day's meals added up (Food Nutrition Analyzer): a calorie ring, where the calories came from,
    each nutrient against its general daily value, and the meals themselves. *meals* are dicts with
    time, foods, portion, calories and photo (True when the estimate came from a photo).
    """
    calories = totals['calories']
    share = percent(shown(calories, 'kcal'), CALORIE_REFERENCE)
    ring = f'''<div class="food-ring" style="--share:{min(100, share):.1f}%" role="img" aria-label="{quantity(calories, 'kcal')} kcal, {share:.0f}% of a {CALORIE_REFERENCE:,} kcal reference">
<div class="food-ring-center"><i style="--icon:url({icon_uri('food/calories.svg')})" aria-hidden="true"></i>
<strong>{quantity(calories, 'kcal')}</strong><small>kcal · {share:.0f}% of {CALORIE_REFERENCE:,}</small></div></div>'''
    energy = {'Protein': totals['protein'] * 4, 'Carbs': totals['carbs'] * 4, 'Fat': totals['fat'] * 9}
    split = ''
    if sum(energy.values()):
        parts = {name: value / sum(energy.values()) * 100 for name, value in energy.items()}
        tones = {'Protein': '#2E9477', 'Carbs': '#C98F1E', 'Fat': '#5B8FA3'}
        split = ('<div class="food-split-label">Where calories came from</div><div class="food-split" aria-hidden="true">'
                 + ''.join(f'<span style="width:{part:.1f}%;background:{tones[name]}"></span>' for name, part in parts.items())
                 + '</div><div class="food-split-legend">'
                 + ''.join(f'<span>{name} {part:.0f}%</span>' for name, part in parts.items()) + '</div>')
    groups = ''.join(f'<div class="food-group">{escape(title)}</div>'
                     + ''.join(_nutrient_row(*row, totals[row[0]], is_today) for row in rows)
                     for title, rows in NUTRIENT_GROUPS)
    meal_rows = ''.join(
        f'''<div class="food-meal"><span class="food-meal-icon" title="{'From a photo' if meal['photo'] else 'Entered by hand'}">
<i style="--icon:url({icon_uri('food/photo.svg' if meal['photo'] else 'food/manual.svg')})" aria-hidden="true"></i></span>
<time>{escape(meal['time'])}</time><span>{escape(meal['foods'])} <small>· {escape(meal['portion'])}</small></span>
<em>{quantity(meal['calories'], 'kcal')} kcal</em></div>''' for meal in meals)
    st.html(f'''<div class="food-day"><div class="food-ring-col">{ring}{split}</div>
<div class="food-groups">{groups}</div></div><div class="food-meals">{meal_rows}</div>''')


def adherence_chart(doses):
    if not doses: return
    df=pd.DataFrame(doses)
    counts=df.groupby(['date','status']).size().reset_index(name='Doses')
    fig=px.bar(counts,x='date',y='Doses',color='status',title='Dose log timeline',color_discrete_map={'taken':'#2E9477','skipped':'#C98F1E','missed':'#C75D5D','upcoming':'#0E9AA7'})
    show_chart(style_figure(fig,280,legend=True))
