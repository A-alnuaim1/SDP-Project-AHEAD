"""Account-scoped in-app notifications."""
import streamlit as st
from datetime import timedelta
from ahead.components import go_to,page_header
from ahead.i18n import tr
from ahead.visuals import metric
from ahead.platform_data import mark_notification,notifications,lifestyle_history,get_goals,challenge_days,notify,local_today
from ahead.pages.medications import refresh_medication_notifications


def render_notifications():
    owner=st.session_state.user['id']
    page_header('Patient updates','Notifications','Reminders and changes for your account.')
    refresh_patient_notifications(owner)
    rows=notifications(owner)
    a,b=st.columns(2)
    with a:metric('Unread',sum(not r['read_at'] for r in rows),'Updates waiting for your attention','#8B5CF6','◉')
    with b:metric('All notifications',len(rows),'Your personal update center','#2563EB','◷')
    unread_only=st.toggle('Show unread only',value=False)
    if unread_only:rows=[r for r in rows if not r['read_at']]
    if not rows:st.info('No notifications yet. Reminders appear here while you use the app.')
    for row in rows:
        with st.container(border=True):
            st.write(('🔵 ' if not row['read_at'] else '')+f"**{row['title']}** · {row['created_at'][:16]}")
            st.caption(row['description'])
            c1,c2=st.columns(2)
            if not row['read_at'] and c1.button(tr('Mark as read'),key=f"read_{row['id']}"):
                mark_notification(owner,row['id']);st.rerun()
            if row['action'] and c2.button(tr('Open'),key=f"open_{row['id']}"):
                mark_notification(owner,row['id']);go_to(row['action'])


def refresh_patient_notifications(owner):
    refresh_medication_notifications(owner)
    today=local_today()
    history=lifestyle_history(owner)
    if history and history[0]['created_at'][:10] < (today-timedelta(days=7)).isoformat():
        notify(owner,'assessment','Lifestyle check-in','Your last lifestyle check-in was over a week ago. Update it when convenient.','wellness',f'weekly-checkin-{today.isocalendar().year}-{today.isocalendar().week}')
    if get_goals(owner) and (not history or history[0]['created_at'][:10] < today.isoformat()):
        notify(owner,'goal','Lifestyle goal reminder','Review your goals and record today’s progress if you wish.','wellness',f'goal-{today.isoformat()}')
    days=challenge_days(owner)
    if days and today.isoformat() not in days:
        notify(owner,'challenge','30×30 activity','You can manually record your activity minutes for today.','wellness',f'challenge-{today.isoformat()}')
