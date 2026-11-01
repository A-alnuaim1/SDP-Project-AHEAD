"""Transparent lifestyle heuristics for education, never a clinical risk score."""
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from datetime import datetime


def clamp(value):
    return max(0, min(100, round(value)))


def evaluate(v, goals=None):
    goals=goals or {}
    minutes=v['exercise_days']*v['exercise_minutes']
    step_goal=max(1000,int(goals.get('steps',8000)))
    water_goal=max(0.5,float(goals.get('water',2.0)))
    sleep_goal=max(6,float(goals.get('sleep',8)))
    weekly_goal=max(30,int(goals.get('exercise',150)))
    nutrition=clamp(50+min(v['fruit_veg'],5)*10-min(v['fast_food'],2)*14-min(v['sugary_food'],4)*7)
    activity=clamp(15+min(v['steps']/step_goal,1)*35+min(minutes/weekly_goal,1)*45-min(v['sedentary_hours'],16)*1.5)
    sleep=clamp(100-abs(v['sleep_hours']-sleep_goal)*13+(v['sleep_quality']-3)*7)
    hydration=clamp(100*min(v['water_litres']/water_goal,1))
    habits=clamp(90-v['stress']*6-min(v['sedentary_hours'],16)*2)
    categories={'Nutrition':nutrition,'Activity':activity,'Sleep':sleep,'Hydration':hydration,'Habits':habits}
    overall=round(sum(categories.values())/len(categories))
    why={
        'Nutrition':f"Fruit/vegetable servings: {v['fruit_veg']}/day; fast food: {v['fast_food']}/day; sugary foods/drinks: {v['sugary_food']}/day.",
        'Activity':f"{v['steps']:,} steps/day against your {step_goal:,} goal, and {minutes} exercise minutes/week against {weekly_goal}.",
        'Sleep':f"{v['sleep_hours']} hours/night against your {sleep_goal:g}-hour goal; quality rated {v['sleep_quality']}/5.",
        'Hydration':f"{v['water_litres']:g} L/day against your {water_goal:g} L personal goal.",
        'Habits':f"Stress rated {v['stress']}/10; sitting {v['sedentary_hours']:g} hours/day."
    }
    suggestions=[]
    if activity<70:suggestions.append('If comfortable, add a short walk or another activity you enjoy and increase gradually.')
    if sleep<70:suggestions.append('Try a consistent bedtime and discuss persistent sleep problems with a professional.')
    if hydration<70:suggestions.append('Keep water available and drink regularly; ask a clinician about fluid limits if you have one.')
    if nutrition<70:suggestions.append('Try adding fruit or vegetables to meals and reducing frequent sugary or fast-food choices.')
    if habits<70:suggestions.append('Consider short movement breaks and stress-management support.')
    if not suggestions:suggestions.append('Keep up habits that work for you and review your goals as circumstances change.')
    return {'overall':overall,'categories':categories,'explanations':why,'recommendations':suggestions,'bmi':round(v['weight_kg']/(v['height_cm']/100)**2,1),'exercise_weekly_minutes':minutes}


def challenge_summary(days, start=None, today=None):
    today=today or datetime.now(ZoneInfo('Asia/Dubai')).date()
    start=start or (min((date.fromisoformat(x) for x in days),default=today))
    elapsed=max(1,min(30,(today-start).days+1))
    completed=sum(days.get((start+timedelta(days=i)).isoformat(),0)>=30 for i in range(elapsed))
    streak=longest=0
    for i in range(elapsed):
        if days.get((start+timedelta(days=i)).isoformat(),0)>=30:
            streak+=1;longest=max(longest,streak)
        else:streak=0
    return {'day':elapsed,'completed':completed,'percent':round(completed/30*100),'streak':streak,'longest':longest,'start':start}
