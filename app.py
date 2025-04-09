import streamlit as st
import json
import datetime
from openai import OpenAI
from typing import List, Dict
import os
from collections import Counter
import pandas as pd
import altair as alt
import time
import random

# --- DeepSeek API Setup ---
client = OpenAI(api_key="sk-fedfd017b2074130a55a15c251670da0", base_url="https://api.deepseek.com")

TASKS_FILE = "tasks.json"
DONE_FILE = "done_tasks.json"

# --- Load & Save Functions ---
def load_data():
    tasks = []
    done_tasks = []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r") as f:
            tasks = json.load(f)
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            done_tasks = json.load(f)
    return tasks, done_tasks

def save_data(tasks, done_tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)
    with open(DONE_FILE, "w") as f:
        json.dump(done_tasks, f, indent=2)

# --- AI Call ---
def ask_deepseek(prompt: str, system_prompt: str, temperature=0.4) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

# --- App Logic ---
def summarize_today(tasks: List[Dict], done_tasks: List[Dict]) -> str:
    today = datetime.date.today().isoformat()
    completed_today = {
        t["task"] for t in done_tasks if t.get("completed_on") == today
    }
    filtered = [t for t in tasks if not (t.get("recurring") and t["task"] in completed_today)]
    system = """You're an intelligent planner.
From the user's task list, suggest a plan for today with 2–4 tasks under 4 hours. Show priority and estimated time.
Format:
### Suggested Focus for Today:
- **Primary:** <task> — ~<time> (<reason>)
- ...
⏲ Total time: ~X hours
"""
    return ask_deepseek(json.dumps(filtered, indent=2), system)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Task Planner", layout="centered")
st.title("🧠 AI Task Planner")

if "tasks" not in st.session_state or "done_tasks" not in st.session_state:
    st.session_state.tasks, st.session_state.done_tasks = load_data()

# --- Add Task ---
st.subheader("Add a Task")
with st.form("add_task"):
    task = st.text_input("Task")
    category = st.selectbox("Category", ["Work", "Personal", "Other"])
    priority = st.selectbox("Priority", ["High", "Medium", "Low"])
    deadline = st.date_input("Deadline", value=None, format="YYYY-MM-DD")
    recurring = st.selectbox("Recurring", ["No", "daily", "weekly"])
    submitted = st.form_submit_button("Add Task")
    if submitted and task:
        st.session_state.tasks.append({
            "task": task,
            "category": category,
            "priority": priority,
            "deadline": str(deadline) if deadline else None,
            "recurring": recurring if recurring != "No" else None,
            "subtasks": [],
            "reasoning": "User entered manually"
        })
        save_data(st.session_state.tasks, st.session_state.done_tasks)
        st.success("Task added!")

# --- Show Tasks ---
st.subheader("Your Tasks")
for i, t in enumerate(st.session_state.tasks):
    col1, col2 = st.columns([4, 1])
    with col1:
        rec = f" 🌀{t['recurring']}" if t.get("recurring") else ""
        st.markdown(f"**{i+1}. {t['task']}**{rec} [{t['priority']} | {t['category']}]")
    with col2:
        if st.button("Done", key=f"done_{i}"):
            st.session_state.done_tasks.append({
                **t, "completed_on": datetime.date.today().isoformat()
            })
            if not t.get("recurring"):
                st.session_state.tasks.pop(i)
            save_data(st.session_state.tasks, st.session_state.done_tasks)
            st.rerun()

# --- Today Plan ---
st.subheader("🗓️ Today's Plan")
if st.button("Generate Today Plan"):
    today_plan = summarize_today(st.session_state.tasks, st.session_state.done_tasks)
    st.markdown(today_plan)

# --- Done Tasks ---
st.subheader("✅ Completed Tasks (This Week)")
week_ago = datetime.date.today() - datetime.timedelta(days=7)
recent_done = [t for t in st.session_state.done_tasks if "completed_on" in t and datetime.date.fromisoformat(t["completed_on"]) >= week_ago]

if not recent_done:
    st.write("No tasks completed this week.")
else:
    for t in recent_done:
        st.markdown(f"- {t['task']} ({t['completed_on']})")

# --- Weekly Review Chart ---
st.subheader("📊 Weekly Review")
if recent_done:
    chart_data = pd.DataFrame(recent_done)
    chart_data["completed_on"] = pd.to_datetime(chart_data["completed_on"])
    chart_data["day"] = chart_data["completed_on"].dt.strftime("%A")
    chart_data["count"] = 1
    heatmap = chart_data.groupby("day")["count"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index()

    chart = alt.Chart(heatmap).mark_bar().encode(
        x=alt.X("day", sort=list(heatmap["day"])),
        y="count",
        tooltip=["day", "count"]
    ).properties(title="📅 Tasks Completed by Day of Week")

    st.altair_chart(chart, use_container_width=True)

    # --- Button to Generate Review ---
    if st.button("🧠 Ask AI for Weekly Review"):
        review_prompt = json.dumps(recent_done, indent=2)
        review_summary = ask_deepseek(
            review_prompt,
            "You are a helpful assistant. Analyze the user's completed tasks this week and give encouraging, constructive feedback."
        )
        st.subheader("💬 AI Feedback on Your Week")
        st.markdown(review_summary)

# --- Focus Mode ---
st.subheader("🎯 Focus Mode")
if st.session_state.tasks:
    task_options = [f"{i+1}. {t['task']} [{t['priority']}, {t['category']}]" for i, t in enumerate(st.session_state.tasks)]
    selected_index = st.selectbox("Choose a task to focus on:", range(len(task_options)), format_func=lambda i: task_options[i])
    duration = st.slider("Focus time (minutes)", 5, 60, 25)

    if st.button("Start Focus Session"):
        focused_task = st.session_state.tasks[selected_index]
        st.success(f"🧠 Now focusing on: **{focused_task['task']}**")
        if focused_task.get("subtasks"):
            st.markdown(f"🔹 Subtask suggestion: `{random.choice(focused_task['subtasks'])}`")
        st.write(f"⏳ Timer started for **{duration} minutes**...")

        for remaining in range(duration, 0, -1):
            st.info(f"{remaining} minute(s) left...")
            time.sleep(60)

        st.success("✅ Time's up!")
        if st.button("Mark as done?"):
            st.session_state.done_tasks.append({**focused_task, "completed_on": datetime.date.today().isoformat()})
            if not focused_task.get("recurring"):
                st.session_state.tasks.pop(selected_index)
            save_data(st.session_state.tasks, st.session_state.done_tasks)
            st.success("Task marked as done!")
            st.rerun()
else:
    st.info("No tasks available for focus mode.")

# --- Ask Anything ---
st.subheader("🤔 Ask Anything About Your Tasks")

user_question = st.text_input("Type your question (e.g., 'What deadlines are coming up?')")

if st.button("Ask DeepSeek"):
    context = {
        "tasks": st.session_state.tasks,
        "done_tasks": st.session_state.done_tasks,
        "question": user_question
    }

    system = """You are an intelligent assistant. The user will ask a question in natural language about their tasks.
You have access to two lists:
- tasks: active/incomplete tasks
- done_tasks: completed tasks with date info

Answer the question clearly and helpfully using those lists.
"""

    response = ask_deepseek(json.dumps(context, indent=2), system)
    st.markdown("**💬 AI Response:**")
    st.markdown(response)
