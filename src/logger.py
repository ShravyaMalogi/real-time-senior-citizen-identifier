import csv
import os
from datetime import datetime, timedelta
import streamlit as st

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'visit_log.csv')

def log_visit(age, gender, cooldown=30, age_tolerance=5):
    """
    Log a visit only if not logged in the last `cooldown` seconds.
    Uses st.session_state to maintain the log across reruns.
    Age predictions are considered the same if within `age_tolerance`.
    """
    if 'last_logged' not in st.session_state:
        st.session_state.last_logged = {}

    timestamp = datetime.now()
    file_exists = os.path.isfile(LOG_FILE)

    key = (round(age / age_tolerance) * age_tolerance, gender)

    if key in st.session_state.last_logged:
        if (timestamp - st.session_state.last_logged[key]) < timedelta(seconds=cooldown):
            return

    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists or os.path.getsize(LOG_FILE) == 0:
                writer.writerow(['Age', 'Gender', 'Timestamp'])
            writer.writerow([age, gender, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            st.session_state.last_logged[key] = timestamp  

    except Exception as e:
        print(f"[LOGGER ERROR] Failed to write log: {e}")
