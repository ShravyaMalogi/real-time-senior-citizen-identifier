import csv
import os
from datetime import datetime, timedelta

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'visit_log.csv')

last_logged = {}

def log_visit(age, gender, cooldown=30, age_tolerance=2):
    """
    Log a visit only if not logged in the last `cooldown` seconds.
    Age predictions are considered the same if within `age_tolerance`.
    """
    global last_logged
    timestamp = datetime.now()
    file_exists = os.path.isfile(LOG_FILE)

    key = (round(age / age_tolerance), gender)

    if key in last_logged:
        if (timestamp - last_logged[key]) < timedelta(seconds=cooldown):
            print(f"[LOGGER] Skipping duplicate log for approx age {age} ({gender})")
            return  

    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Age', 'Gender', 'Timestamp'])
            writer.writerow([age, gender, timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"[LOGGER] Logged: {age}, {gender}")
            last_logged[key] = timestamp  
            
    except Exception as e:
        print(f"[LOGGER ERROR] Failed to write log: {e}")
