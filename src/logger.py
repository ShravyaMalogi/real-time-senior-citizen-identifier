import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'visit_log.csv')

def log_visit(age, gender):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)

    try:
        print(f"[LOGGER] Logging to: {LOG_FILE}")
        print(f"[LOGGER] Age: {age}, Gender: {gender}, Timestamp: {timestamp}")

        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                print("[LOGGER] Writing header...")
                writer.writerow(['Age', 'Gender', 'Timestamp'])
            writer.writerow([age, gender, timestamp])
            print("[LOGGER] Log entry written successfully.")

    except Exception as e:
        print(f"[LOGGER ERROR] Failed to write log: {e}")
