import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from detector import detect_faces
from predictor import predict_age_gender
from logger import log_visit

st.title("👵 Real-Time Senior Citizen Identifier")

video_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

if video_file:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.name)
    with open(temp_path, 'wb') as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty()
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    processed_faces = set()
    total_detected = 0
    frame_skip = 10  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        start_time = time.time()
        total_seniors = 0
        
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                age, gender = predict_age_gender(face_img)
            except:
                continue

            label = f"{gender}, {age} yrs"

            if (x, y) not in processed_faces:
                log_visit(age, gender)  
                processed_faces.add((x, y))

                if age >= 60:
                    total_seniors += 1  

            color = (0, 0, 255) if age >= 60 else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / (time.time() - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success(f"✅ Done! Total Senior Citizens Detected: {total_seniors}")
