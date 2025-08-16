import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
import time
from collections import defaultdict, deque
from mtcnn import MTCNN  # ✅ Face detector

# Add src/ folder to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from detector import FaceTracker
from predictor import predict_age_gender
from logger import log_visit

# ------------------------- STREAMLIT APP -------------------------
st.title("👵 Real-Time Senior Citizen Identifier (MTCNN + Face Tracking)")

# Upload video file
video_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

if video_file:
    # Save uploaded file to a temporary path
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.name)
    with open(temp_path, 'wb') as f:
        f.write(video_file.read())

    # OpenCV video reader
    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty()
    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_seniors = 0
    frame_skip = 10  # ✅ process every 10th frame (better speed/accuracy tradeoff)

    # Face tracker & detector
    tracker = FaceTracker(max_distance=80)
    logged_faces = set()  # Keep track of already-logged face IDs
    detector = MTCNN()

    # ✅ Buffers for smoothing predictions
    SMOOTHING_WINDOW = 5
    prediction_buffer = defaultdict(lambda: {
        "ages": deque(maxlen=SMOOTHING_WINDOW),
        "genders": deque(maxlen=SMOOTHING_WINDOW)
    })

    # ✅ Moving average for FPS
    fps_history = deque(maxlen=30)

    # ------------------------- FRAME LOOP -------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # skip frames for speed

        # Resize for efficiency
        frame = cv2.resize(frame, (640, 360))
        start_time = time.time()

        # ✅ Detect faces with MTCNN
        detections = detector.detect_faces(frame)
        boxes = []
        for res in detections:
            x, y, w, h = res["box"]
            if w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))  # (x0, y0, x1, y1)

        # Track faces across frames
        tracked_faces = tracker.update(boxes)  # [(face_id, (x0,y0,x1,y1)), ...]

        for face_id, (x0, y0, x1, y1) in tracked_faces:
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                continue  # skip invalid boxes

            face_img = frame[y0:y1, x0:x1]
            if face_img.size == 0:
                continue

            try:
                age, gender = predict_age_gender(face_img, face_id=face_id)
            except Exception as e:
                print(f"[ERROR] Prediction failed for {face_id}: {e}")
                continue

            # ✅ Smooth predictions
            buf = prediction_buffer[face_id]
            buf["ages"].append(age)
            buf["genders"].append(gender)

            stable_age = int(np.mean(buf["ages"]))
            stable_gender = max(set(buf["genders"]), key=buf["genders"].count)

            # Prepare label
            label = f"{stable_gender}, {stable_age} yrs"

            # ✅ Log each face once
            if face_id not in logged_faces:
                log_visit(stable_age, stable_gender)
                logged_faces.add(face_id)

                if stable_age >= 60:
                    total_seniors += 1

            # Draw bounding box + label
            color = (0, 0, 255) if stable_age >= 60 else (0, 255, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{label}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ✅ Stable FPS display
        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_history.append(fps)
        smooth_fps = np.mean(fps_history)

        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.success(f"✅ Done! Total Senior Citizens Detected: {total_seniors}")
