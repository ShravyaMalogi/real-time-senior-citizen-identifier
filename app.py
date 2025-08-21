import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
import time
from collections import defaultdict, deque
from facenet-mtcnn import MTCNN
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from detector import FaceTracker
from predictor import predict_age_gender
from logger import log_visit

st.sidebar.title("System Status")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.sidebar.success(f"✅ GPU Detected: {len(gpus)} device(s) found. Using GPU for processing.")
    except RuntimeError as e:
        st.sidebar.error(f"GPU Error: {e}")
else:
    st.sidebar.warning("⚠️ No GPU detected. Processing will run on CPU, which may be slow.")

st.title("👵 Real-Time Senior Citizen Identifier")

video_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

last_logged = {}  
def should_log(face_id, age, gender):
    if face_id not in last_logged:
        last_logged[face_id] = (age, gender)
        return True
    last_age, last_gender = last_logged[face_id]
    if abs(age - last_age) >= 15 or gender != last_gender:
        last_logged[face_id] = (age, gender)
        return True
    return False

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
    total_seniors = 0
    frame_skip = 10

    tracker = FaceTracker(max_distance=80)
    detector = MTCNN()

    SMOOTHING_WINDOW = 5
    prediction_buffer = defaultdict(lambda: {
        "ages": deque(maxlen=SMOOTHING_WINDOW),
        "genders": deque(maxlen=SMOOTHING_WINDOW),
        "confidences": deque(maxlen=SMOOTHING_WINDOW)
    })

    active_faces = {} 

    fps_history = deque(maxlen=30)
    FACE_MARGIN = 0.3

    output_path = os.path.join(temp_dir, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        original_frame = frame.copy()
        display_frame = cv2.resize(frame, (640, 360))
        start_time = time.time()

        detections = detector.detect_faces(frame)
        boxes = []
        for res in detections:
            x, y, w, h = res["box"]
            x_margin = int(w * FACE_MARGIN)
            y_margin = int(h * FACE_MARGIN)
            x0 = max(0, x - x_margin)
            y0 = max(0, y - y_margin)
            x1 = min(frame.shape[1], x + w + x_margin)
            y1 = min(frame.shape[0], y + h + y_margin)
            boxes.append((x0, y0, x1, y1))

        tracked_faces = tracker.update(boxes)
        current_face_ids = set([fid for fid, _ in tracked_faces])

        disappeared = set(active_faces.keys()) - current_face_ids
        for fid in disappeared:
            last_age, last_gender = active_faces[fid]["age"], active_faces[fid]["gender"]
            if last_age is not None and should_log(fid, last_age, last_gender):
                log_visit(last_age, last_gender)
                if last_age >= 60:
                    total_seniors += 1
            del active_faces[fid]

        for face_id, (x0, y0, x1, y1) in tracked_faces:
            if x1 <= x0 or y1 <= y0:
                continue
            face_img = original_frame[y0:y1, x0:x1]
            if face_img.size == 0:
                continue

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                age, gender, gender_conf = predict_age_gender(face_img, face_id=face_id)
            except Exception as e:
                print(f"[ERROR] Prediction failed for {face_id}: {e}")
                continue

            active_faces[face_id] = {"age": age, "gender": gender}

            label = f"{gender}, {age} yrs"
            color = (0, 0, 255) if age >= 60 else (0, 255, 0)

            cv2.rectangle(original_frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(original_frame, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            scale_x = display_frame.shape[1] / frame_width
            scale_y = display_frame.shape[0] / frame_height
            cv2.rectangle(display_frame,
                          (int(x0*scale_x), int(y0*scale_y)),
                          (int(x1*scale_x), int(y1*scale_y)),
                          color, 2)
            cv2.putText(display_frame, label,
                        (int(x0*scale_x), int(y0*scale_y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / (time.time() - start_time + 1e-6)
        fps_history.append(fps)
        smooth_fps = np.mean(fps_history)
        cv2.putText(display_frame, f"FPS: {smooth_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(original_frame, f"FPS: {smooth_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        progress.progress(min(frame_count / total_frames, 1.0))

        out.write(original_frame)

    for fid, data in active_faces.items():
        if data["age"] is not None and should_log(fid, data["age"], data["gender"]):
            log_visit(data["age"], data["gender"])
            if data["age"] >= 60:
                total_seniors += 1

    cap.release()
    out.release()
    st.success(f"✅ Done! Total Senior Citizens Detected: {total_seniors}")

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
