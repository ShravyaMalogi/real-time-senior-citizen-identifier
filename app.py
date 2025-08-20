import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
from collections import defaultdict, deque

# --- App Constants ---
# (No constants needed for this version)

# Add src/ folder to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from detector import FaceTracker, MTCNNFaceDetector
from predictor import predict_age_gender
from logger import log_visit

st.title("👵 Real-Time Senior Citizen Identifier")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Processing Controls")
quality = st.sidebar.slider("Processing Quality", 10, 100, 50, 10, help="Lower quality to speed up processing by resizing the input. Affects both webcam and video.")
resize_scale = quality / 100.0

# --- Input Selection ---
input_source = st.radio("Choose input source:", ("Upload Video", "Webcam"), horizontal=True)

# --- Session State Initialization ---
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'tracker' not in st.session_state:
    st.session_state.tracker = FaceTracker(max_distance=80)
if 'detector' not in st.session_state:
    st.session_state.detector = MTCNNFaceDetector()
if 'active_faces' not in st.session_state:
    st.session_state.active_faces = {}
if 'logged_face_ids' not in st.session_state:
    st.session_state.logged_face_ids = set()
if 'total_seniors' not in st.session_state:
    st.session_state.total_seniors = 0
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None
if 'last_logged' not in st.session_state:
    st.session_state.last_logged = {}

# --- Helper Functions ---
def should_log(face_id):
    """Decide if a new visit should be logged to avoid duplicates."""
    if face_id in st.session_state.logged_face_ids:
        return False
    st.session_state.logged_face_ids.add(face_id)
    return True

def get_expanded_crop(frame, box, expand_ratio=0.1):
    """Get an expanded crop of a face from the frame."""
    h, w, _ = frame.shape
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    expand_w, expand_h = int(box_w * expand_ratio), int(box_h * expand_ratio)
    
    crop_x0 = max(0, x0 - expand_w)
    crop_y0 = max(0, y0 - expand_h)
    crop_x1 = min(w, x1 + expand_w)
    crop_y1 = min(h, y1 + expand_h)
    
    return frame[crop_y0:crop_y1, crop_x0:crop_x1]

def draw_boxes_on_frame(display_frame, faces_to_draw):
    """Helper function to draw face boxes and labels on a frame."""
    scale_x = display_frame.shape[1] / (st.session_state.get('original_frame_width', display_frame.shape[1]))
    scale_y = display_frame.shape[0] / (st.session_state.get('original_frame_height', display_frame.shape[0]))

    for face_id, data in faces_to_draw.items():
        box = data.get("box")
        if not box:
            continue

        label = f"{data['gender']}, {data['age']}"
        color = (0, 0, 255) if data['age'] >= 60 else (0, 255, 0)
        
        disp_box = [int(c * s) for c, s in zip(box, (scale_x, scale_y, scale_x, scale_y))]
        
        cv2.rectangle(display_frame, (disp_box[0], disp_box[1]), (disp_box[2], disp_box[3]), color, 2)
        cv2.putText(display_frame, label, (disp_box[0], disp_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display_frame

def process_frame(frame, display_frame):
    """Detects, tracks, and processes faces in a single frame."""
    boxes = st.session_state.detector.detect_faces(frame)
    
    # Store original frame dimensions for accurate scaling
    st.session_state.original_frame_width = frame.shape[1]
    st.session_state.original_frame_height = frame.shape[0]

    tracked_faces = st.session_state.tracker.update(boxes)
    current_face_ids = {fid for fid, _ in tracked_faces}

    # Handle disappeared faces
    disappeared_ids = set(st.session_state.active_faces.keys()) - current_face_ids
    for fid in disappeared_ids:
        last_data = st.session_state.active_faces.pop(fid)
        if should_log(fid):
            log_visit(last_data['age'], last_data['gender'])
            if last_data['age'] >= 60:
                st.session_state.total_seniors += 1

    # Process currently tracked faces
    for face_id, box in tracked_faces:
        age_face_img = get_expanded_crop(frame, box, expand_ratio=0.12)
        gender_face_img = get_expanded_crop(frame, box, expand_ratio=0.35)

        if age_face_img.size == 0 or gender_face_img.size == 0:
            continue

        try:
            age, _, _ = predict_age_gender(age_face_img)
            _, gender, _ = predict_age_gender(gender_face_img)

            # Store the most recent prediction AND the box coordinates
            st.session_state.active_faces[face_id] = {"age": age, "gender": gender, "box": box}

        except Exception as e:
            print(f"Could not process face {face_id}: {e}")

    # Draw all active faces on the display frame
    display_frame = draw_boxes_on_frame(display_frame, st.session_state.active_faces)

    return display_frame

# --- Main UI Logic ---
def main():
    if input_source == "Webcam":
        st.header("📷 Live Webcam Feed")
        col1, col2 = st.columns(2)
        if col1.button("Start Webcam", key="start_cam"):
            st.session_state.webcam_running = True
        if col2.button("Stop Webcam", key="stop_cam"):
            st.session_state.webcam_running = False

        stframe = st.empty()
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam not found. Please grant access and try again.")
            else:
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break
                    
                    display_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale) if resize_scale < 1.0 else frame.copy()
                    processed_frame = process_frame(frame, display_frame)
                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                cap.release()
                # Log any remaining faces when stopping
                for fid, data in list(st.session_state.active_faces.items()):
                    if should_log(fid):
                        log_visit(data['age'], data['gender'])
                        if data['age'] >= 60:
                            st.session_state.total_seniors += 1
                st.session_state.active_faces.clear()

    elif input_source == "Upload Video":
        st.header("🎥 Video File Upload")
        frame_skip = st.sidebar.slider("Frame Skip", 1, 30, 5, help="Process every Nth frame for faster analysis.")
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

        if st.session_state.processed_video_path:
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name=os.path.basename(st.session_state.processed_video_path),
                    mime="video/mp4"
                )


        if video_file and not st.session_state.video_processing:
            st.session_state.video_processing = True
            st.session_state.processed_video_path = None # Reset path at the start
            
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(video_file.read())
                    video_path = tfile.name

                cap = cv2.VideoCapture(video_path)
                
                # Get video properties for writer
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Adjust dimensions based on resize_scale for the output video
                out_width = int(frame_width * resize_scale) if resize_scale < 1.0 else frame_width
                out_height = int(frame_height * resize_scale) if resize_scale < 1.0 else frame_height

                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

                stframe = st.empty()
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_num += 1
                    
                    # Frame skipping logic
                    display_frame = cv2.resize(frame, (out_width, out_height)) if resize_scale < 1.0 else frame.copy()
                    if frame_num % frame_skip == 0:
                        # Process the original full-res frame, but draw on the scaled display_frame
                        processed_frame = process_frame(frame, display_frame)
                    else:
                        # On skipped frames, just draw the last known boxes
                        processed_frame = draw_boxes_on_frame(display_frame, st.session_state.active_faces)

                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    out.write(processed_frame) # Write the processed frame to the output video
                    progress_bar.progress(frame_num / total_frames)

                cap.release()
                out.release()
                
                # Log remaining faces after processing
                for fid, data in list(st.session_state.active_faces.items()):
                    if should_log(fid):
                        log_visit(data['age'], data['gender'])
                        if data['age'] >= 60:
                            st.session_state.total_seniors += 1
                
                os.remove(video_path)
                st.session_state.processed_video_path = output_video_path
                st.success("Video processing complete!")
                st.rerun() # Rerun to show the download button

            except Exception as e:
                st.error(f"Error during video processing: {e}")
            finally:
                st.session_state.video_processing = False
                st.session_state.active_faces.clear()
                st.session_state.logged_face_ids.clear()

    # --- Display Stats ---
    st.sidebar.header("📊 Statistics")
    st.sidebar.metric("Total Senior Citizens Detected", st.session_state.total_seniors)

    if st.sidebar.button("Reset Tracker & Stats"):
        # Clear all session state variables to reset the app
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.info("Note: A 'senior citizen' is counted if their estimated age is 60 or above.")

if __name__ == "__main__":
    main()

