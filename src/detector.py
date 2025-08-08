import cv2
import os

# Load OpenCV's Haar cascade for frontal face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"[ERROR] Haar cascade not found at {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    """
    Detects faces in the input frame using Haar cascades.
    
    Args:
        frame (np.ndarray): Input image/frame in BGR format.

    Returns:
        list of tuples: Bounding boxes [(x, y, w, h), ...] for each detected face.
    """
    if frame is None:
        print("[WARNING] Empty frame passed to detect_faces.")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


import numpy as np
import uuid

class FaceTracker:
    def __init__(self, max_distance=50):
        self.faces = {}  # id: {'centroid': (x, y), 'age_buffer': []}
        self.max_distance = max_distance

    def update(self, boxes):
        updated_ids = []

        for box in boxes:
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

            matched_id = None
            for face_id, face_data in self.faces.items():
                fx, fy = face_data['centroid']
                dist = np.linalg.norm([cx - fx, cy - fy])
                if dist < self.max_distance:
                    matched_id = face_id
                    break

            if matched_id is None:
                matched_id = str(uuid.uuid4())[:8]
                self.faces[matched_id] = {'centroid': (cx, cy), 'age_buffer': []}
            else:
                self.faces[matched_id]['centroid'] = (cx, cy)

            updated_ids.append((matched_id, box))

        return updated_ids
