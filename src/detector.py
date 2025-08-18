# src/detector.py
import numpy as np
import math
from mtcnn import MTCNN

class MTCNNFaceDetector:
    def __init__(self, min_face_size=55, conf_threshold=0.85):
        self.detector = MTCNN(min_face_size=min_face_size)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
        detections = self.detector.detect_faces(rgb)

        results = []
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < self.conf_threshold:
                continue

            x, y, w, h = det["box"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = x1 + max(1, w), y1 + max(1, h)

            results.append((x1, y1, x2, y2))  # return just tuple

        return results


class FaceTracker:
    def __init__(self, max_distance=50, max_age=10):
        self.next_face_id = 0
        self.tracks = {}  # face_id -> {"box": (x0,y0,x1,y1), "age": 0}
        self.max_distance = max_distance
        self.max_age = max_age

    def _center(self, box):
        x0, y0, x1, y1 = box
        return (int((x0 + x1) / 2), int((y0 + y1) / 2))

    def _distance(self, boxA, boxB):
        cA = self._center(boxA)
        cB = self._center(boxB)
        return math.sqrt((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2)

    def update(self, detections):
        updated_tracks = {}
        results = []

        for det in detections:
            assigned_id = None
            best_dist = 1e9

            for face_id, data in self.tracks.items():
                prev_det = data["box"]
                dist = self._distance(det, prev_det)

                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    assigned_id = face_id

            if assigned_id is None:
                assigned_id = self.next_face_id
                self.next_face_id += 1

            updated_tracks[assigned_id] = {"box": det, "age": 0}
            results.append((assigned_id, det))

        # Increment "age" for unmatched tracks (memory effect)
        for face_id, data in self.tracks.items():
            if face_id not in updated_tracks:
                data["age"] += 1
                if data["age"] < self.max_age:
                    updated_tracks[face_id] = data  # keep it for a while

        self.tracks = updated_tracks
        return results


