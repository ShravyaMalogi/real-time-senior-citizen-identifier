# src/detector.py
import numpy as np
from mtcnn import MTCNN

class MTCNNFaceDetector:
    def __init__(self, min_face_size=40, conf_threshold=0.85):
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
    def __init__(self, max_distance=80):
        self.next_face_id = 0
        self.tracks = {}  # face_id → (x,y)

        self.max_distance = max_distance

    def _center(self, box):
        x0, y0, x1, y1 = box
        return (int((x0 + x1) / 2), int((y0 + y1) / 2))

    def update(self, detections):
        """
        Args:
            detections: list of (x0, y0, x1, y1) tuples
        Returns:
            list of (face_id, (x0, y0, x1, y1))
        """
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        updated_tracks = {}
        results = []

        for det in detections:
            assigned_id = None
            best_iou = 0
            for face_id, prev_det in self.tracks.items():
                iou_score = iou(det, prev_det)
                if iou_score > best_iou and iou_score > 0.3:  # threshold
                    best_iou = iou_score
                    assigned_id = face_id

            if assigned_id is None:
                assigned_id = self.next_face_id
                self.next_face_id += 1

            updated_tracks[assigned_id] = det
            results.append((assigned_id, det))

        self.tracks = updated_tracks
        return results
