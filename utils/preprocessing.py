# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_face(face_img):
    """
    Preprocess face image for UTKFace grayscale model
    Returns (1, 128, 128, 1)
    """
    # Convert to grayscale if it's not already
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    elif len(face_img.shape) == 2:
        gray_face = face_img
    else:
        raise ValueError(f"Invalid face_img shape: {face_img.shape}")

    resized_face = cv2.resize(gray_face, (128, 128))
    normalized_face = resized_face.astype('float32') / 255.0
    normalized_face = np.expand_dims(normalized_face, axis=-1)  # (128,128,1)
    normalized_face = np.expand_dims(normalized_face, axis=0)   # (1,128,128,1)
    return normalized_face
