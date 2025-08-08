import cv2
import numpy as np

def preprocess_face(face_img):
    """
    Preprocess face image for model input.
    - Convert to grayscale
    - Resize to 128x128
    - Normalize pixel values
    - Add channel and batch dimensions

    Returns:
        np.ndarray: Preprocessed image of shape (1, 128, 128, 1)
    """
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (128, 128))
    normalized_face = resized_face.astype('float32') / 255.0
    normalized_face = np.expand_dims(normalized_face, axis=-1)  # (128, 128, 1)
    batched_face = np.expand_dims(normalized_face, axis=0)      # (1, 128, 128, 1)
    return batched_face
