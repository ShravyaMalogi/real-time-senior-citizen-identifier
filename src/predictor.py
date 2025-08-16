import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.preprocessing import preprocess_face
from keras.config import enable_unsafe_deserialization

# Allow unsafe deserialization (needed for some custom lambda layers)
enable_unsafe_deserialization()

# Handle custom objects in model
custom_objects = {
    '<lambda>': lambda x: x
}

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'age_gender_model_v.keras')
model = load_model(model_path, custom_objects=custom_objects, compile=False)

# Constants
GENDER_LABELS = ['Male', 'Female']
AGE_SMOOTHING_BUFFER = 7   # Increased buffer for smoother results
GENDER_VOTE_BUFFER = 7     # Added buffer to stabilize gender too

# Dictionary to keep track of ages and genders per face_id
face_predictions = {}  # face_id: {'ages': [int], 'genders': [str]}


def predict_age_gender(face_img, face_id=None):
    """
    Predicts age and gender from a cropped face image.
    Uses smoothing across frames to reduce fluctuations.
    """
    # Preprocess the face image for model input
    face = preprocess_face(face_img)

    # Get model predictions
    gender_pred, age_pred = model.predict(face, verbose=0)

    raw_age = age_pred[0][0]
    raw_gender = GENDER_LABELS[int(round(float(gender_pred[0])))]

    smoothed_age = int(raw_age)
    stable_gender = raw_gender

    if face_id:
        if face_id not in face_predictions:
            face_predictions[face_id] = {'ages': [], 'genders': []}

        # Append new predictions
        face_predictions[face_id]['ages'].append(raw_age)
        face_predictions[face_id]['genders'].append(raw_gender)

        # Keep buffer size fixed
        if len(face_predictions[face_id]['ages']) > AGE_SMOOTHING_BUFFER:
            face_predictions[face_id]['ages'].pop(0)
        if len(face_predictions[face_id]['genders']) > GENDER_VOTE_BUFFER:
            face_predictions[face_id]['genders'].pop(0)

        # Smooth age (average of last N values)
        smoothed_age = int(np.mean(face_predictions[face_id]['ages']))

        # Stabilize gender (majority vote of last N predictions)
        stable_gender = max(set(face_predictions[face_id]['genders']),
                            key=face_predictions[face_id]['genders'].count)

    return smoothed_age, stable_gender
