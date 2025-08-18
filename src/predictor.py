import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.preprocessing import preprocess_face

# Try to allow unsafe deserialization if supported
try:
    from keras.saving import enable_unsafe_deserialization
    enable_unsafe_deserialization()
except ImportError:
    pass

custom_objects = {'<lambda>': lambda x: x}

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'age_gender_model_v.keras')
model = load_model(model_path, custom_objects=custom_objects, compile=False)

# Constants
AGE_SMOOTHING_BUFFER = 7
GENDER_VOTE_BUFFER = 30  # longer buffer for stability

# Dictionary to store predictions per face
face_predictions = {}  # face_id: {'ages': [], 'genders': [], 'confidences': [], 'locked_gender': None}


def predict_age_gender(face_img, face_id=None, female_conf_threshold=0.9, gender_switch_thresh=0.7):
    """
    Predict age and gender for a face image with stable smoothing and locked gender.
    """
    face = preprocess_face(face_img)

    preds = model.predict(face, verbose=0)
    if isinstance(preds, list) and len(preds) == 2:
        gender_pred, age_pred = preds
    else:
        gender_pred, age_pred = preds[:, 0], preds[:, 1:]

    # Raw predictions
    raw_age = int(age_pred[0][0])
    female_prob = float(gender_pred[0])
    raw_gender = "Female" if female_prob >= 0.5 else "Male"
    if raw_gender == "Female" and female_prob < female_conf_threshold:
        raw_gender = "Male"
    gender_confidence = max(female_prob, 1 - female_prob)

    # Initialize buffer
    if face_id not in face_predictions:
        face_predictions[face_id] = {'ages': [], 'genders': [], 'confidences': [], 'locked_gender': None}

    buf = face_predictions[face_id]
    buf['ages'].append(raw_age)
    buf['genders'].append(raw_gender)
    buf['confidences'].append(gender_confidence)

    # Maintain buffer sizes
    buf['ages'] = buf['ages'][-AGE_SMOOTHING_BUFFER:]
    buf['genders'] = buf['genders'][-GENDER_VOTE_BUFFER:]
    buf['confidences'] = buf['confidences'][-GENDER_VOTE_BUFFER:]

    # Smoothed age
    smoothed_age = int(np.mean(buf['ages']))

    # Stable gender: lock once confident
    if buf['locked_gender'] is None:
        from collections import Counter
        gender_counts = Counter(buf['genders'])
        mode_gender, count = gender_counts.most_common(1)[0]
        proportion = count / len(buf['genders'])
        if proportion >= gender_switch_thresh and gender_confidence > 0.85:
            buf['locked_gender'] = mode_gender

    stable_gender = buf['locked_gender'] if buf['locked_gender'] else raw_gender
    stable_confidence = np.mean(buf['confidences'])

    return smoothed_age, stable_gender, stable_confidence
