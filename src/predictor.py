import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.preprocessing import preprocess_face
from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()

custom_objects = {
    '<lambda>': lambda x: x
}

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'age_gender_model_v.keras')
model = load_model(model_path, custom_objects=custom_objects, compile=False)

GENDER_LABELS = ['Male', 'Female']
AGE_SMOOTHING_BUFFER = 5  # Last N ages to average

# Dictionary to keep track of ages per face_id
face_predictions = {}  # face_id: {'ages': [int], 'gender': str}


def predict_age_gender(face_img, face_id=None):
    face = preprocess_face(face_img)
    gender_pred, age_pred = model.predict(face, verbose=0)

    raw_age = age_pred[0][0]
    smoothed_age = int(raw_age)

    gender = GENDER_LABELS[int(round(float(gender_pred[0])))]

    if face_id:
        if face_id not in face_predictions:
            face_predictions[face_id] = {'ages': [], 'gender': gender}

        face_predictions[face_id]['ages'].append(raw_age)

        # Keep only last N values
        if len(face_predictions[face_id]['ages']) > AGE_SMOOTHING_BUFFER:
            face_predictions[face_id]['ages'].pop(0)

        # Smooth the age
        smoothed_age = int(np.mean(face_predictions[face_id]['ages']))
        gender = face_predictions[face_id]['gender']

    return smoothed_age, gender
