import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.preprocessing import preprocess_face
import cv2

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
GENDER_LABELS = ['Male', 'Female']

def softmax(x):
    """Compute softmax values for a set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_age_gender(face_img, face_id=None):
    """
    Predicts age and gender, now with a defensive check to prevent crashes
    when the model returns an incomplete prediction.
    """
    # 1. Preprocess the image
    print(f"[DEBUG] Cropped face shape: {face_img.shape}, dtype: {face_img.dtype}, min: {face_img.min()}, max: {face_img.max()}")
    # Defensive: check input
    if face_img is None or face_img.size == 0:
        return "N/A", 0
    # Preprocess: grayscale, resize, scale
    import cv2
    import numpy as np
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_LINEAR)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    # 2. Get prediction array
    output = model.predict(face)
    print(f"[DEBUG] model.predict(face) type: {type(output)}, length: {len(output) if hasattr(output, '__len__') else 'N/A'}")
    print(f"[DEBUG] model.predict(face) raw: {output}")

    # 3. If dual output, unpack
    if isinstance(output, (list, tuple)) and len(output) == 2:
        age_pred = output[1]
        gender_pred = output[0]
        # Gender: sigmoid output, shape (1,1)
        # Extract scalar from array if needed
        if isinstance(gender_pred, np.ndarray):
            gender_score = float(gender_pred.squeeze())
        else:
            gender_score = float(gender_pred)
        if isinstance(age_pred, np.ndarray):
            age_scalar = float(age_pred.squeeze())
        else:
            age_scalar = float(age_pred)
        age = int(round(age_scalar))
        # Defensive: age should be positive
        if age < 0 or age > 120:
            age = 0
        gender = "Female" if gender_score > 0.8 else "Male"
        gender_confidence = float(gender_score) if gender == "Female" else float(1 - gender_score)
    else:
        # Fallback for single output
        age = int(output[0][0]) if hasattr(output[0], '__len__') else int(output[0])
        gender = "N/A"
        gender_confidence = 0.0

    return age, gender, gender_confidence
