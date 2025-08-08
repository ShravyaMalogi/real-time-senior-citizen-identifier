import gdown
import os

model_path = "age_gender_model_v.keras"

file_id = "1lE4_59f9c75i_HdPd4im981IL5_kOMp6"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
else:
    print("Model already exists at", model_path)
