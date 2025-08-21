# 👴👵 Real-Time Senior Citizen Identifier  

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)  
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io/)           
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> Detect multiple people, predict their age & gender, and identify **senior citizens** in real-time.  

---

## 📌 Overview  
This project detects faces in a video, predicts age and gender using a deep learning model, and flags individuals aged **60+** as senior citizens.  

It also logs the **age, gender, and timestamp** of each senior citizen to a CSV file for record-keeping — Ideal for analyzing CCTV footage in stores, malls, and other public spaces.

---

## 📂 Repository Structure

```

real-time-senior-citizen-identifier/
├── app.py
│
├── src/
│ ├── detector.py
│ ├── predictor.py
│ └── logger.py
│
├── utils/
│ └── preprocessing.py
│
├── model/
│ └── age_gender_model_v.keras
│
├── notebooks/
│ ├── age_gender_model.ipynb
│ └── evaluation.ipynb
│
├── data/
│ └── visit_log.csv
│
├── samples/
│ ├── input/
│ │ └── sample_input.mp4
│ ├── output/
│ │ └── sample_output.mp4
│ └── Visualization.jpg
│
├── results/
│ ├── MAE & Accuracy.png
│ ├── classification report.png
│ ├── confusion matrix.png
│ ├── evaluation metrics.png
│ ├── scatter plot.png
│ └── Actual vs Predicted.png
│
├── requirements.txt
└── README.md

```

---

## 🚀 Features  
- 🎥 **Real-time** face detection and prediction  
- 🧓 Flags individuals **60 years & older**  
- 📊 Logs details (age, gender, timestamp) to CSV file 
- 👥 Detects **multiple people** at once  
- ⚡ Works with video files

---

## 📂 Dataset  
- **Source:** [UTK-Face Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)  
- **Format Used:** Pre-cropped faces about 20,000+ in number  

---

## 📜 How It Works  
1. **Detection** → MTCNN detects faces in the frame  
2. **Prediction** → Age & gender prediction model processes each face  
3. **Filtering** → Marks people aged 60+ as seniors  
4. **Logging** → Saves details in `visit_log.csv`  

---

## 🖼 Example Output  
<p align="center">
  <a href="samples/input/sample_input.mp4">📥 Full sample Input</a><br>
  <a href="samples/output/sample_output.mp4">📤 Full sample Output</a>
</p>

<img src="/samples/Visualization.jpg" alt="Sample Output" width="1200"/>

---

## ▶️ How to Run  
```bash
# Clone the repository
git clone https://github.com/ShravyaMalogi/real-time-senior-citizen-identifier.git
cd real-time-senior-citizen-identifier

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
