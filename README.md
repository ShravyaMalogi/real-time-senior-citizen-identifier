# 👴👵 Real-Time Senior Citizen Identifier  

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)  
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)](https://pjreddie.com/darknet/yolo/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> Detect multiple people in real-time, predict their age & gender, and identify **senior citizens** instantly.  

---

## 📌 Overview  
This project detects faces in a live camera feed (or video), predicts age using a deep learning model, and flags individuals aged **60+** as senior citizens.  

It also logs the **age, gender, and timestamp** of each senior citizen to a CSV/Excel file for record-keeping — perfect for public places, security systems, or event monitoring.  

---

## 🚀 Features  
- 🎥 **Real-time** face detection & age prediction  
- 🧓 Flags individuals **60 years & older**  
- 📊 Logs details (age, gender, timestamp) to CSV/Excel  
- 👥 Detects **multiple people** at once  
- ⚡ Works with **webcam** or **video files**

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
├── models/
│ └── my_model.keras
│
├── notebooks/
│ └── train_model.ipynb
│
├── data/
│ ├── input/
│ │ └── sample_input.mp4
│ └── output/
│   └── sample_output.mp4
│
├── results/
│ ├── metrics.txt
│ └── confusion_matrix.png
│
├── requirements.txt
└── README.md

```

---

## 📂 Dataset  
- **Source:** [UTK-Face Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)  
- **Format Used:** Pre-cropped faces about 20,000 in number  

---

## 🛠 Tech Stack  
- **Python 3.x**  
- **OpenCV** for real-time video processing  
- **YOLO** for fast face/person detection  
- **TensorFlow/Keras** for age & gender prediction  
- **Pandas** for logging data  

---

## 📜 How It Works  
1. **Detection** → YOLO detects faces in the frame  
2. **Prediction** → Age & gender prediction model processes each face  
3. **Filtering** → Marks people aged 60+ as seniors  
4. **Logging** → Saves detected senior details in `log.csv`  

---

## 📈 Example Output  

<img src="/samples/Visualization.jpg" alt="Sample Output" width="1200"/>

---

## 🏃 How to Run  
```bash
# Clone the repo
git clone https://github.com/ShravyaMalogi/real-time-senior-citizen-identifier.git
cd real-time-senior-citizen-identifier

# Install dependencies
pip install -r requirements.txt

# Run with webcam
python app.py --source 0

# Run on video file
python app.py --source path/to/video.mp4
