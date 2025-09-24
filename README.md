# ✋🤖 Real-Time Sign Detection with MediaPipe & TensorFlow  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg?logo=opencv)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 🌟 Overview  

This repository implements a **real-time sign detection system** using **Google’s MediaPipe Holistic model** combined with a **TensorFlow deep learning classifier**.  

The system can:  
- 🖐️ Detect **hand landmarks** (21 points per hand).  
- 👤 Track **pose landmarks** (body keypoints).  
- 😀 Capture **face mesh landmarks** for additional context.  
- 🔎 Train a classifier to recognize **hand signs or gestures** from landmark sequences.  
- 📹 Operate in **real-time** on video streams using OpenCV.  

---

## 🎯 Motivation  

Sign language is a critical communication tool, yet many ML datasets are **scarce** and **imbalanced**.  
This project aims to:  

- ✅ Provide an **accessible, real-time pipeline** for gesture/sign detection.  
- ✅ Combine **classical CV (OpenCV)** with **deep learning (TensorFlow)** and **pose estimation (MediaPipe)**.  
- ✅ Demonstrate **end-to-end training**, from landmark extraction → dataset building → model training → real-time inference.  

---

## 🚀 Features  

- ⚡ **Real-time landmark detection** with MediaPipe.  
- ✍️ **Custom dataset creation** from video feeds.  
- 🧠 **TensorFlow-based classification** of gestures.  
- 🎥 **Live video demo** with overlayed landmarks + predictions.  
- 📊 **Scikit-learn utilities** for dataset splitting & evaluation.  

---

## 🛠️ Technical Approach  

### 🔹 1. Landmark Extraction with MediaPipe  
- Uses `mp.solutions.holistic.Holistic` for **face, hands, and pose landmarks**.  
- Each video frame is preprocessed with **OpenCV**:  
  - Convert **BGR → RGB** for MediaPipe.  
  - Run through the Holistic model.  
  - Convert back **RGB → BGR** for OpenCV display.  
- Landmarks are drawn using `mp.solutions.drawing_utils`.  

### 🔹 2. Dataset Creation  
- Extracts landmark coordinates for:  
  - **21 hand landmarks (left & right)**  
  - **33 pose landmarks**  
  - **468 face landmarks (optional)**  
- Saves features + labels to disk for supervised training.  

### 🔹 3. Model Training  
- Landmark arrays are flattened and fed into a **TensorFlow/Keras model**.  
- Typical architecture:  
  - Dense layers with ReLU activation.  
  - Dropout layers for regularization.  
  - Softmax output layer for multi-class classification.  
- Training process includes:  
  - Data normalization.  
  - Train/validation split via **scikit-learn**.  
  - Monitoring accuracy & loss.  

### 🔹 4. Real-Time Inference  
- Capture live video stream with OpenCV.  
- Extract landmarks per frame with MediaPipe.  
- Preprocess and feed to the trained model.  
- Display predictions (e.g., “Hello”, “Yes”, “No”) directly on the video window.  

---

## 📊 Example Workflow  

```python
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained classifier
model = load_model("sign_classifier.h5")

# Initialize mediapipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Preprocess frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # TODO: Extract landmarks → preprocess → predict with model
    
    cv2.imshow("Sign Detection", image)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
