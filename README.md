## 📌 Author
Rozina Mohsin Pathan – Project

# multiclass-fish-image-classification
Multiclass Fish Image Classification using CNN and Transfer Learning (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0) with Streamlit deployment.
# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
This project focuses on classifying fish images into multiple species using Deep Learning and Transfer Learning.  
We trained:
- A custom CNN from scratch
- Five pretrained models: **VGG16**, **ResNet50**, **MobileNetV2**, **InceptionV3**, and **EfficientNetB0**

The best-performing model (**MobileNetV2**) was deployed as a **Streamlit web application** for real-time predictions.

---

## 🎯 Problem Statement
The goal is to develop a robust multiclass image classification model that can accurately predict the species of a fish from an image.  
The solution should be deployment-ready and provide confidence scores for predictions.

---

## 📊 Skills Learned
- Deep Learning with TensorFlow/Keras
- Transfer Learning & Fine-tuning
- Data Preprocessing & Augmentation
- Model Evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- Visualization of training history
- Model Deployment with Streamlit
- Python scripting & modular coding

---

## 📂 Dataset
- Images are organized in class-wise folders (one folder per fish species).
- Loaded using TensorFlow's `ImageDataGenerator`.
- Preprocessing:
  - Rescale to [0, 1]
  - Augment with rotation, zoom, and horizontal flip

*Note:* Dataset provided as part of the internship.

---

## 🛠️ Approach

### 1️⃣ Data Preprocessing & Augmentation
- Rescaling to normalize pixel values
- Data augmentation to improve robustness

### 2️⃣ Model Training
- **Custom CNN** from scratch
- **Transfer Learning Models:**
  - VGG16
  - ResNet50
  - MobileNetV2
  - InceptionV3
  - EfficientNetB0
- Fine-tuning the best-performing model

### 3️⃣ Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion Matrix visualization
- Accuracy & Loss plots

### 4️⃣ Deployment
- Streamlit app (`app.py`) to:
  - Upload an image
  - Predict fish species
  - Display confidence scores

---

## 📈 Results

| Model           | Test Accuracy | Test Loss |
|-----------------|--------------|-----------|
| CNN Scratch     | XX.XX%       | X.XXXX    |
| VGG16           | XX.XX%       | X.XXXX    |
| ResNet50        | XX.XX%       | X.XXXX    |
| **MobileNetV2** ✅ | XX.XX%   | X.XXXX    |
| InceptionV3     | XX.XX%       | X.XXXX    |
| EfficientNetB0  | XX.XX%       | X.XXXX    |

Best Model: **MobileNetV2** with highest accuracy.

---

## 📷 Visualizations
All visualizations are stored in the `reports/` folder:
- Confusion matrices for all models
- Accuracy curves
- Loss curves

---

## 🚀 How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
