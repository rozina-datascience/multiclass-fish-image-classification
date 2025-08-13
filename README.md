## 📌 Author
Rozina Mohsin Pathan – Project

# multiclass-fish-image-classification
Multiclass Fish Image Classification using CNN and Transfer Learning (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0) with Streamlit deployment.

## 📌 Project Overview
This project focuses on classifying fish images into multiple categories using deep learning models.  
It involves training a CNN from scratch and applying transfer learning with five pre-trained models to improve accuracy.  
The best-performing model is deployed as a **Streamlit web application** for real-time predictions.

---

## 🎯 Skills Gained
- Deep Learning (CNNs)
- Python
- TensorFlow / Keras
- Transfer Learning
- Data Preprocessing & Augmentation
- Model Evaluation & Visualization
- Streamlit App Development
- Model Deployment

---

## 📂 Dataset
The dataset contains images of different fish species stored in separate class folders.  
Images were preprocessed using **ImageDataGenerator** with:
- Rescaling to [0, 1]
- Data Augmentation (rotation, zoom, flipping)

---

## 🛠️ Approach
1. **Data Preprocessing**
   - Rescale images
   - Apply data augmentation for better generalization

2. **Model Training**
   - Train a CNN model from scratch
   - Train 5 transfer learning models:
     - VGG16
     - ResNet50
     - MobileNetV2
     - InceptionV3
     - EfficientNetB0
   - Fine-tune the best models
   - Save best model as `.h5`

3. **Model Evaluation**
   - Compare Accuracy, Precision, Recall, F1-score
   - Plot training/validation curves
   - Generate confusion matrices

4. **Deployment**
   - Deploy best model (MobileNetV2) to **Streamlit Cloud**
   - User can upload an image and get prediction + confidence score

---

## 📊 Results
| Model          | Test Accuracy (%) |
|----------------|-------------------|
| MobileNetV2    | 97.14              |
| InceptionV3    | 96.52              |
| VGG16          | 73.21              |
| ResNet50       | 15.95              |
| EfficientNetB0 | 15.95              |

**✅ Best Model:** MobileNetV2 with 97.14% accuracy

---

## 📦 Project Structure

## 📷 Visualizations
All visualizations are stored in the `reports/` folder:
- Confusion matrices for all models
- Accuracy curves
- Loss curves

---

## 🚀 Links
- **Streamlit App**: https://multiclass-fish-image-classification-hz3xdwhpz896pd4is8a9pb.streamlit.app/
- **GitHub Repository**:


## 🚀 How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt




