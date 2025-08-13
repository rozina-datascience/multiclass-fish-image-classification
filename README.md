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
├── app.py # Streamlit app
├── MobileNetV2_best.h5 # Best trained model
├── requirements.txt # Dependencies
├── multiclass_fish.ipynb # Training and evaluation notebook
├── model_comparison.csv # Metrics comparison
├── reports/ # Confusion matrices & plots
└── README.md # Project documentation

## 📷 Visualizations
All visualizations are stored in the `reports/` folder:
- Confusion matrices for all models
- Accuracy curves
- Loss curves

---
## 📌 Evaluation & Conclusion
- The experiment compared **five pre-trained CNN architectures** for multiclass fish image classification.
- **MobileNetV2** achieved the highest accuracy and was selected for deployment.
- The model generalizes well on unseen test images and can be used for real-world fish species identification.
- This project shows **transfer learning is effective** for classification problems with limited data.
- Training plots and confusion matrices are available in the `reports/` folder.

## 🚀 Links
- **Streamlit App**: https://multiclass-fish-image-classification-hz3xdwhpz896pd4is8a9pb.streamlit.app/
- **GitHub Repository**:https://github.com/rozina-datascience/multiclass-fish-image-classification.git

💻 How to Run Locally
1. Clone the repository

git clone https://github.com/rozina-datascience/multiclass-fish-image-classification.git
cd multiclass-fish-image-classification

2. Install dependencies
```bash
pip install -r requirements.txt



3.Run Streamlit App
streamlit run app.py

