
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
MODEL_PATH = "MobileNetV2_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10', 'Class11']

# App title
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image and get instant prediction with confidence scores.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100

    st.write(f"**Prediction:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
