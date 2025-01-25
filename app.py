import streamlit as st
import joblib
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import sklearn

# Load model
model = joblib.load("svm_model.joblib")
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def preprocess_image(image):
    resized_image = resize(image, (64, 64), anti_aliasing=True)
    features, _ = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features.reshape(1, -1)

st.title("Brain Tumor Classification Using SVM")

uploaded_file = st.file_uploader("Upload an MRI image for classification", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image = imread(uploaded_file, as_gray=True)
    features = preprocess_image(image)
    prediction = model.predict(features)
    result = class_names[prediction[0]]

    st.subheader(f"Prediction: {result}")
