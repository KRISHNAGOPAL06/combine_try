import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image
import numpy as np
import io

st.title("Python or Anaconda Predictor")
st.header("Large Serpent Classifier")
st.text("Upload an Image for image classification as Anaconda or Python")

@st.experimental_singleton
def classify_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    return teachable_machine_classification(image, 'trained_model_accurate.h5')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image_data = uploaded_file.read()
    st.image(image_data, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    with st.spinner('Classifying...'):
        label = classify_image(image_data)
        st.write("Guess:", label)
