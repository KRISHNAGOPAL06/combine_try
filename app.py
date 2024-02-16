import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image
import numpy as np
import io
from flask import Flask
from flask_cors import CORS

# Create a Flask app
app = Flask(__name__)

# Allow requests from all origins
CORS(app)

st.title("Python or Anaconda Predictor")
st.header("Large Serpent Classifier")
st.text("Upload an Image for image classification as Anaconda or Python")

@st.experimental_singleton
def classify_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    return teachable_machine_classification(image, 'trained_model_accurate.h5')

@st.experimental_singleton
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/classify_image', methods=['POST'])
def upload_file():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        label = classify_image(image_data)
        return {"label": label}

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
