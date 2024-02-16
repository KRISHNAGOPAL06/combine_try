from flask import Flask, request, jsonify
from flask_cors import CORS
from img_classification import teachable_machine_classification
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app, origins='http://localhost:3000/culti_doctor')

def classify_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    label = teachable_machine_classification(image, 'trained_model_accurate.h5')
    return label

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/classify_image', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'})
    if file and allowed_file(file.filename):
        image_data = file.read()
        label = classify_image(image_data)
        return jsonify({'label': label})
    else:
        return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
