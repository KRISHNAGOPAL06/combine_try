from flask import Flask, request, jsonify
from flask_cors import CORS
from img_classification import teachable_machine_classification
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file)
    label = teachable_machine_classification(image, 'trained_model_accurate.h5')
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
