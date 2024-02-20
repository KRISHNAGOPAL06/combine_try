from flask import Flask, request, jsonify
from flask_cors import CORS
from img_classification import teachable_machine_classification
from apple import apple_classification
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        crop_name = request.form.get('crop_name')  # Get crop name from form data
        if crop_name == 'apple':
         classes_list=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','invalid']
         model_name='apple_last.h5'
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        image = Image.open(file)
        label = apple_classification(image, model_name, classes_list, crop_name)
        return jsonify({'label': label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
