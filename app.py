from flask import Flask, request, jsonify
from flask_cors import CORS
from img_classification import teachable_machine_classification
from apple import apple_classification
from PIL import Image

app = Flask(__name__)
CORS(app)

# Define a dictionary to store model names and classes lists for different crops
crop_models = {
    'apple': {
        'classes_list': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'invalid'],
        'model_name': 'apple_last.h5'
    },
    # Add more crops and their corresponding model names and classes lists here
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        crop_name = request.form.get('crop_name')  

        print('Received image:', file)
        print('Crop name:', crop_name)

        return jsonify({'message': 'Image and Crop Name received successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
