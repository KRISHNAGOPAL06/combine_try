from flask import Flask, request, jsonify
from flask_cors import CORS
from img_classification import teachable_machine_classification
from apple import apple_classification
from PIL import Image

app = Flask(_name_)
CORS(app)

# Define a dictionary to store model names and classes lists for different crops
crop_models = {
    'apple': {
        'classes_list': ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy', 'invalid'],
        'model_name': 'apple_last.h5'
    },
    'potato': {
        'classes_list': ['Potato__Early_blight','Potato_Late_blight','Potato__healthy','invalid'],
        'model_name': 'potato_last.h5'
    },
    'corn': {
        'classes_list': ['Corn__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn__healthy', 'invalid'],
        'model_name': 'corn_last.h5'
    }
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
        model_info = crop_models[crop_name]
        classes_list = model_info['classes_list']
        model_name = model_info['model_name']
        print(model_name)
        print(classes_list)
        image = Image.open(file)
        label = apple_classification(image, model_name, classes_list)
        print(label)
        

        return jsonify({'label': label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if _name_ == '_main_':
    app.run(debug=True)
