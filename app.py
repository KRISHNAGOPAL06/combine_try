from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO
from apple import apple_classification

app = Flask(__name__)
CORS(app)
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

# Define your API key for the Remove.bg API
REMOVE_BG_API_KEY = 'mh5QtvF6HrKsYaBFPDW6LDje'

# Define the URL for the Remove.bg API
REMOVE_BG_API_URL = 'https://api.remove.bg/v1.0/removebg'

# Function to remove the background of an image using the Remove.bg API
def remove_background(image):
    try:
        # Convert the image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Make a POST request to the Remove.bg API
        response = requests.post(
            REMOVE_BG_API_URL,
            files={'image_file': img_byte_arr},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVE_BG_API_KEY}
        )

        # Check if the request was successful
        if response.status_code == 200:
            # Open the processed image from the response content
            processed_image = Image.open(BytesIO(response.content))

            # Create a new image with the desired background color
            background = Image.new('RGB', processed_image.size, (228, 225, 220))

            # Composite the processed image on top of the background
            processed_with_bg = Image.alpha_composite(background.convert('RGBA'), processed_image.convert('RGBA')).convert('RGB')

            return processed_with_bg
        else:
            # If the request was not successful, return None
            return None
    except Exception as e:
        print('Error removing background:', str(e))
        return None


# Route for prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        crop_name = request.form.get('crop_name')  

        # Open the received image
        image = Image.open(file)

        # Remove the background
        processed_image = remove_background(image)

        if processed_image:
            crop_name = request.form.get('crop_name')  
            print('Received image:', file)
            print('Crop name:', crop_name)
            model_info = crop_models[crop_name]
            classes_list = model_info['classes_list']
            model_name = model_info['model_name']
            print(model_name)
            print(classes_list)
            label = apple_classification(processed_image, model_name, classes_list)
            print(label)
            return jsonify({'label': label}), 200
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
