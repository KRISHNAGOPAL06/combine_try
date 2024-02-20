from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

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
            return processed_image
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
            # Call the prediction model with the processed image
            # Here, you should replace `model_prediction_function` with the appropriate function
            # to predict the label using the processed image
            label = model_prediction_function(processed_image)
            return jsonify({'label': label}), 200
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
