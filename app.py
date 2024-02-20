import requests
import os
from tkinter import filedialog, Tk
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps  # Import ImageOps for image manipulation
from io import BytesIO
from img_classification import teachable_machine_classification
from apple import apple_classification

app = Flask(__name__)
CORS(app)

# Define your API key
API_KEY = 'mh5QtvF6HrKsYaBFPDW6LDje'

# Define the URL for the Remove.bg API
API_URL = 'https://api.remove.bg/v1.0/removebg'

# Define a dictionary to store model names and classes lists for different crops
crop_models = {
    'apple': {
        'classes_list': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'invalid'],
        'model_name': 'apple_last.h5'
    },
    'potato': {
        'classes_list': ['Potato___Early_blight','Potato___Late_blight','Potato___healthy','invalid'],
        'model_name': 'potato_last.h5'
    },
    'corn': {
        'classes_list': ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'invalid'],
        'model_name': 'corn_last.h5'
    }
    # Add more crops and their corresponding model names and classes lists here
}

# Function to select an image file using a file dialog
def select_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()  # Show file dialog to select an image
    return file_path

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
        
        # Open the selected image file
        with open(file, 'rb') as image_file:
            # Make a POST request to the Remove.bg API
            response = requests.post(API_URL, headers={'X-Api-Key': API_KEY}, files={'image_file': image_file}, data={'size': 'auto'})

        # Check if the request was successful
        if response.status_code == 200:
            # Open the processed image from the response content
            processed_image = Image.open(BytesIO(response.content))
            
            # Create a new image with white background
            new_image = Image.new("RGB", processed_image.size, "white")
            
            # Paste the processed image onto the new image with white background
            new_image.paste(processed_image, (0, 0), processed_image)
            
            # Call the apple_classification model
            label = apple_classification(new_image, model_name, classes_list)
            print(label)

            return jsonify({'label': label}), 200
        else:
            # Print the error message
            return jsonify({'error': 'Background removal failed. Error code: ' + str(response.status_code)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
