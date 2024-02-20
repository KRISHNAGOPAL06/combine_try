from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
import io

import numpy as np
import keras
from keras.models import load_model
from PIL import Image, ImageOps

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy','Invalid']
def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)

    # Convert PIL image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Load the image using load_img
    image = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))
    
    # Convert image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get predicted class index
    predicted_class_index = np.argmax(prediction)
    
    # Get class label corresponding to the predicted index
    predicted_class_label = classes[predicted_class_index]
    
    return predicted_class_label
