from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow as tf
import keras

def preprocess_image(img):
    # Convert image to RGBA (if not already)
    img = img.convert("RGBA")
    
    # Create a white background image with the same size
    new_img = Image.new("RGBA", img.size, "WHITE")
    
    # Composite the original image onto the white background
    new_img.paste(img, (0, 0), img)
    
    # Convert RGBA to RGB
    new_img = new_img.convert("RGB")
    
    return new_img

def apple_classification(img, weights_file, classes):
    model = keras.models.load_model(weights_file)
    
    # Preprocess the image to make background white
    img = preprocess_image(img)

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
