from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import keras
from keras.models import load_model
from PIL import Image, ImageOps
def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)

    size = (224, 224)  # New size
    
    # Resize the image
    image = img.resize(size)
  
    # Convert image to array
    image_array = img_to_array(image)
  
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  
    # Expand dimensions to match the model input shape
    data = np.expand_dims(normalized_image_array, axis=0)

    # Make prediction
    prediction = model.predict(data)
    
    return np.argmax(prediction)

