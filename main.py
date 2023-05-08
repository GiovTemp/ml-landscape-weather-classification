import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = 'weather_classification.h5'
img_path = 'test_image.jpg'
img_height, img_width = 224, 224

# Load the model
model = load_model(model_path)

# Load the test image
img = load_img(img_path, target_size=(img_height, img_width))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.

# Predict the class of the test image
preds = model.predict(x)
pred_class = np.argmax(preds, axis=1)[0]

# Print the predicted class and corresponding label
class_labels = {0: 'cloudy', 1: 'rainy', 2: 'shine', 3: 'sunrise'}
print('Predicted class:', class_labels[pred_class])
