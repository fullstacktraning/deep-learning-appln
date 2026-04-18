import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

IMG_SIZE = 224

# Load model ONCE
model = load_model("pneumonia_model.keras")

def predict(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return {
            "label": "Pneumonia",
            "confidence": float(prediction)
        }
    else:
        return {
            "label": "Normal",
            "confidence": float(1 - prediction)
        }