import os
import sys
import numpy as np
import tensorflow as tf
import gdown
import zipfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

app = FastAPI()

# ==========================
# MODEL DOWNLOAD + LOAD
# ==========================

MODEL_DIR = "models/saved_model"
ZIP_PATH = "models/saved_model.zip"
FILE_ID = "1HEQEZsxN6iagYXLNXqCAAEFLwSxtGUG7"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs("models", exist_ok=True)

# Download and Extract Model
if not os.path.exists(MODEL_DIR):
    if not os.path.exists(ZIP_PATH):
        print("Downloading model...")
        gdown.download(URL, ZIP_PATH, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("models")  # Extracting into models folder

# Load Model
print("Loading SavedModel...")
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(
        MODEL_DIR,
        call_endpoint="serving_default"
    )
])
print("Model ready.")

# Warm-up
dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
model(dummy)

# ==========================
# IMAGE PREPROCESSING
# ==========================

IMAGE_SIZE = (160, 160)


def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ==========================
# SERVE FRONTEND
# ==========================

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ==========================
# PREDICTION API
# ==========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        processed = preprocess_image(image)

        # Inference
        prediction_dict = model(processed)
        # Handle TFSMLayer output format (usually a dict or nested array)
        output_key = list(prediction_dict.keys())[0]
        raw_val = prediction_dict[output_key].numpy()[0][0]

        if raw_val > 0.5:
            label = "Dog"
            confidence = float(raw_val)
        else:
            label = "Cat"
            confidence = float(1 - raw_val)

        return {
            "prediction": label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)