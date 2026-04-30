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

# We will extract directly into the 'models' folder
MODEL_BASE_DIR = "models"
# This is where TensorFlow expects to see the .pb file
SAVED_MODEL_PATH = os.path.join(MODEL_BASE_DIR, "saved_model")
ZIP_PATH = "models/saved_model.zip"

FILE_ID = "1HEQEZsxN6iagYXLNXqCAAEFLwSxtGUG7"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs(MODEL_BASE_DIR, exist_ok=True)

# 1. Download and Extract
# We check for the actual .pb file to ensure extraction worked
pb_file_path = os.path.join(SAVED_MODEL_PATH, "saved_model.pb")

if not os.path.exists(pb_file_path):
    if not os.path.exists(ZIP_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(URL, ZIP_PATH, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        # Extracting to 'models' so it creates 'models/saved_model/...'
        zip_ref.extractall(MODEL_BASE_DIR)
        print("Extraction complete.")

# 2. LOAD THE MODEL
# We point TFSMLayer to the directory containing the .pb file
print(f"Loading SavedModel from: {SAVED_MODEL_PATH}")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(
            SAVED_MODEL_PATH,
            call_endpoint="serving_default"
        )
    ])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback: list files to help debug in Render logs
    for root, dirs, files in os.walk(MODEL_BASE_DIR):
        print(f"Contents of {root}: {files}")
    raise e

# Warm-up
dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
model(dummy)


# ==========================
# IMAGE PREPROCESSING
# ==========================

def preprocess_image(image: Image.Image):
    image = image.resize((160, 160))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ==========================
# SERVE FRONTEND
# ==========================

# Ensure these directories exist or app will crash on mount
os.makedirs("static", exist_ok=True)
os.makedirs("assets", exist_ok=True)

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

        prediction_dict = model(processed)
        output_key = list(prediction_dict.keys())[0]
        raw_val = float(prediction_dict[output_key].numpy()[0][0])

        if raw_val > 0.5:
            label, confidence = "Dog", raw_val
        else:
            label, confidence = "Cat", 1 - raw_val

        return {
            "prediction": label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)