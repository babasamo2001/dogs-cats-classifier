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

# Ensure core directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI()

# ==========================
# AUTO-DETECT MODEL PATH
# ==========================
MODEL_BASE_DIR = "models"
ZIP_PATH = os.path.join(MODEL_BASE_DIR, "saved_model.zip")
FILE_ID = "1HEQEZsxN6iagYXLNXqCAAEFLwSxtGUG7"
URL = f"https://drive.google.com/uc?id={FILE_ID}"


def find_model_path(root_dir):
    """Recursively search for the directory containing saved_model.pb"""
    for root, dirs, files in os.walk(root_dir):
        if "saved_model.pb" in files:
            return root
    return None


# 1. Download and Extract
# We only download if we can't find a .pb file anywhere in 'models'
existing_path = find_model_path(MODEL_BASE_DIR)

if not existing_path:
    if not os.path.exists(ZIP_PATH):
        print("Downloading model...")
        gdown.download(URL, ZIP_PATH, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_BASE_DIR)

    # Check again after extraction
    existing_path = find_model_path(MODEL_BASE_DIR)

# 2. Load Model
model = None
if existing_path:
    print(f"Found SavedModel at: {existing_path}")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.TFSMLayer(existing_path, call_endpoint="serving_default")
        ])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("CRITICAL: saved_model.pb not found after extraction!")

# ==========================
# SERVE FRONTEND
# ==========================
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = "templates/index.html"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("index.html missing in templates folder", status_code=404)


# ==========================
# PREDICTION API
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    try:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((160, 160))
        processed = np.array(image).astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)

        prediction_dict = model(processed)
        # Handle dict or array output
        if isinstance(prediction_dict, dict):
            output_key = list(prediction_dict.keys())[0]
            raw_val = float(prediction_dict[output_key].numpy()[0][0])
        else:
            raw_val = float(prediction_dict.numpy()[0][0])

        if raw_val > 0.5:
            label, confidence = "Dog", raw_val
        else:
            label, confidence = "Cat", 1 - raw_val

        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)