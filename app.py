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

# Create directories BEFORE initializing FastAPI
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI()

# ==========================
# MODEL DOWNLOAD + LOAD
# ==========================
MODEL_BASE_DIR = "models"
SAVED_MODEL_PATH = os.path.join(MODEL_BASE_DIR, "saved_model")
ZIP_PATH = os.path.join(MODEL_BASE_DIR, "saved_model.zip")

FILE_ID = "1HEQEZsxN6iagYXLNXqCAAEFLwSxtGUG7"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Only download if the model directory doesn't exist
if not os.path.exists(SAVED_MODEL_PATH):
    if not os.path.exists(ZIP_PATH):
        print("Downloading model...")
        gdown.download(URL, ZIP_PATH, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_BASE_DIR)

    # Cleanup zip to save disk space on Render
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

print("Loading SavedModel...")
try:
    # Use the specific path where saved_model.pb lives
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint="serving_default")
    ])
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    # Don't let it crash silently; show what's inside the folder
    print(f"Folder contents: {os.listdir(SAVED_MODEL_PATH if os.path.exists(SAVED_MODEL_PATH) else '.')}")
    model = None

# ==========================
# SERVE FRONTEND
# ==========================
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = "templates/index.html"
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>index.html not found in templates/</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================
# PREDICTION API
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server."}

    try:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((160, 160))
        processed = np.array(image).astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)

        prediction_dict = model(processed)
        output_key = list(prediction_dict.keys())[0]
        raw_val = float(prediction_dict[output_key].numpy()[0][0])

        if raw_val > 0.5:
            label, confidence = "Dog", raw_val
        else:
            label, confidence = "Cat", 1 - raw_val

        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Ensure port binding is correct for Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)