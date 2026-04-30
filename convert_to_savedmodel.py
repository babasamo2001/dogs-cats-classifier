# convert_to_savedmodel.py

import tensorflow as tf
import os
import shutil

INPUT_MODEL = "models/best_model.keras"
OUTPUT_MODEL = "models/saved_model"

print("TensorFlow version:", tf.__version__)

print("Loading model...")

model = tf.keras.models.load_model(
    INPUT_MODEL,
    compile=False
)

# Remove old SavedModel if exists
if os.path.exists(OUTPUT_MODEL):
    shutil.rmtree(OUTPUT_MODEL)

print("Exporting SavedModel...")

model.export(OUTPUT_MODEL)

print("SavedModel created successfully.")