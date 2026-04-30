import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

# Folders
os.makedirs("assets", exist_ok=True)

MODEL_PATH = "models/final_model.h5"
VAL_DIR    = "data/cats_and_dogs/validation"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded from:", MODEL_PATH)

# Load history
history_file = "models/history.json"
with open(history_file, "r") as f:
    history = json.load(f)

# Plot Accuracy Curve
plt.figure()
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("assets/accuracy_curve.png")
plt.show()

# Plot Loss Curve
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("assets/loss_curve.png")
plt.show()

# Load validation dataset
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Generate Predictions
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.concatenate([model.predict(x) for x, y in val_ds], axis=0)
y_pred_class = (y_pred > 0.5).astype(int)
print("Predictions generated.")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_class)
plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("assets/confusion_matrix.png")
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_class))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("assets/roc_curve.png")
plt.show()

# Validation Accuracy
test_accuracy = accuracy_score(y_true, y_pred_class)
print(f"Validation Accuracy: {test_accuracy:.4f}")