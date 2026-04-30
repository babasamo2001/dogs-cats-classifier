import os
import tensorflow as tf

# Clear Previous TensorFlow Graphs
tf.keras.backend.clear_session()

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from src.data_pipeline import (
    create_tf_data,
    add_augmentation
)

from src.model import (
    create_baseline_cnn
)


# Config

EPOCHS = 30

TRAIN_DIR = "data/cats_and_dogs/train"
VAL_DIR   = "data/cats_and_dogs/validation"

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
MODEL_DIR = "models"


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

LAST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

LOG_PATH = os.path.join(
    LOG_DIR,
    "training_log.csv"
)


# Create Folders Safely

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Load Data

print("Loading datasets...")

# Apply Scaling
train_ds = create_tf_data(
    TRAIN_DIR,
    shuffle=True
)

val_ds = create_tf_data(
    VAL_DIR,
    shuffle=False
)

# Apply Augmentation

train_ds = add_augmentation(
    train_ds
)


# Load or Create Model

if os.path.exists(LAST_MODEL_PATH):

    print("Resuming from last saved model...")

    model = tf.keras.models.load_model(
        LAST_MODEL_PATH
    )

    initial_epoch = 0

else:

    print("Starting fresh training...")

    model = create_baseline_cnn()

    initial_epoch = 0


# Callbacks (Fail-Safe)

callbacks = []


# Save best model

checkpoint_best = ModelCheckpoint(

    BEST_MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1,
    mode="min",
)

callbacks.append(
    checkpoint_best
)


# Save Last model

checkpoint_last = ModelCheckpoint(

    LAST_MODEL_PATH,

    save_best_only=False,

    verbose=0
)

callbacks.append(
    checkpoint_last
)


# Early Stopping

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)

callbacks.append(
    early_stop
)


# Reduce Learning Rate

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)

callbacks.append(
    reduce_lr
)


# CSV Logger (Learning Curves)

csv_logger = CSVLogger(

    LOG_PATH,

    append=True
)

callbacks.append(
    csv_logger
)


# Train Model

print("Training started...")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    initial_epoch=initial_epoch,

    callbacks=callbacks
)


# Save Final Model

model.save(

    FINAL_MODEL_PATH
)

print("Final model saved.")
print("Training completed safely.")