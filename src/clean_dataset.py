import os
import tensorflow as tf

TRAIN_DIR = os.path.join("data", "cats_and_dogs", "train")
VAL_DIR   = os.path.join("data", "cats_and_dogs", "validation")


def is_valid_image(path):
    try:
        img = tf.io.read_file(path)
        tf.image.decode_image(img, channels=3)  # FORCE RGB
        return True
    except:
        return False


def clean_folder(data_dir):
    print(f"\n🔍 Scanning: {data_dir}\n")

    removed = 0
    checked = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            checked += 1

            if not is_valid_image(path):
                print("❌ Deleting bad file:", path)
                os.remove(path)
                removed += 1

    print(f"\nDone: {data_dir}")
    print(f"Checked: {checked}")
    print(f"Removed: {removed}")


clean_folder(TRAIN_DIR)
clean_folder(VAL_DIR)