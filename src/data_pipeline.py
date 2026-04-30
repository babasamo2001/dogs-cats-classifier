import tensorflow as tf
import os

# CONFIGURATION

IMAGE_SIZE = (160, 160)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# CACHE DIRECTORY

os.makedirs("cache", exist_ok=True)


# RESCALING LAYER

rescale = tf.keras.layers.Rescaling(1.0 / 255)


# DATA AUGMENTATION

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomZoom(0.1),

])


# CREATE DATASET

def create_tf_data(
        data_dir: str,
        shuffle: bool = True
) -> tf.data.Dataset:

    dataset = tf.keras.utils.image_dataset_from_directory(

        data_dir,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="binary",

        shuffle=shuffle

    )

    # Apply Rescaling
    dataset = dataset.map(

        lambda x, y: (rescale(x), y),

        num_parallel_calls=AUTOTUNE

    )

    # Cache to DISK (safe for large datasets)
    cache_path = os.path.join(
        "cache",
        os.path.basename(data_dir)
    )

    dataset = dataset.cache(cache_path)

    # Prefetch
    dataset = dataset.prefetch(

        buffer_size=AUTOTUNE

    )

    return dataset


# APPLY AUGMENTATION

def add_augmentation(
        dataset: tf.data.Dataset
) -> tf.data.Dataset:

    def augment(images, labels):

        images = augmentation(
            images,
            training=True
        )

        return images, labels

    return dataset.map(

        augment,

        num_parallel_calls=AUTOTUNE

    )