import tensorflow as tf

IMAGE_SIZE = (160, 160)


def create_baseline_cnn():

    model = tf.keras.Sequential([

        # Input shape only
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),

        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),

        loss="binary_crossentropy",

        metrics=["accuracy"]
    )

    return model