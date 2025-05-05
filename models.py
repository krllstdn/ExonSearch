import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def enhanced_cnn(input_shape=(201, 4)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(96, 15, padding="same", activation="swish")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Inception-like blocks
    for filters in [128, 256, 512]:
        # Parallel convs
        branch1 = tf.keras.layers.Conv1D(
            filters // 2, 5, padding="same", activation="swish"
        )(x)
        branch2 = tf.keras.layers.Conv1D(
            filters // 2, 3, padding="same", activation="swish"
        )(x)
        x = tf.keras.layers.Concatenate()([branch1, branch2])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

    # Attention gate
    att = tf.keras.layers.Conv1D(1, 1, activation="sigmoid")(x)
    x = tf.keras.layers.Multiply()([x, att])

    # x = se_block(x)

    # Classifier
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dense(256, activation="swish")(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    outputs = tf.keras.layers.Dense(1, "sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def se_block(x, ratio=8):
    """Squeeze-and-Excitation (SE) Block"""
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAvgPool1D()(x)
    se = tf.keras.layers.Dense(filters // ratio, activation="swish")(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
    se = tf.keras.layers.Multiply()([x, tf.keras.layers.Reshape((1, filters))(se)])
    return se


def build_attention_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=10, activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu")(x)

    # Attention layer
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def build_simple_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(64, 10, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 5, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_complex_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(64, kernel_size=15, padding="valid", activation="relu")(
        inputs
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="valid", activation="relu")(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding="valid", activation="relu")(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def enhanced_301(input_shape=(301, 4)):
    """
    - Trained slower than the previous model
    - The process was killed after first batch. Likely due to OOM
    I had no time for debug and the previous model was already good enough, so I let it be.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(96, 5, activation="gelu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(192, 11, activation="gelu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    att = tf.keras.layers.Conv1D(1, 25, activation="sigmoid", padding="same")(x)
    x = tf.keras.layers.Multiply()([x, att])

    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, value_dim=192)(x, x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    outputs = tf.keras.layers.Dense(1, "sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
