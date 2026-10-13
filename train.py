import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, callbacks, layers, models

from config import APP_NAME, METRICS_PATH, MODEL_PATH, VERSION

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
VAL_SIZE = 6000

tf.keras.utils.set_random_seed(SEED)


def load_data():
    path = os.path.expanduser("~/.keras/datasets/mnist.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])
    return tf.keras.datasets.mnist.load_data()


def prepare(x):
    return x.reshape((-1, 28, 28, 1)).astype("float32") / 255.0


def conv_block(x, filters, n=2):
    for _ in range(n):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(0.25)(x)


def build_model():
    inputs = Input(shape=(28, 28, 1))

    x = layers.RandomRotation(0.08, fill_mode="constant")(inputs)
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="constant")(x)
    x = layers.RandomZoom(0.12, fill_mode="constant")(x)

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128, n=1)

    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{APP_NAME}_v{VERSION}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = prepare(x_train), prepare(x_test)

    idx = np.random.RandomState(SEED).permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]
    x_val, y_val = x_train[:VAL_SIZE], y_train[:VAL_SIZE]
    x_tr, y_tr = x_train[VAL_SIZE:], y_train[VAL_SIZE:]

    model = build_model()
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=2, min_lr=1e-5, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        x_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=cbs,
    )
    elapsed = time.time() - start
