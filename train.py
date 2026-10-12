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
