import os

APP_NAME = "DigitVision"
VERSION = "1.0.0"
TAGLINE = "Handwritten digit recognition, powered by deep learning"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.keras")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
