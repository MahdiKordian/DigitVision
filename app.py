import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from config import APP_NAME, METRICS_PATH, MODEL_PATH, TAGLINE, VERSION

AUTHOR_NAME = "Mahdi Kordian"
AUTHOR_TITLE = "AI Engineer"
AUTHOR_FOCUS = "ML · DL · Generative AI · LLMs · Python"
AUTHOR_LOCATION = "Iran"
AUTHOR_AVATAR = "https://avatars.githubusercontent.com/u/314552004?v=4"

REPO_URL = "https://github.com/MahdiKordian/DigitVision"
DEMO_URL = "https://digitvision-app.streamlit.app"
AUTHOR_LINKS = {
    "🐙 GitHub": "https://github.com/MahdiKordian",
    "💼 LinkedIn": "https://www.linkedin.com/in/mahdikordian/",
    "✈️ Telegram": "https://t.me/MahdiKordian/",
    "𝕏 X (Twitter)": "https://x.com/MahdiKordian/",
    "▶️ YouTube": "https://www.youtube.com/@MahdiKordian/",
}

st.set_page_config(
    page_title=f"{APP_NAME} v{VERSION}",
    page_icon="🔢",
    layout="wide",
)

st.markdown(
    """
