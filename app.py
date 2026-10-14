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
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 20px; padding: 2rem 2.2rem; color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(124, 58, 237, .25);
}
.hero h1 {margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;}
.hero p {margin: .4rem 0 0 0; opacity: .92; font-size: 1.05rem;}
.badge {
    display: inline-block; background: rgba(255,255,255,.22); border-radius: 999px;
    padding: .15rem .8rem; font-size: .8rem; font-weight: 700; margin-left: .6rem;
    vertical-align: middle; backdrop-filter: blur(4px);
}
.result-card {
    border-radius: 20px; padding: 1.6rem; text-align: center; color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,.15);
}
.result-card .digit {font-size: 6.5rem; font-weight: 800; line-height: 1;}
.result-card .label {opacity: .85; font-size: .9rem; text-transform: uppercase; letter-spacing: 2px;}
.result-card .conf {font-size: 1.3rem; font-weight: 700; margin-top: .6rem;}
.high {background: linear-gradient(135deg, #059669, #10b981);}
.mid  {background: linear-gradient(135deg, #d97706, #f59e0b);}
.low  {background: linear-gradient(135deg, #dc2626, #ef4444);}
.footer {text-align: center; opacity: .55; font-size: .85rem; margin-top: 3rem;}

/* ---- About page ---- */
.about-card {
    border: 1px solid rgba(124, 58, 237, .25); border-radius: 18px;
    padding: 1.4rem 1.6rem; background: rgba(124, 58, 237, .05); height: 100%;
}
.about-card h3 {margin-top: 0;}
.step {
    display: flex; gap: .9rem; align-items: flex-start; margin-bottom: .8rem;
}
.step .num {
    flex: 0 0 2rem; height: 2rem; border-radius: 50%; color: white; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #db2777);
}
.step .txt {padding-top: .15rem;}
.author-name {font-size: 1.8rem; font-weight: 800; margin: 0; letter-spacing: -.5px;}
.author-title {font-weight: 600; color: #7c3aed; margin: 0;}
.author-meta {opacity: .75; margin: .2rem 0 0 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
