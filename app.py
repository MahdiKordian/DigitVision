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
<div class="hero">
  <h1>🔢 {APP_NAME}<span class="badge">v{VERSION}</span></h1>
  <p>{TAGLINE}</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found:\n`{MODEL_PATH}`")
        st.info("Please run `python train.py` first.")
        st.stop()
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


model = get_model()
metrics = get_metrics()


def preprocess_for_mnist(image):
    """
    تصویر BGR می‌گیره و دیکشنری شامل ورودی 28x28 مدل و مراحل میانی برمی‌گردونه.
    اگر رقمی پیدا نشه None برمی‌گردونه.
    """
    h, w = image.shape[:2]
    if max(h, w) > 800:
        s = 800 / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    method = "Red ink"

    if cv2.countNonZero(mask) < max(100, 0.0005 * h * w):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask > 0) > 0.5:
            mask = cv2.bitwise_not(mask)
        method = "Auto threshold (Otsu)"

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_area = max(cv2.contourArea(c) for c in contours)
    keep = [c for c in contours if cv2.contourArea(c) >= 0.1 * max_area]
    pts = np.vstack(keep)
    x, y, bw, bh = cv2.boundingRect(pts)
    if bw * bh < 50:
        return None

    digit = mask[y:y + bh, x:x + bw]

    scale = 20 / max(bw, bh)
    new_w, new_h = max(1, int(bw * scale)), max(1, int(bh * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    sx, sy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[sy:sy + new_h, sx:sx + new_w] = digit

    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28))

    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 120), max(2, w // 200))

    return {
        "input": canvas.astype("float32") / 255.0,
        "mask": mask,
        "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "method": method,
    }


def decode(uploaded):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def predict(processed_input):
    x = processed_input[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    return probs


def render_about():
    """About page: project introduction + about the author."""
    st.markdown("## 🎯 About the project")
    st.markdown(
        f"**{APP_NAME}** recognizes handwritten digits from real photos. A compact, "
        "regularized convolutional neural network trained on MNIST is combined with a "
        "computer-vision preprocessing pipeline that converts messy, real-world images "
        "into clean MNIST-style input, so the model works beyond the dataset itself."
    )

    b1, b2, _ = st.columns([1, 1, 3])
    b1.link_button("⭐ Source code", REPO_URL, use_container_width=True)
    b2.link_button("🚀 Live demo", DEMO_URL, use_container_width=True)

    if metrics:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
        m2.metric("Test loss", f"{metrics['test_loss']:.4f}")
        m3.metric("Parameters", f"{metrics['params']:,}")
        m4.metric("Epochs run", metrics["epochs_run"])

    st.markdown("&nbsp;", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
<div class="about-card">
<h3>⚙️ How it works</h3>
<div class="step"><div class="num">1</div><div class="txt"><b>Segment</b> — finds red ink via HSV masking, or falls back to Otsu thresholding for dark pen on light paper.</div></div>
<div class="step"><div class="num">2</div><div class="txt"><b>Clean up</b> — morphological closing and opening remove noise and fill gaps.</div></div>
<div class="step"><div class="num">3</div><div class="txt"><b>Localize</b> — contours are merged into one bounding box around the digit.</div></div>
<div class="step"><div class="num">4</div><div class="txt"><b>Normalize</b> — scaled to 20 px, padded to 28×28 and centered by its center of mass, like MNIST.</div></div>
<div class="step"><div class="num">5</div><div class="txt"><b>Classify</b> — the CNN returns a probability for each digit 0–9.</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
<div class="about-card">
<h3>🧠 Model &amp; stack</h3>

- **Augmentation:** random rotation, translation and zoom
- **Backbone:** 3 convolutional blocks (32 → 64 → 128 filters) with BatchNorm, ReLU, max-pooling and dropout
- **Head:** Dense(128) + BatchNorm + Dropout(0.4) → softmax over 10 classes
- **Training:** Adam, early stopping and learning-rate scheduling
- **Tools:** TensorFlow / Keras, OpenCV, Streamlit, NumPy, pandas

</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.info(
        "**Good to know:** DigitVision reads a single digit per image. The accuracy shown "
        "above is measured on the MNIST test set; results on real photos depend on lighting, "
        "pen thickness and image quality, which is why a confidence score is always shown."
    )

    st.divider()
    st.markdown("## 👨‍💻 About me")

    pic_col, info_col = st.columns([1, 4], gap="large")
    with pic_col:
        st.image(AUTHOR_AVATAR, width=170)
    with info_col:
        st.markdown(
            f"""
<p class="author-name">{AUTHOR_NAME}</p>
<p class="author-title">{AUTHOR_TITLE}</p>
<p class="author-meta">{AUTHOR_FOCUS}</p>
<p class="author-meta">📍 {AUTHOR_LOCATION}</p>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Hi, I'm {AUTHOR_NAME}. I build machine-learning and deep-learning projects "
            "and love turning models into tools people can actually use. "
            f"{APP_NAME} is one of them. Feel free to explore my work and get in touch."
        )

    link_cols = st.columns(len(AUTHOR_LINKS))
    for col, (label, url) in zip(link_cols, AUTHOR_LINKS.items()):
        col.link_button(label, url, use_container_width=True)


with st.sidebar:
    st.markdown(f"## 🔢 {APP_NAME}")
    st.caption(f"Version {VERSION}")
    st.divider()

    st.markdown("### 📊 Model info")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
        c2.metric("Parameters", f"{metrics['params'] / 1000:.0f}K")
        st.caption(f"Trained for {metrics['epochs_run']} epochs")
        with st.expander("Per-digit accuracy"):
            df = pd.DataFrame(
                {"Accuracy": metrics["per_class_accuracy"]}
            )
            st.bar_chart(df)
    else:
        st.info("Run `train.py` to see model stats.")

    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown(
        "- Write **one** digit, large and clear\n"
        "- Red marker on white paper works best\n"
        "- Avoid shadows and busy backgrounds\n"
        "- Dark pen on light paper also works"
    )

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []

    st.divider()
    st.markdown(f"Built by **[{AUTHOR_NAME}](https://github.com/MahdiKordian)**")
    st.caption(f"[Source on GitHub]({REPO_URL}) · MIT License")

if "history" not in st.session_state:
    st.session_state["history"] = []

tab_app, tab_about = st.tabs(["🔢 Recognize", "ℹ️ About"])

with tab_about:
    render_about()

with tab_app:
    tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Take photo"])

    source = None
    with tab_upload:
        up = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg"], key="file")
        if up is not None:
            source = up
    with tab_camera:
        cam = st.camera_input("Take a photo of a digit", key="cam")
        if cam is not None:
            source = cam

    if source is None:
        st.info("👆 Upload a photo or use your camera to get started.")
    else:
        image = decode(source)
        if image is None:
            st.error("Could not read this image. Try another file.")
        else:
            with st.spinner("Analyzing..."):
                result = preprocess_for_mnist(image)

            if result is None:
                st.error("Could not detect a digit in the image. Try a clearer photo.")
            else:
                probs = predict(result["input"])
                digit = int(np.argmax(probs))
                conf = float(np.max(probs))

                level = "high" if conf >= 0.90 else "mid" if conf >= 0.60 else "low"

                col_img, col_res = st.columns([1.2, 1], gap="large")
                with col_img:
                    st.image(result["annotated"], caption="Detected digit", use_container_width=True)
                with col_res:
                    st.markdown(
                        f"""
<div class="result-card {level}">
  <div class="label">Prediction</div>
  <div class="digit">{digit}</div>
  <div class="conf">{conf:.1%} confident</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    if level == "low":
                        st.warning("Low confidence. Try a clearer, larger, better-lit image.")

                st.markdown("### 📈 Probability of each digit")
                top3 = np.argsort(probs)[::-1][:3]
                tcols = st.columns(3)
                for rank, (col, d) in enumerate(zip(tcols, top3), start=1):
