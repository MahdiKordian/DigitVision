<div align="center">

# 🔢 DigitVision

**Handwritten digit recognition, powered by deep learning.**

Upload a photo or snap one with your camera, and a CNN trained on MNIST tells you which digit it sees, with a confidence score and a full look at how the image was processed.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://digitvision-app.streamlit.app)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Test Accuracy](https://img.shields.io/badge/MNIST%20test%20accuracy-99.32%25-success)](#-model-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Live Demo**](https://digitvision-app.streamlit.app) · [**Report a Bug**](https://github.com/MahdiKordian/DigitVision/issues) · [**Request a Feature**](https://github.com/MahdiKordian/DigitVision/issues)

</div>

<!--
  Tip: add a screenshot or GIF of the app to an `assets/` folder and uncomment the line below.
  <p align="center"><img src="assets/demo.png" alt="DigitVision demo" width="85%"></p>
-->

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [How It Works](#️-how-it-works)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage Tips](#-usage-tips)
- [Project Structure](#-project-structure)
- [Deployment](#️-deployment)
- [Limitations](#️-limitations)
- [License](#-license)
- [Author](#-author)

---

## 🎯 About the Project

MNIST models usually stop at a notebook: great accuracy on clean 28×28 test images, and no idea what happens with a real photo. **DigitVision** closes that gap. It pairs a compact, regularized CNN with a computer-vision preprocessing pipeline that turns messy, real-world photos of handwritten digits into MNIST-style input, and wraps everything in a clean Streamlit web app.

You can try it right now: **[digitvision-app.streamlit.app](https://digitvision-app.streamlit.app)**

## ✨ Features

- **Two input modes**: upload an image (PNG/JPG) or take a photo directly with your camera.
- **Smart digit extraction**: detects red ink via HSV color masking, with an automatic Otsu-threshold fallback for dark pen on light paper.
- **MNIST-style normalization**: crops, scales to 20 px, pads to 28×28 and centers the digit by its center of mass, the same way MNIST was built.
- **Transparent results**: predicted digit, confidence level (high / medium / low), top-3 predictions and a probability chart for all ten classes.
- **"How DigitVision saw your image"**: inspect the detection box, binary mask and the exact 28×28 model input.
- **Model dashboard**: test accuracy, parameter count, epochs and per-digit accuracy in the sidebar.
- **Prediction history**: the last 10 predictions of your session.
- **Reproducible training**: fixed seeds and a single script (`train.py`) that also writes `metrics.json`.

