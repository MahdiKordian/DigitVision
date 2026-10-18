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

## ⚙️ How It Works

```
 Photo / Upload
       │
       ▼
 Resize (max 800 px)
       │
       ▼
 Enough red ink found? ── yes ──▶ HSV red-ink mask ──────┐
       │ no                                              │
       ▼                                                 │
 Grayscale → Gaussian blur → Otsu threshold ─────────────┤
 (auto-inverted if the background is dark)               │
                                                         ▼
                                       Morphological close + open
                                                         │
                                                         ▼
                                    Contour detection → bounding box
                                                         │
                                                         ▼
                           Scale to 20 px → pad to 28×28 → center of mass
                                                         │
                                                         ▼
                                       CNN → softmax (10 classes)
                                                         │
                                                         ▼
                                     Digit + confidence + charts
```

1. **Segmentation**: the image is searched for red ink first; if there isn't enough, it falls back to grayscale + Gaussian blur + Otsu thresholding (auto-inverted when the background is dark).
2. **Cleanup**: morphological closing and opening remove noise and fill small gaps.
3. **Localization**: contours with at least 10 % of the largest contour's area are merged into one bounding box.
4. **Normalization**: the digit is resized so its longest side is 20 px, placed on a 28×28 canvas and shifted so its center of mass sits at the center.
5. **Classification**: the CNN outputs a probability for each digit 0–9.

## 🧠 Model Architecture

A lightweight VGG-style CNN with built-in data augmentation, **289,066 parameters** in total.

| Stage | Layers |
|---|---|
| **Augmentation** (training only) | `RandomRotation(0.08)` → `RandomTranslation(0.1, 0.1)` → `RandomZoom(0.12)` |
| **Block 1** | 2 × [`Conv2D(32, 3×3)` → `BatchNorm` → `ReLU`] → `MaxPool` → `Dropout(0.25)` |
| **Block 2** | 2 × [`Conv2D(64, 3×3)` → `BatchNorm` → `ReLU`] → `MaxPool` → `Dropout(0.25)` |
| **Block 3** | 1 × [`Conv2D(128, 3×3)` → `BatchNorm` → `ReLU`] → `MaxPool` → `Dropout(0.25)` |
| **Classifier** | `Flatten` → `Dense(128)` → `BatchNorm` → `ReLU` → `Dropout(0.4)` → `Dense(10, softmax)` |

**Training setup**

| Setting | Value |
|---|---|
| Optimizer | Adam (`lr = 1e-3`) |
| Loss | Sparse categorical cross-entropy |
| Batch size | 128 |
| Max epochs | 30 |
| Validation split | 6,000 samples from the training set |
| Callbacks | `EarlyStopping` (monitor `val_accuracy`, patience 5, restore best weights), `ReduceLROnPlateau` (factor 0.5, patience 2, min lr 1e-5) |
| Random seed | 42 |

## 📊 Model Performance

Evaluated on the official 10,000-image MNIST test set. Early stopping ended training after **15 epochs**.

| Metric | Value |
|---|---|
| **Test accuracy** | **99.32 %** |
| Test loss | 0.0205 |
| Parameters | 289,066 |
| Epochs run | 15 (of 30 max) |
| Training time | ~23 min (1,376.7 s on the author's machine) |

**Per-digit accuracy**

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Accuracy** | 99.69 % | 99.82 % | 99.32 % | 99.80 % | 99.59 % | 98.77 % | 98.12 % | 99.51 % | 99.38 % | 99.01 % |

All numbers come from [`metrics.json`](metrics.json), which `train.py` generates automatically.

## 🛠 Tech Stack

| Area | Tools |
|---|---|
| Deep learning | TensorFlow / Keras |
| Computer vision | OpenCV (headless) |
| Web app | Streamlit |
| Data handling | NumPy, pandas |
| Language | Python |

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` (and optionally `venv`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MahdiKordian/DigitVision.git
cd DigitVision

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

The repository already includes the trained model (`mnist_model.keras`), so you can start right away:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually <http://localhost:8501>).

### (Optional) Retrain the model

```bash
python train.py
```

This downloads MNIST (if needed), trains the network, and saves:

- `mnist_model.keras`: the trained model
- `metrics.json`: accuracy, loss, parameters, epochs, training time and per-digit accuracy

The app reads both files automatically.

## 💡 Usage Tips

- Write **one** digit, **large and clear**, in the frame.
- A **red marker on white paper** gives the most reliable detection.
- A dark pen on light paper works too.
- Avoid shadows, glare and busy backgrounds.
- If the result card turns orange or red, the model is unsure: retake the photo with better light or a bigger digit.

## 📁 Project Structure

```
DigitVision/
├── app.py              # Streamlit web app (UI, preprocessing, inference)
├── train.py            # Model definition, training and evaluation
├── config.py           # App name, version, tagline and file paths
├── requirements.txt    # Python dependencies
├── mnist_model.keras   # Trained model
├── metrics.json        # Evaluation results shown in the app
├── LICENSE             # MIT License
├── .gitignore
└── README.md
```

## ☁️ Deployment

The live demo is hosted on **Streamlit Community Cloud**: [digitvision-app.streamlit.app](https://digitvision-app.streamlit.app).

To deploy your own copy:

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from your fork.
3. Set the main file to `app.py` and deploy.

> Make sure `mnist_model.keras` and `metrics.json` are committed. The app needs them at startup.

## ⚠️ Limitations

- DigitVision recognizes **a single digit per image**. Multi-digit numbers are not supported.
- The model is trained on MNIST only. The 99.32 % figure is measured on MNIST's own test set; accuracy on real photos depends on lighting, pen thickness and image quality, which is why the preprocessing pipeline and confidence indicator exist.
- Unusual writing styles (for example a crossed 7 or a looped 1) may reduce confidence.

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

