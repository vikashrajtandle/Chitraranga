# 📸 Chitraranga – Image Colorization using U-Net

Chitraranga is a deep-learning project that colorizes grayscale images using a **U-Net neural network**.
The project includes both a **trained model** (`UNet_chitraranga_model.keras`) and a complete **Flask web application** for real-time image colorization.

---

## 🚀 Features

* Converts **black & white images** into color automatically
* Built using **TensorFlow / Keras**
* Lightweight **U-Net** architecture
* Flask interface for quick usage
* Supports image upload and colorized output download
* Works on CPU or GPU

---

## 📂 Project Structure

```
Chitraranga/
│
├── model/
│   ├── UNet_chitraranga_model.keras     # Saved trained U-Net model
│   └── train.ipynb                      # Training notebook
│
├── app/
│   ├── app.py                           # Flask server
│   ├── static/
│   │   ├── styles.css
│   │   └── output.png                   # Last generated output
│   └── templates/
│       └── index.html                   # Web UI for uploads
│
├── utils/
│   └── preprocessing.py                 # Image preprocessing helpers
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture (U-Net)

The U-Net used in this project follows the classic encoder–decoder structure:

* **Encoder:** Convolution + MaxPooling
* **Bottleneck:** Deeper conv layers
* **Decoder:** UpSampling + skip connections
* **Output:** 3-channel (RGB) prediction

It was trained on (example):

* **CIFAR-10** or any dataset converted to grayscale → color pairs
* Input size: **128×128**

> The model performs best on grayscale images that are low–medium resolution.

---

## 🔧 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Chitraranga.git
cd Chitraranga
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask app

```bash
python app/app.py
```

Then open:

```
http://127.0.0.1:5000/
```

---

## 🌈 Usage

1. Open the web app
2. Upload any **grayscale** JPG/PNG image
3. Model processes the image
4. Download or view the colorized result

---

## 📦 Model File

The trained U-Net model is saved as:

```
UNet_chitraranga_model.keras
```

To load it:

```python
from tensorflow.keras.models import load_model

model = load_model("UNet_chitraranga_model.keras")
```

---

## 🛠 Improvements You Can Add

* Replace MSE with **perceptual loss** for sharper results
* Train on a **higher-resolution** dataset (e.g., ImageNet, Places365, CelebA-HQ)
* Add a **GAN-based discriminator** (Pix2Pix)
* Add drag-and-drop UI
* Deploy on HuggingFace Spaces or Railway.app

---

## 🖼 Example (Before / After)

(Add your images here)

```
before/gray.png → after/colorized.png
```

---

## 🧪 Dataset

You can use any dataset for training:

* CIFAR-10 (resized to 128×128)
* ImageNet subset
* CelebA (faces)
* Custom photography dataset

Each image is converted:

```
RGB → grayscale (input)
RGB → original (target)
```

---
