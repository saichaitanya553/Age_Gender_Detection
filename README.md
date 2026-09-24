---
title: Age & Gender Detection
emoji: 👤
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
short_description: CNN-based age and gender detection from face images
---

# VisionAge — Age & Gender Detection

A portfolio-ready Flask web application that estimates age and binary gender from an uploaded image using two custom CNN models.

## What changed for free deployment

The original models were trained and saved as Keras `.h5` files. Running TensorFlow/Keras directly on a 512 MB Render Free instance caused worker timeouts and SIGKILL events during inference.

The deployment version therefore keeps the **same trained model weights and architecture**, but exports the inference weights to compact `.npz` files and runs the CNN forward pass with NumPy. This removes the TensorFlow runtime from the production server and greatly reduces memory requirements.

TensorFlow is **not required at runtime**.

## Architecture

```text
Upload image
    ↓
OpenCV resize → 128 × 128 × 3
    ↓
NumPy CNN inference
   ↙       ↘
Age CNN   Gender CNN
   ↘       ↙
Age + Gender result
    ↓
Flask UI
```

## Models

### Age model

```text
Input 128×128×3
→ Conv2D 32 + ReLU
→ MaxPool 2×2
→ Conv2D 64 + ReLU
→ MaxPool 2×2
→ Conv2D 128 + ReLU
→ MaxPool 2×2
→ Flatten
→ Dense 128 + ReLU
→ Dense 1 + Linear
```

### Gender model

```text
Input 128×128×3
→ Conv2D 32 + ReLU
→ MaxPool 2×2
→ Conv2D 64 + ReLU
→ MaxPool 2×2
→ Flatten
→ Dense 128 + ReLU
→ Dense 1 + Sigmoid
```

The original `.h5` files are retained under `original_models/` for reference/retraining. The live application loads `age_weights.npz` and `gender_weights.npz` instead.

## Tech stack

- Python
- Flask
- NumPy
- OpenCV
- Gunicorn
- Custom CNNs trained with Keras/TensorFlow
- HTML/CSS/JavaScript

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/health
```

## Render deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn --workers 1 --threads 1 --timeout 120 app:app
```

Python version:

```text
3.11.11
```

The production runtime does not install TensorFlow, which is the key change for low-memory free hosting.

## Re-exporting model weights

If the original `.h5` models are retrained or replaced, regenerate the compact runtime files with:

```bash
pip install h5py
python tools/export_weights.py
```

This script is only for model maintenance. It is not required to run the web application.

## Important limitation

The application predicts from the uploaded image after resizing the full image to 128×128. It does not perform a separate face-detection/cropping stage. For best results, upload a clear image where the face is prominent and reasonably centered.

## Project structure

```text
Age-Gender-Detection/
├── app.py
├── numpy_cnn.py
├── age_weights.npz
├── gender_weights.npz
├── original_models/
│   ├── age_model.h5
│   └── gender_model.h5
├── requirements.txt
├── Procfile
├── runtime.txt
├── .python-version
├── templates/
│   └── index.html
├── static/
│   └── uploads/
└── tools/
    └── export_weights.py
```

## Disclaimer

This project is for educational and portfolio purposes. Age and gender predictions are estimates and may be inaccurate, especially with low-quality, non-frontal, or poorly lit images.
