# VisionAge — Age & Gender Detection

An AI-powered **Age & Gender Detection** web application built with a custom CNN architecture, TensorFlow/Keras, OpenCV and Flask.

The application accepts a face image, preprocesses it to **128×128**, and runs two trained neural networks:

- **Age model** — regression model that estimates age.
- **Gender model** — binary classification model that predicts Male/Female.

> This is an academic/portfolio computer-vision project. Predictions are estimates and should not be treated as authoritative demographic identification.

## Features

- Modern responsive web interface
- Drag-and-drop image upload
- Instant client-side image preview
- Custom CNN age prediction
- Custom CNN gender prediction
- Flask server-side inference
- OpenCV image preprocessing
- `/health` endpoint for deployment checks
- Secure generated upload filenames
- 10 MB upload limit
- Render/Gunicorn deployment configuration included

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Computer Vision | OpenCV |
| Deep Learning | TensorFlow / Keras |
| Models | Custom CNNs |
| Production Server | Gunicorn |
| Deployment | Render |

## Project Structure

```text
.
├── app.py
├── age_model.h5
├── gender_model.h5
├── train_model.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── static/
│   └── uploads/
│       └── .gitkeep
└── templates/
    └── index.html
```

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Age-Gender-Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Flask

```bash
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

## Deploy on Render

1. Push the project to GitHub.
2. Create a **Web Service** on Render.
3. Connect the GitHub repository.
4. Use **Python 3** / the version specified in `runtime.txt`.
5. Build command:

```bash
pip install -r requirements.txt
```

6. Start command:

```bash
gunicorn --workers 1 --threads 2 --timeout 120 app:app
```

7. Deploy and open the generated Render URL.

The project intentionally uses one Gunicorn worker because TensorFlow model loading consumes memory. The two trained `.h5` files are loaded once when the Flask process starts.

## Model Pipeline

```text
Uploaded image
      ↓
OpenCV image read
      ↓
Resize to 128 × 128
      ↓
Normalize pixels to 0–1
      ↓
Age CNN ───────→ Estimated age
      ↓
Gender CNN ────→ Male / Female
```

## API

### `GET /health`

Returns a simple health response:

```json
{
  "status": "healthy",
  "service": "Age & Gender Detection API"
}
```

### `POST /predict`

Upload an image using the `image` form field. The application returns the rendered result page with the predicted age and gender.

## Notes

- The training dataset is not included in this deployment repository.
- `train_model.py` is retained for reference/retraining.
- Uploaded images are stored temporarily under `static/uploads/` and are not intended as permanent storage.
- Render's filesystem is ephemeral, so production applications requiring persistent uploads should use object storage.
- Model predictions depend heavily on image quality, face framing, lighting and the training data used to build the models.

## Author

**Papishetty Sai Chaitanya**  
B.Tech — Computer Science & Engineering
