import os
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained custom CNN models once when the application starts.
age_model = load_model(os.path.join(BASE_DIR, "age_model.h5"), compile=False)
gender_model = load_model(os.path.join(BASE_DIR, "gender_model.h5"), compile=False)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_from_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("The uploaded file could not be read as an image.")

    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    age_prediction = age_model.predict(img, verbose=0)
    gender_prediction = gender_model.predict(img, verbose=0)

    age = max(0, int(round(float(age_prediction[0][0]))))
    gender = "Female" if float(gender_prediction[0][0]) > 0.5 else "Male"

    return age, gender


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "healthy", "service": "Age & Gender Detection API"}


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if not file or not file.filename:
        return render_template("index.html", error="Please choose an image first.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP.",
        )

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        age, gender = predict_from_image(filepath)
    except (ValueError, cv2.error) as exc:
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template("index.html", error=str(exc))
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template(
            "index.html",
            error="Prediction failed. Please try another clear face image.",
        )

    return render_template(
        "index.html",
        age=age,
        gender=gender,
        image=f"/static/uploads/{unique_name}",
    )


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        error="Image is too large. Please upload an image smaller than 10 MB.",
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
