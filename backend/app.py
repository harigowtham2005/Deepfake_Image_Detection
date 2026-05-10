import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
from mtcnn import MTCNN

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "deepfake_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
IMG_SIZE = 224

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL + FACE DETECTOR
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)
detector = MTCNN()

# ==============================
# CHECK FILE TYPE
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# EXTRACT FACE (CRITICAL FIX)
# ==============================
def extract_face(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces is None or len(faces) == 0:
            return None

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)

        face = img[y:y+h, x:x+w]

        if face.shape[0] < 40 or face.shape[1] < 40:
            return None

        return face

    except Exception as e:
        print("Face extraction error:", e)
        return None

# ==============================
# MAIN ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_filename = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        # File validation
        if not file or file.filename == "":
            error = "No file selected"
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Unsupported file format. Use JPG, JPEG, PNG."
            return render_template("index.html", error=error)

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        image_filename = filename

        # 🔥 FACE EXTRACTION (IMPORTANT)
        face = extract_face(filepath)

        if face is None:
            error = "No face detected. Upload a clear frontal face image."
            return render_template("index.html", error=error)

        # Resize + preprocess
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = tf.keras.applications.efficientnet.preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Prediction
        result = model.predict(face)[0][0]

        print("Raw Prediction:", result)  # Debug

        prediction = "Real" if result > 0.5 else "Fake"
        confidence = round((result if result > 0.5 else 1 - result) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_filename=image_filename,
        error=error
    )

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)