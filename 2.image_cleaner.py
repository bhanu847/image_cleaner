"""
image_cleaner.py
────────────────
Flask web app to denoise & binarise scanned images for better OCR.
"""

from flask import (
    Flask, request, send_file,
    render_template_string, flash, redirect
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)
app.secret_key = "change‑me‑in‑production"

# ── Allowed file extensions ──────────────────────────────────────────────
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"}

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ── Bootstrap HTML template held in a string ─────────────────────────────
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clean Noisy Scan</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet">
</head>
<body class="bg-light py-5">
  <div class="container">
    <h1 class="mb-4 text-center">Clean Noisy Scan for OCR</h1>

    {% with msgs = get_flashed_messages() %}
      {% if msgs %}
        <div class="alert alert-warning">{{ msgs[0] }}</div>
      {% endif %}
    {% endwith %}

    <form method="POST" enctype="multipart/form-data" class="card p-4 shadow-sm">
      <div class="mb-3">
        <input class="form-control" type="file" name="image" required>
      </div>
      <button class="btn btn-primary">Process &amp; Download</button>
    </form>
  </div>
</body>
</html>
"""


# ── Core OpenCV cleaning pipeline (your exact steps) ─────────────────────
def clean_scan(file_bytes: bytes) -> bytes:
    """
    Apply grayscale → denoise → adaptive threshold → upscale.
    Returns JPEG bytes of the processed image.
    """
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh   = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=35, C=15
    )
    resized  = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Could not encode image")

    return buffer.tobytes()


# ── Flask route ──────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        if not allowed(file.filename):
            flash("Unsupported file type.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        processed_bytes = clean_scan(file.read())

        img_io = BytesIO(processed_bytes)
        img_io.seek(0)

        download_name = os.path.splitext(filename)[0] + "_clean.jpg"
        return send_file(
            img_io,
            mimetype="image/jpeg",
            as_attachment=True,
            download_name=download_name
        )

    # GET: show upload form
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    # pip install flask opencv-python numpy
    app.run(debug=True)
