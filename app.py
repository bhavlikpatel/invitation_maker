from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image, ImageDraw
import cv2
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# -------------------------------
# Face detector (reliable)
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def make_circle(img):
    size = img.size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)

    result = Image.new("RGBA", size)
    result.paste(img, (0,0), mask)
    return result


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # biggest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

    # extra padding for turban/hair
    pad = int(h * 0.35)
    y = max(0, y - pad)
    h = min(img.shape[0] - y, h + pad*2)

    return img[y:y+h, x:x+w]

# -------------------------------
# API
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if "photo" not in request.files:
        return jsonify({"error": "Photo missing"}), 400

    # Crop values from CropperJS
    x = int(float(request.form["x"]))
    y = int(float(request.form["y"]))
    w = int(float(request.form["w"]))
    h = int(float(request.form["h"]))

    # Load uploaded photo
    img = Image.open(request.files["photo"]).convert("RGBA")

    # Crop selected face
    face = img.crop((x, y, x + w, y + h))
    face = face.resize((300,300))

    # Make circular mask
    mask = Image.new("L", (300,300), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0,0,300,300), fill=255)

    face_circle = Image.new("RGBA", (300,300))
    face_circle.paste(face, (0,0), mask)

    # Load template
    bg = Image.open("static/template.png").convert("RGBA")

    # bg_w, bg_h = bg.size

    # Dynamic position for large template
    circle_size = 900   # ~470 px
    x = 100             # ~120 px from left
    y = 2550            # ~2410 px from top

    face = face.resize((circle_size, circle_size))

    mask = Image.new("L", (circle_size, circle_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0,0,circle_size,circle_size), fill=255)

    face_circle = Image.new("RGBA", (circle_size, circle_size))
    face_circle.paste(face, (0,0), mask)

    bg.paste(face_circle, (x, y), face_circle)



    # Return image
    output = io.BytesIO()
    bg.save(output, format="PNG")
    output.seek(0)
    return send_file(output, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

