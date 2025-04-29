import os
from flask import Flask, render_template, request, session, send_from_directory
from werkzeug.utils import secure_filename
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img

def fallback_tesseract(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
        doc = DocumentFile.from_images(file_path)
    elif ext == ".pdf":
        doc = DocumentFile.from_pdf(file_path)
    else:
        return "Unsupported file type"

    result = ocr_model(doc)

    lines_with_pos = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            line_text = " ".join([word.value for word in line.words])
            x_min = line.geometry[0][0]
            y_min = line.geometry[0][1]
            lines_with_pos.append((y_min, x_min, line_text))

    lines_with_pos.sort(key=lambda x: (round(x[0], 2), x[1]))

    grouped_lines = defaultdict(list)
    for y, x, text in lines_with_pos:
        key = round(y * 100)
        grouped_lines[key].append((x, text))

    formatted_lines = []
    for key in sorted(grouped_lines.keys()):
        line_parts = sorted(grouped_lines[key], key=lambda t: t[0])
        formatted_lines.append(" ".join([t[1] for t in line_parts]))

    return "\n".join(formatted_lines)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            text_output = extract_text(save_path)
            session["chat_history"].append({
                "filename": filename,
                "text": text_output
            })
            session.modified = True

    return render_template("index.html", messages=session.get("chat_history", []))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
