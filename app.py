from flask import Flask, request, jsonify
from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1.types import Image as VisionImage
import os
import uuid

# Initialize Flask app
app = Flask(__name__)

# Environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT_NAME")
ROBOFLOW_VERSION_NUMBER = os.getenv("ROBOFLOW_VERSION_NUMBER")

# Initialize Roboflow and Google Cloud Vision client
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_PROJECT_NAME)
model = project.version(ROBOFLOW_VERSION_NUMBER).model
vision_client = vision.ImageAnnotatorClient()

# Configure image saving directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Utility to save debug images
def save_debug_image(image, step, label_id=None):
    filename = f"{LOG_DIR}/{step}_{label_id or str(uuid.uuid4())}.png"
    cv2.imwrite(filename, image)
    return filename

# Function for perspective correction
def perspective_correction(label_region):
    def extract_blue_edges(cropped):
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blurred = cv2.GaussianBlur(blue_mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges, blue_mask

    def find_correct_line(edges, blue_mask, cropped):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:  # Near-horizontal line
                    return (x1, y1, x2, y2), angle
        return None, 0

    def apply_affine_deskew(cropped, line):
        x1, y1, x2, y2 = line
        src_pts = np.float32([[x1, y1], [x2, y2], [x1, y1 + 10]])
        dst_pts = np.float32([[x1, y1], [x2, y1], [x1, y1 + 10]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        h, w = cropped.shape[:2]
        aligned = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return aligned

    edges, blue_mask = extract_blue_edges(label_region)
    save_debug_image(edges, "blue_edges_debug")
    line, _ = find_correct_line(edges, blue_mask, label_region)
    if line:
        corrected = apply_affine_deskew(label_region, line)
        return corrected
    return label_region

# Function to preprocess labels
def preprocess_label(image, bbox, label_id=None):
    x, y, w, h = bbox
    x_min = int(x - w / 2)
    y_min = int(y - h / 2)
    x_max = int(x + w / 2)
    y_max = int(y + h / 2)

    label_region = image[y_min:y_max, x_min:x_max]
    save_debug_image(label_region, "original_label", label_id)

    corrected = perspective_correction(label_region)
    save_debug_image(corrected, "corrected_label", label_id)
    return corrected

# Function to extract text using Google Vision API
def extract_text(label_image, label_id=None):
    temp_image_path = f"{LOG_DIR}/temp_label_image_{label_id}.png"
    cv2.imwrite(temp_image_path, label_image)

    with open(temp_image_path, 'rb') as image_file:
        content = image_file.read()
        image = VisionImage(content=content)

    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description.strip()
    return ""

# Flask route
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']
    image_path = f"{LOG_DIR}/uploaded_image.png"
    image_file.save(image_path)

    image = cv2.imread(image_path)
    result = model.predict(image_path, confidence=40, overlap=30).json()
    detections = result.get("predictions", [])

    results = []
    for i, detection in enumerate(detections):
        bbox = detection['x'], detection['y'], detection['width'], detection['height']
        label_image = preprocess_label(image, bbox, f"label_{i+1}")
        text = extract_text(label_image, f"label_{i+1}")
        results.append({"label_id": f"label_{i+1}", "text": text})

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)