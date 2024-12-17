from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
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

@app.route('/')
def home():
    return "The API is live! Use the /process-image endpoint."

# Flask route
@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if raw image bytes are provided
    if not request.data:
        return jsonify({"error": "No image data provided."}), 400

    # Decode image from raw bytes
    try:
        np_arr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Save uploaded image for debugging
    uploaded_image_path = f"{LOG_DIR}/uploaded_image.png"
    cv2.imwrite(uploaded_image_path, image)

    # Predict using Roboflow model
    result = model.predict(uploaded_image_path, confidence=40, overlap=30).json()
    detections = result.get("predictions", [])

    # Process each detection
    results = []
    for i, detection in enumerate(detections):
        bbox = detection['x'], detection['y'], detection['width'], detection['height']
        label_image = preprocess_label(image, bbox, f"label_{i+1}")
        text = extract_text(label_image, f"label_{i+1}")
        results.append({"label_id": f"label_{i+1}", "text": text})

    return jsonify(results)

# Fetch environment variables or raise an error
DB_USER = os.getenv("DB_USER")
if not DB_USER:
    raise ValueError("Environment variable DB_USER is not set")

DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise ValueError("Environment variable DB_PASSWORD is not set")

DB_HOST = os.getenv("DB_HOST", "localhost")  # Fallback is fine for HOST
DB_PORT = os.getenv("DB_PORT", "5432")       # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")
if not DB_NAME:
    raise ValueError("Environment variable DB_NAME is not set")

# Configure the connection string
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Initialize the database
db = SQLAlchemy(app)

# Initialize Database
db = SQLAlchemy(app)

# Product Model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

# Employee Model
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

# Create Tables
with app.app_context():
    db.create_all()

# New Endpoint to Populate Database
@app.route("/populate", methods=["POST"])
def populate_database():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Extract products and employees
    products = data.get("products", [])
    employees = data.get("employees", [])

    try:
        # Add products to database
        for product in products:
            product_name = product.get("name")
            if product_name:
                db.session.add(Product(name=product_name))

        # Add employees to database
        for employee in employees:
            employee_name = employee.get("name")
            if employee_name:
                db.session.add(Employee(name=employee_name))

        db.session.commit()
        return jsonify({"message": "Database populated successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

# Endpoint to Retrieve Data for Testing
@app.route("/products", methods=["GET"])
def get_products():
    products = Product.query.all()
    return jsonify([{"id": p.id, "name": p.name} for p in products])

@app.route("/employees", methods=["GET"])
def get_employees():
    employees = Employee.query.all()
    return jsonify([{"id": e.id, "name": e.name} for e in employees])


if __name__ == "__main__":
    app.run(debug=True)
