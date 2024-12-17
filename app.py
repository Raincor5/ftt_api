from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from roboflow import Roboflow
from google.cloud import vision
from google.cloud.vision_v1.types import Image as VisionImage
import cv2
import numpy as np
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Fetch environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not all([DB_USER, DB_PASSWORD, DB_NAME, GOOGLE_APPLICATION_CREDENTIALS]):
    raise ValueError("Database credentials or Google credentials are not set properly.")

# Configure database connection
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize Roboflow and Google Cloud Vision client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT_NAME")
ROBOFLOW_VERSION_NUMBER = os.getenv("ROBOFLOW_VERSION_NUMBER")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_PROJECT_NAME)
model = project.version(ROBOFLOW_VERSION_NUMBER).model
vision_client = vision.ImageAnnotatorClient()

# Configure image saving directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Models
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

# Create Tables
with app.app_context():
    db.create_all()

# Utility Functions
def save_debug_image(image, step, label_id=None):
    filename = f"{LOG_DIR}/{step}_{label_id or str(uuid.uuid4())}.png"
    cv2.imwrite(filename, image)
    return filename

def extract_text(label_image, label_id=None):
    temp_image_path = f"{LOG_DIR}/temp_label_image_{label_id}.png"
    cv2.imwrite(temp_image_path, label_image)

    with open(temp_image_path, 'rb') as image_file:
        content = image_file.read()
        image = VisionImage(content=content)

    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""

# Routes
@app.route("/")
def home():
    return "The API is live! Use /process-image or /populate endpoints."

@app.route("/populate", methods=["POST"])
def populate_database():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    products = data.get("products", [])
    employees = data.get("employees", [])

    try:
        for product in products:
            product_name = product.get("name")
            if product_name:
                db.session.merge(Product(name=product_name))

        for employee in employees:
            employee_name = employee.get("name")
            if employee_name:
                db.session.merge(Employee(name=employee_name))

        db.session.commit()
        return jsonify({"message": "Database populated successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/products", methods=["GET"])
def get_products():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    products = Product.query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify([{"id": p.id, "name": p.name} for p in products.items])

@app.route("/employees", methods=["GET"])
def get_employees():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    employees = Employee.query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify([{"id": e.id, "name": e.name} for e in employees.items])

@app.route("/process-image", methods=["POST"])
def process_image():
    if not request.data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        np_arr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        uploaded_image_path = f"{LOG_DIR}/uploaded_image.png"
        cv2.imwrite(uploaded_image_path, image)

        # Roboflow detection
        result = model.predict(uploaded_image_path, confidence=40, overlap=30).json()
        detections = result.get("predictions", [])

        results = []
        for i, detection in enumerate(detections):
            bbox = detection['x'], detection['y'], detection['width'], detection['height']
            label_image = image[int(detection['y']):int(detection['y'] + detection['height']),
                                int(detection['x']):int(detection['x'] + detection['width'])]
            text = extract_text(label_image, f"label_{i+1}")
            results.append({"label_id": f"label_{i+1}", "text": text})

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
