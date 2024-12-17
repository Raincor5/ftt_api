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
import difflib
from datetime import datetime
import re

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
    return "The API is live! Use the /process-image or /populate endpoint."


@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if raw image bytes are provided
    if not request.data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        # Decode image from raw bytes
        np_arr = np.frombuffer(request.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Save uploaded image for debugging
        uploaded_image_path = f"{LOG_DIR}/uploaded_image.png"
        cv2.imwrite(uploaded_image_path, image)

        # Predict using Roboflow model
        result = model.predict(uploaded_image_path, confidence=40, overlap=30).json()
        detections = result.get("predictions", [])

        # Fetch product and employee names from the database
        products = [product.name for product in Product.query.all()]
        employees = [employee.name for employee in Employee.query.all()]

        # Process each detection
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['x'], detection['y'], detection['width'], detection['height']
            label_image = preprocess_label(image, bbox, f"label_{i+1}")
            raw_text = extract_text(label_image, f"label_{i+1}")

            # Skip empty raw_text
            if not raw_text:
                continue

            # Parse label text and validate against database
            parsed_data = parse_label_text(raw_text, products, employees)
            parsed_data['employee_name'] = find_closest_match(parsed_data['employee_name'], employees)
            parsed_data['product_name'] = find_closest_match(parsed_data['product_name'], products)

            # Check if the label data is fully constructed
            if (parsed_data['product_name'] and
                parsed_data['employee_name'] and
                parsed_data['dates']):
                results.append({
                    "label_id": f"label_{i+1}",
                    "raw_text": raw_text,
                    "parsed_data": parsed_data
                })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Extract products and employees
    products = data.get("products", [])
    employees = data.get("employees", [])

    try:
        # Add products to database
        for product in products:
            product_name = product.get("name")
            if product_name:
                db.session.merge(Product(name=product_name))

        # Add employees to database
        for employee in employees:
            employee_name = employee.get("name")
            if employee_name:
                db.session.merge(Employee(name=employee_name))

        db.session.commit()
        return jsonify({"message": "Database populated successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

# Endpoint to Retrieve Data for Testing
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

def find_closest_match(input_string, candidates):
    """Find the closest match for a string from a list of candidates."""
    closest_match = difflib.get_close_matches(input_string, candidates, n=1, cutoff=0.6)
    return closest_match[0] if closest_match else ""

def extract_dates_with_details(lines):
    """Extract dates while preserving time and additional info like EOD."""
    date_pattern = r"(\d{1,2}/\d{1,2}/\d{2}(?:\s+\d{2}:\d{2})?(?:\s+[A-Za-z.\s]+)?)"
    dates = []
    for line in lines:
        matches = re.findall(date_pattern, line)
        dates.extend([match.strip() for match in matches])
    return dates

def extract_batch_number(lines):
    """Extract batch number using flexible matching."""
    batch_pattern = r"(Batch No[:\s]*)([^\n]+)"
    for line in lines:
        match = re.search(batch_pattern, line, re.IGNORECASE)
        if match:
            batch_no = match.group(2).strip()
            # Ensure batch number has at least 2 characters
            if len(batch_no) < 2:
                return "N/A"
            return batch_no
    return "N/A"


def parse_label_text(text, product_names, employee_names):
    """Parse and extract data from label text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Extract product name and RTE type
    product_name = ""
    rte_status = ""
    for line in lines:
        closest_product = find_closest_match(line, product_names)
        if closest_product:
            product_name = closest_product
            rte_status = "RTE" if "RTE" in line else ""
            break

    # Identify label type
    label_type = "Defrosted" if any("DEFROST" in line.upper() for line in lines) else "Normal"

    # Extract employee name
    employee_name = ""
    for line in lines:
        closest_employee = find_closest_match(line, employee_names)
        if closest_employee:
            employee_name = closest_employee
            break

    # Extract dates
    extracted_dates = extract_dates_with_details(lines)

    # Extract batch number
    batch_no = extract_batch_number(lines)

    # Find day of the week for the expiry date
    expiry_date = None
    expiry_day = "N/A"
    if extracted_dates:
        try:
            expiry_date_str = extracted_dates[-1].split()[0]  # Take only the date part
            expiry_date = datetime.strptime(expiry_date_str, "%d/%m/%y")
            expiry_day = expiry_date.strftime("%A").upper()
        except ValueError:
            pass  # Ignore invalid date formats

    return {
        "product_name": product_name,
        "rte_status": rte_status,
        "employee_name": employee_name,
        "label_type": label_type,
        "dates": extracted_dates,
        "batch_no": batch_no,
        "expiry_day": expiry_day
    }



if __name__ == "__main__":
    app.run(debug=True)