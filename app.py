# Title : TraceXpert: Real-Time Facial Recognition and Case Management for Missing Person Recovery
# FY25-Q1-S1-01/20-Ver:1.0
# Author : Ramateja(Developer,Tester),Dinesh(Developer,Tester),Rohitha(UI Developer),Bhuvaneswari Devi(Model Training)
# FY25-Q1-S1-01/20-Version 1.0 – Enhancement: Implemented login and registration functionality to manage user access.
# FY25-Q1-S1-01/21-Version 1.0 – Bug Fix: Resolved an issue in converting login and registration data from Google Sheets to CSV files.
# FY25-Q1-S2-01/22-Version 2.0 – Enhancement: Developed user-specific dashboards and session-based logout features.
# FY25-Q1-S3-01/24-Version 2.0 – Enhancement: Enabled users to submit complaints with unique ticket IDs.
# FY25-Q1-S4-01/27-Version 2.0 – Enhancement: Added functionality for users to report suspected cases for matching.
# FY25-Q1-S5-01/28-Version 2.0 – Enhancement: Implemented ticket retrieval and search functionality.
# FY25-Q1-S6-01/30-Version 2.0 – Enhancement: Gathered and structured data for AI model training.
# FY25-Q1-S7-01/31-Version 2.0 – Enhancement: Trained an AI model for image feature extraction.
# FY25-Q1-S8-02/03-Version 2.0 – Enhancement: Implemented image feature extraction and similarity comparison.
# FY25-Q1-S9-02/05-Version 2.0 – Enhancement: Automated the movement of matched cases to the "Found" folder and added user notifications.
# FY25-Q1-S10-02/07-Version 3.0 – Enhancement: Improved ticket search and detailed lookup functionalities.
# FY25-Q1-S11-02/10-Version 3.0 – Enhancement: Implemented automatic timestamp generation for records.
# FY25-Q1-S12-02/11-Version 3.0 – Enhancement: Developed a system for sending and managing notifications.
# FY25-Q1-S13-02/12-Version 3.0 – Enhancement: Created a simulated backend response for integration with the frontend.
# FY25-Q1-S14-02/14-Version 4.0 – Enhancement: Updated CSS styling for the dashboard to improve UI.
# FY25-Q1-S15-02/17-Version 4.0 – Enhancement: Displayed user statistics & graphs retrieved from CSV files.
# FY25-Q1-S16-02/18-Version 4.0 – Enhancement: Upgraded the AI model to use ArchFace with a fallback to FaceNet for improved accuracy.
# FY25-Q1-S17-02/19-Version 4.0 – Enhancement: Admin dashboard displaying overall application usage statistics based on user data from CSV files.
# FY25-Q1-S18-02/20-Version 4.0 – Enhancement: Shows pending complaints, pending suspected cases, and found data extracted from respective CSV files.
# FY25-Q1-S19-02/21-Version 4.0 – Enhancement: Visualizes data through graphs for the admin's overview.
# FY25-Q1-S20-02/25-Version 4.0 – Enhancement: Introduced a help page with detailed instructions, acting as a user manual to assist users in navigating the application
# FY25-Q1-S21-02/26-Version 4.0 – Enhancement: OTP-based user authentication setup for registration.

from flask import Flask, render_template, request, redirect, flash, url_for, session ,jsonify
import csv
import os
import re
import time
from werkzeug.utils import secure_filename
import random
import numpy as np
import shutil
import traceback
import tensorflow as tf
import cv2
import pandas as pd
from keras_facenet import FaceNet
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime


# Initializes the Flask application with secret key and configurations.
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# File paths of register.csv and login.csv
REGISTER_CSV = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\register.csv"
LOGIN_CSV = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\login.csv"

if not os.path.exists(REGISTER_CSV):
    with open(REGISTER_CSV, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Email", "Name", "Phone", "Password"])

if not os.path.exists(LOGIN_CSV):
    with open(LOGIN_CSV, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Email", "Password"])

@app.route("/",methods=["GET"])
def landing():
    return render_template("home.html")

CSV_FILE = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\otpverification.csv"
HARD_CODED_OTP = "898989"  # This OTP will always work

# Ensure CSV file exists
if not os.path.exists(CSV_FILE):
    open(CSV_FILE, 'w').close()

def save_otp(phone, otp):
    """Save or update OTP for a phone number in the CSV file."""
    rows = []
    updated = False

    # Read existing data
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]

    # Update OTP if phone exists, else append
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            if row and row[0] == phone:
                writer.writerow([phone, otp])  # Update OTP
                updated = True
            else:
                writer.writerow(row)
        if not updated:
            writer.writerow([phone, otp])  # Append new OTP

def read_otp(phone):
    """Retrieve OTP for a given phone number."""
    if not os.path.exists(CSV_FILE):
        return None
    with open(CSV_FILE, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == phone:
                return row[1]  # Return OTP
    return None

@app.route('/otpverification', methods=['POST'])
def otp_verification():
    """Handles OTP request and verification."""
    data = request.get_json()
    phone_number = data.get("phone")
    entered_otp = data.get("otp")
    
    if not phone_number:
        return jsonify({"status": "Phone number required"}), 400

    if entered_otp:  # Verify OTP
        stored_otp = read_otp(phone_number)
        if stored_otp and entered_otp == stored_otp or entered_otp == HARD_CODED_OTP:
            return jsonify({"status": "OTP Verified"}), 200
        return jsonify({"status": "Invalid OTP"}), 401

    # Generate a new OTP and save it
    otp = str(random.randint(100000, 999999))
    save_otp(phone_number, otp)

    print(f"OTP for {phone_number}: {otp}")  # Simulate sending OTP
    return jsonify({"status": "OTP Sent"}), 200


# Handles user authentication by validating credentials against login CSV data.
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate email format
        if "@gmail.com" not in email:
            flash("Invalid email format", "error")
            return redirect(url_for("login"))

        with open(LOGIN_CSV, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == email:
                    if row[1] == password:
                        session["email"] = email
                        return redirect(url_for("dashboard"))
                    else:
                        flash("Incorrect password", "error")
                        return redirect(url_for("login"))
            flash("Email not found", "error")
            return redirect(url_for("login"))
    return render_template("login.html")


# Manages user registration by storing validated details in register and login CSVs.
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        name = request.form.get("name")
        phone = request.form.get("phone")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        is_verified = request.form.get("is_verified", "false")

        # Validate input fields
        if "@gmail.com" not in email:
            flash("Invalid email format", "error")
            return redirect(url_for("register"))
        if len(phone) != 10 or not phone.isdigit():
            flash("Invalid phone number", "error")
            return redirect(url_for("register"))
        if not is_valid_password(password):
            flash("Password must be at least 8 characters: 1 upper, 1 lower, 1 digit, 1 special symbol.", "error")
            return redirect(url_for("register"))
        # Prevent registration if phone is not verified
        if is_verified != "true":
            flash("Please verify your phone number before registering.", "error")
            return redirect(url_for("register"))

        existing_emails = []
        existing_phones = []
        with open(REGISTER_CSV, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    existing_emails.append(row[0]) 
                    existing_phones.append(row[2])

        if email in existing_emails:
            flash("Email already exists", "error")
            return redirect(url_for("register"))
        if phone in existing_phones:
            flash("Phone number already exists", "error")
            return redirect(url_for("register"))
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        #Stores the data in register CSV file
        with open(REGISTER_CSV, "a", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([email, name, phone, password])

        #Stores the email and password in login CSV file
        with open(LOGIN_CSV, "a", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([email, password])

        flash("Registration successful", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Validating password
def is_valid_password(password):
    # Regex for password validation
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    return re.match(pattern, password)


# Displays dashboard (admin/regular) after checking session authentication.
@app.route("/dashboard")
def dashboard():
    # Retrieve current user's details from session 
    email = session.get("email")  
    if not email:
        flash("Please log in to access the dashboard", "error")
        return redirect(url_for("login"))

    user_details = None
    with open(REGISTER_CSV, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == email:  # Match the session email
                user_details = {
                    "Email": row[0],
                    "Name": row[1],
                    "Phone": row[2]
                }
                break

    if not user_details:
        flash("User not found", "error")
        return redirect(url_for("login"))

    return  render_template("admindash.html", user=user_details) if(email=="admin@gmail.com") else  render_template("dashboard.html", user=user_details)


# File paths of Complaint CSV and Complaint_data folder
CSV_FILE1 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\complaints.csv"
UPLOAD_FOLDER1 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\static\complaint_data"
os.makedirs(UPLOAD_FOLDER1, exist_ok=True)

# File paths of Suspected CSV and Suspected_data folder
CSV_FILE2 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\suspected.csv"
UPLOAD_FOLDER2 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\static\suspected_data"
os.makedirs(UPLOAD_FOLDER2, exist_ok=True)


# Processes lost child reports: Checks for matches in suspected entries if not stores data in complaints CSV file.
@app.route("/submit_complaint", methods=['POST'])
def submit_complaint():
    try:
        picture = request.files.get('picture')
        name = request.form.get('name', '').strip()
        address = request.form.get('address', '').strip()
        phone = request.form.get('phone', '').strip()
        description = request.form.get('description', '').strip()
        email = session.get("email")
        
        if not email:
            return jsonify({"error": "Unauthorized access"}), 401
        date_part, time_part = get_timestamp_and_split()

        complaint_id = generate_unique_id("MCR", [CSV_FILE1, CSV_FILE2, CSV_FILE3])

        if picture and picture.filename:
            img_filename = secure_filename(picture.filename)
            img_path = os.path.join(UPLOAD_FOLDER1, img_filename)
            picture.save(img_path)

            match_found, matched_details = is_image_in_suspected_folder(img_path)
            if matched_details == "invalid":
                result=simulate_backend_response(None, None, None, None, None, None)
                return jsonify(result)
            if match_found:

                suspected_id = matched_details[0]
                
                process_result = process_found_case1(complaint_id, img_filename, email, name, address, phone, description, suspected_id,date_part, time_part)
                if process_result[1] == True:
                    # Correct indices based on process_found_case2's return value
                    img_filename = process_result[0][0]  # Complaint image filename
                    name = process_result[0][1]          # Complaint name
                    address = process_result[0][2]       # Complaint address
                    phone = process_result[0][3]         # Complaint phone
                    ticket_id = process_result[0][4]     # Complaint ticket ID (MCR-...)
                    work = process_result[0][5]          # "completed"
                    
                    result = simulate_backend_response(img_filename, name, address, phone, ticket_id, work)
                    return jsonify(result)
                else:
                    return jsonify({"error": process_result}), 500
            
            with open(CSV_FILE1, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([complaint_id, img_filename, email, name, address, phone, description, "In Progress",date_part, time_part])
        work = "In Progress"
        result = simulate_backend_response(img_filename, name, address, phone, complaint_id, work)
        return jsonify(result)
    except Exception as e:
        print("Error in submit_complaint:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Processes found child reports: Checks for matches in complaint entries if not stores data in suspected CSV file.
@app.route("/submit_suspected", methods=['POST'])
def submit_suspected():
    try:
        picture = request.files.get('picture')
        name = request.form.get('name', '').strip()
        address = request.form.get('location', '').strip()
        phone = request.form.get('phone', '').strip()
        nameofchild = request.form.get('nameofchild', '').strip()
        email = session.get("email")
        if not email:
            return jsonify({"error": "Unauthorized access"}), 401
        date_part, time_part = get_timestamp_and_split()
        # Generate a unique suspected child ID
        suspected_id = generate_unique_id("SCF", [CSV_FILE1, CSV_FILE2])

        if picture and picture.filename:
            img_filename = secure_filename(picture.filename)
            img_path = os.path.join(UPLOAD_FOLDER2, img_filename)
            picture.save(img_path)

            match_found, matched_details = is_image_in_complaint_folder(img_path)
            if matched_details == "invalid":
                result=simulate_backend_response(None, None, None, None, None, None)
                return jsonify(result)
            
            if match_found:
                complaint_id = matched_details[0]  # Extract the complaint_id from matched details
                process_result = process_found_case2(suspected_id, img_filename, email, name, address, nameofchild, phone, complaint_id,date_part, time_part)
                if process_result[1] == True:
                    # Correct indices based on process_found_case1's return value
                    img_filename = process_result[0][0]  # Suspected image filename
                    name = process_result[0][1]          # Suspected name
                    address = process_result[0][2]       # Suspected address
                    phone = process_result[0][3]         # Suspected phone
                    ticket_id = process_result[0][4]     # Suspected ticket ID (SCF-...)
                    work = process_result[0][5]          # "completed"
                    
                    result = simulate_backend_response(img_filename, name, address, phone, ticket_id, work)
                    return jsonify(result)
                else:
                    return jsonify({"error": process_result}), 500
            
            with open(CSV_FILE2, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([suspected_id, img_filename, email, name, address, nameofchild, phone, "In Progress",date_part, time_part])
        work = "In Progress"
        result = simulate_backend_response(img_filename, name, address, phone, suspected_id, work)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Generates current date and time strings for timestamping records.
def get_timestamp_and_split():
    # Get the current timestamp
    timestamp = datetime.now()
    
    # Format date and time separately
    date_str = timestamp.strftime("%Y-%m-%d") 
    time_str = timestamp.strftime("%H:%M:%S")  
    
    return date_str, time_str


# Creates unique ticket IDs (MCR-XXXXXXXX for complaints, SCF-XXXXXXXX for suspects) avoiding duplicates across CSVs.
def generate_unique_id(prefix, csv_files):
    while True:
        random_id = f"{prefix}-{random.randint(10000000, 99999999)}"

        # Ensure all CSV files exist before checking for duplicates
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                with open(csv_file, "w", encoding="utf-8", newline='') as file:
                    writer = csv.writer(file)
                    if "found.csv" in csv_file:
                        writer.writerow(["complaint_id", "img_filename", "email", "name", "address", "phone", "description", "status", "suspected_id"])

        # Check uniqueness in all CSV files
        found = False
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as file:
                if random_id in file.read():
                    found = True
                    break  # ID already exists, generate again

        if not found:
            return random_id


# 



import cv2
import numpy as np
import insightface
from insightface.model_zoo import ArcFaceONNX
import mediapipe as mp
import onnxruntime as ort
import os
import csv
from scipy.spatial.distance import cosine

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Buffalo and w600k_r50 ONNX model
def load_models():
    print("[INFO] Loading models...")

    buffalo_model = insightface.app.FaceAnalysis(name='buffalo_l')
    buffalo_model.prepare(ctx_id=-1)  # Use CPU

    arcface_onnx_path = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\model\w600k_r50.onnx"
    arcface_session = ort.InferenceSession(arcface_onnx_path)

    return buffalo_model, arcface_session

buffalo, arcface_session = load_models()

# Detect if a face exists using MediaPipe
def detect_face_mediapipe(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    return results.detections is not None and len(results.detections) > 0

# Extract face embedding using Buffalo_l
def extract_face_embedding_buffalo(image_path):
    image = cv2.imread(image_path)
    faces = buffalo.get(image)

    if len(faces) == 0:
        return None  # No face detected by Buffalo
    return faces[0]['embedding']

# Extract face embedding using w600k_r50 ONNX
def extract_face_embedding_w600k(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (112, 112))
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0).astype(np.float32)

    image = (image - 127.5) / 127.5  # Normalize

    input_name = arcface_session.get_inputs()[0].name
    outputs = arcface_session.run(None, {input_name: image})

    return outputs[0][0]

# Extract embeddings with failover strategy
def extract_face_embedding(image_path):
    # Step 1: Try Buffalo first
    embedding = extract_face_embedding_buffalo(image_path)
    if embedding is not None:
        print("[INFO] Face embedding extracted using Buffalo_l.")
        return embedding  # ✅ Return immediately

    # Step 2: If Buffalo fails, check with MediaPipe and use w600k_r50 if a face exists
    if detect_face_mediapipe(image_path):
        print("[INFO] Face detected using MediaPipe. Extracting embedding with w600k_r50.")
        embedding = extract_face_embedding_w600k(image_path)

        if embedding is not None:
            print("[INFO] Face embedding extracted using w600k_r50.")
            return embedding  # ✅ Return immediately

    print("[OOPS!] No face embedding extracted.")
    return None  # ❌ Fix the incorrect return value


# Compare embeddings using cosine similarity
def compare_faces(embedding1, embedding2, threshold=0.60):
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity >= threshold, similarity

# Check if an image exists in the Complaint Folder
def is_image_in_complaint_folder(suspected_image_path):
    suspected_embedding = extract_face_embedding(suspected_image_path)

    if suspected_embedding is None:
        return False, "invalid"

    with open(CSV_FILE1, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            complaint_image = row[1]
            complaint_image_path = os.path.join(UPLOAD_FOLDER1, complaint_image)

            if os.path.isfile(complaint_image_path):
                complaint_embedding = extract_face_embedding(complaint_image_path)

                if complaint_embedding is not None:
                    is_same_person, similarity = compare_faces(suspected_embedding, complaint_embedding)
                    if is_same_person:
                        print("Cosine Similarity:", similarity)
                        return True, row

    return False, None

# Check if an image exists in the Suspected Folder
def is_image_in_suspected_folder(complaint_image_path):
    complaint_embedding = extract_face_embedding(complaint_image_path)

    if complaint_embedding is None:
        return False, "invalid"

    with open(CSV_FILE2, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            suspected_image = row[1]
            suspected_image_path = os.path.join(UPLOAD_FOLDER2, suspected_image)

            if os.path.isfile(suspected_image_path):
                suspected_embedding = extract_face_embedding(suspected_image_path)

                if suspected_embedding is not None:
                    is_same_person, similarity = compare_faces(complaint_embedding, suspected_embedding)
                    if is_same_person:
                        print("Cosine Similarity:", similarity)
                        return True, row

    return False, None








# File paths of Found CSV and Found_data folder
UPLOAD_FOLDER3 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\static\found_data"
CSV_FILE3 = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\found.csv"
os.makedirs(UPLOAD_FOLDER3, exist_ok=True)


# Moves matched complaint data to "found" folder and Found CSV, updates Complaints CSV, and sends notifications when a suspected match is found.
def process_found_case1(complaint_id, complaint_img_filename, complaint_email, complaint_name, complaint_address, complaint_phone, complaint_description, suspected_id,date_part, time_part):
    try:
        # Move the complaint image to Found Folder
        src_complaint_path = os.path.join(UPLOAD_FOLDER1, complaint_img_filename)
        dst_complaint_path = os.path.join(UPLOAD_FOLDER3, complaint_img_filename)
        if os.path.exists(src_complaint_path):
            shutil.move(src_complaint_path, dst_complaint_path)
        
        # Store the found complaint details in found.csv
        with open(CSV_FILE3, "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([complaint_id, complaint_img_filename, complaint_email, complaint_name, complaint_address, complaint_phone, complaint_description, "Completed", suspected_id,date_part, time_part])

        # Find the suspected case in suspected.csv
        suspected_data = []
        suspected_entry = None
        with open(CSV_FILE2, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == suspected_id:
                    suspected_entry = row  # Store the found suspected row
                else:
                    suspected_data.append(row)
        
        if suspected_entry:
            suspected_img_filename = suspected_entry[1]
            suspected_email = suspected_entry[2]
            suspected_name = suspected_entry[3]
            suspected_address = suspected_entry[4]
            suspected_child_name = suspected_entry[5]
            suspected_phone = suspected_entry[6]
            suspected_date_part = suspected_entry[8]
            suspected_time_part = suspected_entry[9]
            ticket_id = suspected_id
            ticket_id2=complaint_id
            work="Completed"
            suspected_details=[suspected_img_filename,suspected_name,suspected_address,suspected_phone,ticket_id,work]
            send_notification(suspected_email,ticket_id,ticket_id2)
           
            # Move the suspected image to Found Folder
            src_suspected_path = os.path.join(UPLOAD_FOLDER2, suspected_img_filename)
            dst_suspected_path = os.path.join(UPLOAD_FOLDER3, suspected_img_filename)
            if os.path.exists(src_suspected_path):
                shutil.move(src_suspected_path, dst_suspected_path)
            
            # Store the found suspected details in found.csv
            with open(CSV_FILE3, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([suspected_id, suspected_img_filename, suspected_email, suspected_name, suspected_address, suspected_child_name, suspected_phone, "Completed", complaint_id,suspected_date_part,suspected_time_part])
        
        # Overwrite suspected.csv without the found entry
        with open(CSV_FILE2, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(suspected_data)
        
        return [suspected_details,True]
    except Exception as e:
        print("Error in process_found_case1:", traceback.format_exc())
        return str(e)


# Moves matched suspected data to "found" folder and Found CSV, updates Suspected CSV, and sends notifications when a complaint match is found.
def process_found_case2(suspected_id, img_filename, email, name, address, nameofchild, phone, complaint_id,date_part, time_part):
    try:
        # Move the suspected image to Found Folder
        src_suspected_path = os.path.join(UPLOAD_FOLDER2, img_filename)
        dst_suspected_path = os.path.join(UPLOAD_FOLDER3, img_filename)
        if os.path.exists(src_suspected_path):
            shutil.move(src_suspected_path, dst_suspected_path)
        
        # Store the found suspected details in found.csv
        with open(CSV_FILE3, "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([suspected_id, img_filename, email, name, address, nameofchild, phone, "Completed", complaint_id,date_part, time_part])
        
        # Find and update the complaint data
        complaint_data = []
        complaint_entry = None
        with open(CSV_FILE1, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == complaint_id:
                    complaint_entry = row  # Store the found complaint row
                else:
                    complaint_data.append(row)
        
        if complaint_entry:
            complaint_img_filename = complaint_entry[1]
            complaint_email = complaint_entry[2]
            complaint_name = complaint_entry[3]
            complaint_address = complaint_entry[4]
            complaint_phone = complaint_entry[5]
            complaint_description = complaint_entry[6]
            complaint_date_part = complaint_entry[8]
            complaint_time_part = complaint_entry[9]
            ticket_id = complaint_id
            work="Completed"
            complaint_details=[complaint_img_filename,complaint_name,complaint_address,complaint_phone,ticket_id,work]
            ticket_id2=suspected_id
            send_notification(complaint_email,ticket_id,ticket_id2)
            # Move the complaint image to Found Folder
            src_complaint_path = os.path.join(UPLOAD_FOLDER1, complaint_img_filename)
            dst_complaint_path = os.path.join(UPLOAD_FOLDER3, complaint_img_filename)
            if os.path.exists(src_complaint_path):
                shutil.move(src_complaint_path, dst_complaint_path)
            
            # Store the found complaint details in found.csv
            with open(CSV_FILE3, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([complaint_id, complaint_img_filename, complaint_email, complaint_name, complaint_address, complaint_phone, complaint_description, "Completed", suspected_id,complaint_date_part,complaint_time_part])
        
        # Overwrite complaint.csv without the found entry
        with open(CSV_FILE1, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(complaint_data)
        
        return [complaint_details,True]
    except Exception as e:
        print("Error in process_found_case2:", traceback.format_exc())
        return str(e)


# Returns the response to the fronted-end modal animation for found/notfound cases.
def simulate_backend_response(matched_image, name, address, phone, ticket_id, work):
    # Simulate some processing time
    time.sleep(10)
    if work==None:
        details={
            "status":"INVALID"
        }
        return details
    
    if work.lower() == "completed":

        # Directly return "MATCH FOUND" if work is "completed"
        details = {
            "status": "MATCH FOUND",
            "matched_image": matched_image,
            "details": {
                "name": name,
                "address": address,
                "phone": phone,
                "ticket_id": ticket_id
            }
        }
    else:
        # Return "NOT FOUND" if work is "in progress"
        details = {
            "status": "NOT FOUND",
            "ticket_id": ticket_id
        }
    
    return details


# Fetches all tickets (complaints/suspected/found) associated with the logged-in user from CSVs.
@app.route('/get_tickets', methods=['GET'])
def get_tickets():
    email = session.get("email")
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    user_tickets = []

    # Function to fetch ticket details from a CSV file
    def fetch_tickets_from_csv(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[2] == email:  # Email is in the 3rd column
                    ticket_id = row[0]  # Ticket ID is in the 1st column
                    status = row[7].strip()  # Status is in the 8th column

                    user_tickets.append({
                        "ticket_id": ticket_id,
                        "status": status
                    })

    # Read from all three CSV files
    fetch_tickets_from_csv(CSV_FILE1)
    fetch_tickets_from_csv(CSV_FILE2)
    fetch_tickets_from_csv(CSV_FILE3)

    return jsonify({"tickets": user_tickets})


# Searches for a ticket ID across all CSVs and returns matched records with image paths.
@app.route('/search_ticket', methods=['POST'])
def search_ticket():
    ticket_id = request.json.get('ticket_id')
    if not ticket_id:
        return jsonify({"error": "Ticket ID is required"}), 400

    ticket_id = ticket_id.strip().lower()
    # Search in found.csv
    found_details = search_in_csv(CSV_FILE3, ticket_id)
    if found_details:
        # Fetch complaint and suspected details from found.csv
        complaint_id = found_details[0].strip().lower()
        suspected_id = found_details[8].strip().lower()
        if complaint_id.startswith("scf"):
            complaint_id,suspected_id=suspected_id,complaint_id

        # Fetch complaint details
        complaint_details = search_in_csv(CSV_FILE3, complaint_id)
        # Fetch suspected details
        if complaint_details:
            complaint_details.append("static/found_data")
        suspected_details = search_in_csv(CSV_FILE3, suspected_id)
        if suspected_details:
            suspected_details.append("static/found_data")

        if complaint_details and suspected_details:
            return jsonify({
                "status": "match_found",
                "complaint_details": complaint_details,
                "suspected_details": suspected_details
            })

    # Search in complaints.csv
    complaint_details = search_in_csv(CSV_FILE1, ticket_id)
    if complaint_details:
        complaint_details.append("static/complaint_data")
        return jsonify({
            "status": "complaint_found",
            "complaint_details": complaint_details,
            "suspected_details": None
        })

    # Search in suspected.csv
    suspected_details = search_in_csv(CSV_FILE2, ticket_id)
    if suspected_details:
        suspected_details.append("static/suspected_data")  # Add image path
        return jsonify({
            "status": "suspected_found",
            "complaint_details": None,
            "suspected_details": suspected_details
        })

    return jsonify({"error": "Ticket ID not found"}), 404


# Helper function to search for a ticket ID in a specified CSV file.
def search_in_csv(file_path, ticket_id):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].strip().lower() == ticket_id:
                    return row
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None


# Aggregates resolved cases from found.csv for admin dashboard display.
@app.route('/get_found_data', methods=['GET'])
def get_found_data():
    try:
        found_data = []
        mcr_records = []
        scf_records = {}

        # Read the CSV file and separate MCR and SCF records
        with open(CSV_FILE3, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0].startswith("MCR"):  # Complaint records
                    mcr_records.append({
                        "complaint_id": row[0],  
                        "complaint_image": row[1],  
                        "complaint_email": row[2],  
                        "name_of_child": row[3],  
                        "address": row[4],          
                        "phone": row[5],  
                        "suspected_id": row[8], 
                        "status": row[7],
                        "date":row[9]  
                    })
                elif row[0].startswith("SCF"):  # Suspected records
                    scf_records[row[0]] = {
                        "suspected_image": row[1],  
                        "founder_email": row[2],  
                        "founder_name": row[3],  
                        "location_found": row[4],  
                        "founder_phone": row[6], 
                        "found_date" : row[9]
                    }

        # Combine MCR and SCF records
        for mcr in mcr_records:
            suspected_id = mcr["suspected_id"]
            if suspected_id in scf_records:
                scf = scf_records[suspected_id]
                found_data.append({
                    "complaint_id": mcr["complaint_id"],
                    "complaint_image": mcr["complaint_image"],
                    "complaint_email": mcr["complaint_email"],
                    "name_of_child": mcr["name_of_child"],
                    "address": mcr["address"],
                    "phone": mcr["phone"],
                    "date":mcr["date"],
                    "suspected_image": scf["suspected_image"],
                    "founder_email": scf["founder_email"],
                    "founder_name": scf["founder_name"],
                    "location_found": scf["location_found"],
                    "founder_phone": scf["founder_phone"],
                    "suspected_id": suspected_id,
                    "found_date":scf["found_date"],
                    "status": mcr["status"]
                })
            # else:
            #     # If no matching SCF record, include only MCR data
            #     found_data.append({
            #         "complaint_id": mcr["complaint_id"],
            #         "complaint_image": mcr["complaint_image"],
            #         "complaint_email": mcr["complaint_email"],
            #         "name_of_child": mcr["name_of_child"],
            #         "address": mcr["address"],
            #         "phone": mcr["phone"],
            #         "suspected_image": "",
            #         "founder_email": "",
            #         "founder_name": "",
            #         "location_found": "",
            #         "founder_phone": "",
            #         "suspected_id": suspected_id,
            #         "status": mcr["status"]
            #     })

        return jsonify(found_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Returns all "In Progress" complaints from complaints.csv for admin review.
@app.route('/get_complaints_pending', methods=['GET'])
def get_complaints_pending():
    try:
        complaints_data = []
        with open(CSV_FILE1, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[7] == "In Progress":
                    complaints_data.append({
                        "complaint_id": row[0],
                        "image": row[1],
                        "email": row[2],
                        "name_of_child": row[3],
                        "address": row[4],
                        "phone": row[5],
                        "description": row[6],
                        "date": row[8],
                        "status": row[7]
                    })
        # print(complaints_data)
        return jsonify(complaints_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Returns all "In Progress" suspected entries from suspected.csv for admin review.
@app.route('/get_suspected_pending', methods=['GET'])
def get_suspected_pending():
    try:
        suspected_data = []
        with open(CSV_FILE2, "r") as file:
            reader = csv.reader(file)
            # print(*reader)
            for row in reader:
                # return row
                if row[7] == "In Progress":
                    suspected_data.append({
                        "suspected_id": row[0],
                        "image": row[1],
                        "email": row[2],
                        "founder_name": row[3],
                        "location_found": row[4],
                        "name_of_child": row[5],
                        "phone": row[6],
                        "found_date": row[8],
                        "status": row[7]
                    })
        # print(suspected_data)
        return jsonify(suspected_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# File paths of Notification CSV file.
NOTIFICATIONS_FILE = r"C:\Users\Tejap\OneDrive\Desktop\Ram\TraceMaster\notification.csv"

# Ensure the file exists
if not os.path.exists(NOTIFICATIONS_FILE):
    with open(NOTIFICATIONS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["email", "message", "status"])  # Column headers


# Writes match notifications to notification.csv when cases are resolved.
def send_notification(email,ticket_id,ticket_id2):
    
    if ticket_id.startswith("MCR"):
        message = f"For your Ticket_Id {ticket_id}, a Match is Found with {ticket_id2}"
    if ticket_id.startswith("SCF"):
        message = f"For your Ticket_Id {ticket_id}, a Match is Found with {ticket_id2}"

    if not email or not message:
        return jsonify({"error": "Missing email or message"}), 400

    # Append notification to CSV
    with open(NOTIFICATIONS_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([email, message, "unread"])

    return jsonify({"message": "Notification sent successfully"}), 200


# Retrieves unread/read notifications for the logged-in user from notification CSV.
@app.route("/get_notifications", methods=["GET"])
def get_notifications():
    email = session.get("email")
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    notifications = []

    # Check if file exists and has data
    if os.path.exists(NOTIFICATIONS_FILE) and os.path.getsize(NOTIFICATIONS_FILE) > 0:
        with open(NOTIFICATIONS_FILE, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            for row in reader:
                if row and row[0] == email:
                    notifications.append({"message": row[1], "status": row[2]})

    return jsonify(notifications)


#Removes all notifications for the current user from notification.csv.
@app.route("/clear_notifications", methods=["POST"])
def clear_notifications():
    email = session.get("email")
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    if not os.path.exists(NOTIFICATIONS_FILE) or os.path.getsize(NOTIFICATIONS_FILE) == 0:
        return jsonify({"message": "No notifications to clear"})

    notifications = []
    
    with open(NOTIFICATIONS_FILE, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)  # Skip header safely

        for row in reader:
            if row and row[0] != email:
                notifications.append(row)

    # Write back the remaining notifications
    with open(NOTIFICATIONS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)  # Re-add the header
        writer.writerows(notifications)

    return jsonify({"message": "Notifications cleared successfully"})


# Collects feedback from users and store in feedback.csv
# @app.route("/submit_feedback", methods=["POST"])
# def submit_feedback():
#     email = session.get("email")
#     if not email:
#         return jsonify({"error": "Unauthorized"}), 401

#     feedback = request.form.get("feedback")
#     if not feedback:
#         return jsonify({"error": "Feedback is required"}), 400

#     with open("feedback.csv", "a", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([email, feedback])

#     return jsonify({"message": "Feedback submitted successfully"}), 200


# Calculates the Admin and User Statistics
@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    try:
        email = session.get("email")  # Replace with actual session email
        if not email:
            return jsonify({"error": "Unauthorized"}), 401

        # Initialize counters
        mcr_pending = 0
        mcr_pending_admin=0
        scf_pending = 0
        scf_pending_admin = 0
        mcr_found = 0
        mcr_found_admin = 0
        scf_found = 0
        scf_found_admin = 0
        total_found_admin=0
        total_users = 0

        with open(REGISTER_CSV, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    total_users += 1 # Subtract header row


        # Count MCR and SCF cases in complaint.csv and suspected.csv
        with open(CSV_FILE1, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0].startswith("MCR"):
                    mcr_pending_admin+=1
                if row and row[2] == email:  # Email is at index 2
                    mcr_pending += 1


        
        with open(CSV_FILE2, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                scf_pending_admin+=1
                if row and row[2] == email:  # Email is at index 2
                    scf_pending += 1

        # Count MCR and SCF cases in found.csv
        with open(CSV_FILE3, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or len(row) < 3:  # Ensure row is not empty and has at least 3 columns
                    continue  # Skip to the next row
                total_found_admin+=1
                if row and row[2] == email:  # Email is at index 2
                    if row[0].startswith("MCR"):
                        mcr_found += 1
                    elif row[0].startswith("SCF"):
                        scf_found += 1
                if row[0].startswith("MCR"):
                        mcr_found_admin+=1
                elif row[0].startswith("SCF"):
                        scf_found_admin+=1
            



        # Prepare data for graphs
        statistics = {
            "total_cases": mcr_pending + scf_pending + mcr_found + scf_found,
            "mcr_total": mcr_pending + mcr_found ,
            "scf_total": scf_pending + scf_found ,
            "mcr_found": mcr_found,
            "scf_found": scf_found,
            "mcr_pending": mcr_pending,
            "scf_pending": scf_pending,
            "total_users": total_users,
            "total_found_admin": total_found_admin,
            "mcr_pending_admin":mcr_pending_admin,
            "scf_pending_admin":scf_pending_admin,
            "mcr_found_admin":mcr_found_admin,
            "scf_found_admin":scf_found_admin,
            "admin_total_cases" : total_found_admin + mcr_pending_admin +scf_pending_admin ,
            "total_pending" : mcr_pending_admin +scf_pending_admin 
            

        }


        print("Statistics:", statistics)  # Debugging: Print statistics
        return jsonify(statistics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# To display the Admin graphs
@app.route('/get_all_graphs')
def get_all_graphs():
    all_data, complaint_df, suspected_df, found_df = load_data()
    
    # Donut Chart Data
    mcr_pending = len(complaint_df[complaint_df['Type'] == 'MCR'])
    mcr_found = len(found_df[found_df['Type'] == 'MCR'])
    scf_pending = len(suspected_df[suspected_df['Type'] == 'SCF'])
    scf_found = len(found_df[found_df['Type'] == 'SCF'])

    # Stacked Bar Data
    stacked_data = all_data.groupby(['Date', 'Type', 'Status']).size().reset_index(name='Count')

    # Area Chart Data
    tickets_over_time = all_data.groupby('Date').size().reset_index(name='Count')

    # Convert Date and Time columns properly
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data['Time'] = pd.to_datetime(all_data['Time'], format="%H:%M:%S", errors='coerce').dt.time

    # Heatmap Data: 30-minute intervals
    time_slots = pd.date_range("00:00", "23:30", freq="30min").strftime('%H:%M').tolist()

    # **Create 'DateTime' and Assign to 30-Minute Slots**
    all_data["DateTime"] = pd.to_datetime(all_data["Date"].astype(str) + " " + all_data["Time"].astype(str), errors='coerce')
    all_data = all_data.dropna(subset=["DateTime"])  # Remove invalid datetime rows
    all_data["TimeSlot"] = all_data["DateTime"].dt.floor("30min").dt.strftime('%H:%M')

    # **Count Tickets per 30-Minute Slot**
    traffic_data = all_data.groupby("TimeSlot").size().reindex(time_slots, fill_value=0).reset_index()
    traffic_data.columns = ["TimeSlot", "Tickets"]

    # Scatter Plot: Case Resolution Time
    linked_cases = []
    found_df["DateTime"] = pd.to_datetime(found_df["Date"].astype(str) + " " + found_df["Time"].astype(str), errors='coerce')
    
    for _, row in found_df.iterrows():
        ticket_id = row["TicketID"]
        linked_id = row["SuspectedID"]
        linked_row = found_df[found_df["TicketID"] == linked_id]

        if not linked_row.empty:
            linked_row = linked_row.iloc[0]
            time1, time2 = row["DateTime"], linked_row["DateTime"]
            
            # Determine latest and earliest ticket
            if time1 > time2:
                latest_ticket, earliest_ticket = row, linked_row
            else:
                latest_ticket, earliest_ticket = linked_row, row

            # Compute time difference
            time_diff = abs(time1 - time2)
            days, hours, minutes = time_diff.days, time_diff.seconds // 3600, (time_diff.seconds % 3600) // 60
            time_diff_str = f"{days}d {hours}h {minutes}m"
            
            linked_cases.append({
                "latest_id": latest_ticket["TicketID"],
                "earliest_id": earliest_ticket["TicketID"],
                "latest_time": latest_ticket["DateTime"].strftime('%H:%M:%S'),
                "earliest_time": earliest_ticket["DateTime"].strftime('%H:%M:%S'),
                "latest_date": latest_ticket["DateTime"].strftime('%Y-%m-%d'),
                "earliest_date": earliest_ticket["DateTime"].strftime('%Y-%m-%d'),
                "time_diff": time_diff_str
            })

    scatter_data = {
        "latest_ids": [case["latest_id"] for case in linked_cases],
        "earliest_ids": [case["earliest_id"] for case in linked_cases],
        "latest_times": [case["latest_time"] for case in linked_cases],
        "earliest_times": [case["earliest_time"] for case in linked_cases],
        "latest_dates": [case["latest_date"] for case in linked_cases],
        "earliest_dates": [case["earliest_date"] for case in linked_cases],
        "time_diffs": [case["time_diff"] for case in linked_cases]
    }

    # Prepare all graph data
    data = {
        "donut": {
            "names": ["MCR Pending", "MCR Found", "SCF Pending", "SCF Found"],
            "values": [mcr_pending, mcr_found, scf_pending, scf_found]
        },
        "pie": {
            "names": ["Pending Cases", "Found Cases"],
            "values": [mcr_pending + scf_pending, len(found_df)]
        },
        "pending_found_bar": {
            "categories": ["MCR Pending", "MCR Found", "Total MCR", "SCF Pending", "SCF Found", "Total SCF"],
            "values": [mcr_pending, mcr_found, mcr_pending + mcr_found, scf_pending, scf_found, scf_pending + scf_found]
        },
        "stacked_bar": {
            "dates": stacked_data["Date"].astype(str).tolist(),
            "counts": stacked_data["Count"].astype(int).tolist(),
            "statuses": stacked_data["Status"].tolist(),
            "types": stacked_data["Type"].tolist()
        },
        "area": {
            "dates": tickets_over_time["Date"].astype(str).tolist(),
            "counts": tickets_over_time["Count"].tolist()
        },
        "heatmap":{
        "x_labels": time_slots,  # Fixed 30-minute time intervals
        "y_labels": ["Ticket Volume"],  # Single row for heatmap
        "z_values": [traffic_data["Tickets"].tolist()]  # Ticket counts in list format
    },
        "scatter": scatter_data,
    }

    return jsonify(data)


# Loads the graph data from CSV files.
def load_data():
    # Load CSV files
    complaint_df = pd.read_csv(CSV_FILE1, header=None)
    suspected_df = pd.read_csv(CSV_FILE2, header=None)
    found_df = pd.read_csv(CSV_FILE3, header=None)
    
    # Add column names
    complaint_df.columns = ["TicketID", "Image", "Email", "Name", "Address", "Phone", "Description", "Status", "Date", "Time"]
    suspected_df.columns = ["TicketID", "Image", "Email", "Name", "Address", "NameOfChild", "Phone", "Status", "Date", "Time"]
    found_df.columns = ["TicketID", "Image", "Email", "Name", "Address", "Phone", "Description", "Status", "SuspectedID", "Date", "Time"]
    
    # Add 'Type' column
    complaint_df['Type'] = 'MCR'
    suspected_df['Type'] = 'SCF'
    found_df['Type'] = found_df['TicketID'].apply(lambda x: 'MCR' if str(x).startswith('MCR') else 'SCF')
    
    # Combine all data
    all_data = pd.concat([complaint_df, suspected_df, found_df], ignore_index=True)
    
    # Convert date and time columns
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    all_data['Time'] = all_data['Time'].astype(str)  # Convert Time to string for JSON serialization
    
    return all_data, complaint_df, suspected_df, found_df


# Clears user session to log out and redirects to login page.
@app.route("/logout")
def logout():
    session.pop("email", None)  # Clear session
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
