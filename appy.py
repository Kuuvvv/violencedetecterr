from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import threading
import time
import urllib.request
from datetime import datetime
from collections import deque

app = Flask(__name__, static_folder='static')
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = 'fight_detection_model.h5'
IMG_SIZE = 224
SEQUENCE_LENGTH = 64
THRESHOLD = 0.85
SKIP_FRAMES = 2
VIOLENCE_CLASS_INDEX = 1
DETECTIONS_FOLDER = "static/detections"
HISTORY_FILE = "history.json"

# Ensure all necessary directories exist
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# --- PRIVACY LAYER: OpenCV Deep Learning (DNN) SSD Model ---
def load_face_dnn():
    """Auto-downloads and loads the highly accurate OpenCV Deep Learning Face Model."""
    model_dir = "dnn_model"
    os.makedirs(model_dir, exist_ok=True)
    
    prototxt = os.path.join(model_dir, "deploy.prototxt")
    caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download files if they don't exist
    if not os.path.exists(prototxt):
        print("--- Downloading DNN Face Prototxt... ---")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", prototxt)
    if not os.path.exists(caffemodel):
        print("--- Downloading DNN Face Weights... ---")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", caffemodel)
        
    return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

print("Initializing Deep Learning Face Detector...")
face_net = load_face_dnn()

# --- GLOBALS ---
model = None
cap = None
stream_active = False
stream_lock = threading.Lock()
latest_frame = None
latest_score = 0.0
latest_label = "NORMAL"

def load_model():
    """Loads the 3D-CNN Violence model into memory."""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("--- SENTINEL Violence AI Loaded Successfully ---")
    except Exception as e:
        print(f"!!! CRITICAL ERROR: Could not load model. {e}")

def apply_privacy_blur(frame):
    """Detects faces using the DNN model and applies heavy Gaussian blur. (PARANOIA MODE)"""
    h, w = frame.shape[:2]
    
    # 1. UPSCALED VISION: Changed (300, 300) to (400, 400) 
    # This forces the AI to look closer at background pixels for distant faces
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 1.0, (400, 400), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    # Loop over all detected potential faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # 2. AGGRESSIVE THRESHOLD: Dropped from 0.40 to 0.15
        # If it's even 15% confident it's a face, it gets blurred.
        if confidence > 0.15: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Expand the bounding box slightly to blur the hair/ears too (adds +10 pixels padding)
            startX, startY = max(0, startX - 10), max(0, startY - 15)
            endX, endY = min(w, endX + 10), min(h, endY + 10)
            
            # Extract and blur
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size > 0:
                # 99x99 kernel = Massive blur
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 50)
                frame[startY:endY, startX:endX] = blurred_face
                
    return frame

def load_history():
    """Reads the JSON archive of past incidents."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: return json.load(f)
            except: return []
    return []

def save_to_history(entry):
    """Saves a new incident to the top of the JSON archive."""
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def save_evidence(frames, location_label="Unknown Zone", source_label="Stream"):
    """Anonymizes and saves incident screenshots with location metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    indices = [0, 20, 40, 60]
    
    for i in indices:
        if i < len(frames):
            anonymized_frame = apply_privacy_blur(frames[i].copy())
            fname = f"incident_{timestamp}_f{i}.jpg"
            fpath = os.path.join(DETECTIONS_FOLDER, fname)
            cv2.imwrite(fpath, anonymized_frame)
            saved_files.append(f"detections/{fname}")

    entry = {
        "id": timestamp,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location_label,
        "source": source_label,
        "screenshots": saved_files,
        "score": round(float(latest_score), 3),
        "privacy_protected": True
    }
    save_to_history(entry)
    print(f"--- Alert Archived & Anonymized: {location_label} ---")
    return saved_files

def process_stream(source, source_label, location_label):
    """The core engine that feeds frames to the AI and handles the sliding window."""
    global cap, stream_active, latest_frame, latest_score, latest_label

    processed_buffer = deque(maxlen=SEQUENCE_LENGTH)
    raw_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    cooldown_timer = 0

    cap = cv2.VideoCapture(source)
    while stream_active:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0

        processed_buffer.append(normalized)
        raw_buffer.append(frame.copy())

        if model and len(processed_buffer) == SEQUENCE_LENGTH and frame_count % SKIP_FRAMES == 0:
            tensor = np.expand_dims(list(processed_buffer), axis=0)
            prediction = model.predict(tensor, verbose=0)[0]
            score = float(prediction[VIOLENCE_CLASS_INDEX])
            latest_score = score

            if score > THRESHOLD:
                latest_label = "VIOLENCE DETECTED"
                if cooldown_timer == 0:
                    save_evidence(list(raw_buffer), location_label, source_label)
                    cooldown_timer = 90 
            else:
                latest_label = "NORMAL"

        if cooldown_timer > 0: 
            cooldown_timer -= 1

        color = (0, 0, 255) if latest_label == "VIOLENCE DETECTED" else (0, 220, 80)
        cv2.putText(frame, f"ZONE: {location_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, latest_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        with stream_lock:
            latest_frame = frame.copy()

    if cap: cap.release()
    stream_active = False

def generate_frames():
    """Generator to push video frames to the web frontend."""
    while True:
        with stream_lock:
            frame = latest_frame
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.05)

# --- WEB API ROUTES ---

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/public')
def public_portal():
    return send_from_directory('static', 'public.html')

@app.route('/api/start/webcam', methods=['POST'])
def start_webcam():
    global stream_active
    data = request.json or {}
    location = data.get('location', 'Main Entrance')
    if stream_active: return jsonify({"error": "Already active"}), 400
    stream_active = True
    threading.Thread(target=process_stream, args=(0, "Webcam", location), daemon=True).start()
    return jsonify({"status": "started", "location": location})

@app.route('/api/start/file', methods=['POST'])
def start_file():
    global stream_active
    if stream_active: return jsonify({"error": "Already active"}), 400
    location = request.form.get('location', 'File Upload')
    file = request.files.get('video')
    if not file: return jsonify({"error": "File missing"}), 400
    
    upload_path = os.path.join('static', 'uploads', file.filename)
    file.save(upload_path)
    
    stream_active = True
    threading.Thread(target=process_stream, args=(upload_path, file.filename, location), daemon=True).start()
    return jsonify({"status": "started", "source": file.filename})

@app.route('/api/stop', methods=['POST'])
def stop_stream():
    global stream_active
    stream_active = False
    return jsonify({"status": "stopped"})

@app.route('/api/status')
def status():
    return jsonify({"active": stream_active, "score": round(latest_score, 3), "label": latest_label})

@app.route('/api/history')
def history():
    return jsonify(load_history())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)