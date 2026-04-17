import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from collections import deque

# --- CONFIGURATION (Based on your model summary) ---
MODEL_PATH = 'fight_detection_model.h5'
IMG_SIZE = 224
SEQUENCE_LENGTH = 64  # Your model expects 64 frames
THRESHOLD = 0.85      # Sensitivity
SKIP_FRAMES = 2       # Process every 3rd frame to reduce CPU load

# --- CLASS TOGGLE ---
# If sitting still triggers an alert, swap these:
VIOLENCE_CLASS_INDEX = 1 # Usually 1 is violence, 0 is normal. 

def save_evidence(frames, folder="detections"):
    """Saves 4 frames as JPG screenshots for local storage."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Grab 4 frames spread throughout the sequence
    indices = [0, 20, 40, 60] 
    
    for i in indices:
        if i < len(frames):
            file_path = f"{folder}/alert_{timestamp}_frame{i}.jpg"
            cv2.imwrite(file_path, frames[i])
    print(f"--- SCREENSHOTS SAVED TO {folder} ---")

def run_app():
    # Load Model
    print("Loading model... this may take a moment.")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0) # Change to "video.mp4" for file input
    
    # Deques automatically pop the oldest item when a new one is added
    processed_buffer = deque(maxlen=SEQUENCE_LENGTH)
    raw_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    frame_count = 0
    cooldown_timer = 0 # Prevents saving 100 photos for 1 fight

    print("Detector Active. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # 1. Preprocessing
        # Convert BGR to RGB (Most AI models are trained on RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0 # Rescale pixels to [0, 1]

        # 2. Add to buffers
        processed_buffer.append(normalized)
        raw_buffer.append(frame)

        # 3. Prediction (Only every SKIP_FRAMES to maintain speed)
        if len(processed_buffer) == SEQUENCE_LENGTH and frame_count % SKIP_FRAMES == 0:
            # Reshape for 3D-CNN: (1, 64, 224, 224, 3)
            input_tensor = np.expand_dims(list(processed_buffer), axis=0)
            
            prediction = model.predict(input_tensor, verbose=0)[0]
            
            score_0 = prediction[0]
            score_1 = prediction[1]
            violence_score = prediction[VIOLENCE_CLASS_INDEX]

            # Visual Feedback
            status_color = (0, 255, 0) # Green
            label = "NORMAL"

            if violence_score > THRESHOLD:
                status_color = (0, 0, 255) # Red
                label = "VIOLENCE DETECTED"
                
                # Save screenshots if not in cooldown
                if cooldown_timer == 0:
                    save_evidence(list(raw_buffer))
                    cooldown_timer = 30 # Wait 30 processed cycles before next save

            # Display scores for debugging the "sitting still" issue
            cv2.putText(frame, f"C0: {score_0:.2f} | C1: {score_1:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, label, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

        if cooldown_timer > 0:
            cooldown_timer -= 1

        cv2.imshow('Violence Detector Dashboard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()