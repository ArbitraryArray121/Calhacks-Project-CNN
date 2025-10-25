import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# Load trained model
print("Loading model...")
model = load_model('asl_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

norm_params = np.load('normalization_params.npy')
X_min, X_max = norm_params[0], norm_params[1]

print(f"Ready! Classes: {label_encoder.classes_}")

# Initialize MediaPipe with optimized settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60

# Prediction smoothing buffer (temporal filtering)
prediction_buffer = deque(maxlen=5)  # Last 5 predictions
confidence_buffer = deque(maxlen=5)

def normalize_landmarks_improved(landmarks):
    """
    Improved normalization that's more robust to scale variations.
    Uses hand span as a reference for scale-invariant features.
    """
    landmark_array = []
    
    # Extract all landmarks
    for landmark in landmarks:
        landmark_array.append([landmark.x, landmark.y, landmark.z])
    
    landmark_array = np.array(landmark_array)
    
    # Use wrist as origin (make translation-invariant)
    wrist = landmark_array[0]
    landmark_array = landmark_array - wrist
    
    # Calculate hand span (distance from wrist to middle finger tip) for scale normalization
    middle_finger_tip = landmark_array[12]  # Middle finger tip
    hand_span = np.linalg.norm(middle_finger_tip)
    
    # Avoid division by zero
    if hand_span < 0.001:
        hand_span = 1.0
    
    # Make scale-invariant by normalizing by hand span
    landmark_array = landmark_array / hand_span
    
    # Flatten for model input
    return landmark_array.flatten()

def get_smooth_prediction(predicted_label, confidence):
    """Apply temporal smoothing to reduce jitter"""
    prediction_buffer.append(predicted_label)
    confidence_buffer.append(confidence)
    
    # Use majority voting for more stable predictions
    if len(prediction_buffer) >= 3:
        # Count occurrences
        unique, counts = np.unique(list(prediction_buffer), return_counts=True)
        most_common_idx = np.argmax(counts)
        smoothed_label = unique[most_common_idx]
        smoothed_confidence = np.mean(list(confidence_buffer))
        
        return smoothed_label, smoothed_confidence
    
    return predicted_label, confidence

def get_hand_bbox(hand_landmarks, frame_shape, padding=0.3):
    """Extract bounding box around hand with padding"""
    h, w = frame_shape[:2]
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add padding
    x_min = max(0, x_min - padding * width)
    x_max = min(1, x_max + padding * width)
    y_min = max(0, y_min - padding * height)
    y_max = min(1, y_max + padding * height)
    
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    
    return x1, y1, x2, y2

# Single MediaPipe instance with balanced settings
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,  # Balanced - not too strict
    min_tracking_confidence=0.5,   # Allow tracking even with slight movements
    model_complexity=1              # Good balance of speed and accuracy
)

print("Single-pass optimized detection ready!")

# Increase camera resolution for better detail
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization for better detection in varying lighting
        rgb_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        rgb_yuv[:,:,0] = cv2.equalizeHist(rgb_yuv[:,:,0])
        rgb_enhanced = cv2.cvtColor(rgb_yuv, cv2.COLOR_YUV2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_enhanced)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw landmarks on main frame
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Get bounding box for visualization
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, frame.shape, padding=0.3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create zoomed view for better visualization
                hand_crop = frame[y1:y2, x1:x2]
                if hand_crop.size > 0:
                    hand_resized = cv2.resize(hand_crop, (300, 300))
                    cv2.imshow('Zoomed Hand', hand_resized)
                
                # Use improved normalization
                landmark_features = normalize_landmarks_improved(hand_landmarks.landmark)
                
                # Reshape for model
                # First apply the training normalization
                landmarks_array = landmark_features.reshape(1, -1)
                landmarks_normalized = (landmarks_array - X_min) / (X_max - X_min + 1e-8)
                landmarks_reshaped = landmarks_normalized.reshape(1, 21, 3, 1)
                
                # Predict
                prediction = model.predict(landmarks_reshaped, verbose=0)
                predicted_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                predicted_label = label_encoder.classes_[predicted_idx]
                
                # Apply temporal smoothing
                smoothed_label, smoothed_confidence = get_smooth_prediction(predicted_label, confidence)
                
                # Quitting logic with smoothed prediction
                if smoothed_label == 'DEL' and smoothed_confidence > 70:
                    stop_gesture_counter += 1
                    countdown_text = f"Stopping in: {((STOP_GESTURE_THRESHOLD - stop_gesture_counter) // 20)}s"
                    cv2.putText(frame, countdown_text, (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    stop_gesture_counter = 0
                
                if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
                    print("Stop gesture held! Exiting...")
                    break
                
                # Display smoothed prediction on main frame
                label_text = f'{smoothed_label}: {smoothed_confidence:.1f}%'
                
                # Color code by confidence
                if smoothed_confidence > 80:
                    color = (0, 255, 0)  # Green - high confidence
                elif smoothed_confidence > 60:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
                
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Show raw vs smoothed prediction for debugging
                debug_text = f"Raw: {predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, debug_text, (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            # Clear buffers when no hand detected
            prediction_buffer.clear()
            confidence_buffer.clear()
            stop_gesture_counter = 0

        # Add instructions
        cv2.putText(frame, "Hold DEL gesture for 3s to quit", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('ASL Recognition (Improved)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
            break

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
