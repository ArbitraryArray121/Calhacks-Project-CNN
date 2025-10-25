import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load trained model
print("Loading model...")
model = load_model('asl_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

norm_params = np.load('normalization_params.npy')
X_min, X_max = norm_params[0], norm_params[1]

print(f"Ready! Classes: {label_encoder.classes_}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60

def get_hand_bbox(hand_landmarks, frame_shape, padding=0.4):
    """Extract bounding box around hand with padding"""
    h, w = frame_shape[:2]
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    x_min = max(0, x_min - padding * width)
    x_max = min(1, x_max + padding * width)
    y_min = max(0, y_min - padding * height)
    y_max = min(1, y_max + padding * height)
    
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    
    return x1, y1, x2, y2

# Two MediaPipe instances: one for detection, one for refined tracking
hands_detect = mp.solutions.hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0  # Faster model for initial detection
)

hands_refine = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1  # More accurate model for cropped region
)

print("Two-pass detection ready!")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # PASS 1: Quick detection on full frame
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_detect = hands_detect.process(rgb_full)

        if results_detect.multi_hand_landmarks:
            for hand_landmarks_rough in results_detect.multi_hand_landmarks:
                
                # Get bounding box from rough detection
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks_rough, frame.shape, padding=0.4)
                
                # Draw rough bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Crop the hand region
                hand_crop = frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    # PASS 2: Re-detect on cropped/zoomed region
                    hand_resized = cv2.resize(hand_crop, (640, 640))  # Larger for better detection
                    rgb_crop = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
                    results_refine = hands_refine.process(rgb_crop)
                    
                    if results_refine.multi_hand_landmarks:
                        # Use the REFINED landmarks from cropped image
                        hand_landmarks = results_refine.multi_hand_landmarks[0]
                        
                        # Draw landmarks on the zoomed view
                        mp_drawing.draw_landmarks(
                            hand_resized, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Show zoomed hand with landmarks
                        cv2.imshow('Zoomed Hand (Refined)', hand_resized)
                        
                        # Extract landmarks for CNN prediction
                        landmark_list = []
                        wrist = hand_landmarks.landmark[0]
                        
                        for landmark in hand_landmarks.landmark:
                            landmark_list.extend([
                                landmark.x - wrist.x,
                                landmark.y - wrist.y,
                                landmark.z - wrist.z
                            ])
                        
                        # Preprocess
                        landmarks_array = np.array(landmark_list).reshape(1, -1)
                        landmarks_normalized = (landmarks_array - X_min) / (X_max - X_min)
                        landmarks_reshaped = landmarks_normalized.reshape(1, 21, 3, 1)
                        
                        # Predict
                        prediction = model.predict(landmarks_reshaped, verbose=0)
                        predicted_idx = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        predicted_label = label_encoder.classes_[predicted_idx]
                        
                        # Quitting
                        if predicted_label == 'DEL' and confidence > 80:
                            stop_gesture_counter += 1
                            countdown_text = f"Stopping in: {((STOP_GESTURE_THRESHOLD - stop_gesture_counter) // 20)}s"
                            cv2.putText(frame, countdown_text, (20, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            stop_gesture_counter = 0
                        
                        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
                            print("Stop gesture held! Exiting...")
                            break
                        
                        # Display prediction on main frame
                        label_text = f'{predicted_label}: {confidence:.1f}%'
                        cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
                        cv2.putText(frame, label_text, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        
                        # Draw refined landmarks back on main frame (scaled back)
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            # Convert cropped coordinates back to original frame
                            lm_x = int(x1 + landmark.x * (x2 - x1))
                            lm_y = int(y1 + landmark.y * (y2 - y1))
                            cv2.circle(frame, (lm_x, lm_y), 3, (0, 255, 0), -1)

        cv2.imshow('ASL Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
            break

finally:
    hands_detect.close()
    hands_refine.close()
    cap.release()
    cv2.destroyAllWindows()
