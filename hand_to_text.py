import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import time

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

# Recording variables
recorded_text = []
current_letter = None
letter_counter = 0
LETTER_THRESHOLD = 8  # Hold for ~1 second to confirm letter (increased for stability)

stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60  # ~3 seconds at 20fps

# Confidence threshold for recording
MIN_CONFIDENCE = 65.0  # Slightly lower but with smoothing

# Prediction smoothing
prediction_buffer = deque(maxlen=5)
confidence_buffer = deque(maxlen=5)

# Increase camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

print("\n" + "="*50)
print("ASL RECORDING MODE - IMPROVED")
print("="*50)
print("Hold a gesture steady to record it")
print("Show 'DEL' gesture for 3 seconds to finish")
print("Progress bar shows hold time")
print("="*50 + "\n")

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1
)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Apply histogram equalization for better detection in varying lighting
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
        frame_enhanced = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        
        rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with better visualization
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                # Use improved normalization
                landmark_features = normalize_landmarks_improved(hand_landmarks.landmark)
                
                # Reshape for model
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
                
                # Check for stop gesture (DEL)
                if smoothed_label == 'DEL' and smoothed_confidence > 70:
                    stop_gesture_counter += 1
                    countdown_text = f"Stopping in: {((STOP_GESTURE_THRESHOLD - stop_gesture_counter) // 20)}s"
                    cv2.putText(frame, countdown_text, (20, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Reset letter counter when showing stop gesture
                    letter_counter = 0
                    current_letter = None
                else:
                    stop_gesture_counter = 0
                    
                    # Recording logic - only for confident predictions with smoothing
                    if smoothed_confidence > MIN_CONFIDENCE and smoothed_label != 'DEL':
                        if smoothed_label == current_letter:
                            letter_counter += 1
                            
                            # If held long enough, record it
                            if letter_counter == LETTER_THRESHOLD:
                                recorded_text.append(smoothed_label)
                                print(f"✓ Recorded: {smoothed_label} | Full text: {''.join(recorded_text)}")
                                
                                # Visual feedback - flash green
                                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
                                
                                # Reset to allow recording the same letter again
                                letter_counter = 0
                                current_letter = None
                        else:
                            # New letter detected
                            current_letter = smoothed_label
                            letter_counter = 1
                    else:
                        # Reset if confidence drops
                        if letter_counter > 0:
                            letter_counter = max(0, letter_counter - 1)  # Gradual decay
                        if letter_counter == 0:
                            current_letter = None

                # Check if stop threshold reached
                if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
                    print("\n" + "="*50)
                    print("RECORDING FINISHED!")
                    print("="*50)
                    print(f"Final text: {''.join(recorded_text)}")
                    print("="*50 + "\n")
                    
                    # Save to file
                    with open('asl_recording.txt', 'w') as f:
                        f.write(''.join(recorded_text))
                    print("Saved to: asl_recording.txt")
                    
                    break

                # Display current prediction with color coding
                if smoothed_confidence > 80:
                    color = (0, 255, 0)  # Green
                elif smoothed_confidence > 60:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                label_text = f'{smoothed_label}: {smoothed_confidence:.1f}%'
                cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Show raw vs smoothed for debugging
                debug_text = f"Raw: {predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, debug_text, (460, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Show hold progress bar with better visualization
                if letter_counter > 0 and current_letter and current_letter != 'DEL':
                    progress_percent = (letter_counter / LETTER_THRESHOLD)
                    progress_width = int(progress_percent * 400)
                    
                    # Background
                    cv2.rectangle(frame, (20, 170), (420, 210), (50, 50, 50), -1)
                    # Progress
                    cv2.rectangle(frame, (20, 170), (20 + progress_width, 210), (0, 255, 0), -1)
                    # Border
                    cv2.rectangle(frame, (20, 170), (420, 210), (255, 255, 255), 2)
                    
                    # Text
                    hold_text = f"Recording: {current_letter} ({letter_counter}/{LETTER_THRESHOLD})"
                    cv2.putText(frame, hold_text, (25, 195), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No hand detected - clear buffers and decay counters
            prediction_buffer.clear()
            confidence_buffer.clear()
            stop_gesture_counter = 0
            if letter_counter > 0:
                letter_counter = max(0, letter_counter - 2)  # Faster decay when no hand
            if letter_counter == 0:
                current_letter = None

        # Display recorded text at bottom with better formatting
        recorded_display = ''.join(recorded_text)
        if len(recorded_display) > 40:
            recorded_display = '...' + recorded_display[-37:]
        
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Recorded: {recorded_display}", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow('ASL Recording (Improved)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
        # Already handled above
        pass

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

# Print final recording
print("\n" + "="*50)
print("SESSION ENDED")
print("="*50)
print(f"Recorded text: {''.join(recorded_text)}")
print("="*50)
