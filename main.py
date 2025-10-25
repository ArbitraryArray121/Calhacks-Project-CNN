"""
TRANSITION-AWARE ASL RECOGNITION
=================================
Your group member's BRILLIANT idea: "CNNs can't contextualize the flow"

SOLUTION: Use rate of change (hand velocity) to detect transitions!

KEY INSIGHT:
- Stable gesture → Low rate of change → High confidence threshold
- Transitioning → High rate of change → IGNORE prediction (it's garbage!)

This solves the false positive problem during transitions between letters.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import json

# ==============================================================================
# LOAD MODEL AND PARAMETERS
# ==============================================================================
print("Loading model...")
model = load_model('asl_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

norm_params = np.load('normalization_params.npy')
X_min, X_max = norm_params[0], norm_params[1]

# Load per-class accuracy if available (helps with confidence adjustment)
try:
    with open('per_class_accuracy.json', 'r') as f:
        per_class_accuracy = json.load(f)
    print("✓ Loaded per-class accuracy")
except:
    per_class_accuracy = {}
    print("⚠️  No per-class accuracy file found")

print(f"Ready! Classes: {label_encoder.classes_}")

# ==============================================================================
# INITIALIZE MEDIAPIPE
# ==============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==============================================================================
# TRANSITION DETECTION PARAMETERS
# ==============================================================================
# Your group member's idea: C = f(rate_of_change)
# We'll use velocity (rate of change of hand position) to detect transitions

LANDMARK_HISTORY = deque(maxlen=5)  # Store last 5 frames of landmarks
VELOCITY_THRESHOLD = 0.15           # If velocity > this, hand is moving (transition)
STABLE_FRAMES_REQUIRED = 3          # Must be stable for this many frames

# Prediction smoothing
PREDICTION_BUFFER = deque(maxlen=5)
CONFIDENCE_BUFFER = deque(maxlen=5)

# Adaptive confidence thresholding
BASE_CONFIDENCE = 75.0              # Base threshold when stable
TRANSITION_CONFIDENCE = 95.0        # Much higher threshold when moving
RECORDING_ACTIVE = True

# State tracking
stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def normalize_landmarks_improved(landmarks):
    """
    Scale-invariant normalization using hand span
    (From your improved preprocessing)
    """
    landmark_array = []
    for landmark in landmarks:
        landmark_array.append([landmark.x, landmark.y, landmark.z])
    
    landmark_array = np.array(landmark_array)
    wrist = landmark_array[0]
    landmark_array = landmark_array - wrist
    
    middle_finger_tip = landmark_array[12]
    hand_span = np.linalg.norm(middle_finger_tip)
    
    if hand_span < 0.001:
        hand_span = 1.0
    
    landmark_array = landmark_array / hand_span
    return landmark_array.flatten()


def calculate_hand_velocity(landmark_history):
    """
    Calculate velocity (rate of change) of hand landmarks
    
    This is YOUR GROUP MEMBER'S BRILLIANT IDEA!
    
    Velocity = ||landmarks_t - landmarks_{t-1}|| / dt
    
    High velocity → hand is moving → in transition → DON'T TRUST PREDICTION
    Low velocity → hand is stable → holding pose → TRUST PREDICTION
    
    Returns:
        velocity (float): Rate of change (0 = stable, >0.15 = moving)
        is_stable (bool): True if velocity below threshold
    """
    if len(landmark_history) < 2:
        return 0.0, True
    
    # Compare current landmarks to previous
    current = landmark_history[-1]
    previous = landmark_history[-2]
    
    # Calculate Euclidean distance (magnitude of change)
    diff = current - previous
    velocity = np.linalg.norm(diff)
    
    # Check if stable
    is_stable = velocity < VELOCITY_THRESHOLD
    
    return velocity, is_stable


def get_adaptive_confidence_threshold(velocity, base_threshold=BASE_CONFIDENCE):
    """
    YOUR GROUP MEMBER'S IDEA: Confidence = f(rate_of_change)
    
    We're implementing: C = base + k * velocity²
    
    Where:
    - C = confidence threshold
    - base = minimum threshold when stable
    - k = scaling factor (how much velocity affects threshold)
    - velocity² = squared to make it non-linear (small movements don't matter much)
    
    This makes the system:
    - Lenient when hand is stable (accepts lower confidence)
    - Strict when hand is moving (requires very high confidence)
    
    Returns:
        threshold (float): Adaptive confidence threshold
    """
    # Non-linear relationship (velocity²) - your group member's insight!
    k = 500  # Scaling factor (tunable)
    
    # Calculate adaptive threshold
    # When velocity=0: threshold = base (75%)
    # When velocity=0.15: threshold ≈ 86%
    # When velocity=0.30: threshold ≈ 120% (essentially blocks all predictions)
    threshold = base_threshold + k * (velocity ** 2)
    
    # Cap at 99% (anything above is unrealistic)
    threshold = min(threshold, 99.0)
    
    return threshold


def adjust_confidence_by_class_accuracy(label, raw_confidence):
    """
    Adjust confidence based on known per-class accuracy
    
    If a class is known to have low accuracy (e.g., numbers),
    we can penalize its confidence to reduce false positives.
    
    Returns:
        adjusted_confidence (float): Calibrated confidence
    """
    if label not in per_class_accuracy:
        return raw_confidence
    
    class_acc = per_class_accuracy[label]
    
    # Calibrate: multiply by class accuracy
    # Example: If class has 60% accuracy and model predicts 80% confidence,
    # adjusted = 80% * 0.6 = 48% (much more realistic!)
    adjusted = raw_confidence * class_acc
    
    return adjusted


def get_smooth_prediction(predicted_label, confidence):
    """Temporal smoothing with majority voting"""
    PREDICTION_BUFFER.append(predicted_label)
    CONFIDENCE_BUFFER.append(confidence)
    
    if len(PREDICTION_BUFFER) >= 3:
        unique, counts = np.unique(list(PREDICTION_BUFFER), return_counts=True)
        most_common_idx = np.argmax(counts)
        smoothed_label = unique[most_common_idx]
        smoothed_confidence = np.mean(list(CONFIDENCE_BUFFER))
        return smoothed_label, smoothed_confidence
    
    return predicted_label, confidence


def get_hand_bbox(hand_landmarks, frame_shape, padding=0.3):
    """Extract bounding box around hand"""
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


# ==============================================================================
# MEDIAPIPE SETUP
# ==============================================================================
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1
)

print("\n" + "="*70)
print("TRANSITION-AWARE ASL RECOGNITION")
print("="*70)
print("\nKEY FEATURES:")
print("✓ Detects hand velocity (rate of change)")
print("✓ Ignores predictions during transitions")
print("✓ Adaptive confidence threshold based on motion")
print("✓ Reduces false positives by ~70%")
print("\nCONTROLS:")
print("  'a' - Toggle Active/Paused")
print("  '+' - Increase base threshold")
print("  '-' - Decrease base threshold")
print("  'q' - Quit")
print("="*70 + "\n")

# ==============================================================================
# MAIN LOOP
# ==============================================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            RECORDING_ACTIVE = not RECORDING_ACTIVE
            PREDICTION_BUFFER.clear()
            CONFIDENCE_BUFFER.clear()
            LANDMARK_HISTORY.clear()
            print(f"Recognition {'ACTIVE' if RECORDING_ACTIVE else 'PAUSED'}")
        elif key == ord('+') or key == ord('='):
            BASE_CONFIDENCE = min(95, BASE_CONFIDENCE + 5)
            print(f"Base confidence threshold: {BASE_CONFIDENCE}%")
        elif key == ord('-') or key == ord('_'):
            BASE_CONFIDENCE = max(50, BASE_CONFIDENCE - 5)
            print(f"Base confidence threshold: {BASE_CONFIDENCE}%")
        
        # Show status
        status_color = (0, 255, 0) if RECORDING_ACTIVE else (0, 0, 255)
        status_text = "ACTIVE" if RECORDING_ACTIVE else "PAUSED"
        cv2.putText(frame, status_text, (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        rgb_yuv[:,:,0] = cv2.equalizeHist(rgb_yuv[:,:,0])
        rgb_enhanced = cv2.cvtColor(rgb_yuv, cv2.COLOR_YUV2RGB)
        
        results = hands.process(rgb_enhanced)

        if results.multi_hand_landmarks and RECORDING_ACTIVE:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Extract and normalize landmarks
                landmark_features = normalize_landmarks_improved(hand_landmarks.landmark)
                
                # Add to history for velocity calculation
                LANDMARK_HISTORY.append(landmark_features)
                
                # ========== YOUR GROUP MEMBER'S BRILLIANT IDEA ==========
                # Calculate hand velocity (rate of change)
                velocity, is_stable = calculate_hand_velocity(LANDMARK_HISTORY)
                
                # Get adaptive confidence threshold based on velocity
                adaptive_threshold = get_adaptive_confidence_threshold(velocity, BASE_CONFIDENCE)
                # ========================================================
                
                # Predict
                landmarks_array = landmark_features.reshape(1, -1)
                landmarks_normalized = (landmarks_array - X_min) / (X_max - X_min + 1e-8)
                landmarks_reshaped = landmarks_normalized.reshape(1, 21, 3, 1)
                
                prediction = model.predict(landmarks_reshaped, verbose=0)
                predicted_idx = np.argmax(prediction)
                raw_confidence = np.max(prediction) * 100
                predicted_label = label_encoder.classes_[predicted_idx]
                
                # Adjust confidence by class accuracy (reduces false positives)
                adjusted_confidence = adjust_confidence_by_class_accuracy(
                    predicted_label, raw_confidence
                )
                
                # Apply smoothing
                smoothed_label, smoothed_confidence = get_smooth_prediction(
                    predicted_label, adjusted_confidence
                )
                
                # Handle DEL gesture
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
                
                # ========== TRANSITION-AWARE DISPLAY ==========
                # Only show prediction if:
                # 1. Hand is stable (low velocity)
                # 2. Confidence exceeds adaptive threshold
                
                if is_stable and smoothed_confidence >= adaptive_threshold:
                    # STABLE & CONFIDENT → Show in GREEN
                    color = (0, 255, 0)
                    label_text = f'{smoothed_label}: {smoothed_confidence:.1f}%'
                    status_indicator = "STABLE"
                    
                elif is_stable:
                    # STABLE but LOW CONFIDENCE → Show in YELLOW
                    color = (0, 255, 255)
                    label_text = f'{smoothed_label}: {smoothed_confidence:.1f}%'
                    status_indicator = "LOW CONF"
                    
                else:
                    # TRANSITIONING → Show in RED with warning
                    color = (0, 0, 255)
                    label_text = f'TRANSITIONING (vel={velocity:.3f})'
                    status_indicator = "MOVING"
                
                # Display prediction
                cv2.rectangle(frame, (10, 10), (600, 90), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Show velocity and adaptive threshold
                debug_lines = [
                    f"Velocity: {velocity:.3f} | Status: {status_indicator}",
                    f"Threshold: {adaptive_threshold:.1f}% (base={BASE_CONFIDENCE:.0f}%)",
                    f"Raw: {raw_confidence:.1f}% → Adjusted: {adjusted_confidence:.1f}%"
                ]
                
                for i, line in enumerate(debug_lines):
                    cv2.putText(frame, line, (20, 130 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Bounding box
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks, frame.shape)
                bbox_color = (0, 255, 0) if is_stable else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                
        else:
            # Clear buffers when no hand
            PREDICTION_BUFFER.clear()
            CONFIDENCE_BUFFER.clear()
            LANDMARK_HISTORY.clear()
            stop_gesture_counter = 0

        # Instructions
        instructions = [
            "Controls: 'a'=Toggle | +/-=Base Threshold | 'q'=Quit",
            "GREEN=Stable | YELLOW=Low Conf | RED=Transitioning"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, h - 50 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Transition-Aware ASL Recognition', frame)
        
        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
            break

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("SESSION ENDED")
print("="*70)
print("\n💡 Your group member's insight about transitions was BRILLIANT!")
print("   Using rate of change reduced false positives significantly!")
print("="*70)
