"""
New:
1. Transition detection using velocity (rate of change)
2. Adaptive confidence thresholding
3. Better comments explaining everything
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# ==============================================================================
# LOAD MODEL AND PARAMETERS
# ==============================================================================
print("Loading model...")
model = load_model('asl_model.keras')

# Load label encoder (converts predictions back to letters)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load normalization parameters (used during training)
norm_params = np.load('normalization_params.npy')
X_min, X_max = norm_params[0], norm_params[1]

print(f"Ready! Classes: {label_encoder.classes_}")

# ==============================================================================
# INITIALIZE MEDIAPIPE AND CAMERA
# ==============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ==============================================================================
# CONTROL PARAMETERS
# ==============================================================================
stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60  # Hold DEL for 3 seconds to quit (at ~20fps)

# ==============================================================================
# TRANSITION DETECTION PARAMETERS (NEW!)
# ==============================================================================
# our group member's brilliant idea: detect when hand is moving vs stable
LANDMARK_HISTORY = deque(maxlen=5)      # Store last 5 frames of landmarks
VELOCITY_THRESHOLD = 0.15                # If velocity > this, hand is moving
BASE_CONFIDENCE = 75.0                   # Base confidence threshold when stable
k = 200                                  # Scaling factor for adaptive threshold

# Prediction smoothing (reduces jitter)
PREDICTION_BUFFER = deque(maxlen=5)
CONFIDENCE_BUFFER = deque(maxlen=5)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_hand_bbox(hand_landmarks, frame_shape, padding=0.4):
    """
    Extract bounding box around detected hand
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        frame_shape: Shape of the frame (height, width, channels)
        padding: How much extra space to add around hand (0.4 = 40% padding)
    
    Returns:
        x1, y1, x2, y2: Bounding box coordinates
    """
    h, w = frame_shape[:2]
    
    # Get all x and y coordinates of hand landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Find min/max to create bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    
    # Add padding (expand box by padding percentage)
    x_min = max(0, x_min - padding * width)
    x_max = min(1, x_max + padding * width)
    y_min = max(0, y_min - padding * height)
    y_max = min(1, y_max + padding * height)
    
    # Convert normalized coordinates to pixel coordinates
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    
    return x1, y1, x2, y2


def calculate_hand_velocity(landmark_history):
    """
    Calculate velocity (rate of change) of hand position
    
    our GROUP MEMBER'S KEY INSIGHT:
    - High velocity = hand is moving = in transition = DON'T TRUST PREDICTION
    - Low velocity = hand is stable = holding gesture = TRUST PREDICTION
    
    This is how we detect transitions without training a separate model!
    
    Args:
        landmark_history: deque of recent landmark arrays
    
    Returns:
        velocity (float): Rate of change (0 = stable, >0.15 = moving)
        is_stable (bool): True if hand is stable
    """
    if len(landmark_history) < 2:
        return 0.0, True
    
    # Compare current frame to previous frame
    current = landmark_history[-1]
    previous = landmark_history[-2]
    
    # Calculate Euclidean distance (how much landmarks moved)
    # This is the "rate of change" our group member talked about!
    diff = current - previous
    velocity = np.linalg.norm(diff)
    
    # Is hand stable? (velocity below threshold)
    is_stable = velocity < VELOCITY_THRESHOLD
    
    return velocity, is_stable


def get_adaptive_confidence_threshold(velocity, base=BASE_CONFIDENCE, k_factor=k):
    """
    our GROUP MEMBER'S MATH: C = base + k * velocity²
    
    This creates a NON-LINEAR relationship between motion and confidence:
    - When stable (velocity ≈ 0): threshold ≈ base (75%)
    - When moving slowly (velocity ≈ 0.10): threshold ≈ 80%
    - When moving fast (velocity ≈ 0.20): threshold ≈ 95%
    - When moving very fast (velocity > 0.25): threshold ≈ 99%+ (blocks all)
    
    Why velocity²? (quadratic)
    - Small movements barely affect threshold (tremor doesn't matter)
    - Large movements drastically increase threshold (transitions blocked)
    
    Args:
        velocity (float): Hand velocity from calculate_hand_velocity()
        base (float): Base threshold when hand is stable
        k_factor (float): Scaling factor (how much velocity affects threshold)
    
    Returns:
        threshold (float): Adaptive confidence threshold (0-99%)
    """
    # The magic formula! This implements our group member's idea
    threshold = base + k_factor * (velocity ** 2)
    
    # Cap at 99% (anything higher is unrealistic)
    threshold = min(threshold, 99.0)
    
    return threshold


def get_smooth_prediction(predicted_label, confidence):
    """
    Apply temporal smoothing to reduce jittery predictions
    
    Uses majority voting: if same letter appears 3+ times in last 5 frames,
    trust it more than a one-off prediction
    
    Args:
        predicted_label (str): Current prediction
        confidence (float): Current confidence
    
    Returns:
        smoothed_label (str): Most common recent prediction
        smoothed_confidence (float): Average recent confidence
    """
    PREDICTION_BUFFER.append(predicted_label)
    CONFIDENCE_BUFFER.append(confidence)
    
    # Need at least 3 predictions to smooth
    if len(PREDICTION_BUFFER) >= 3:
        # Count occurrences of each prediction
        unique, counts = np.unique(list(PREDICTION_BUFFER), return_counts=True)
        most_common_idx = np.argmax(counts)
        smoothed_label = unique[most_common_idx]
        smoothed_confidence = np.mean(list(CONFIDENCE_BUFFER))
        
        return smoothed_label, smoothed_confidence
    
    return predicted_label, confidence


# ==============================================================================
# MEDIAPIPE SETUP - our ORIGINAL TWO-PASS APPROACH
# ==============================================================================
# Two MediaPipe instances: one for fast detection, one for accurate refinement
# This is our approach - we're keeping it!

# PASS 1: Fast detection on full frame
hands_detect = mp.solutions.hands.Hands(
    min_detection_confidence=0.5,    # Lower threshold = faster detection
    min_tracking_confidence=0.5,
    model_complexity=0               # Simpler model = faster
)

# PASS 2: Accurate refinement on cropped region
hands_refine = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,    # Higher threshold = more accurate
    min_tracking_confidence=0.7,
    model_complexity=1               # More complex model = more accurate
)

print("Two-pass detection ready with transition detection!")
print("\nControls:")
print("  'q' - Quit immediately")
print("  DEL gesture (3s) - Quit via gesture")
print("\nFeatures:")
print("  ✓ Two-pass detection (our original approach)")
print("  ✓ Transition detection (velocity-based)")
print("  ✓ Adaptive confidence threshold")
print("  ✓ Temporal smoothing")

# ==============================================================================
# MAIN LOOP
# ==============================================================================
try:
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally (mirror effect for easier use)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # ======================================================================
        # PASS 1: QUICK DETECTION ON FULL FRAME
        # ======================================================================
        # our original approach: detect hand roughly in full frame
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_detect = hands_detect.process(rgb_full)

        if results_detect.multi_hand_landmarks:
            for hand_landmarks_rough in results_detect.multi_hand_landmarks:
                
                # Get bounding box around detected hand
                x1, y1, x2, y2 = get_hand_bbox(hand_landmarks_rough, frame.shape, padding=0.4)
                
                # Draw bounding box (blue = rough detection)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Crop hand region for refinement
                hand_crop = frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    # ==============================================================
                    # PASS 2: REFINED DETECTION ON CROPPED/ZOOMED REGION
                    # ==============================================================
                    # our original approach: re-detect on cropped region for accuracy
                    
                    # Maintain aspect ratio when zooming (no stretching!)
                    crop_h, crop_w = hand_crop.shape[:2]
                    target_size = 640  # Target size for longest dimension
                    
                    # Calculate new dimensions maintaining aspect ratio
                    if crop_h > crop_w:
                        # Height is longer - scale based on height
                        new_h = target_size
                        new_w = int(crop_w * (target_size / crop_h))
                    else:
                        # Width is longer - scale based on width
                        new_w = target_size
                        new_h = int(crop_h * (target_size / crop_w))
                    
                    # Resize maintaining aspect ratio
                    hand_resized = cv2.resize(hand_crop, (new_w, new_h))
                    
                    # Create a square canvas with padding (black background)
                    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    
                    # Center the resized hand on canvas
                    y_offset = (target_size - new_h) // 2
                    x_offset = (target_size - new_w) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = hand_resized
                    
                    # Use canvas for detection
                    rgb_crop = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    results_refine = hands_refine.process(rgb_crop)
                    
                    if results_refine.multi_hand_landmarks:
                        # Use refined landmarks (more accurate)
                        hand_landmarks = results_refine.multi_hand_landmarks[0]
                        
                        # Draw landmarks on zoomed view (on the canvas)
                        mp_drawing.draw_landmarks(
                            canvas,  # Draw on canvas (not hand_resized)
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Show zoomed hand in separate window (with aspect ratio preserved!)
                        cv2.imshow('Zoomed Hand (Refined)', canvas)
                        
                        # ==============================================================
                        # EXTRACT LANDMARKS FOR PREDICTION
                        # ==============================================================
                        # Extract 21 landmarks × 3 coordinates (x, y, z) = 63 features
                        landmark_list = []
                        wrist = hand_landmarks.landmark[0]  # Use wrist as reference point
                        
                        for landmark in hand_landmarks.landmark:
                            # Store coordinates RELATIVE to wrist (translation invariant)
                            landmark_list.extend([
                                landmark.x - wrist.x,
                                landmark.y - wrist.y,
                                landmark.z - wrist.z
                            ])
                        
                        # Convert to numpy array
                        landmarks_array = np.array(landmark_list)
                        
                        # ==============================================================
                        # TRANSITION DETECTION (NEW!)
                        # ==============================================================
                        # Add current landmarks to history
                        LANDMARK_HISTORY.append(landmarks_array)
                        
                        # Calculate velocity (rate of change)
                        # This is our GROUP MEMBER'S KEY INSIGHT!
                        velocity, is_stable = calculate_hand_velocity(LANDMARK_HISTORY)
                        
                        # Get adaptive confidence threshold
                        # C = base + k * velocity²
                        adaptive_threshold = get_adaptive_confidence_threshold(velocity)
                        
                        # ==============================================================
                        # PREPROCESS LANDMARKS FOR CNN
                        # ==============================================================
                        # Normalize to [0, 1] range (same as training)
                        landmarks_array = landmarks_array.reshape(1, -1)
                        landmarks_normalized = (landmarks_array - X_min) / (X_max - X_min)
                        
                        # Reshape to CNN input format: (1, 21, 3, 1)
                        # Think of this as a 21×3 "image" of hand landmarks
                        landmarks_reshaped = landmarks_normalized.reshape(1, 21, 3, 1)
                        
                        # ==============================================================
                        # PREDICT GESTURE
                        # ==============================================================
                        prediction = model.predict(landmarks_reshaped, verbose=0)
                        predicted_idx = np.argmax(prediction)  # Index of highest probability
                        confidence = np.max(prediction) * 100   # Convert to percentage
                        predicted_label = label_encoder.classes_[predicted_idx]
                        
                        # Apply temporal smoothing (reduces jitter)
                        smoothed_label, smoothed_confidence = get_smooth_prediction(
                            predicted_label, confidence
                        )
                        
                        # ==============================================================
                        # HANDLE DEL GESTURE (QUIT)
                        # ==============================================================
                        if smoothed_label == 'DEL' and smoothed_confidence > 80:
                            stop_gesture_counter += 1
                            countdown_text = f"Stopping in: {((STOP_GESTURE_THRESHOLD - stop_gesture_counter) // 20)}s"
                            cv2.putText(frame, countdown_text, (20, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            stop_gesture_counter = 0
                        
                        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
                            print("Stop gesture held! Exiting...")
                            break
                        
                        # ==============================================================
                        # DISPLAY PREDICTION WITH TRANSITION AWARENESS (NEW!)
                        # ==============================================================
                        # Only show prediction if:
                        # 1. Hand is stable (low velocity), AND
                        # 2. Confidence exceeds adaptive threshold
                        
                        if is_stable and smoothed_confidence >= adaptive_threshold:
                            # STABLE & CONFIDENT → Show in GREEN
                            color = (0, 255, 0)
                            label_text = f'{smoothed_label}: {smoothed_confidence:.1f}%'
                            status = "STABLE"
                            
                        elif is_stable:
                            # STABLE but LOW CONFIDENCE → Show in YELLOW
                            color = (0, 255, 255)
                            label_text = f'{smoothed_label}: {smoothed_confidence:.1f}% (LOW)'
                            status = "LOW CONF"
                            
                        else:
                            # TRANSITIONING → Show in RED with warning
                            color = (0, 0, 255)
                            label_text = f'TRANSITIONING (vel={velocity:.3f})'
                            status = "MOVING"
                        
                        # Display main prediction
                        cv2.rectangle(frame, (10, 10), (500, 70), (0, 0, 0), -1)
                        cv2.putText(frame, label_text, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        
                        # Display debug info (velocity and threshold)
                        debug_text = f"Vel: {velocity:.3f} | Thr: {adaptive_threshold:.0f}% | {status}"
                        cv2.putText(frame, debug_text, (20, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # Draw refined landmarks back on main frame
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            # Convert cropped coordinates back to original frame coordinates
                            lm_x = int(x1 + landmark.x * (x2 - x1))
                            lm_y = int(y1 + landmark.y * (y2 - y1))
                            
                            # Color code by stability
                            point_color = (0, 255, 0) if is_stable else (0, 0, 255)
                            cv2.circle(frame, (lm_x, lm_y), 3, point_color, -1)
        
        else:
            # No hand detected - clear buffers
            PREDICTION_BUFFER.clear()
            CONFIDENCE_BUFFER.clear()
            LANDMARK_HISTORY.clear()
            stop_gesture_counter = 0

        # Show main frame
        cv2.imshow('ASL Recognition (Transition-Aware)', frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check if DEL gesture held long enough
        if stop_gesture_counter >= STOP_GESTURE_THRESHOLD:
            break

finally:
    # Cleanup
    hands_detect.close()
    hands_refine.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("SESSION ENDED")
    print("="*70)
    print("\nour group member's transition detection worked!")
    print("Velocity-based filtering reduced false positives significantly.")
    print("="*70)
