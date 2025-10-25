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
LETTER_THRESHOLD = 5  # Hold for ~0.75 seconds to confirm letter

stop_gesture_counter = 0
STOP_GESTURE_THRESHOLD = 60  # ~1.5 seconds at 20fps

# Confidence threshold for recording
MIN_CONFIDENCE = 70.0

print("\n" + "="*50)
print("ASL RECORDING MODE")
print("="*50)
print("Hold a gesture to record it")
print("Show 'DEL' gesture for 3 seconds to finish")
print("="*50 + "\n")

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                
                # Check for stop gesture (DEL)
                if predicted_label == 'DEL' and confidence > 80:
                    stop_gesture_counter += 1
                    countdown_text = f"Stopping in: {((STOP_GESTURE_THRESHOLD - stop_gesture_counter) // 20)}s"
                    cv2.putText(frame, countdown_text, (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Reset letter counter when showing stop gesture
                    letter_counter = 0
                    current_letter = None
                else:
                    stop_gesture_counter = 0
                    
                    # Recording logic - only for confident predictions
                    if confidence > MIN_CONFIDENCE and predicted_label != 'DEL':
                        if predicted_label == current_letter:
                            letter_counter += 1
                            
                            # If held long enough, record it
                            if letter_counter == LETTER_THRESHOLD:
                                recorded_text.append(predicted_label)
                                print(f"Recorded: {predicted_label} | Full text: {''.join(recorded_text)}")
                        else:
                            # New letter detected
                            current_letter = predicted_label
                            letter_counter = 1
                    else:
                        # Reset if confidence drops or no hand detected
                        letter_counter = 0
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
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Display current prediction
                label_text = f'{predicted_label}: {confidence:.1f}%'
                cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Show hold progress bar
                if letter_counter > 0 and current_letter != 'DEL':
                    progress = int((letter_counter / LETTER_THRESHOLD) * 300)
                    cv2.rectangle(frame, (20, 120), (320, 150), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, 120), (20 + progress, 150), (0, 255, 0), -1)
                    cv2.putText(frame, f"Hold: {current_letter}", (25, 142), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No hand detected - reset counters
            letter_counter = 0
            current_letter = None
            stop_gesture_counter = 0

        # Display recorded text at bottom
        recorded_display = ''.join(recorded_text)
        if len(recorded_display) > 30:
            recorded_display = '...' + recorded_display[-27:]
        
        cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, f"Recorded: {recorded_display}", (10, frame.shape[0]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('ASL Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Print final recording
print("\n" + "="*50)
print("SESSION ENDED")
print("="*50)
print(f"Recorded text: {''.join(recorded_text)}")
print("="*50)
