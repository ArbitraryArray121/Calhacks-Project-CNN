import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                       min_detection_confidence=0.7)

# Path to your dataset folder
# Structure: asl_dataset/A/img1.jpg, asl_dataset/B/img1.jpg, etc.
dataset_path = '/Users/jerrywang/Desktop/UCB_Documents/CalHacks/Root/Type_01_(Raw_Gesture)'  # Change this to your path

data = []
labels = []

print("Extracting landmarks from images...")

# Loop through each class folder (A, B, C, ..., 0, 1, 2, ...)
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    print(f"Processing class: {class_name}")
    
    # Loop through images in this class
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hand landmarks
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks (normalized relative to wrist)
            landmark_list = []
            wrist = landmarks.landmark[0]
            
            for landmark in landmarks.landmark:
                landmark_list.extend([
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])
            
            data.append(landmark_list)
            labels.append(class_name)

hands.close()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('asl_landmarks.csv', index=False)

print(f"\nExtraction complete! Total samples: {len(data)}")
print(f"Saved to: asl_landmarks.csv")
print(f"Classes found: {sorted(set(labels))}")