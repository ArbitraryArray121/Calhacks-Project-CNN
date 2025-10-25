"""
ASL Model Training Script - IMPROVED
=====================================
This script trains a CNN to recognize ASL gestures from MediaPipe hand landmarks.

KEY IMPROVEMENTS:
1. Added class weights to handle imbalanced data
2. Better comments explaining each step
3. Improved model architecture
4. Saves per-class accuracy for analysis
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle

print("="*70)
print("ASL MODEL TRAINING")
print("="*70)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("\n[1/7] Loading landmark data...")
df = pd.read_csv('asl_landmarks.csv')

# X = features (63 landmark coordinates: 21 landmarks × 3 coordinates each)
# y = labels (the ASL letter/gesture)
X = df.iloc[:, :-1].values  # All columns except last (the features)
y = df['label'].values       # Last column (the labels)

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {len(X)}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {sorted(np.unique(y))}")

# ==============================================================================
# STEP 2: NORMALIZE FEATURES
# ==============================================================================
print("\n[2/7] Normalizing landmark coordinates...")
# Normalization scales all values to [0, 1] range
# This helps the neural network learn better (prevents large values from dominating)
X_min = X.min()  # Minimum value across all landmarks
X_max = X.max()  # Maximum value across all landmarks
X_normalized = (X - X_min) / (X_max - X_min)

print(f"Original range: [{X_min:.3f}, {X_max:.3f}]")
print(f"Normalized range: [0.000, 1.000]")

# ==============================================================================
# STEP 3: ENCODE LABELS
# ==============================================================================
print("\n[3/7] Encoding labels...")
# LabelEncoder converts text labels (like 'A', 'B', '0') to numbers (0, 1, 2, ...)
# Neural networks need numerical labels, not text
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

print(f"\nLabel Mapping (Original → Encoded):")
for original, encoded in zip(le.classes_, range(num_classes)):
    count = np.sum(y == original)  # Count samples per class
    print(f"  {original:6s} → {encoded:2d}  ({count:5d} samples)")

# NOTE: The numerical labels like '0', '1', '2' are STRINGS in the original data
# LabelEncoder treats them as any other label (no confusion with encoded values)
# Example: The string '0' might be encoded as 37, and the string 'A' as 0

# Save label encoder for use during inference
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n✓ Label encoder saved: label_encoder.pkl")

# ==============================================================================
# STEP 4: CALCULATE CLASS WEIGHTS (CRITICAL FOR IMBALANCED DATA!)
# ==============================================================================
print("\n[4/7] Computing class weights for imbalanced data...")
# Class weights tell the model to care more about underrepresented classes
# Without this, model ignores rare classes (like '0' with only 16 samples)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))

print("\nClass weights (higher = more important):")
for idx, (label, weight) in enumerate(zip(le.classes_, class_weights)):
    count = np.sum(y_encoded == idx)
    print(f"  {label:6s} (n={count:5d}): weight={weight:.3f}")

print(f"\nWeight ratio (max/min): {class_weights.max()/class_weights.min():.1f}:1")

# ==============================================================================
# STEP 5: RESHAPE DATA FOR CNN
# ==============================================================================
print("\n[5/7] Reshaping data for CNN input...")
# Reshape from (N, 63) to (N, 21, 3, 1)
# This creates a 2D "image" where:
#   - 21 rows = 21 hand landmarks
#   - 3 columns = x, y, z coordinates
#   - 1 channel = single "color" channel (grayscale-like)
X_reshaped = X_normalized.reshape(-1, 21, 3, 1)
print(f"Reshaped from {X_normalized.shape} to {X_reshaped.shape}")

# ==============================================================================
# STEP 6: SPLIT DATA
# ==============================================================================
print("\n[6/7] Splitting data into train/validation/test sets...")
# Split: 80% train, 10% validation, 10% test
# Stratify ensures each set has same class distribution as original
X_train, X_temp, y_train, y_temp = train_test_split(
    X_reshaped, y_encoded, 
    test_size=0.2,           # 20% for validation + test
    random_state=42,         # For reproducibility
    stratify=y_encoded       # Maintain class distribution
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,           # Split the 20% in half (10% each)
    random_state=42,
    stratify=y_temp
)

# Convert labels to one-hot encoding for categorical cross-entropy loss
# Example: class 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, ...]
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"Training set:   {X_train.shape[0]:5d} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]:5d} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]:5d} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ==============================================================================
# STEP 7: BUILD MODEL
# ==============================================================================
print("\n[7/7] Building CNN model...")
print("\nModel Architecture:")
print("-" * 70)

model = Sequential([
    # INPUT SHAPE: (21, 3, 1)
    # Think of this as a 21×3 "image" of hand landmarks
    
    # CONVOLUTIONAL BLOCK 1
    # Learns spatial patterns in landmarks (like "index finger above thumb")
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(21, 3, 1)),
    BatchNormalization(),  # Normalizes activations (helps training)
    MaxPooling2D((1, 1)),  # No downsampling (data too small)
    
    # CONVOLUTIONAL BLOCK 2
    # Learns more complex patterns (combinations of simple patterns)
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((1, 1)),
    
    # CONVOLUTIONAL BLOCK 3
    # Learns high-level features (like "closed fist" vs "spread fingers")
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    
    # GLOBAL AVERAGE POOLING
    # Converts 2D feature maps to 1D feature vector
    GlobalAveragePooling2D(),
    
    # FULLY CONNECTED LAYERS
    # Makes final decision based on learned features
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevents overfitting (randomly drops 50% of neurons)
    Dense(64, activation='relu'),
    Dropout(0.3),  # Less aggressive dropout
    
    # OUTPUT LAYER
    # Outputs probability for each class (sums to 1.0)
    Dense(num_classes, activation='softmax')
])

# Compile model: choose optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),      # Adam optimizer with learning rate 0.001
    loss='categorical_crossentropy',          # Standard loss for multi-class classification
    metrics=['accuracy']                      # Track accuracy during training
)

model.summary()

# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================
print("\n" + "="*70)
print("TRAINING CALLBACKS")
print("="*70)

# EARLY STOPPING: Stop training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor='val_loss',            # Watch validation loss
    patience=20,                   # Wait 20 epochs before stopping
    restore_best_weights=True,     # Revert to best model
    verbose=1
)

# CHECKPOINT: Save best model during training
checkpoint = ModelCheckpoint(
    'best_asl_model.keras',
    monitor='val_accuracy',        # Watch validation accuracy
    save_best_only=True,           # Only save if accuracy improves
    mode='max',                    # Higher is better
    verbose=1
)

# LEARNING RATE REDUCTION: Reduce learning rate when stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                    # Reduce by half
    patience=10,                   # Wait 10 epochs
    min_lr=1e-6,                   # Don't go below this
    verbose=1
)

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("\n⚠️  This may take 10-30 minutes depending on your hardware")
print("📊 Watch for validation accuracy to increase\n")

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=200,                           # Max epochs (early stopping may end sooner)
    batch_size=32,                        # Process 32 samples at a time
    class_weight=class_weight_dict,       # ⭐ USE CLASS WEIGHTS!
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# ==============================================================================
# EVALUATE MODEL
# ==============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ==============================================================================
# PER-CLASS ANALYSIS
# ==============================================================================
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Identify problematic classes
print("\n⚠️  Classes with low accuracy (<70%):")
for idx, label in enumerate(le.classes_):
    mask = y_test == idx
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == y_test[mask]).mean()
        if acc < 0.7:
            print(f"  {label}: {acc*100:.1f}% (n={mask.sum()})")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)

# ==============================================================================
# SAVE EVERYTHING
# ==============================================================================
print("\n" + "="*70)
print("SAVING MODEL AND PARAMETERS")
print("="*70)

model.save('asl_model.keras')
print("✓ Model saved: asl_model.keras")

np.save('normalization_params.npy', np.array([X_min, X_max]))
print("✓ Normalization params saved: normalization_params.npy")

# Save training history for analysis
import json
history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)
print("✓ Training history saved: training_history.json")

# Save per-class accuracy for runtime use
per_class_acc = {}
for idx, label in enumerate(le.classes_):
    mask = y_test == idx
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == y_test[mask]).mean()
        per_class_acc[label] = float(acc)

with open('per_class_accuracy.json', 'w') as f:
    json.dump(per_class_acc, f)
print("✓ Per-class accuracy saved: per_class_accuracy.json")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\n💡 Next steps:")
print("1. Check per-class accuracy in classification report")
print("2. If some classes have low accuracy, collect more data for them")
print("3. Use the model with main_controlled.py for inference")
print("4. Monitor false positives - if high, increase confidence threshold")
print("="*70)
