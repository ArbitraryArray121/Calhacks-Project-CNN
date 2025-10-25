import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle

print("Loading landmark data...")
df = pd.read_csv('asl_landmarks.csv')

# Separate features and labels
X = df.iloc[:, :-1].values  # All columns except last
y = df['label'].values

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {sorted(np.unique(y))}")

# Normalize landmarks to [0, 1]
X_min = X.min()
X_max = X.max()
X_normalized = (X - X_min) / (X_max - X_min)

# Encode labels What is this doing? Would the relabing cause problem since therere numerical ones existing.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

""" ===== ADDED: PRINT LABEL MAPPING =====
print("\n" + "="*50)
print("LABEL MAPPING DETAILS:")
print("="*50)

# Create a mapping table
print("\nOriginal Labels → Encoded Labels:")
print("-" * 35)
for original_label, encoded_label in zip(le.classes_, range(num_classes)):
    print(f"  '{original_label}' (type: {type(original_label).__name__}) → {encoded_label}")

# Show sample mapping for first few instances
print(f"\nSample of actual transformations (first 10 instances):")
print("-" * 50)
sample_indices = min(10, len(y))
for i in range(sample_indices):
    print(f"  Instance {i}: '{y[i]}' → {y_encoded[i]}")

print(f"\nTotal classes: {num_classes}")
print("="*50)
===== END ADDED CODE ====="""
# ===== ADDED: SIMPLE LABEL MAPPING DISPLAY =====
print("\nLabel Mapping (Original → Encoded):")
for original, encoded in zip(le.classes_, range(num_classes)):
    print(f"  {original} → {encoded}")
# ===== END ADDED CODE =====

# Save label encoder for later use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\nLabel encoding: {dict(zip(le.classes_, range(num_classes)))}")

# Reshape for CNN input: (N, 21, 3, 1)
X_reshaped = X_normalized.reshape(-1, 21, 3, 1)

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\nTraining set: {X_train.shape} samples")
print(f"Validation set: {X_val.shape} samples")
print(f"Test set: {X_test.shape} samples")

# Build the CNN model
print("\nBuilding CNN model...")
model = Sequential([
    # Input: (21, 3, 1)
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(21, 3, 1)),
    BatchNormalization(),
    MaxPooling2D((1, 1)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((1, 1)),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    
    # Global Average Pooling
    GlobalAveragePooling2D(),
    
    # Fully Connected Layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    # Output Layer
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_asl_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save final model
model.save('asl_model.keras')
print("\nModel saved as: asl_model.keras")

# Detailed classification report
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Save normalization parameters
np.save('normalization_params.npy', np.array([X_min, X_max]))
print("\nNormalization parameters saved: normalization_params.npy")
print("Training complete!")
