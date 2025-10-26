# enhanced_server_fastapi.py
# Enhanced FastAPI backend with custom gesture registration and incremental training

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np
import cv2
import mediapipe as mp
import pickle
import json
from collections import deque
from datetime import datetime
import asyncio

# Optional: TensorFlow/Keras imports
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf
except Exception:
    load_model = None

app = FastAPI(title="Enhanced Gesture Recognition API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ========== Global State ==========
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "message": ""
}

# Storage for registration data
REGISTRATION_DATA_DIR = "registration_data"
os.makedirs(REGISTRATION_DATA_DIR, exist_ok=True)


# ========== Models ==========
class PredictionResult(BaseModel):
    gesture: Optional[str]
    confidence: float
    status: str
    velocity: float
    adaptive_threshold: float


class RegistrationRequest(BaseModel):
    gesture_name: str
    num_samples: int = 10


class TrainingRequest(BaseModel):
    learning_rate: float = 0.001
    epochs: int = 50
    include_old_data: bool = True


class GestureInfo(BaseModel):
    name: str
    sample_count: int
    registered_at: str


# ========== Data Augmentation ==========
class DataAugmentor:
    '''Applies data augmentation to increase dataset diversity'''

    @staticmethod
    def augment_landmarks(landmarks_array, num_augmentations=3):
        '''Generate augmented versions of landmark data'''
        augmented_data = [landmarks_array]  # Original

        for _ in range(num_augmentations):
            augmented = landmarks_array.copy()

            # Add small random noise (simulating slight hand movement)
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise

            # Random scaling (simulating zoom)
            scale = np.random.uniform(0.95, 1.05)
            augmented *= scale

            # Random rotation (2D rotation around wrist)
            angle = np.random.uniform(-15, 15) * np.pi / 180
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            augmented = augmented.reshape(-1, 3) @ rotation_matrix.T
            augmented = augmented.flatten()

            augmented_data.append(augmented)

        return augmented_data


# ========== Gesture Engine (Enhanced) ==========
class EnhancedGestureEngine:
    def __init__(self):
        # Load model files if available
        self.demo_mode = False
        self.model = None
        self.label_encoder = None
        self.X_min = None
        self.X_max = None

        try:
            if load_model is None:
                raise RuntimeError("TensorFlow not available")
            self.model = load_model('asl_model.keras')
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            norm_params = np.load('normalization_params.npy')
            self.X_min, self.X_max = norm_params[0], norm_params[1]
        except Exception as e:
            print(f"[WARN] Running in DEMO mode: {e}")
            self.demo_mode = True

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands_detect = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.hands_refine = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )

        # Transition detection
        self.LANDMARK_HISTORY = deque(maxlen=5)
        self.VELOCITY_THRESHOLD = 0.15
        self.BASE_CONFIDENCE = 75.0
        self.k = 200
        self.PREDICTION_BUFFER = deque(maxlen=5)
        self.CONFIDENCE_BUFFER = deque(maxlen=5)

    def extract_landmarks(self, frame_bgr):
        '''Extract hand landmarks from a frame'''
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res_detect = self.hands_detect.process(rgb_full)

        if not res_detect.multi_hand_landmarks:
            return None

        hand_rough = res_detect.multi_hand_landmarks[0]
        x1, y1, x2, y2 = self.get_hand_bbox(hand_rough, frame_bgr.shape)
        hand_crop = frame_bgr[y1:y2, x1:x2]

        if hand_crop.size == 0:
            return None

        # Resize and refine
        crop_h, crop_w = hand_crop.shape[:2]
        target = 640
        if crop_h > crop_w:
            new_h, new_w = target, int(crop_w * (target / crop_h))
        else:
            new_w, new_h = target, int(crop_h * (target / crop_w))

        hand_resized = cv2.resize(hand_crop, (new_w, new_h))
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = hand_resized

        rgb_crop = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res_refine = self.hands_refine.process(rgb_crop)

        hand_lm = res_refine.multi_hand_landmarks[0] if res_refine.multi_hand_landmarks else hand_rough

        # Extract landmarks relative to wrist
        landmark_list = []
        wrist = hand_lm.landmark[0]
        for landmark in hand_lm.landmark:
            landmark_list.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        return np.array(landmark_list)

    @staticmethod
    def get_hand_bbox(hand_landmarks, frame_shape, padding=0.4):
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

    def predict_from_frame(self, frame_bgr):
        landmarks = self.extract_landmarks(frame_bgr)
        if landmarks is None:
            return None, 0.0, "no_hand_detected", 0.0, self.BASE_CONFIDENCE

        if self.demo_mode:
            # Simulate S, O, S for testing without a model
            demo_gestures = ["S", "O", "S"]
            import random
            label = random.choice(demo_gestures)
            conf = 95.0
            return label, conf, "stable", 0.0, self.BASE_CONFIDENCE

        # Predict
        arr = landmarks.reshape(1, -1)
        arr_norm = (arr - self.X_min) / (self.X_max - self.X_min)
        arr_cnn = arr_norm.reshape(1, 21, 3, 1)

        pred = self.model.predict(arr_cnn, verbose=0)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred) * 100)
        label = str(self.label_encoder.classes_[idx])

        return label, conf, "stable", 0.0, self.BASE_CONFIDENCE


engine = EnhancedGestureEngine()


# ========== NEW: Emergency Function ==========
def call_police():
    """
    Placeholder function to simulate calling emergency services.
    In a real application, this would integrate with a service like Twilio
    to send an SMS or make an automated call.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! EMERGENCY ALERT: SOS DETECTED at {timestamp} !!!")
    print(f"!!! SIMULATING CALL TO 911...                  !!!")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# ========== Registration Endpoints ==========
@app.post("/register-gesture")
async def register_gesture(
        gesture_name: str = Form(...),
        files: List[UploadFile] = File(...)
):
    '''Register a new custom gesture by uploading multiple sample images'''

    if not gesture_name or len(files) == 0:
        raise HTTPException(status_code=400, detail="Gesture name and files required")

    # Create directory for this gesture
    gesture_dir = os.path.join(REGISTRATION_DATA_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    landmarks_list = []
    saved_count = 0

    for file in files:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Extract landmarks
        landmarks = engine.extract_landmarks(frame)
        if landmarks is not None:
            landmarks_list.append(landmarks)
            saved_count += 1

    if saved_count == 0:
        raise HTTPException(status_code=400, detail="No valid hand landmarks detected")

    # Apply data augmentation
    augmentor = DataAugmentor()
    augmented_landmarks = []
    for lm in landmarks_list:
        augmented = augmentor.augment_landmarks(lm, num_augmentations=3)
        augmented_landmarks.extend(augmented)

    # Save landmarks to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    landmarks_file = os.path.join(gesture_dir, f"landmarks_{timestamp}.npy")
    np.save(landmarks_file, np.array(augmented_landmarks))

    # Save metadata
    metadata = {
        "gesture_name": gesture_name,
        "original_samples": saved_count,
        "augmented_samples": len(augmented_landmarks),
        "timestamp": timestamp
    }
    metadata_file = os.path.join(gesture_dir, f"metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "gesture_name": gesture_name,
        "original_samples": saved_count,
        "augmented_samples": len(augmented_landmarks),
        "message": f"Gesture '{gesture_name}' registered successfully with {len(augmented_landmarks)} total samples"
    }


@app.get("/gestures")
async def list_gestures():
    '''List all registered gestures'''
    gestures = []

    if not os.path.exists(REGISTRATION_DATA_DIR):
        return {"gestures": []}

    for gesture_name in os.listdir(REGISTRATION_DATA_DIR):
        gesture_dir = os.path.join(REGISTRATION_DATA_DIR, gesture_name)
        if not os.path.isdir(gesture_dir):
            continue

        # Count samples
        sample_count = 0
        latest_timestamp = None

        for file in os.listdir(gesture_dir):
            if file.startswith("landmarks_"):
                landmarks = np.load(os.path.join(gesture_dir, file))
                sample_count += len(landmarks)
            elif file.startswith("metadata_"):
                with open(os.path.join(gesture_dir, file), 'r') as f:
                    meta = json.load(f)
                    latest_timestamp = meta.get("timestamp", "unknown")

        gestures.append({
            "name": gesture_name,
            "sample_count": sample_count,
            "registered_at": latest_timestamp or "unknown"
        })

    return {"gestures": gestures}


# ========== Training Endpoints ==========
def train_model_background(learning_rate, epochs, include_old_data):
    '''Background task for incremental training'''
    global training_status

    try:
        training_status["is_training"] = True
        training_status["status"] = "loading_data"
        training_status["progress"] = 10

        # Load all registered gesture data
        X_train = []
        y_train = []

        for gesture_name in os.listdir(REGISTRATION_DATA_DIR):
            gesture_dir = os.path.join(REGISTRATION_DATA_DIR, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue

            for file in os.listdir(gesture_dir):
                if file.startswith("landmarks_"):
                    landmarks = np.load(os.path.join(gesture_dir, file))
                    X_train.extend(landmarks)
                    y_train.extend([gesture_name] * len(landmarks))

        if len(X_train) == 0:
            training_status["status"] = "error"
            training_status["message"] = "No training data available"
            return

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        training_status["status"] = "preparing_model"
        training_status["progress"] = 30

        # Load existing model or create new one
        if engine.model is not None and not engine.demo_mode:
            model = engine.model
        else:
            # Create new model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout

            model = Sequential([
                Dense(128, activation='relu', input_shape=(63,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(set(y_train)), activation='softmax')
            ])

        # Update label encoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)

        # Normalize
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        X_normalized = (X_train - X_min) / (X_max - X_min + 1e-8)
        X_cnn = X_normalized.reshape(-1, 21, 3, 1)

        training_status["status"] = "training"
        training_status["progress"] = 50

        # Compile with smaller learning rate for fine-tuning
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train
        model.fit(X_cnn, y_encoded, epochs=epochs, batch_size=16, verbose=0)

        training_status["status"] = "saving"
        training_status["progress"] = 90

        # Save updated model
        model.save('asl_model_updated.keras')
        with open('label_encoder_updated.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        np.save('normalization_params_updated.npy', np.array([X_min, X_max]))

        # Reload model in engine
        engine.model = model
        engine.label_encoder = label_encoder
        engine.X_min = X_min
        engine.X_max = X_max
        engine.demo_mode = False

        training_status["is_training"] = False
        training_status["status"] = "completed"
        training_status["progress"] = 100
        training_status["message"] = f"Model trained successfully with {len(set(y_train))} gestures"

    except Exception as e:
        training_status["is_training"] = False
        training_status["status"] = "error"
        training_status["message"] = str(e)


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    '''Trigger incremental training on registered gestures'''

    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Reset status
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["status"] = "starting"
    training_status["message"] = ""

    # Start background training
    background_tasks.add_task(
        train_model_background,
        request.learning_rate,
        request.epochs,
        request.include_old_data
    )

    return {
        "status": "training_started",
        "message": "Model training initiated in background"
    }


@app.get("/training-status")
async def get_training_status():
    '''Get current training status'''
    return training_status


# ========== Existing Endpoints ==========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "demo_mode": engine.demo_mode,
        "model_loaded": (not engine.demo_mode),
        "version": "2.0.0"
    }


@app.post("/predict-image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        label, conf, status, vel, thr = engine.predict_from_frame(frame)
        return PredictionResult(
            gesture=label,
            confidence=conf,
            status=status,
            velocity=vel,
            adaptive_threshold=thr
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    # MODIFIED: Per-client sequence tracker for SOS
    sos_sequence = deque(maxlen=3)

    try:
        while True:
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"type": "error", "message": "Invalid frame"})
                continue

            label, conf, status, vel, thr = engine.predict_from_frame(frame)

            # MODIFIED: Check for SOS sequence
            if label in ["S", "O"] and conf >= 85.0:
                # Add to sequence only if it's a new gesture
                if len(sos_sequence) == 0 or label != sos_sequence[-1]:
                    sos_sequence.append(label)

            # Check if the sequence is "S", "O", "S"
            if list(sos_sequence) == ["S", "O", "S"]:
                call_police()  # Call the emergency function
                # Send a special alert to the client
                await ws.send_json({
                    "type": "sos_alert",
                    "message": "SOS Detected. Emergency services notified."
                })
                sos_sequence.clear()  # Reset the sequence
                continue  # Skip sending the regular prediction for this frame

            await ws.send_json({
                "type": "prediction",
                "gesture": label,
                "confidence": conf,
                "status": status,
                "velocity": vel,
                "adaptive_threshold": thr
            })
    except WebSocketDisconnect:
        print("Client disconnected.")
        return
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await ws.close(code=1011, reason=str(e))
