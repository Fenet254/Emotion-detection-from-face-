"""
Emotion Detection Model Training - Fast Version
Uses simple pixel-based features with RandomForest for quick training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "EDFF_dataset"
IMAGE_SIZE = 32  # Very small for fast processing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Emotion classes
EMOTION_CLASSES = ['angry', 'confused', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

print("=" * 60)
print("Emotion Detection Model Training (Fast Version)")
print("=" * 60)

def load_dataset():
    """Load dataset with simple features"""
    print("\n[1/5] Loading dataset...")
    
    data_path = Path(DATA_DIR)
    images = []
    labels = []
    
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = data_path / emotion
        if not emotion_dir.exists():
            continue
        
        img_files = list(emotion_dir.glob("*.jpg"))[:500]  # Limit per class for speed
        
        for img_path in img_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize and convert to grayscale
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Flatten pixel values as features
                features = gray.flatten()
                images.append(features)
                labels.append(emotion)
                
            except Exception as e:
                pass
    
    print(f"  Total images: {len(images)}")
    return np.array(images), np.array(labels)

def main():
    start_time = time.time()
    
    # Load data
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Error: No images loaded!")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    print("\n[2/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"  Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Train
    print("\n[3/5] Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"  Validation Accuracy: {accuracy*100:.2f}%")
    
    # Save
    print("\n[4/5] Saving model...")
    model_data = {
        'model': model,
        'label_encoder': le,
        'emotion_classes': EMOTION_CLASSES,
        'image_size': IMAGE_SIZE,
        'model_type': 'RandomForest',
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, "best_emotion_model.pth")
    print("  Model saved to: best_emotion_model.pth")
    
    # Summary
    total_time = time.time() - start_time
    print("\n[5/5] Summary")
    print("=" * 60)
    print(f"Time: {total_time:.2f}s")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("=" * 60)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
