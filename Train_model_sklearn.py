"""
Emotion Detection Model Training Script - scikit-learn Version
Alternative to PyTorch for systems with compatibility issues
Uses HOG features + SVM/RandomForest for emotion classification
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================

# Dataset path
DATA_DIR = "EDFF_dataset"

# Training settings
IMAGE_SIZE = 64  # Smaller size for faster processing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Emotion classes
EMOTION_CLASSES = [
    'angry', 'confused', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'shy', 'surprise'
]

NUM_CLASSES = len(EMOTION_CLASSES)
print(f"Number of emotion classes: {NUM_CLASSES}")
print(f"Emotion classes: {EMOTION_CLASSES}")

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_hog_features(image, size=(64, 64)):
    """Extract HOG (Histogram of Oriented Gradients) features"""
    # Resize image
    img = cv2.resize(image, size)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute HOG features
    win_size = (size[0], size[1])
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(gray)
    
    return features.flatten()

def extract_lbp_features(image, size=(64, 64)):
    """Extract LBP (Local Binary Pattern) features - simplified version"""
    # Resize image
    img = cv2.resize(image, size)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Simple LBP-like feature: compare each pixel with neighbors
    features = []
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary_val = 0
            binary_val |= (gray[i-1, j-1] > center) << 7
            binary_val |= (gray[i-1, j] > center) << 6
            binary_val |= (gray[i-1, j+1] > center) << 5
            binary_val |= (gray[i, j+1] > center) << 4
            binary_val |= (gray[i+1, j+1] > center) << 3
            binary_val |= (gray[i+1, j] > center) << 2
            binary_val |= (gray[i+1, j-1] > center) << 1
            binary_val |= (gray[i, j-1] > center) << 0
            features.append(binary_val)
    
    # Create histogram
    hist, _ = np.histogram(features, bins=256, range=(0, 256))
    return hist

def extract_color_histogram(image, size=(64, 64)):
    """Extract color histogram features"""
    img = cv2.resize(image, size)
    
    features = []
    for i in range(3):  # BGR channels
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    return np.array(features)

def extract_all_features(image):
    """Combine all feature extractors"""
    # Resize image first
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Extract features
    hog_feat = extract_hog_features(img)
    color_feat = extract_color_histogram(img)
    
    # Combine features
    features = np.concatenate([hog_feat, color_feat])
    
    return features

# =====================================================
# DATA LOADING
# =====================================================

def load_dataset():
    """Load and preprocess the emotion dataset"""
    print("\n[1/5] Loading dataset...")
    
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    
    images = []
    labels = []
    total_loaded = 0
    
    for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = data_path / emotion
        if not emotion_dir.exists():
            print(f"  Warning: {emotion} directory not found")
            continue
        
        # Get all jpg files
        img_files = list(emotion_dir.glob("*.jpg"))
        
        for img_path in img_files:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Extract features
                features = extract_all_features(img)
                images.append(features)
                labels.append(emotion)
                total_loaded += 1
                
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
        
        print(f"  {emotion}: {len([l for l in labels if l == emotion])} images")
    
    print(f"  Total images loaded: {total_loaded}")
    
    if total_loaded == 0:
        raise ValueError("No images were loaded from the dataset")
    
    return np.array(images), np.array(labels)

# =====================================================
# MODEL TRAINING
# =====================================================

def train_model(X_train, y_train, X_val, y_val):
    """Train emotion detection model"""
    print("\n[2/5] Training model...")
    
    # Normalize features
    print("  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Try multiple models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=RANDOM_STATE
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        # Use scaled data for SVM, original for tree-based
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        elapsed = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy,
            'time': elapsed
        }
        
        print(f"    Accuracy: {accuracy*100:.2f}%")
        print(f"    Time: {elapsed:.2f}s")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\n  Best model: {best_name} with {best_accuracy*100:.2f}% accuracy")
    
    return best_model, scaler, best_name, results

# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def main():
    print("=" * 70)
    print("Emotion Detection Model Training (scikit-learn version)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load dataset
    X, y = load_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    print("\n[3/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    # Train model
    model, scaler, best_name, results = train_model(X_train, y_train, X_val, y_val)
    
    # Save model
    print("\n[4/5] Saving model...")
    model_path = "best_emotion_model.pth"
    
    # Save model with all necessary components
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'emotion_classes': EMOTION_CLASSES,
        'image_size': IMAGE_SIZE,
        'model_type': best_name,
        'accuracy': results[best_name]['accuracy']
    }
    
    joblib.dump(model_data, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Print summary
    total_time = time.time() - start_time
    
    print("\n[5/5] Training Summary")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {results[best_name]['accuracy']*100:.2f}%")
    print(f"Model type: {best_name}")
    print(f"Model saved to: {model_path}")
    print("=" * 70)
    
    # Save training history
    history = {
        'model_type': best_name,
        'accuracy': results[best_name]['accuracy'],
        'training_time': total_time,
        'all_results': {k: {'accuracy': v['accuracy'], 'time': v['time']} for k, v in results.items()}
    }
    
    with open("training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training complete!")
    
    return model_data

if __name__ == "__main__":
    main()
