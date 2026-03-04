"""
Emotion Detection Model Training - Advanced scikit-learn Version
Optimized version with better feature extraction and ensemble methods
Designed for systems where PyTorch has compatibility issues
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================

# Dataset path
DATA_DIR = "EDFF_dataset"

# Training settings - Optimized
IMAGE_SIZE = 64  # Slightly larger for better features
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Resume training
CHECKPOINT_PATH = "best_emotion_model.pth"
RESUME_TRAINING = False  # Set to True to try loading previous model

# Emotion classes
EMOTION_CLASSES = [
    'angry', 'confused', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'shy', 'surprise'
]

NUM_CLASSES = len(EMOTION_CLASSES)
print(f"Number of emotion classes: {NUM_CLASSES}")
print(f"Emotion classes: {EMOTION_CLASSES}")

# =====================================================
# ADVANCED FEATURE EXTRACTION
# =====================================================

def extract_hog_features(image, size=(64, 64)):
    """Extract HOG (Histogram of Oriented Gradients) features"""
    img = cv2.resize(image, size)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute HOG features with optimized parameters
    win_size = (size[0], size[1])
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(gray)
    
    return features.flatten()


def extract_lbp_histogram(image, size=(64, 64)):
    """Extract Local Binary Pattern histogram features"""
    img = cv2.resize(image, size)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute LBP
    lbp = np.zeros_like(gray)
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
            lbp[i, j] = binary_val
    
    # Create histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)  # Normalize
    
    return hist


def extract_color_histogram(image, size=(64, 64)):
    """Extract color histogram features from multiple color spaces"""
    img = cv2.resize(image, size)
    
    features = []
    
    # BGR histogram
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.extend(hist)
    
    # HSV histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.extend(hist)
    
    # LAB histogram
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [16], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            features.extend(hist)
    except:
        features.extend([0] * 48)
    
    return np.array(features)


def extract_gabor_features(image, size=(64, 64)):
    """Extract Gabor filter features"""
    img = cv2.resize(image, size)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    gray = gray.astype(np.float32) / 255.0
    
    # Apply Gabor filters at different orientations
    features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 2]:
            for freq in [0.1, 0.2]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0/freq, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(filtered.mean())
                features.append(filtered.std())
    
    return np.array(features)


def extract_edge_histogram(image, size=(64, 64)):
    """Extract edge histogram features"""
    img = cv2.resize(image, size)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Create histograms
    edge_hist, _ = np.histogram(edges.ravel(), bins=16, range=(0, 256))
    mag_hist, _ = np.histogram(magnitude.ravel(), bins=16)
    
    edge_hist = edge_hist.astype(float) / (edge_hist.sum() + 1e-7)
    mag_hist = mag_hist.astype(float) / (mag_hist.sum() + 1e-7)
    
    return np.concatenate([edge_hist, mag_hist])


def extract_all_features(image):
    """Combine all feature extractors"""
    # Resize image first
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Extract all features
    hog_feat = extract_hog_features(img)
    lbp_feat = extract_lbp_histogram(img)
    color_feat = extract_color_histogram(img)
    gabor_feat = extract_gabor_features(img)
    edge_feat = extract_edge_histogram(img)
    
    # Also add resized grayscale pixels as features (spatial information)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (16, 16))
    pixel_feat = small_gray.flatten() / 255.0
    
    # Combine all features
    features = np.concatenate([hog_feat, lbp_feat, color_feat, gabor_feat, edge_feat, pixel_feat])
    
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
        
        count = len([l for l in labels if l == emotion])
        print(f"  {emotion}: {count} images")
    
    print(f"  Total images loaded: {total_loaded}")
    
    if total_loaded == 0:
        raise ValueError("No images were loaded from the dataset")
    
    return np.array(images), np.array(labels)


# =====================================================
# MODEL TRAINING - ENSEMBLE APPROACH
# =====================================================

def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train ensemble of multiple models"""
    print("\n[2/5] Training ensemble models...")
    
    # Normalize features
    print("  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define base models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=40,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=40,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=RANDOM_STATE
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
    }
    
    trained_models = {}
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        # Use scaled data for SVM and MLP, original for tree-based
        if name in ['SVM', 'MLP']:
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
        
        trained_models[name] = model
        print(f"    Accuracy: {accuracy*100:.2f}%")
        print(f"    Time: {elapsed:.2f}s")
    
    # Create voting ensemble
    print("\n  Creating voting ensemble...")
    
    # For voting, we need all models trained on scaled data
    ensemble_models = [
        ('rf', trained_models['RandomForest']),
        ('et', trained_models['ExtraTrees']),
    ]
    
    # Add SVM and MLP which were trained on scaled data
    if 'SVM' in trained_models:
        ensemble_models.append(('svm', trained_models['SVM']))
    if 'MLP' in trained_models:
        ensemble_models.append(('mlp', trained_models['MLP']))
    
    # Soft voting ensemble
    voting_clf = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',
        n_jobs=-1
    )
    
    print("  Training voting classifier...")
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_ensemble = voting_clf.predict(X_val_scaled)
    ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
    
    print(f"  Ensemble accuracy: {ensemble_accuracy*100:.2f}%")
    results['Ensemble'] = {'accuracy': ensemble_accuracy, 'time': 0}
    
    # Find best model
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_name]['accuracy']
    
    if best_name == 'Ensemble':
        best_model = voting_clf
        best_scaler = scaler
    elif best_name in ['SVM', 'MLP']:
        best_model = trained_models[best_name]
        best_scaler = scaler
    else:
        best_model = trained_models[best_name]
        best_scaler = None
    
    print(f"\n  Best model: {best_name} with {best_accuracy*100:.2f}% accuracy")
    
    return best_model, best_scaler, best_name, results, voting_clf, scaler


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def main():
    print("=" * 70)
    print("Emotion Detection Model Training (Advanced sklearn version)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load dataset
    X, y = load_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data with stratification
    print("\n[3/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    # Train model
    model, scaler, best_name, results, ensemble_model, ensemble_scaler = train_ensemble_model(
        X_train, y_train, X_val, y_val
    )
    
    # Determine which model to save (ensemble or best single)
    # Save the ensemble if it's better
    if results['Ensemble']['accuracy'] >= results[best_name]['accuracy']:
        final_model = ensemble_model
        final_scaler = ensemble_scaler
        final_name = 'Ensemble'
        final_accuracy = results['Ensemble']['accuracy']
    else:
        final_model = model
        final_scaler = scaler if best_name in ['SVM', 'MLP'] else None
        final_name = best_name
        final_accuracy = results[best_name]['accuracy']
    
    # Save model
    print("\n[4/5] Saving model...")
    model_path = "best_emotion_model.pth"
    
    # Save model with all necessary components
    model_data = {
        'model': final_model,
        'scaler': final_scaler,
        'label_encoder': le,
        'emotion_classes': EMOTION_CLASSES,
        'image_size': IMAGE_SIZE,
        'model_type': final_name,
        'accuracy': final_accuracy,
        'all_results': {k: {'accuracy': v['accuracy'], 'time': v['time']} for k, v in results.items()}
    }
    
    joblib.dump(model_data, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Print summary
    total_time = time.time() - start_time
    
    print("\n[5/5] Training Summary")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {final_accuracy*100:.2f}%")
    print(f"Model type: {final_name}")
    print(f"Model saved to: {model_path}")
    print("\n  All model results:")
    for name, result in results.items():
        print(f"    {name}: {result['accuracy']*100:.2f}%")
    print("=" * 70)
    
    # Save training history
    history = {
        'model_type': final_name,
        'accuracy': final_accuracy,
        'training_time': total_time,
        'all_results': {k: {'accuracy': v['accuracy'], 'time': v['time']} for k, v in results.items()}
    }
    
    with open("training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✅ Training complete!")
    
    return model_data


if __name__ == "__main__":
    main()

