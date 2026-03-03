"""
Emotion Detection Web Application - Updated Version
Flask-based web UI for emotion detection with image, video, and webcam support
"""

import os
import io
import base64
import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from PIL import Image

# Try to import PyTorch, but make it optional
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None
    models = None

# Try to import sklearn (for sklearn-based model)
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception as e:
    print(f"Warning: sklearn not available: {e}")
    SKLEARN_AVAILABLE = False
    joblib = None
    StandardScaler = None

# Model type tracker
MODEL_TYPE = None  # 'pytorch' or 'sklearn'
SKLEARN_MODEL_DATA = None


app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-detection-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Global variables
MODEL = None
DEVICE = None
EMOTION_CLASSES = [
    'angry', 'confused', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'shy', 'surprise'
]
NUM_CLASSES = len(EMOTION_CLASSES)
IMAGE_SIZE = 224

# Prediction history
prediction_history = []

# Training status
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'current_acc': 0.0,
    'best_acc': 0.0
}

# =====================================================
# MODEL DEFINITION
# =====================================================

def create_model(num_classes):
    """Create emotion detection model using EfficientNet-B0"""
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(weights=None)
    
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def load_trained_model():
    """Load the trained model"""
    global MODEL, DEVICE
    
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Cannot load model.")
        return False
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    MODEL = create_model(NUM_CLASSES)
    
    model_path = 'best_emotion_model.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            if 'emotion_classes' in checkpoint:
                global EMOTION_CLASSES
                EMOTION_CLASSES = checkpoint['emotion_classes']
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try with weights_only=True
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
                MODEL.load_state_dict(checkpoint)
                print(f"Model loaded (weights only) from {model_path}")
            except Exception as e2:
                print(f"Could not load model: {e2}")
                MODEL = None
    else:
        print("No trained model found. Please train the model first.")
        MODEL = None
    
    if MODEL:
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
    
    return MODEL is not None


def preprocess_image(image):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_emotion(image_tensor):
    """Predict emotion from image tensor"""
    if MODEL is None:
        return None, 0.0, None
    
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    emotion = EMOTION_CLASSES[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    return emotion, confidence_score, all_probs


def load_sklearn_model():
    """Load sklearn model"""
    global SKLEARN_MODEL_DATA, MODEL_TYPE
    
    if not SKLEARN_AVAILABLE:
        print("sklearn is not available.")
        return False
    
    model_path = 'best_emotion_model.pth'
    if os.path.exists(model_path):
        try:
            SKLEARN_MODEL_DATA = joblib.load(model_path)
            MODEL_TYPE = 'sklearn'
            print(f"sklearn model loaded from {model_path}")
            print(f"Model type: {SKLEARN_MODEL_DATA.get('model_type', 'Unknown')}")
            print(f"Accuracy: {SKLEARN_MODEL_DATA.get('accuracy', 0)*100:.2f}%")
            return True
        except Exception as e:
            print(f"Error loading sklearn model: {e}")
            SKLEARN_MODEL_DATA = None
            return False
    return False


def preprocess_for_sklearn(image):
    """Preprocess image for sklearn model"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    img_size = SKLEARN_MODEL_DATA.get('image_size', 32) if SKLEARN_MODEL_DATA else 32
    resized = cv2.resize(gray, (img_size, img_size))
    features = resized.flatten()
    return features


def predict_emotion_sklearn(image):
    """Predict emotion using sklearn model"""
    if SKLEARN_MODEL_DATA is None:
        return None, 0.0, None
    
    try:
        features = preprocess_for_sklearn(image)
        features = features.reshape(1, -1)
        
        if 'scaler' in SKLEARN_MODEL_DATA:
            features = SKLEARN_MODEL_DATA['scaler'].transform(features)
        
        model = SKLEARN_MODEL_DATA['model']
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs)
        else:
            confidence = 1.0
            probs = np.zeros(len(EMOTION_CLASSES))
            probs[prediction] = confidence
        
        if 'label_encoder' in SKLEARN_MODEL_DATA:
            emotion = SKLEARN_MODEL_DATA['label_encoder'].inverse_transform([prediction])[0]
        else:
            emotion = EMOTION_CLASSES[prediction]
        
        return emotion, confidence, probs
        
    except Exception as e:
        print(f"sklearn prediction error: {e}")
        return None, 0.0, None


def is_model_loaded():
    """Check if any model is loaded"""
    return MODEL is not None or SKLEARN_MODEL_DATA is not None


def get_prediction(image):
    """Get prediction from whichever model is loaded"""
    if MODEL is not None:
        image_tensor = preprocess_image(image)
        return predict_emotion(image_tensor)
    elif SKLEARN_MODEL_DATA is not None:
        return predict_emotion_sklearn(image)
    return None, 0.0, None


def detect_faces(frame, face_cascade):
    """Detect faces in a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', 
                           emotion_classes=EMOTION_CLASSES,
                           model_loaded=is_model_loaded())


@app.route('/upload')
def upload_page():
    """Image/Video upload page"""
    return render_template('upload.html', emotion_classes=EMOTION_CLASSES)


@app.route('/webcam')
def webcam_page():
    """Webcam detection page"""
    return render_template('webcam.html', emotion_classes=EMOTION_CLASSES)


@app.route('/training')
def training_page():
    """Training dashboard page"""
    return render_template('training.html', 
                           emotion_classes=EMOTION_CLASSES,
                           training_status=training_status)


@app.route('/results')
def results_page():
    """Results and history page"""
    return render_template('results.html', 
                           history=prediction_history[-50:][::-1],
                           emotion_classes=EMOTION_CLASSES)


# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/model/status')
def model_status():
    """Get model loading status"""
    return jsonify({
        'loaded': is_model_loaded(),
        'model_type': MODEL_TYPE,
        'device': str(DEVICE) if DEVICE else None,
        'emotion_classes': EMOTION_CLASSES
    })


@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """Predict emotion from uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not is_model_loaded():
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        image = Image.open(filepath).convert('RGB')
        emotion, confidence, all_probs = get_prediction(image)
        
        result = {
            'success': True,
            'emotion': emotion,
            'confidence': float(confidence * 100),
            'all_emotions': {EMOTION_CLASSES[i]: float(all_probs[i] * 100) 
                           for i in range(len(EMOTION_CLASSES))},
            'image_url': f"/static/uploads/{saved_filename}",
            'timestamp': datetime.now().isoformat()
        }
        
        prediction_history.append({
            'type': 'image',
            'emotion': emotion,
            'confidence': float(confidence * 100),
            'filename': saved_filename,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/video', methods=['POST'])
def predict_video():
    """Predict emotions from uploaded video"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not is_model_loaded():
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        output_filename = f"result_{saved_filename}"
        output_path = os.path.join('results', output_filename)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        cap = cv2.VideoCapture(filepath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        emotion_counts = {e: 0 for e in EMOTION_CLASSES}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 5 == 0:
                faces = detect_faces(frame, face_cascade)
                
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        emotion, confidence, _ = get_prediction(face_roi)
                        
                        if emotion:
                            emotion_counts[emotion] += 1
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            label = f"{emotion}: {confidence*100:.1f}%"
                            cv2.putText(frame, label, (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        total_detections = sum(emotion_counts.values())
        if total_detections > 0:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            dominant_percentage = emotion_counts[dominant_emotion] / total_detections * 100
        else:
            dominant_emotion = "No faces detected"
            dominant_percentage = 0
        
        prediction_history.append({
            'type': 'video',
            'emotion': dominant_emotion,
            'confidence': float(dominant_percentage),
            'filename': saved_filename,
            'result_file': output_filename,
            'emotion_counts': emotion_counts,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'dominant_emotion': dominant_emotion,
            'confidence': float(dominant_percentage),
            'emotion_counts': emotion_counts,
            'total_frames': frame_count,
            'result_video': f"/results/{output_filename}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/webcam/feed')
def webcam_feed():
    """Generate webcam feed with emotion detection"""
    def generate_frames():
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Please allow camera access", (170, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = detect_faces(frame, face_cascade)
                
                if is_model_loaded():
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            emotion, confidence, _ = get_prediction(face_roi)
                            
                            if emotion:
                                color = (0, 255, 0)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label = f"{emotion}: {confidence*100:.1f}%"
                                cv2.putText(frame, label, (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "Model not loaded - Training required", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Webcam error: {e}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Error: {str(e)}", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            if cap is not None:
                cap.release()
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/statistics')
def get_statistics():
    """Get prediction statistics"""
    if not prediction_history:
        return jsonify({
            'total_predictions': 0,
            'emotion_distribution': {e: 0 for e in EMOTION_CLASSES},
            'recent_predictions': []
        })
    
    emotion_dist = {e: 0 for e in EMOTION_CLASSES}
    for pred in prediction_history:
        if pred['emotion'] in emotion_dist:
            emotion_dist[pred['emotion']] += 1
    
    return jsonify({
        'total_predictions': len(prediction_history),
        'emotion_distribution': emotion_dist,
        'recent_predictions': prediction_history[-10:][::-1]
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    def training_thread():
        import subprocess
        subprocess.run(['python', 'Train_model_optimized.py'], shell=True)
        training_status['is_training'] = False
    
    training_status['is_training'] = True
    thread = threading.Thread(target=training_thread)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/training/status')
def training_status_api():
    """Get training status"""
    return jsonify(training_status)


@app.route('/api/history')
def get_history():
    """Get prediction history"""
    return jsonify(prediction_history[-100:][::-1])


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global prediction_history
    prediction_history = []
    return jsonify({'success': True})


# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    print("Loading model...")
    
    model_loaded = load_trained_model()
    
    if not model_loaded and SKLEARN_AVAILABLE:
        print("Trying sklearn model...")
        model_loaded = load_sklearn_model()
    
    if not model_loaded:
        print("WARNING: No model loaded. Please train the model.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
