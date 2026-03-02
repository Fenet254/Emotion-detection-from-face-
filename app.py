"""
Emotion Detection Web Application
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

# =====================================================
# CONFIGURATION
# =====================================================

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
                           model_loaded=MODEL is not None)


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
        'loaded': MODEL is not None,
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
    
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Preprocess and predict
        image = Image.open(filepath).convert('RGB')
        image_tensor = preprocess_image(image)
        emotion, confidence, all_probs = predict_emotion(image_tensor)
        
        # Prepare response
        result = {
            'success': True,
            'emotion': emotion,
            'confidence': float(confidence * 100),
            'all_emotions': {EMOTION_CLASSES[i]: float(all_probs[i] * 100) 
                           for i in range(len(EMOTION_CLASSES))},
            'image_url': f"/static/uploads/{saved_filename}",
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
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
    
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400
    
    try:
        # Save uploaded video
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Process video
        output_filename = f"result_{saved_filename}"
        output_path = os.path.join('results', output_filename)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        emotion_counts = {e: 0 for e in EMOTION_CLASSES}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and predict every 5 frames for speed
            if frame_count % 5 == 0:
                faces = detect_faces(frame, face_cascade)
                
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        image_tensor = preprocess_image(face_roi)
                        emotion, confidence, _ = predict_emotion(image_tensor)
                        
                        if emotion:
                            emotion_counts[emotion] += 1
                            
                            # Draw bounding box
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            label = f"{emotion}: {confidence*100:.1f}%"
                            cv2.putText(frame, label, (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Find dominant emotion
        total_detections = sum(emotion_counts.values())
        if total_detections > 0:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            dominant_percentage = emotion_counts[dominant_emotion] / total_detections * 100
        else:
            dominant_emotion = "No faces detected"
            dominant_percentage = 0
        
        # Add to history
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
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    def generate_frames():
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = detect_faces(frame, face_cascade)
            
            # Process each face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    image_tensor = preprocess_image(face_roi)
                    emotion, confidence, _ = predict_emotion(image_tensor)
                    
                    if emotion:
                        # Draw bounding box
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        label = f"{emotion}: {confidence*100:.1f}%"
                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
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
    # Try to load model on startup
    print("Loading model...")
    load_trained_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
