"""
Emotion Detection Inference Script for Video
Loads a trained model and performs emotion detection on video streams
Uses face detection to find faces and classify emotions
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import argparse
import time


# Configuration
MODEL_PATH = "best_emotion_model.pth"
IMAGE_SIZE = 224

# Emotion classes (must match training)
EMOTION_CLASSES = [
    'angry', 'confused', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'shy', 'surprise'
]

NUM_CLASSES = len(EMOTION_CLASSES)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_model(num_classes):
    """Create emotion detection model using ResNet18"""
    model = models.resnet18(weights=None)
    
    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    model = create_model(NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def preprocess_face(face_image):
    """Preprocess face image for inference"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL if needed
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    else:
        face_image = face_image.convert('RGB')
    
    image_tensor = transform(face_image).unsqueeze(0)
    return image_tensor


def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    try:
        image_tensor = preprocess_face(face_image)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = EMOTION_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        return emotion, confidence_score, probabilities[0].cpu().numpy()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Unknown", 0.0, None


def detect_faces(frame, face_cascade):
    """Detect faces in a frame using Haar cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def draw_emotion_label(frame, x, y, w, h, emotion, confidence):
    """Draw emotion label with bounding box"""
    # Draw bounding box
    color = (0, 255, 0)  # Green
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    label = f"{emotion}: {confidence:.1f}%"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y - 10, label_size[1])
    
    cv2.rectangle(
        frame, 
        (x, label_y - label_size[1] - 5), 
        (x + label_size[0], label_y + 5), 
        color, 
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame, 
        label, 
        (x, label_y - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 0, 0), 
        1
    )


def process_video(model, video_path, output_path=None, show_display=True):
    """Process video file for emotion detection"""
    # Load OpenCV's pre-trained face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print("Warning: Could not load face cascade classifier")
        return
    
    # Open video capture
    if video_path == '0':
        cap = cv2.VideoCapture(0)  # Webcam
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print("Press 'q' to quit")
    
    frame_count = 0
    emotion_counts = {emotion: 0 for emotion in EMOTION_CLASSES}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detect_faces(frame, face_cascade)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Predict emotion
                emotion, confidence, _ = predict_emotion(model, face_roi)
                
                # Update emotion counts
                if emotion != "Unknown":
                    emotion_counts[emotion] += 1
                
                # Draw emotion label
                draw_emotion_label(frame, x, y, w, h, emotion, confidence * 100)
        
        # Add frame number
        cv2.putText(
            frame, 
            f"Frame: {frame_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Add emotion statistics
        if sum(emotion_counts.values()) > 0:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            cv2.putText(
                frame, 
                f"Most common: {dominant_emotion}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        # Write to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if show_display:
            cv2.imshow('Emotion Detection', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\n--- Detection Summary ---")
    print(f"Total frames processed: {frame_count}")
    print("Emotion distribution:")
    total_detections = sum(emotion_counts.values())
    for emotion, count in emotion_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {emotion}: {count} ({percentage:.1f}%)")


def process_webcam(model, output_path=None):
    """Process webcam stream for emotion detection"""
    process_video(model, '0', output_path, show_display=True)


def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Inference for Video')
    parser.add_argument('--video', type=str, help='Path to input video file (or "0" for webcam)')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using Train_model.py")
        return
    
    # Load model
    model = load_model(args.model)
    
    # Process video
    if args.video:
        process_video(
            model, 
            args.video, 
            output_path=args.output, 
            show_display=not args.no_display
        )
    else:
        # Default: use webcam
        print("No video specified, using webcam...")
        process_webcam(model, args.output)


if __name__ == "__main__":
    main()
