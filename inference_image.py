"""
Emotion Detection Inference Script for Images
Loads a trained model and performs emotion detection on input images
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


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


def preprocess_image(image_path):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_emotion(model, image_tensor):
    """Predict emotion from image tensor"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    emotion = EMOTION_CLASSES[predicted.item()]
    confidence_score = confidence.item()
    
    return emotion, confidence_score, probabilities[0].cpu().numpy()


def predict_from_image(model, image_path, show_all=False):
    """Predict emotion from an image file"""
    print(f"\nProcessing: {image_path}")
    print("-" * 50)
    
    try:
        image_tensor = preprocess_image(image_path)
        emotion, confidence, all_probs = predict_emotion(model, image_tensor)
        
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
        if show_all:
            print("\nAll emotions:")
            for i, prob in enumerate(all_probs):
                print(f"  {EMOTION_CLASSES[i]}: {prob * 100:.2f}%")
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


def predict_from_folder(model, folder_path):
    """Predict emotions for all images in a folder"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            emotion, confidence = predict_from_image(model, image_path, show_all=True)
            results.append({
                'filename': filename,
                'emotion': emotion,
                'confidence': confidence
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Inference for Images')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--show-all', action='store_true', help='Show all emotion probabilities')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using Train_model.py")
        return
    
    # Load model
    model = load_model(args.model)
    
    # Perform inference
    if args.image:
        predict_from_image(model, args.image, show_all=args.show_all)
    elif args.folder:
        results = predict_from_folder(model, args.folder)
        print(f"\nProcessed {len(results)} images")
    else:
        print("Please provide either --image or --folder argument")
        parser.print_help()


if __name__ == "__main__":
    main()
